"""Standalone, process-isolated worker embedded in NB03.

One worker trains at most a small epoch chunk, writes an atomic resume checkpoint, and exits.
The notebook parent owns Hugging Face uploads and launches the next chunk in a fresh process.
"""

NB03_WORKER_SRC = r'''
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def current_rss_gb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 2**20
    except Exception:
        pass
    return float("nan")


def release_memory():
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def close_dataset(ds):
    base = getattr(ds, "base", ds)
    cache = getattr(base, "_cache", None)
    if not isinstance(cache, dict):
        return
    for rec in list(cache.values()):
        data = getattr(rec, "data", None)
        try:
            if hasattr(data, "close"):
                data.close()
            mmap = getattr(data, "_mmap", None)
            if mmap is not None and not getattr(mmap, "closed", False):
                mmap.close()
        except Exception:
            pass
    cache.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fold", required=True, type=int)
    args = ap.parse_args()

    ctx = json.loads(Path(args.context).read_text(encoding="utf-8"))
    cfg = ctx["cfg"]
    work = Path(ctx["work"])
    data = Path(ctx["data"])
    out = work / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    status_path = work / "worker_status.json"
    sys.path.insert(0, str(work))

    from crvs_data import WindowDataset, FS
    from crvs_engine import Trainer, seed_all
    from crvs_losses import MSEOnly, CompositeLoss
    from crvs_metrics import seg_metrics, detect_r_peaks, hrv_from_peaks, peak_detection_scores
    from crvs_models import build_baseline, count_params

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # A child never needs pinned pages or loader subprocesses: its lifetime is deliberately
    # short, and mmap reads are already much faster than GPU compute.
    base_loader = Trainer._loader
    base_validation = Trainer._validation
    base_write_epoch = Trainer._write_epoch

    def isolated_loader(self, *a, **kw):
        loader = base_loader(self, *a, **kw)
        loader.pin_memory = False
        return loader

    def isolated_validation(self, *a, **kw):
        try:
            return base_validation(self, *a, **kw)
        finally:
            release_memory()

    def migrate_metric_names(record):
        for old, new in (("train_mse", "train_loss_mse"),
                         ("val_mse", "val_loss_mse")):
            if old in record:
                record.setdefault(new, record[old])
                record.pop(old, None)

    def portable_write_epoch(self, rec):
        for old_rec in self.state.get("history", []):
            migrate_metric_names(old_rec)
        migrate_metric_names(rec)
        base_write_epoch(self, rec)

    Trainer._loader = isolated_loader
    Trainer._validation = isolated_validation
    Trainer._write_epoch = portable_write_epoch

    class TargetConventionDataset:
        def __init__(self, base, target_01=False):
            self.base = base
            self.target_01 = bool(target_01)
            self.index = base.index
        def __len__(self):
            return len(self.base)
        def set_epoch(self, epoch):
            self.base.set_epoch(epoch)
        def __getitem__(self, i):
            x, y, pk, rr = self.base[i]
            if self.target_01:
                y = (y + 1.0) * 0.5
            return x, y, pk, rr

    class BaselineOutputConvention(torch.nn.Module):
        def __init__(self, base, target_01=False):
            super().__init__()
            self.base = base
            self.target_01 = bool(target_01)
        def forward(self, x):
            pred = self.base(x)
            if self.target_01:
                pred["wave"] = (pred["wave"] + 1.0) * 0.5
                if "aux" in pred:
                    pred["aux"] = [torch.sigmoid(v) for v in pred["aux"]]
            return pred

    W = pd.read_parquet(data / "windows.parquet")
    norm_all = json.loads((data / "norm_stats.json").read_text(encoding="utf-8"))
    expinfo = json.loads((data / "experiments.json").read_text(encoding="utf-8"))
    experiments = expinfo["experiments"]
    rec_dir = data / "recordings"

    sub = W[W["scenario_canon"].isin(experiments[args.experiment])]
    te_g, va_g = args.fold % cfg["N_FOLDS"], (args.fold + 1) % cfg["N_FOLDS"]
    tri = sub[~sub["fold_group"].isin([te_g, va_g])]
    vai = sub[(sub["fold_group"] == va_g) & sub["no_overlap"]]
    tei = sub[(sub["fold_group"] == te_g) & sub["no_overlap"]]
    if set(tri["subject"]) & set(tei["subject"]):
        raise RuntimeError("SUBJECT LEAK")
    norm = norm_all.get(f"{args.experiment}|{args.fold}")
    if norm is None:
        raise RuntimeError(f"no normalisation stats for {args.experiment}|{args.fold}")
    channel_idx = [expinfo["channels"].index(c) for c in cfg["CHANNELS"]]
    sub_norm = {"mean": [norm["mean"][i] for i in channel_idx],
                "std": [norm["std"][i] for i in channel_idx]}

    def make_dataset(frame, augment):
        base = WindowDataset(rec_dir, frame, sub_norm, cfg["CHANNELS"], augment=augment,
                             seed=cfg["SEED"] + args.fold)
        return TargetConventionDataset(base, cfg["TARGET_01"])

    tr_ds, va_ds, te_ds = (make_dataset(tri, True), make_dataset(vai, False),
                            make_dataset(tei, False))

    def evaluate(Y, P):
        rows = []
        idx_frame = tei.reset_index(drop=True)
        subjects = idx_frame["subject"].to_numpy()
        scenarios = idx_frame["scenario_canon"].to_numpy()
        for i in range(len(Y)):
            metric = seg_metrics(Y[i], P[i], FS)
            metric["subject"] = subjects[i] if i < len(subjects) else "?"
            metric["scenario"] = scenarios[i] if i < len(scenarios) else "?"
            rows.append(metric)
        dfw = pd.DataFrame(rows)
        dfw.to_parquet(out / "metrics_windows.parquet", index=False)
        numeric = [c for c in dfw.columns if dfw[c].dtype.kind in "fi"]
        agg = {c: float(dfw[c].mean()) for c in numeric}
        agg.update({c + "_std": float(dfw[c].std()) for c in numeric})
        hr_rows = []
        for rec_id, group in idx_frame.assign(_row=np.arange(len(idx_frame))).groupby("rec_id"):
            group = group.sort_values("start")
            pos = group["_row"].to_numpy()
            if len(pos) < 2:
                continue
            yg, pg = np.concatenate(Y[pos]), np.concatenate(P[pos])
            gt_hrv = hrv_from_peaks(detect_r_peaks(yg, FS), FS)
            pr_hrv = hrv_from_peaks(detect_r_peaks(pg, FS), FS)
            peak = peak_detection_scores(yg, pg, FS)
            hr_rows.append({"rec_id": rec_id, "subject": str(group["subject"].iloc[0]),
                            "scenario": str(group["scenario_canon"].iloc[0]),
                            **{f"gt_{k}": v for k, v in gt_hrv.items()},
                            **{f"pr_{k}": v for k, v in pr_hrv.items()}, **peak})
        dfh = pd.DataFrame(hr_rows)
        if len(dfh):
            dfh.to_parquet(out / "metrics_recordings.parquet", index=False)
            numeric = [c for c in dfh.columns if dfh[c].dtype.kind in "fi"]
            dfh.groupby("subject", as_index=False)[numeric].mean().to_parquet(
                out / "metrics_subjects.parquet", index=False)
            for key in ("F1", "precision", "recall", "accuracy", "missed_rate",
                        "timing_err_ms_median"):
                if key in dfh:
                    agg["peak_" + key] = float(dfh[key].mean())
            for key in ("mean_hr_bpm", "rmssd_ms"):
                if f"gt_{key}" in dfh and f"pr_{key}" in dfh:
                    agg["MAE_" + key] = float(
                        (dfh[f"gt_{key}"] - dfh[f"pr_{key}"]).abs().mean())
        return agg

    tr = None
    try:
        print(f"[isolated worker] {args.run_id} | pid={os.getpid()} | "
              f"start RSS={current_rss_gb():.2f} GB", flush=True)
        print(f"  train {len(tr_ds):,} | val {len(va_ds):,} | test {len(te_ds):,}",
              flush=True)
        seed_all(cfg["SEED"] + args.fold)
        base = build_baseline(args.model, in_ch=len(cfg["CHANNELS"]), out_ch=1,
                              base=cfg["BASE"], levels=cfg["LEVELS"])
        model = BaselineOutputConvention(base, cfg["TARGET_01"])
        loss_fn = MSEOnly() if cfg["LOSS"] == "mse" else CompositeLoss()
        run_config = {"protocol_id": cfg["PROTOCOL_ID"], "experiment": args.experiment,
                      "model": args.model, "fold": args.fold, "epochs": cfg["EPOCHS"],
                      "channels": cfg["CHANNELS"], "target_01": cfg["TARGET_01"],
                      "base": cfg["BASE"], "levels": cfg["LEVELS"], "loss": cfg["LOSS"],
                      "lr": cfg["LR"], "batch": cfg["BATCH"],
                      "seed": cfg["SEED"] + args.fold,
                      "data_index_sha256": ctx["data_hash"],
                      "library_sha256": ctx["module_hashes"],
                      "train_subjects": sorted(map(str, tri["subject"].unique())),
                      "val_subjects": sorted(map(str, vai["subject"].unique())),
                      "test_subjects": sorted(map(str, tei["subject"].unique()))}
        tr = Trainer(model, loss_fn, out, args.run_id, sync=None, lr=cfg["LR"],
                     weight_decay=cfg["WEIGHT_DECAY"], epochs=cfg["EPOCHS"],
                     patience=cfg["PATIENCE"], batch_size=cfg["BATCH"],
                     num_workers=0, amp=cfg["AMP"], multi_gpu=cfg["MULTI_GPU"],
                     log_every=cfg["LOG_EVERY"],
                     checkpoint_every_steps=cfg["CHECKPOINT_EVERY_STEPS"],
                     checkpoint_every_s=cfg["CHECKPOINT_EVERY_S"],
                     seed=cfg["SEED"] + args.fold,
                     require_dual_gpu=cfg["REQUIRE_DUAL_T4"], run_config=run_config)
        tr.load()
        for rec in tr.state.get("history", []):
            migrate_metric_names(rec)
        if tr.state.get("stop_reason") in ("host_memory_guard", "process_chunk_boundary"):
            tr.state.pop("stop_reason", None)
            if str(tr.state.get("last_error", "")).startswith("HostMemoryGuard:"):
                tr.state.pop("last_error", None)

        full_epochs = int(cfg["EPOCHS"])
        start_epoch = int(tr.state.get("epoch", 0))
        already_finished = (start_epoch >= full_epochs or
                            tr.state.get("stop_reason") == "early_stopping")
        if not already_finished:
            chunk_end = min(full_epochs, start_epoch + int(cfg["EPOCHS_PER_PROCESS"]))
            print(f"  isolated epoch chunk: {start_epoch + 1}..{chunk_end} / {full_epochs}",
                  flush=True)
            tr.epochs = chunk_end
            tr.fit(tr_ds, va_ds)

        naturally_finished = (int(tr.state.get("epoch", 0)) >= full_epochs or
                              tr.state.get("stop_reason") == "early_stopping")
        if not naturally_finished:
            tr.epochs = full_epochs
            tr.state["done"] = False
            tr.state.pop("finished_utc", None)
            tr.state["stop_reason"] = "process_chunk_boundary"
            tr.save("state", reason="process-chunk-boundary")
            atomic_json(status_path, {"status": "chunk_complete", "run_id": args.run_id,
                                      "epoch": int(tr.state["epoch"]),
                                      "rss_gb": current_rss_gb(),
                                      "pid": os.getpid(),
                                      "finished_utc": datetime.now(timezone.utc).isoformat()})
            print(f"[isolated worker] chunk saved at epoch {tr.state['epoch']}; exiting", flush=True)
            return 0

        tr.epochs = full_epochs
        Y, P = tr.predict(te_ds)
        agg = evaluate(Y, P)
        keep = min(200, len(Y))
        sel = np.linspace(0, len(Y) - 1, keep).astype(int)
        np.savez_compressed(out / "preds_sample.npz", y=Y[sel].astype(np.float32),
                            p=P[sel].astype(np.float32),
                            subject=tei["subject"].to_numpy()[sel].astype(str))
        budget = ctx.get("budget_by_model", {}).get(args.model, {})
        summary = {"run_id": args.run_id, "protocol_id": cfg["PROTOCOL_ID"],
                   "config_hash": tr.config_hash,
                   "memory_runtime_version": ctx["memory_runtime_version"],
                   "process_isolated": True, "epochs_per_process": cfg["EPOCHS_PER_PROCESS"],
                   "data_loader_workers": 0, "pin_memory": False,
                   "experiment": args.experiment, "model": args.model, "fold": args.fold,
                   "channels": cfg["CHANNELS"], "target_01": cfg["TARGET_01"],
                   "loss": cfg["LOSS"], "epochs_run": tr.state["epoch"],
                   "best_epoch": tr.state["best_epoch"], "best_val": tr.state["best"],
                   "params": count_params(tr.raw_model),
                   "gflops_per_window": budget.get("gflops_per_window"),
                   "forward_ms_batch4": budget.get("forward_ms_batch4"),
                   "n_train": len(tr_ds), "n_val": len(va_ds), "n_test": len(te_ds),
                   "test_subjects": sorted(map(str, tei["subject"].unique())),
                   "metrics": agg, "finished_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(out / "summary.json", summary)
        atomic_json(status_path, {"status": "run_complete", "run_id": args.run_id,
                                  "epoch": int(tr.state["epoch"]), "metrics": agg,
                                  "rss_gb": current_rss_gb(), "pid": os.getpid(),
                                  "finished_utc": datetime.now(timezone.utc).isoformat()})
        print(f"  --> CC_t {agg['CC_temporal']:.2f}  CC_s {agg['CC_spectral']:.2f}  "
              f"MAE {agg['MAE']:.5f}  MSE {agg['MSE']:.5f}", flush=True)
        return 0
    except KeyboardInterrupt:
        if tr is not None and getattr(tr, "_active", False):
            try:
                tr.emergency_checkpoint()
            except Exception:
                pass
        atomic_json(status_path, {"status": "interrupted", "run_id": args.run_id,
                                  "rss_gb": current_rss_gb(), "pid": os.getpid(),
                                  "finished_utc": datetime.now(timezone.utc).isoformat()})
        return 130
    except Exception as exc:
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        atomic_json(status_path, {"status": "failed", "run_id": args.run_id,
                                  "error": f"{type(exc).__name__}: {exc}",
                                  "rss_gb": current_rss_gb(), "pid": os.getpid(),
                                  "finished_utc": datetime.now(timezone.utc).isoformat()})
        traceback.print_exc()
        return 1
    finally:
        for ds in (locals().get("tr_ds"), locals().get("va_ds"), locals().get("te_ds")):
            if ds is not None:
                close_dataset(ds)
        release_memory()


if __name__ == "__main__":
    raise SystemExit(main())
'''
