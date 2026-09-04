"""Standalone, process-isolated worker embedded in NB04.

One worker trains at most a small epoch chunk, writes an atomic resume checkpoint, and exits.
The notebook parent owns Hugging Face uploads and launches the next chunk in a fresh process.
"""

NB04_WORKER_SRC = r'''
import argparse
import gc
import json
import os
import sys
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
    cache = getattr(ds, "_cache", None)
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
    ap.add_argument("--variant", required=True)
    ap.add_argument("--fold", required=True, type=int)
    args = ap.parse_args()

    ctx = json.loads(Path(args.context).read_text(encoding="utf-8"))
    cfg = ctx["cfg"]
    variants = ctx["variants"]
    spec = variants[args.variant]
    work = Path(ctx["work"])
    data = Path(ctx["data"])
    out = work / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    status_path = Path(ctx.get("status_path", work / "worker_status.json"))
    sys.path.insert(0, str(work))

    from crvs_cmnet import build_cmnet
    from crvs_data import WindowDataset, FS
    from crvs_engine import Trainer, seed_all
    from crvs_losses import CompositeLoss, MSEOnly
    from crvs_metrics import seg_metrics, detect_r_peaks, hrv_from_peaks, peak_detection_scores
    from crvs_models import build_baseline, count_params

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Worker lifetime is deliberately short. Avoid loader subprocesses and pinned host pages;
    # mmap reads are already faster than the model compute for these one-dimensional windows.
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
        # Keep evaluation MSE (`val_MSE`) distinct from the MSE loss component.
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

    W = pd.read_parquet(data / "windows.parquet")
    norm_all = json.loads((data / "norm_stats.json").read_text(encoding="utf-8"))
    expinfo = json.loads((data / "experiments.json").read_text(encoding="utf-8"))
    experiments = expinfo["experiments"]
    rec_dir = data / "recordings"

    def split_for(exp, fold, n_folds=None):
        n_folds = n_folds or cfg["N_FOLDS"]
        if exp == "D_loso":
            ids = expinfo.get("loso_values", sorted(map(int, W["loso_id"].unique())))
            fold = int(fold)
            vid = ids[(ids.index(fold) + 1) % len(ids)]
            subset = W[W["scenario_canon"].isin(experiments["C_all5"])]
            train = subset[~subset["loso_id"].isin([fold, vid])]
            val = subset[(subset["loso_id"] == vid) & subset["no_overlap"]]
            test = subset[(subset["loso_id"] == fold) & subset["no_overlap"]]
            return train, val, test
        if exp.startswith("F_cross:"):
            test_scenario = exp.split(":", 1)[1]
            scenarios = expinfo.get("cross_scenarios", list(experiments["C_all5"]))
            val_scenario = scenarios[(scenarios.index(test_scenario) + 1) % len(scenarios)]
            train = W[~W["scenario_canon"].isin([test_scenario, val_scenario])]
            val = W[(W["scenario_canon"] == val_scenario) & W["no_overlap"]]
            test = W[(W["scenario_canon"] == test_scenario) & W["no_overlap"]]
            return train, val, test
        subset = W[W["scenario_canon"].isin(experiments[exp])]
        test_group, val_group = fold % n_folds, (fold + 1) % n_folds
        train = subset[~subset["fold_group"].isin([test_group, val_group])]
        val = subset[(subset["fold_group"] == val_group) & subset["no_overlap"]]
        test = subset[(subset["fold_group"] == test_group) & subset["no_overlap"]]
        if set(train["subject"]) & set(test["subject"]):
            raise RuntimeError("SUBJECT LEAK")
        return train, val, test

    train_index, val_index, test_index = split_for(args.experiment, args.fold)
    if not args.experiment.startswith("F_cross:"):
        if set(train_index["subject"]) & set(test_index["subject"]):
            raise RuntimeError("SUBJECT LEAK train/test")
        if set(val_index["subject"]) & set(test_index["subject"]):
            raise RuntimeError("SUBJECT LEAK val/test")
    norm_key = (f"{args.experiment}|0" if args.experiment.startswith("F_cross:")
                else f"{args.experiment}|{args.fold}")
    norm = norm_all.get(norm_key)
    if norm is None:
        raise RuntimeError(f"no normalisation stats for {norm_key} -- re-run NB02 v2")
    channels = spec["channels"]
    channel_idx = [expinfo["channels"].index(c) for c in channels]
    sub_norm = {"mean": [norm["mean"][i] for i in channel_idx],
                "std": [norm["std"][i] for i in channel_idx]}

    def make_dataset(frame, augment):
        return WindowDataset(rec_dir, frame, sub_norm, channels, augment=augment,
                             seed=cfg["SEED"] + args.fold)

    train_ds = make_dataset(train_index, True)
    val_ds = make_dataset(val_index, False)
    test_ds = make_dataset(test_index, False)

    def make_model():
        if spec["kind"] == "baseline":
            return build_baseline(spec["model"], in_ch=len(channels), out_ch=1,
                                  base=64, levels=cfg["LEVELS"])
        return build_cmnet(in_ch=len(channels), base=cfg["BASE"], levels=cfg["LEVELS"],
                           d_ssm=cfg["D_SSM"], ssm_blocks=cfg["SSM_BLOCKS"],
                           d_state=cfg["D_STATE"], bottleneck=spec.get("bottleneck", "ssm"),
                           use_wavelet=spec.get("wavelet", True),
                           multitask=spec.get("multitask", True),
                           use_film=spec.get("film", True), dropout=cfg["DROPOUT"])

    def evaluate(Y, P):
        frame = test_index.reset_index(drop=True)
        subjects = frame["subject"].to_numpy()
        scenarios = frame["scenario_canon"].to_numpy()
        rows = []
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
        recording_rows = []
        for rec_id, group in frame.assign(_row=np.arange(len(frame))).groupby("rec_id"):
            group = group.sort_values("start")
            pos = group["_row"].to_numpy()
            if len(pos) < 2:
                continue
            truth, pred = np.concatenate(Y[pos]), np.concatenate(P[pos])
            true_hrv = hrv_from_peaks(detect_r_peaks(truth, FS), FS)
            pred_hrv = hrv_from_peaks(detect_r_peaks(pred, FS), FS)
            recording_rows.append({
                "rec_id": rec_id, "subject": str(group["subject"].iloc[0]),
                "scenario": str(group["scenario_canon"].iloc[0]),
                **{f"gt_{k}": v for k, v in true_hrv.items()},
                **{f"pr_{k}": v for k, v in pred_hrv.items()},
                **peak_detection_scores(truth, pred, FS),
            })
        dfr = pd.DataFrame(recording_rows)
        if len(dfr):
            dfr.to_parquet(out / "metrics_recordings.parquet", index=False)
            numeric = [c for c in dfr.columns if dfr[c].dtype.kind in "fi"]
            dfr.groupby("subject", as_index=False)[numeric].mean().to_parquet(
                out / "metrics_subjects.parquet", index=False)
            for key in ("F1", "precision", "recall", "accuracy", "missed_rate",
                        "timing_err_ms_median"):
                if key in dfr:
                    agg["peak_" + key] = float(dfr[key].mean())
            for key in ("mean_hr_bpm", "rmssd_ms"):
                if f"gt_{key}" in dfr and f"pr_{key}" in dfr:
                    agg["MAE_" + key] = float(
                        (dfr[f"gt_{key}"] - dfr[f"pr_{key}"]).abs().mean())
        return agg

    trainer = None
    try:
        print(f"[isolated worker] {args.run_id} | pid={os.getpid()} | "
              f"start RSS={current_rss_gb():.2f} GB", flush=True)
        print(f"  in_ch {len(channels)} | train {len(train_ds):,} | val {len(val_ds):,} | "
              f"test {len(test_ds):,}", flush=True)
        seed_all(cfg["SEED"] + args.fold)
        model = make_model()
        loss_fn = (CompositeLoss(
            **{f"w_{k}": v for k, v in cfg["W"].items()},
            huber_delta=cfg["HUBER_DELTA"], peak_weight=cfg["PEAK_WEIGHT"])
            if spec["loss"] == "composite" else MSEOnly())
        run_config = {
            "experiment": args.experiment, "variant": args.variant, "fold": args.fold,
            "epochs": cfg["ACTIVE_EPOCHS"], "spec": spec, "base": cfg["BASE"],
            "d_ssm": cfg["D_SSM"], "ssm_blocks": cfg["SSM_BLOCKS"],
            "d_state": cfg["D_STATE"], "levels": cfg["LEVELS"],
            "dropout": cfg["DROPOUT"], "loss_weights": cfg["W"],
            "lr": cfg["LR"], "batch": cfg["BATCH"],
            "monitor": cfg["MONITOR"], "monitor_mode": cfg["MONITOR_MODE"],
            "min_epochs": cfg["MIN_EPOCHS"],
            "seed": cfg["SEED"] + args.fold,
            "data_index_sha256": ctx["data_hash"],
            "library_sha256": ctx["module_hashes"],
            "train_subjects": sorted(map(str, train_index["subject"].unique())),
            "val_subjects": sorted(map(str, val_index["subject"].unique())),
            "test_subjects": sorted(map(str, test_index["subject"].unique())),
        }
        trainer = Trainer(
            model, loss_fn, out, args.run_id, sync=None, lr=cfg["LR"],
            weight_decay=cfg["WEIGHT_DECAY"], epochs=cfg["ACTIVE_EPOCHS"],
            patience=cfg["PATIENCE"], batch_size=cfg["BATCH"], num_workers=0,
            amp=cfg["AMP"], multi_gpu=cfg["MULTI_GPU"], log_every=cfg["LOG_EVERY"],
            checkpoint_every_steps=cfg["CHECKPOINT_EVERY_STEPS"],
            checkpoint_every_s=cfg["CHECKPOINT_EVERY_S"], seed=cfg["SEED"] + args.fold,
            require_dual_gpu=cfg["REQUIRE_DUAL_T4"], run_config=run_config,
            monitor=cfg["MONITOR"], monitor_mode=cfg["MONITOR_MODE"],
            min_epochs=cfg["MIN_EPOCHS"])
        trainer.load()
        for rec in trainer.state.get("history", []):
            migrate_metric_names(rec)
        if trainer.state.get("stop_reason") == "process_chunk_boundary":
            trainer.state.pop("stop_reason", None)

        full_epochs = int(cfg["ACTIVE_EPOCHS"])
        start_epoch = int(trainer.state.get("epoch", 0))
        already_finished = (start_epoch >= full_epochs or
                            trainer.state.get("stop_reason") == "early_stopping")
        if not already_finished:
            chunk_end = min(full_epochs, start_epoch + int(cfg["EPOCHS_PER_PROCESS"]))
            print(f"  isolated epoch chunk: {start_epoch + 1}..{chunk_end} / {full_epochs}",
                  flush=True)
            trainer.epochs = chunk_end
            trainer.fit(train_ds, val_ds)

        naturally_finished = (int(trainer.state.get("epoch", 0)) >= full_epochs or
                              trainer.state.get("stop_reason") == "early_stopping")
        if not naturally_finished:
            trainer.epochs = full_epochs
            trainer.state["done"] = False
            trainer.state.pop("finished_utc", None)
            trainer.state["stop_reason"] = "process_chunk_boundary"
            trainer.save("state", reason="process-chunk-boundary")
            atomic_json(status_path, {
                "status": "chunk_complete", "run_id": args.run_id,
                "epoch": int(trainer.state["epoch"]), "rss_gb": current_rss_gb(),
                "pid": os.getpid(), "finished_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(f"[isolated worker] chunk saved at epoch {trainer.state['epoch']}; exiting",
                  flush=True)
            return 0

        trainer.epochs = full_epochs
        Y, P = trainer.predict(test_ds)
        agg = evaluate(Y, P)
        keep = min(200, len(Y))
        sel = np.linspace(0, len(Y) - 1, keep).astype(int)
        np.savez_compressed(out / "preds_sample.npz", y=Y[sel].astype(np.float32),
                            p=P[sel].astype(np.float32),
                            subject=test_index["subject"].to_numpy()[sel].astype(str))
        budget = ctx.get("budget_by_variant", {}).get(args.variant, {})
        summary = {
            "run_id": args.run_id, "config_hash": trainer.config_hash,
            "memory_runtime_version": ctx["memory_runtime_version"],
            "process_isolated": True, "epochs_per_process": cfg["EPOCHS_PER_PROCESS"],
            "data_loader_workers": 0, "pin_memory": False,
            "experiment": args.experiment, "variant": args.variant, "fold": args.fold,
            "spec": spec, "params": count_params(trainer.raw_model),
            "epochs_run": trainer.state["epoch"], "best_epoch": trainer.state["best_epoch"],
            "monitor": trainer.monitor, "monitor_mode": trainer.monitor_mode,
            "best_monitor": trainer.state["best"],
            # Kept for compatibility with old readers; it now means the selected monitor value.
            "best_val": trainer.state["best"],
            "gflops_per_window": budget.get("gflops_per_window"),
            "forward_ms_batch2": budget.get("fwd_ms_batch2"),
            "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
            "test_subjects": sorted(map(str, test_index["subject"].unique())),
            "metrics": agg, "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(out / "summary.json", summary)
        atomic_json(status_path, {
            "status": "run_complete", "run_id": args.run_id,
            "epoch": int(trainer.state["epoch"]), "metrics": agg,
            "rss_gb": current_rss_gb(), "pid": os.getpid(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"  --> CC_t {agg['CC_temporal']:.2f}  CC_s {agg['CC_spectral']:.2f}  "
              f"MAE {agg['MAE']:.5f}  RRMSE_t {agg['RRMSE_temporal']:.4f}  "
              f"F1 {agg.get('peak_F1', float('nan')):.3f}  "
              f"dRMSSD {agg.get('MAE_rmssd_ms', float('nan')):.1f} ms", flush=True)
        return 0
    except KeyboardInterrupt:
        if trainer is not None and getattr(trainer, "_active", False):
            try:
                trainer.emergency_checkpoint()
            except Exception:
                pass
        atomic_json(status_path, {
            "status": "interrupted", "run_id": args.run_id,
            "rss_gb": current_rss_gb(), "pid": os.getpid(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        return 130
    except Exception as exc:
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        atomic_json(status_path, {
            "status": "failed", "run_id": args.run_id,
            "error": f"{type(exc).__name__}: {exc}", "rss_gb": current_rss_gb(),
            "pid": os.getpid(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        traceback.print_exc()
        return 1
    finally:
        for ds in (locals().get("train_ds"), locals().get("val_ds"),
                   locals().get("test_ds")):
            if ds is not None:
                close_dataset(ds)
        release_memory()


if __name__ == "__main__":
    raise SystemExit(main())
'''
