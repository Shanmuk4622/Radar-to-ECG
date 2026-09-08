
# crvs_engine.py -- deterministic dual-GPU engine with mid-epoch recovery and telemetry.
import csv, json, math, time, os, random, shutil, subprocess, threading, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ENGINE_VERSION = 7

def _autocast(device_type, enabled):
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)

def _grad_scaler(device_type, enabled):
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)

def pick_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        return torch.device("cuda"), n, names
    return torch.device("cpu"), 0, []

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=str, allow_nan=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, float) and not math.isfinite(v): return None
    return v

def _append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean = {str(k): _jsonable(v) for k, v in record.items()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean, default=str, allow_nan=False) + "\n"); f.flush()

def _mean_parts(sums, n, prefix):
    return {f"{prefix}_{k}": float(v) / max(int(n), 1) for k, v in sums.items()}

def _system_stats(out_dir):
    d = shutil.disk_usage(Path(out_dir))
    rec = {"disk_free_gb": d.free / 2**30, "disk_used_gb": d.used / 2**30}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rec["process_peak_rss_gb"] = rss / 2**20  # Linux ru_maxrss is KiB
    except Exception:
        pass
    try:
        load = os.getloadavg(); rec.update(cpu_load_1m=load[0], cpu_load_5m=load[1], cpu_load_15m=load[2])
    except Exception:
        pass
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            rec[f"gpu{i}_peak_alloc_gb"] = torch.cuda.max_memory_allocated(i) / 2**30
            rec[f"gpu{i}_peak_reserved_gb"] = torch.cuda.max_memory_reserved(i) / 2**30
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,temperature.gpu,power.draw,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True,
                               timeout=10, check=False)
            for line in q.stdout.strip().splitlines():
                vals = [x.strip() for x in line.split(",")]
                if len(vals) == 5:
                    i = vals[0]
                    for key, val in zip(("util_pct", "temp_c", "power_w", "mem_used_mb"), vals[1:]):
                        try: rec[f"gpu{i}_{key}"] = float(val)
                        except ValueError: pass
        except Exception:
            pass
    return rec

class Trainer:
    def __init__(self, model, loss_fn, out_dir, run_id, sync=None, lr=5e-4, weight_decay=1e-4,
                 epochs=120, patience=20, batch_size=64, num_workers=2, amp=True,
                 multi_gpu=True, grad_clip=1.0, min_lr=1e-6, log_every=25,
                 checkpoint_every_steps=50, checkpoint_every_s=300, seed=42,
                 require_dual_gpu=False, run_config=None, monitor="val_total",
                 monitor_mode="min", min_epochs=0, recovery_lr_factor=0.25,
                 max_numerical_recoveries=3, disable_amp_after_recoveries=2,
                 max_nonfinite_grad_batches=8):
        self.device, self.ngpu, self.gpu_names = pick_device()
        if require_dual_gpu and self.ngpu < 2:
            raise RuntimeError("This training notebook requires Kaggle GPU T4 x2. "
                               "Choose Settings > Accelerator > GPU T4 x2, then restart.")
        self.raw_model = model.to(self.device); self.model = self.raw_model
        if multi_gpu and self.ngpu > 1:
            self.model = nn.DataParallel(self.raw_model)
        self.loss_fn = loss_fn
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id; self.sync = sync
        self.epochs = int(epochs); self.patience = int(patience)
        self.monitor = str(monitor)
        self.monitor_mode = str(monitor_mode).lower()
        if self.monitor_mode not in ("min", "max"):
            raise ValueError("monitor_mode must be 'min' or 'max'")
        self.min_epochs = max(0, int(min_epochs))
        self.recovery_lr_factor = float(recovery_lr_factor)
        if not 0.0 < self.recovery_lr_factor < 1.0:
            raise ValueError("recovery_lr_factor must be between 0 and 1")
        self.max_numerical_recoveries = max(1, int(max_numerical_recoveries))
        self.disable_amp_after_recoveries = max(1, int(disable_amp_after_recoveries))
        self.max_nonfinite_grad_batches = max(1, int(max_nonfinite_grad_batches))
        self.initial_lr = float(lr)
        self.bs = int(batch_size); self.nw = int(num_workers); self.seed = int(seed)
        self.amp = bool(amp and self.device.type == "cuda"); self.grad_clip = grad_clip
        self.opt = torch.optim.AdamW(self.raw_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=self.epochs, eta_min=min_lr)
        self.scaler = _grad_scaler(self.device.type, self.amp)
        self.log_every = max(1, int(log_every))
        self.checkpoint_every_steps = max(1, int(checkpoint_every_steps))
        self.checkpoint_every_s = max(30, int(checkpoint_every_s))
        self.config = dict(run_config or {})
        self.config_hash = hashlib.sha256(json.dumps(
            self.config, sort_keys=True, default=str).encode()).hexdigest()
        self.state = {"schema_version": 5, "engine_version": ENGINE_VERSION,
                      "epoch": 0, "active_epoch": 0, "batch_in_epoch": 0,
                      "global_step": 0,
                      "best": float("inf") if self.monitor_mode == "min" else -float("inf"),
                      "best_epoch": -1, "monitor": self.monitor,
                      "monitor_mode": self.monitor_mode, "min_epochs": self.min_epochs,
                      "bad_epochs": 0, "history": [], "partial": {},
                      "run_id": run_id, "done": False, "config_hash": self.config_hash,
                      "created_utc": datetime.now(timezone.utc).isoformat()}
        self._save_lock = threading.Lock(); self._last_checkpoint = time.time()
        self._active = False
        _atomic_json(self.out / "run_config.json", self.config)
        _atomic_json(self.out / "environment.json", {
            "engine_version": ENGINE_VERSION, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu_count": self.ngpu, "gpu_names": self.gpu_names,
            "amp": self.amp, "python": os.sys.version, "config_hash": self.config_hash,
            "recovery_lr_factor": self.recovery_lr_factor,
            "max_numerical_recoveries": self.max_numerical_recoveries,
            "disable_amp_after_recoveries": self.disable_amp_after_recoveries,
            "max_nonfinite_grad_batches": self.max_nonfinite_grad_batches})

    @property
    def ckpt(self):
        return self.out / "state.pt"

    def _payload(self):
        return {"model": self.raw_model.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                "amp_enabled": bool(self.amp),
                "state": self.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "np_rng": np.random.get_state(), "python_rng": random.getstate(),
                "config": self.config, "config_hash": self.config_hash,
                "saved_utc": datetime.now(timezone.utc).isoformat()}

    def save(self, tag="state", reason="checkpoint"):
        with self._save_lock:
            path = self.out / (tag + ".pt"); tmp = path.with_suffix(".pt.tmp")
            torch.save(self._payload(), tmp); os.replace(tmp, path)
            _atomic_json(self.out / "state.json", self.state)
            self._last_checkpoint = time.time()
        if self.sync:
            self.sync.mark_dirty(f"{self.run_id}:{reason}")

    def emergency_checkpoint(self):
        if self._active:
            self.save("state", reason="interrupt-emergency")

    @staticmethod
    def _resume_config_view(config):
        # Library hashes and the recovery policy may change only to repair the runtime. Model,
        # data, split, loss and optimizer semantics must remain byte-for-byte compatible.
        clean = dict(config or {})
        clean.pop("library_sha256", None)
        clean.pop("recovery_policy", None)
        return json.dumps(clean, sort_keys=True, default=str)

    def _check_resume_config(self, payload):
        got_hash = payload.get("config_hash", payload.get("state", {}).get("config_hash"))
        if not got_hash or got_hash == self.config_hash:
            return
        old = self._resume_config_view(payload.get("config", {}))
        new = self._resume_config_view(self.config)
        if old != new:
            raise RuntimeError("checkpoint training configuration differs from this run. "
                               "Use a new RUN_ID or restore the original configuration.")
        print("  compatible engine/recovery-policy upgrade detected; preserving the run")

    @staticmethod
    def _model_payload_is_finite(payload):
        for value in payload.get("model", {}).values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                return False
        return True

    def _restore_payload(self, payload):
        self.raw_model.load_state_dict(payload["model"], strict=True)
        self.opt.load_state_dict(payload["opt"])
        self.sched.load_state_dict(payload["sched"])
        # A disabled GradScaler intentionally serializes to {}. Older engine-v6
        # checkpoints did not persist the AMP flag, so loading that empty state into
        # a newly enabled scaler raises "source state dict is empty" before the run
        # can resume. Treat an old empty scaler as proof that AMP had been disabled;
        # new checkpoints also carry the explicit flag.
        scaler_state = payload.get("scaler") or {}
        saved_amp = payload.get("amp_enabled")
        if saved_amp is None:
            saved_amp = bool(scaler_state)
        self.amp = bool(saved_amp and self.device.type == "cuda")
        self.scaler = _grad_scaler(self.device.type, self.amp)
        if self.amp and scaler_state:
            self.scaler.load_state_dict(scaler_state)
        self.state = payload["state"]
        if payload.get("torch_rng") is not None:
            torch.set_rng_state(payload["torch_rng"].cpu())
        if payload.get("np_rng") is not None:
            np.random.set_state(payload["np_rng"])
        if payload.get("python_rng") is not None:
            random.setstate(payload["python_rng"])
        if torch.cuda.is_available() and payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng"]])

    def load(self):
        if not self.ckpt.exists():
            return False
        try:
            current = torch.load(self.ckpt, map_location=self.device, weights_only=False)
            self._check_resume_config(current)
            failed_state = dict(current.get("state", {}))
            last_error = str(failed_state.get("last_error", ""))
            numerical_error = ("non-finite" in last_error.lower() or
                               "floatingpointerror" in last_error.lower())
            poisoned_model = not self._model_payload_is_finite(current)
            recovery_needed = numerical_error or poisoned_model
            if recovery_needed:
                prior_count = int(failed_state.get("recovery_count", 0) or 0)
                if prior_count >= self.max_numerical_recoveries:
                    raise RuntimeError(
                        f"numerical recovery limit ({self.max_numerical_recoveries}) reached; "
                        "do not silently finalize this run")
                best_path = self.out / "best.pt"
                if not best_path.exists():
                    raise RuntimeError("numerical checkpoint failure and best.pt is unavailable")
                chosen = torch.load(best_path, map_location=self.device, weights_only=False)
                self._check_resume_config(chosen)
                if not self._model_payload_is_finite(chosen):
                    raise RuntimeError("best.pt also contains non-finite model parameters")
                self._restore_payload(chosen)
                recovery_count = prior_count + 1
                eta_min = float(getattr(self.sched, "eta_min", 1e-6))
                target_lr = max(eta_min, self.initial_lr *
                                (self.recovery_lr_factor ** recovery_count))
                for group in self.opt.param_groups:
                    group["lr"] = min(float(group["lr"]), target_lr)
                if hasattr(self.sched, "base_lrs"):
                    self.sched.base_lrs = [min(float(v), target_lr)
                                           for v in self.sched.base_lrs]
                if hasattr(self.sched, "_last_lr"):
                    self.sched._last_lr = [float(g["lr"]) for g in self.opt.param_groups]
                amp_disabled_by_policy = recovery_count >= self.disable_amp_after_recoveries
                if amp_disabled_by_policy:
                    self.amp = False
                    self.scaler = _grad_scaler(self.device.type, False)
                event = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "reason": last_error or "non-finite model parameters",
                    "failed_epoch": failed_state.get("active_epoch", failed_state.get("epoch")),
                    "failed_batch": failed_state.get("batch_in_epoch", 0),
                    "rollback_epoch": int(self.state.get("epoch", 0)),
                    "rollback_best_epoch": int(self.state.get("best_epoch", -1)),
                    "recovery_count": recovery_count,
                    "effective_lr": float(self.opt.param_groups[0]["lr"]),
                    "amp_disabled": bool(not self.amp),
                    "engine_version": ENGINE_VERSION,
                }
                previous_events = list(failed_state.get("recovery_events", []))
                self.state["recovery_events"] = previous_events + [event]
                self.state["recovery_count"] = recovery_count
                self.state["last_recovered_error"] = event["reason"]
                self.state.pop("last_error", None)
                self.state.pop("stop_reason", None)
                self.state.pop("finished_utc", None)
                self.state.update(done=False, active_epoch=int(self.state.get("epoch", 0)),
                                  batch_in_epoch=0, partial={}, engine_version=ENGINE_VERSION,
                                  config_hash=self.config_hash)
                _append_jsonl(self.out / "recovery_events.jsonl", event)
                self.save("state", reason="numerical-rollback-to-best")
                print(f"  RECOVERED {self.run_id}: epoch "
                      f"{event['failed_epoch']} -> best epoch {event['rollback_best_epoch']}; "
                      f"lr={event['effective_lr']:.2e}; AMP={'on' if self.amp else 'off'}")
            else:
                self._restore_payload(current)
                self.state.update(engine_version=ENGINE_VERSION, config_hash=self.config_hash)
                self.state.pop("last_error", None)
                print(f"  resumed {self.run_id}: completed_epoch={self.state['epoch']}, "
                      f"active_epoch={self.state.get('active_epoch')}, "
                      f"completed_batches={self.state.get('batch_in_epoch', 0)}")
            return True
        except Exception as e:
            raise RuntimeError(f"Checkpoint exists but cannot be resumed safely: "
                               f"{type(e).__name__}: {e}") from e

    def _loader(self, ds, shuffle, epoch=0):
        if len(ds) == 0:
            raise RuntimeError("empty dataset -- check the subject split")
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        drop = bool(shuffle) and len(ds) > self.bs
        gen = torch.Generator(); gen.manual_seed(self.seed + int(epoch) * 1000003)
        return DataLoader(ds, batch_size=min(self.bs, max(len(ds), 1)), shuffle=shuffle,
                          generator=gen, num_workers=self.nw,
                          pin_memory=(self.device.type == "cuda"), drop_last=drop,
                          persistent_workers=False)

    def _step(self, batch, train):
        x, y, pk, rr = [b.to(self.device, non_blocking=True) for b in batch]
        with _autocast(self.device.type, self.amp):
            pred = self.model(x)
            if isinstance(pred, dict) and "aux" in pred and not train:
                pred = {k: v for k, v in pred.items() if k != "aux"}
            loss, parts = self.loss_fn(pred, y, pk, rr)
        return loss, {k: float(v) for k, v in parts.items()}, pred, y, pk, rr

    def _write_batch(self, rec):
        _append_jsonl(self.out / "batch_metrics.jsonl", rec)

    def _validation(self, val_ds, epoch):
        self.model.eval(); sums = {}; n = 0; Y = []; P = []; PK = []; PH = []; RR = []; RH = []
        t0 = time.time()
        with torch.no_grad():
            for batch in self._loader(val_ds, False, epoch):
                loss, parts, pred, y, pk, rr = self._step(batch, False)
                parts = {"total": float(loss), **parts}
                for k, v in parts.items(): sums[k] = sums.get(k, 0.0) + float(v)
                n += 1; Y.append(y.squeeze(1).float().cpu().numpy())
                P.append(pred["wave"].squeeze(1).float().cpu().numpy())
                PK.append(pk.squeeze(1).float().cpu().numpy())
                if "peak" in pred: PH.append(pred["peak"].squeeze(1).float().cpu().numpy())
                RR.append(rr.squeeze(1).float().cpu().numpy())
                if "rr" in pred: RH.append(pred["rr"].squeeze(1).float().cpu().numpy())
        if n == 0: raise RuntimeError("validation loader yielded zero batches")
        y = np.concatenate(Y); p = np.concatenate(P); pk = np.concatenate(PK)
        rec = _mean_parts(sums, n, "val")
        rec["val_seconds"] = time.time() - t0; rec["val_windows"] = len(y)
        rec.update(val_true_mean=float(y.mean()), val_true_std=float(y.std()),
                   val_true_min=float(y.min()), val_true_max=float(y.max()),
                   val_pred_mean=float(p.mean()), val_pred_std=float(p.std()),
                   val_pred_min=float(p.min()), val_pred_max=float(p.max()),
                   val_pred_bias=float((p-y).mean()))
        try:
            from crvs_metrics import seg_metrics, peak_detection_scores, detect_r_peaks, hrv_from_peaks
            global_m = seg_metrics(y.reshape(-1), p.reshape(-1))
            rec.update({"val_" + k: v for k, v in global_m.items()})
            # Per-window waveform diagnostics are compact and retained for every epoch.
            den_y = np.sqrt(np.sum((y - y.mean(1, keepdims=True)) ** 2, axis=1))
            den_p = np.sqrt(np.sum((p - p.mean(1, keepdims=True)) ** 2, axis=1))
            cc = np.sum((y-y.mean(1, keepdims=True))*(p-p.mean(1, keepdims=True)), axis=1) / (den_y*den_p+1e-12)
            rec.update(val_window_CC_temporal_mean=float(100 * np.mean(cc)),
                       val_window_CC_temporal_median=float(100 * np.median(cc)),
                       val_window_CC_temporal_std=float(100 * np.std(cc)),
                       val_window_CC_temporal_p05=float(100 * np.percentile(cc, 5)),
                       val_window_MAE_mean=float(np.mean(np.abs(y-p))),
                       val_window_MSE_mean=float(np.mean((y-p)**2)))
            win = {"epoch": np.full(len(y), epoch + 1), "window": np.arange(len(y)),
                   "mae": np.mean(np.abs(y-p), 1), "mse": np.mean((y-p)**2, 1),
                   "cc_temporal": 100*cc}
            try:
                import pandas as pd
                frame = pd.DataFrame(win)
                if hasattr(val_ds, "index") and len(val_ds.index) == len(frame):
                    for col in ("rec_id", "subject", "scenario_canon", "start"):
                        if col in val_ds.index: frame[col] = val_ds.index[col].to_numpy()
                vd = self.out / "validation_windows"; vd.mkdir(exist_ok=True)
                frame.to_parquet(vd / f"epoch_{epoch+1:04d}.parquet", index=False)
            except Exception as e:
                rec["val_window_table_error"] = f"{type(e).__name__}: {e}"
            # Never concatenate different recordings: that fabricates a beat interval at
            # each boundary. Compute peak/HRV metrics per recording, then macro-average.
            record_rows = []
            if hasattr(val_ds, "index") and len(val_ds.index) == len(y):
                ix = val_ds.index.reset_index(drop=True).assign(_row=np.arange(len(y)))
                for rid, grp in ix.groupby("rec_id"):
                    pos = grp.sort_values("start")["_row"].to_numpy()
                    if len(pos) < 2: continue
                    yg = np.concatenate(y[pos]); pg = np.concatenate(p[pos])
                    peak_s = peak_detection_scores(yg, pg)
                    gt_hrv = hrv_from_peaks(detect_r_peaks(yg))
                    pr_hrv = hrv_from_peaks(detect_r_peaks(pg))
                    rr = {"rec_id": rid, "subject": str(grp["subject"].iloc[0]), **peak_s}
                    for k in ("mean_rr_ms", "sd_rr_ms", "mean_hr_bpm", "sd_hr_bpm", "rmssd_ms"):
                        rr[f"true_{k}"] = gt_hrv[k]; rr[f"pred_{k}"] = pr_hrv[k]
                        rr[f"abs_error_{k}"] = abs(pr_hrv[k]-gt_hrv[k])
                    record_rows.append(rr)
            if record_rows:
                import pandas as pd
                rdf = pd.DataFrame(record_rows)
                rd = self.out / "validation_recordings"; rd.mkdir(exist_ok=True)
                rdf.to_parquet(rd / f"epoch_{epoch+1:04d}.parquet", index=False)
                for k in ("TP", "FP", "FN", "precision", "recall", "F1", "accuracy",
                          "timing_err_ms_median", "timing_err_ms_iqr", "missed_rate"):
                    rec[f"val_wave_peak_{k}"] = float(rdf[k].mean())
                for k in ("mean_rr_ms", "sd_rr_ms", "mean_hr_bpm", "sd_hr_bpm", "rmssd_ms"):
                    rec[f"val_hrv_true_{k}"] = float(rdf[f"true_{k}"].mean())
                    rec[f"val_hrv_pred_{k}"] = float(rdf[f"pred_{k}"].mean())
                    rec[f"val_hrv_abs_error_{k}"] = float(rdf[f"abs_error_{k}"].mean())
        except Exception as e:
            rec["val_signal_metrics_error"] = f"{type(e).__name__}: {e}"
        if PH:
            ph = np.concatenate(PH); prob = 1 / (1 + np.exp(-np.clip(ph, -30, 30)))
            truth = pk >= 0.5; guess = prob >= 0.5
            tp = int(np.sum(truth & guess)); fp = int(np.sum(~truth & guess)); fn = int(np.sum(truth & ~guess))
            prec = tp / max(tp+fp, 1); recall = tp / max(tp+fn, 1)
            rec.update(val_peak_head_TP=tp, val_peak_head_FP=fp, val_peak_head_FN=fn,
                       val_peak_head_precision=prec, val_peak_head_recall=recall,
                       val_peak_head_F1=2*prec*recall/max(prec+recall, 1e-12))
        if RH:
            rr = np.concatenate(RR); rh = np.concatenate(RH); mask = rr > 0
            if np.any(mask):
                rec["val_rr_head_mae_ms"] = float(np.mean(np.abs(rr[mask]-rh[mask]))*1000)
                rec["val_rr_head_rmse_ms"] = float(np.sqrt(np.mean((rr[mask]-rh[mask])**2))*1000)
        return rec

    def _write_epoch(self, rec):
        _append_jsonl(self.out / "epoch_metrics.jsonl", rec)
        # CSV is convenient in Kaggle; JSONL remains the lossless schema-of-record.
        rows = self.state["history"]
        keys = sorted({k for r in rows for k in r})
        tmp = self.out / "epoch_metrics.csv.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for row in rows: w.writerow({k: _jsonable(row.get(k)) for k in keys})
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.out / "epoch_metrics.csv")

    def fit(self, train_ds, val_ds):
        start = int(self.state["epoch"])
        if start >= self.epochs:
            print(f"  {self.run_id} already complete at epoch {start}/{self.epochs}")
            return self.state
        if self.state.get("done"):
            self.state["done"] = False
        self._active = True
        if self.sync: self.sync.set_before_final_flush(self.emergency_checkpoint)
        run_t0 = time.time()
        try:
            for ep in range(start, self.epochs):
                if torch.cuda.is_available():
                    for gpu_i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(gpu_i)
                tl = self._loader(train_ds, True, ep)
                resume_batch = int(self.state.get("batch_in_epoch", 0)) if int(self.state.get("active_epoch", ep)) == ep else 0
                partial = self.state.get("partial", {}) if resume_batch else {}
                sums = {k: float(v) for k, v in partial.get("sums", {}).items()}
                n = int(partial.get("n", 0)); samples = int(partial.get("samples", 0))
                grad_sum = float(partial.get("grad_sum", 0)); clip_events = int(partial.get("clip_events", 0))
                nonfinite_grad_skips = int(partial.get("nonfinite_grad_skips", 0))
                epoch_t0 = time.time(); data_t = 0.0; compute_t = 0.0; last_end = time.time()
                self.state.update(active_epoch=ep, batch_in_epoch=resume_batch, done=False)
                self.model.train()
                for i, batch in enumerate(tl):
                    data_t += time.time() - last_end
                    if i < resume_batch:
                        last_end = time.time(); continue
                    step_t0 = time.time(); self.opt.zero_grad(set_to_none=True)
                    loss, parts, _, _, _, _ = self._step(batch, True)
                    if not torch.isfinite(loss):
                        self.save("state", reason="non-finite-loss")
                        raise FloatingPointError(f"non-finite loss at epoch {ep+1}, batch {i+1}")
                    self.scaler.scale(loss).backward(); self.scaler.unscale_(self.opt)
                    grad = float(torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), self.grad_clip or float("inf")))
                    if not math.isfinite(grad):
                        # GradScaler normally skips this update. Make that behavior explicit,
                        # checkpointable and bounded so a bad batch cannot poison model weights.
                        self.opt.zero_grad(set_to_none=True)
                        self.scaler.update()
                        nonfinite_grad_skips += 1
                        self.state["batch_in_epoch"] = i + 1
                        self.state["partial"] = {
                            "sums": sums, "n": n, "samples": samples,
                            "grad_sum": grad_sum, "clip_events": clip_events,
                            "nonfinite_grad_skips": nonfinite_grad_skips,
                        }
                        self._write_batch({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "run_id": self.run_id, "epoch": ep+1, "batch": i+1,
                            "batches": len(tl), "event": "nonfinite_gradient_skipped",
                            "grad_norm": grad, "lr": self.opt.param_groups[0]["lr"],
                            "amp_scale": float(self.scaler.get_scale()),
                            "skip_count_epoch": nonfinite_grad_skips,
                        })
                        compute_t += time.time() - step_t0
                        if nonfinite_grad_skips >= self.max_nonfinite_grad_batches:
                            raise FloatingPointError(
                                f"{nonfinite_grad_skips} non-finite gradient batches at "
                                f"epoch {ep+1}; rolling back to best checkpoint")
                        last_end = time.time()
                        continue
                    if self.grad_clip and grad > self.grad_clip: clip_events += 1
                    self.scaler.step(self.opt); self.scaler.update()
                    compute_t += time.time() - step_t0
                    values = {"total": float(loss.detach()), **parts}
                    for k, v in values.items(): sums[k] = sums.get(k, 0.0) + float(v)
                    n += 1; samples += int(batch[0].shape[0]); grad_sum += grad
                    self.state["global_step"] = int(self.state.get("global_step", 0)) + 1
                    self.state["batch_in_epoch"] = i + 1
                    self.state["partial"] = {"sums": sums, "n": n, "samples": samples,
                                             "grad_sum": grad_sum, "clip_events": clip_events,
                                             "nonfinite_grad_skips": nonfinite_grad_skips}
                    if (i + 1) % self.log_every == 0 or i + 1 == len(tl):
                        brec = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
                                "epoch": ep+1, "batch": i+1, "batches": len(tl),
                                "global_step": self.state["global_step"], "loss": float(loss),
                                "grad_norm": grad, "lr": self.opt.param_groups[0]["lr"],
                                "amp_scale": float(self.scaler.get_scale()),
                                "windows_per_s": int(batch[0].shape[0])/max(time.time()-step_t0, 1e-9)}
                        brec.update({"loss_"+k: v for k, v in parts.items()}); self._write_batch(brec)
                    due_step = self.state["global_step"] % self.checkpoint_every_steps == 0
                    due_time = time.time() - self._last_checkpoint >= self.checkpoint_every_s
                    if due_step or due_time:
                        self.save("state", reason="mid-epoch")
                    last_end = time.time()
                if n == 0: raise RuntimeError("training loader yielded zero batches")
                self.sched.step(); val = self._validation(val_ds, ep)
                rec = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
                       "epoch": ep+1, "epochs_planned": self.epochs,
                       "global_step": self.state["global_step"], "train_batches": n,
                       "train_windows": samples, "train_grad_norm_mean": grad_sum/max(n,1),
                       "train_grad_clip_events": clip_events,
                       "train_nonfinite_grad_skips": nonfinite_grad_skips,
                       "train_data_seconds": data_t, "train_compute_seconds": compute_t,
                       "train_windows_per_s": samples/max(compute_t, 1e-9),
                       "epoch_seconds": time.time()-epoch_t0,
                       "elapsed_seconds": time.time()-run_t0,
                       "lr": self.opt.param_groups[0]["lr"],
                       "amp_scale": float(self.scaler.get_scale())}
                rec.update(_mean_parts(sums, n, "train")); rec.update(val); rec.update(_system_stats(self.out))
                with torch.no_grad():
                    rec["model_parameter_l2"] = math.sqrt(sum(
                        float(torch.sum(p.detach().float() ** 2)) for p in self.raw_model.parameters()))
                va = float(rec["val_total"])
                if self.monitor not in rec:
                    raise KeyError(f"configured monitor '{self.monitor}' is absent from validation metrics")
                monitored = float(rec[self.monitor])
                if not math.isfinite(monitored):
                    raise FloatingPointError(
                        f"non-finite monitor {self.monitor} at epoch {ep+1}: {monitored}")
                best_so_far = float(self.state["best"])
                improved = ((monitored < best_so_far - 1e-6) if self.monitor_mode == "min"
                            else (monitored > best_so_far + 1e-6))
                if improved:
                    self.state["best"] = monitored; self.state["best_epoch"] = ep+1
                    self.state["bad_epochs"] = 0
                else:
                    self.state["bad_epochs"] = int(self.state.get("bad_epochs", 0)) + 1
                rec.update(improved=bool(improved), monitor=self.monitor,
                           monitor_mode=self.monitor_mode, monitor_value=monitored,
                           best_monitor=float(self.state["best"]),
                           best_val=float(self.state["best"]),
                           best_epoch=int(self.state["best_epoch"]),
                           bad_epochs=int(self.state["bad_epochs"]))
                self.state["epoch"] = ep+1; self.state["active_epoch"] = ep+1
                self.state["batch_in_epoch"] = 0; self.state["partial"] = {}
                self.state["history"].append(rec)
                if improved: self.save("best", reason="new-best-local")
                self.save("state", reason="epoch-complete"); self._write_epoch(rec)
                if self.sync:
                    self.sync.log("epoch", run_id=self.run_id, epoch=ep+1,
                                  train_total=rec.get("train_total"), val_total=va,
                                  cc_t=rec.get("val_CC_temporal"), cc_s=rec.get("val_CC_spectral"),
                                  hr_mae_bpm=rec.get("val_hrv_abs_error_mean_hr_bpm"), improved=improved)
                eta = rec["epoch_seconds"] * max(self.epochs-ep-1, 0) / 3600
                print(f"  ep {ep+1:>3}/{self.epochs} train {rec['train_total']:.5f} "
                      f"val {va:.5f} CCt-win {rec.get('val_window_CC_temporal_mean', float('nan')):.1f} "
                      f"CCt-global {rec.get('val_CC_temporal', float('nan')):.1f} "
                      f"CCs {rec.get('val_CC_spectral', float('nan')):.1f} "
                      f"{'*' if improved else ''} {rec['epoch_seconds']:.0f}s ETA {eta:.1f}h")
                if ((ep + 1) >= self.min_epochs and
                        int(self.state["bad_epochs"]) >= self.patience):
                    self.state["stop_reason"] = "early_stopping"; break
            self.state["done"] = True
            self.state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            self.save("state", reason="run-complete")
            if self.sync: self.sync.log("training_complete", run_id=self.run_id)
            return self.state
        except KeyboardInterrupt:
            # SIGINT normally reaches HFSync first: its final hook already saved and pushed.
            # Avoid a duplicate commit; a directly-raised KeyboardInterrupt still takes this path.
            if time.time() - self._last_checkpoint > 2:
                self.save("state", reason="keyboard-interrupt")
            if self.sync:
                if not self.sync.recently_pushed(5):
                    self.sync.flush(final=True, force=True,
                                    msg=f"{self.run_id} stopped by user", run_final_hook=False)
            raise
        except Exception as e:
            self.state["last_error"] = f"{type(e).__name__}: {e}"
            self.save("state", reason="training-error")
            if self.sync:
                self.sync.log("training_error", run_id=self.run_id, error=self.state["last_error"])
                self.sync.flush(final=True, force=True,
                                msg=f"{self.run_id} error checkpoint", run_final_hook=False)
            raise
        finally:
            self._active = False
            if self.sync: self.sync.set_before_final_flush(None)

    @torch.no_grad()
    def predict(self, ds, max_keep=None):
        bp = self.out / "best.pt"
        if bp.exists():
            d = torch.load(bp, map_location=self.device, weights_only=False)
            self.raw_model.load_state_dict(d["model"], strict=True)
        self.model.eval(); Y = []; P = []
        for batch in self._loader(ds, False, 0):
            x, y, pk, rr = [b.to(self.device, non_blocking=True) for b in batch]
            with _autocast(self.device.type, self.amp): out = self.model(x)
            Y.append(y.squeeze(1).float().cpu().numpy())
            P.append(out["wave"].squeeze(1).float().cpu().numpy())
        Y = np.concatenate(Y); P = np.concatenate(P)
        if max_keep is not None: return Y[:max_keep], P[:max_keep]
        return Y, P
