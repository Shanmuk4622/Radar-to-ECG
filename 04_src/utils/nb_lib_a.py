"""Shared notebook-builder helpers + embedded library sources (part A).

IMPORTANT CONVENTION: every *_SRC string below is written into a notebook cell wrapped in
r\"\"\"...\"\"\". Therefore the embedded module code must NEVER contain a triple-double-quote.
Use # comments only inside these modules. (Learned the hard way in NB01.)
"""
import json, sys, ast

# --------------------------------------------------------------------------- builder
class NB:
    def __init__(self):
        self.cells = []

    def _src(self, s):
        s = s.strip("\n")
        lines = s.split("\n")
        return [l + "\n" for l in lines[:-1]] + [lines[-1]]

    def md(self, s):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": self._src(s)})

    def code(self, s):
        body = s.strip("\n")
        try:
            ast.parse(body)
        except SyntaxError as e:
            print(f"!! SYNTAX ERROR in code cell {len(self.cells)}: line {e.lineno}: {e.msg}",
                  file=sys.stderr)
            ln = body.split("\n")
            if e.lineno and e.lineno <= len(ln):
                print("   " + ln[e.lineno - 1], file=sys.stderr)
            raise
        self.cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": self._src(body)})

    def write(self, path, accelerator="nvidiaTeslaT4"):
        nb = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11.0",
                                  "mimetype": "text/x-python", "file_extension": ".py",
                                  "codemirror_mode": {"name": "ipython", "version": 3},
                                  "nbconvert_exporter": "python", "pygments_lexer": "ipython3"},
                "kaggle": {"accelerator": accelerator, "dataSources": [],
                           "isInternetEnabled": True, "language": "python",
                           "sourceType": "notebook"},
            },
            "nbformat": 4, "nbformat_minor": 5,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        with open(path, encoding="utf-8") as f:
            back = json.load(f)
        nmd = sum(1 for c in back["cells"] if c["cell_type"] == "markdown")
        nco = len(back["cells"]) - nmd
        print(f"wrote {path}: {len(back['cells'])} cells ({nmd} md, {nco} code), "
              f"accelerator={accelerator}")


# --------------------------------------------------------------------------- hf_sync
HF_SYNC_SRC = r'''
# crvs_sync.py -- conservative, resumable and interrupt-safe Hugging Face sync.
# A folder upload can involve several HTTP requests, so this deliberately schedules far
# fewer than the nominal API limit: one periodic upload per 30 minutes, plus major stages
# and a best-effort final upload on interrupt. All local writes are atomic.
import os, json, time, random, threading, atexit, signal
from collections import deque
from pathlib import Path
from datetime import datetime, timezone

SYNC_VERSION = 2

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=str)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

class RollingLimiter:
    # Limits upload_folder CALLS, not HTTP requests. The low default leaves a wide margin.
    def __init__(self, calls_per_hour=24, min_gap_s=20):
        self.limit = max(1, int(calls_per_hour)); self.min_gap = float(min_gap_s)
        self.times = deque(); self.lock = threading.Lock()
    def take(self, timeout=1800):
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                now = time.monotonic()
                while self.times and now - self.times[0] >= 3600:
                    self.times.popleft()
                gap = self.min_gap - (now - self.times[-1]) if self.times else 0.0
                window = 3600 - (now - self.times[0]) if len(self.times) >= self.limit else 0.0
                wait = max(0.0, gap, window)
                if wait <= 0:
                    self.times.append(now); return True
            if time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 10.0))

class HFSync:
    def __init__(self, repo_id, local_dir, token, repo_type="dataset", private=False,
                 run_id="run", push_interval_s=1800, max_upload_calls_hour=24,
                 retry_max=6, verbose=True):
        from huggingface_hub import HfApi
        if not token:
            raise RuntimeError("HF_TOKEN is missing. Add it under Kaggle > Add-ons > Secrets.")
        self.api = HfApi(token=token); self.token = token
        self.repo_id = repo_id; self.repo_type = repo_type; self.private = private
        self.run_id = run_id
        self.local = Path(local_dir); self.local.mkdir(parents=True, exist_ok=True)
        self.interval = max(300, int(push_interval_s))
        self.limiter = RollingLimiter(max_upload_calls_hour)
        self.retry_max = retry_max; self.verbose = verbose
        self._last_push = time.time(); self._dirty = threading.Event()
        self._force = threading.Event(); self._wake = threading.Event()
        self._stop = threading.Event(); self._upload_lock = threading.Lock()
        self._log_lock = threading.Lock(); self._dirty_lock = threading.Lock()
        self._dirty_generation = 0; self._before_final = None
        self._pushes = 0; self._failures = 0; self._closed = False
        self.history = self.local / "sync_history.jsonl"
        self.state_path = self.local / "sync_state.json"
        self._ensure_repo(); self._install_handlers()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hf-uploader")
        self._thread.start()
        self.log("sync_started", repo=self.repo_id, private=self.private,
                 interval_s=self.interval, sync_version=SYNC_VERSION)

    def _ensure_repo(self):
        from huggingface_hub import create_repo
        create_repo(self.repo_id, repo_type=self.repo_type, private=self.private,
                    exist_ok=True, token=self.token)

    @property
    def url(self):
        kind = "datasets/" if self.repo_type == "dataset" else ""
        return "https://huggingface.co/" + kind + self.repo_id

    def recently_pushed(self, seconds=10):
        return self._pushes > 0 and (time.time() - self._last_push) <= float(seconds)

    def log(self, event, _mark_dirty=True, **kw):
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "run": self.run_id, "event": event}
        rec.update(kw)
        try:
            with self._log_lock, open(self.history, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n"); f.flush()
        except Exception:
            pass
        if _mark_dirty:
            self._touch()
        if self.verbose and event not in ("heartbeat",):
            print("  [" + event + "] " + " ".join(f"{k}={v}" for k, v in kw.items()))

    def _touch(self):
        with self._dirty_lock:
            self._dirty_generation += 1; self._dirty.set()

    def mark_dirty(self, reason=None):
        if reason:
            self.log("dirty", reason=reason)
        else:
            self._touch()

    def save_state(self, state):
        atomic_json(self.state_path, state); self._touch()

    def load_state(self, default=None):
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception as e:
                self.log("state_read_error", err=type(e).__name__)
        return default if default is not None else {}

    def pull(self, allow_patterns=None, into=None):
        # Download into the real working folder. Call before producing new local files.
        from huggingface_hub import snapshot_download
        target = Path(into or self.local); target.mkdir(parents=True, exist_ok=True)
        try:
            p = snapshot_download(self.repo_id, repo_type=self.repo_type, token=self.token,
                                  local_dir=str(target), allow_patterns=allow_patterns,
                                  max_workers=4)
            self.log("resume_pull_ok", _mark_dirty=False, path=str(p), patterns=allow_patterns)
            return True
        except Exception as e:
            self.log("resume_pull_empty", _mark_dirty=False,
                     err=f"{type(e).__name__}: {str(e)[:240]}")
            return False

    def set_before_final_flush(self, callback):
        # Trainer registers an atomic emergency-checkpoint callback while it is active.
        self._before_final = callback

    def stage_done(self, name, **kw):
        self.log("stage_done", stage=name, **kw)
        self._force.set(); self._wake.set()

    def _run_final_hook(self):
        cb = self._before_final
        if cb is not None:
            try:
                cb()
            except Exception as e:
                self.log("final_checkpoint_error", err=f"{type(e).__name__}: {e}")

    def _do_upload(self, msg):
        from huggingface_hub import upload_folder
        for attempt in range(self.retry_max):
            if not self.limiter.take(timeout=1800):
                self.log("upload_call_limit_timeout"); return False
            try:
                info = upload_folder(folder_path=str(self.local), repo_id=self.repo_id,
                                     repo_type=self.repo_type, token=self.token,
                                     commit_message=msg,
                                     ignore_patterns=["*.tmp", "**/__pycache__/**", ".git*",
                                                      "*.lock", ".cache/**"])
                self._pushes += 1; self._last_push = time.time()
                meta = {"last_push_utc": datetime.now(timezone.utc).isoformat(),
                        "pushes_this_session": self._pushes,
                        "last_commit": str(getattr(info, "oid", "")), "message": msg}
                atomic_json(self.local / "last_push.json", meta)
                self.log("push_ok", _mark_dirty=False, n=self._pushes,
                         commit=meta["last_commit"], msg=msg)
                return True
            except Exception as e:
                self._failures += 1
                wait = min(300, (2 ** attempt) * 5) * (0.7 + 0.6 * random.random())
                self.log("push_retry", attempt=attempt + 1,
                         err=f"{type(e).__name__}: {str(e)[:500]}", sleep=round(wait, 1))
                time.sleep(wait)
        self.log("push_failed_permanently", msg=msg); return False

    def flush(self, final=False, msg=None, force=False, run_final_hook=False):
        if run_final_hook:
            self._run_final_hook()
        if not self._dirty.is_set() and not force:
            return True
        with self._upload_lock:
            if not self._dirty.is_set() and not force:
                return True
            with self._dirty_lock:
                generation = self._dirty_generation
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            label = "final" if final else "checkpoint"
            message = msg or f"{self.run_id} {label} @ {stamp}Z"
            ok = self._do_upload(message)
            if ok:
                self._force.clear()
                with self._dirty_lock:
                    if self._dirty_generation == generation:
                        self._dirty.clear()
            return ok

    def _loop(self):
        while not self._stop.is_set():
            remaining = max(1.0, self.interval - (time.time() - self._last_push))
            self._wake.wait(min(30.0, remaining)); self._wake.clear()
            if self._stop.is_set():
                break
            forced = self._force.is_set()
            due = (time.time() - self._last_push) >= self.interval
            if self._dirty.is_set() and (due or forced):
                try:
                    tag = "major-stage" if forced else "periodic-30min"
                    self.flush(msg=f"{self.run_id} {tag} @ " +
                               datetime.now(timezone.utc).strftime("%H:%M") + "Z")
                except Exception as e:
                    self.log("loop_error", err=f"{type(e).__name__}: {e}")

    def _install_handlers(self):
        def handler(signum, frame):
            self.log("interrupt", signal=int(signum))
            try:
                self.flush(final=True, force=True, run_final_hook=True,
                           msg=f"{self.run_id} interrupted (signal {signum})")
            finally:
                if signum == signal.SIGINT:
                    raise KeyboardInterrupt
                raise SystemExit(128 + int(signum))
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        atexit.register(self.close)

    def close(self):
        if self._closed:
            return
        self._closed = True; self.log("closing")
        self._stop.set(); self._wake.set()
        try:
            self._thread.join(timeout=5)
            self.flush(final=True, force=self._dirty.is_set(), run_final_hook=True)
        except Exception as e:
            print("Final Hugging Face sync failed:", type(e).__name__, e)
'''

# --------------------------------------------------------------------------- data
CRVS_DATA_SRC = r'''
# crvs_data.py -- windowing, folds, normalisation and the torch Dataset.
# Shared by NB03, NB04 and NB05 so every experiment sees byte-identical inputs.
import json, math
import numpy as np
from pathlib import Path

CHANNELS = ["I", "Q", "phi", "dy", "vel", "acc", "amp", "cardiac"]
# One recording is stored as a single UNCOMPRESSED .npy of shape (len(ARRAY_ROWS), n).
# It has to be .npy, not .npz: np.load(..., mmap_mode="r") silently IGNORES mmap_mode on an
# .npz, so every __getitem__ would decompress all 11 arrays to slice 1024 samples out of
# each -- measured at 23 ms per window, which would dominate the GPU time on Kaggle.
ARRAY_ROWS = CHANNELS + ["ecg_norm", "peak_map", "rr_ms"]
ROW = {name: i for i, name in enumerate(ARRAY_ROWS)}
FS       = 128
# Bumped whenever this module changes in a way the notebooks depend on. Every notebook
# asserts it after import, because writing a .py and importing it is NOT idempotent inside
# one kernel: Python caches the module in sys.modules, so a second run silently keeps the
# first version. That is how a stale .npz loader survived a rebuilt notebook once already.
LIB_VERSION = 4
WINDOW   = 1024          # 8.0 s, frozen to Chowdhury et al. 2024 section 2.3.4
HOP_TRAIN = 512          # 50 % overlap on train only
SCENARIOS = ["Resting", "Valsalva", "Apnea", "Tilt-up", "Tilt-down"]

def canon_scenario(s):
    s = str(s).strip().lower()
    for key, out in [("tiltdown", "Tilt-down"), ("tilt_down", "Tilt-down"), ("tilt-down", "Tilt-down"),
                     ("tiltup", "Tilt-up"), ("tilt_up", "Tilt-up"), ("tilt-up", "Tilt-up"),
                     ("valsalva", "Valsalva"), ("apnea", "Apnea"), ("apnoea", "Apnea"),
                     ("rest", "Resting")]:
        if key in s:
            return out
    return str(s)

def range_normalise(x, eps=1e-8):
    # z-score then squash to [-1, 1]; the baseline used [0, 1], we declare the change
    x = np.asarray(x, np.float32)
    sd = float(x.std())
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, np.float32)          # constant input -> 0, not -1
    x = (x - x.mean()) / (sd + eps)
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    x = np.clip(x, lo, hi)
    rng = float(hi - lo)
    if rng < eps:
        return np.zeros_like(x, np.float32)
    return (2.0 * (x - lo) / rng - 1.0).astype(np.float32)

def peak_heatmap(n, peaks, sigma=3.0):
    # Gaussian bumps at each R peak -- the target for the multi-task peak head
    y = np.zeros(n, np.float32)
    if len(peaks) == 0:
        return y
    half = int(math.ceil(3 * sigma))
    g = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2).astype(np.float32)
    for p in np.asarray(peaks, int):
        a, b = max(0, p - half), min(n, p + half + 1)
        y[a:b] = np.maximum(y[a:b], g[a - (p - half): (b - (p - half))])
    return y

def rr_curve(n, peaks, fs=FS, lo_ms=300.0, hi_ms=2000.0):
    # per-sample instantaneous RR interval in ms, linearly interpolated between beats
    out = np.full(n, np.nan, np.float32)
    p = np.asarray(peaks, int)
    if len(p) < 3:
        return np.nan_to_num(out, nan=800.0)
    rr = np.diff(p) / fs * 1000.0
    mid = (p[:-1] + p[1:]) / 2.0
    ok = (rr > lo_ms) & (rr < hi_ms)
    if ok.sum() < 2:
        return np.nan_to_num(out, nan=float(np.median(rr)))
    out = np.interp(np.arange(n), mid[ok], rr[ok]).astype(np.float32)
    return out

_SLOW_WARNED = {"npz": False}

class _Rec:
    # Reads one recording in whichever format is on disk.
    #   .npy (preferred) -- uncompressed, genuinely memory-mapped, ~0.3 ms per window
    #   .npz (legacy)    -- what an earlier NB02 wrote; correct but ~85x slower, because
    #                       np.load ignores mmap_mode on a zip archive and every window
    #                       decompresses all 11 arrays.
    # Both are supported so an existing corpus keeps working without a 400 MB re-upload.
    __slots__ = ("data", "kind")

    def __init__(self, rec_dir, rid):
        d = Path(rec_dir)
        pnpy, pnpz = d / (rid + ".npy"), d / (rid + ".npz")
        if pnpy.exists():
            self.data = np.load(pnpy, mmap_mode="r"); self.kind = "npy"
        elif pnpz.exists():
            self.data = np.load(pnpz); self.kind = "npz"
            if not _SLOW_WARNED["npz"]:
                _SLOW_WARNED["npz"] = True
                print("  note: reading legacy .npz recordings. Correct, but about 85x slower "
                      "per window than .npy -- re-run NB02 to regenerate the corpus and cut "
                      "the data-loading cost.")
        else:
            raise FileNotFoundError(
                f"no recording for '{rid}' in {d} (looked for .npy and .npz). "
                "Either NB02 did not finish, or the snapshot_download allow_patterns in "
                "this notebook do not cover the format NB02 wrote.")

    def rows(self, names, s, e):
        if self.kind == "npy":
            return np.array(self.data[[ROW[n] for n in names], s:e], np.float32)
        return np.stack([np.array(self.data[n][s:e], np.float32) for n in names], 0)

    def one(self, name, s, e):
        if self.kind == "npy":
            return np.array(self.data[ROW[name], s:e], np.float32)
        return np.array(self.data[name][s:e], np.float32)

class WindowDataset:
    # Slices windows on the fly, so changing WINDOW or the overlap never requires
    # re-running NB02.
    def __init__(self, rec_dir, index, norm=None, channels=None, augment=False, seed=0):
        self.rec_dir = Path(rec_dir)
        self.index = index.reset_index(drop=True)
        self.norm = norm
        self.channels = channels or CHANNELS
        self.rows = [ROW[c] for c in self.channels]
        self.augment = augment
        self.seed = int(seed); self.epoch = 0
        self._cache = {}

    def set_epoch(self, epoch):
        # Augmentation is a pure function of (seed, epoch, index). With workers restarted
        # each epoch, an interrupted epoch can replay and skip batches byte-for-byte.
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.index)

    def _rec(self, rid):
        if rid not in self._cache:
            if len(self._cache) > 48:
                self._cache.pop(next(iter(self._cache)))
            self._cache[rid] = _Rec(self.rec_dir, rid)
        return self._cache[rid]

    def __getitem__(self, i):
        import torch
        r = self.index.iloc[i]
        z = self._rec(r["rec_id"])
        s, e = int(r["start"]), int(r["start"]) + WINDOW
        # _Rec.rows / _Rec.one always np.array (copy), never a view into a read-only
        # memmap -- torch.from_numpy on a non-writable array is undefined behaviour.
        x = z.rows(self.channels, s, e)
        if self.norm is not None:
            mu = np.asarray(self.norm["mean"], np.float32)[:, None]
            sd = np.asarray(self.norm["std"], np.float32)[:, None]
            x = (x - mu) / (sd + 1e-6)
        x = np.clip(x, -8.0, 8.0)
        y  = z.one("ecg_norm", s, e)
        pk = z.one("peak_map", s, e)
        rr = z.one("rr_ms", s, e) / 1000.0                           # seconds, O(1) scale
        if self.augment:
            rng = np.random.RandomState(np.random.SeedSequence(
                [self.seed, self.epoch, int(i)]).generate_state(1)[0])
            if rng.rand() < 0.5:
                x = x + rng.randn(*x.shape).astype(np.float32) * 0.01
            if rng.rand() < 0.3:
                g = np.float32(1.0 + 0.1 * rng.randn())
                x = x * g
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(y)[None, :],
                torch.from_numpy(pk)[None, :],
                torch.from_numpy(rr)[None, :])

def compute_norm(rec_dir, index, channels=CHANNELS, max_windows=4000, seed=0):
    # Per-channel mean/std computed on TRAIN WINDOWS ONLY. Computing them over the whole
    # corpus is a classic, invisible source of leakage.
    rng = np.random.RandomState(seed)
    idx = index if len(index) <= max_windows else index.iloc[
        rng.choice(len(index), max_windows, replace=False)]
    n = 0
    s1 = np.zeros(len(channels), np.float64)
    s2 = np.zeros(len(channels), np.float64)
    cache = {}
    rec_dir = Path(rec_dir)
    for _, r in idx.iterrows():
        rid = r["rec_id"]
        if rid not in cache:
            if len(cache) > 48:
                cache.pop(next(iter(cache)))
            cache[rid] = _Rec(rec_dir, rid)
        a, b = int(r["start"]), int(r["start"]) + WINDOW
        x = cache[rid].rows(list(channels), a, b).astype(np.float64)
        s1 += x.sum(1); s2 += (x * x).sum(1); n += x.shape[1]
    mean = s1 / max(n, 1)
    var = np.maximum(s2 / max(n, 1) - mean ** 2, 1e-12)
    return {"mean": mean.tolist(), "std": np.sqrt(var).tolist(),
            "n_samples": int(n), "channels": list(channels)}
'''

# --------------------------------------------------------------------------- metrics
CRVS_METRICS_SRC = r'''
# crvs_metrics.py -- every metric the baseline reports, plus the ones it should have.
import numpy as np
from scipy import signal as ss
from scipy import stats as sstats

def _f(x):
    return np.nan_to_num(np.asarray(x, np.float64), nan=0.0, posinf=0.0, neginf=0.0)

def pearson(a, b):
    a, b = _f(a), _f(b)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def psd(x, fs=128, nperseg=256):
    f, p = ss.welch(_f(x), fs=fs, nperseg=min(nperseg, len(x)))
    return f, p

def seg_metrics(y, yhat, fs=128):
    # One window. Correlations are reported x100 to match the baseline's tables.
    y, yhat = _f(y), _f(yhat)
    mae = float(np.mean(np.abs(y - yhat)))
    mse = float(np.mean((y - yhat) ** 2))
    cct = 100.0 * pearson(y, yhat)
    _, py = psd(y, fs); _, ph = psd(yhat, fs)
    ccs = 100.0 * pearson(py, ph)
    rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, np.float64) ** 2)))
    rr_t = rms(yhat - y) / (rms(y) + 1e-12)
    rr_s = rms(ph - py) / (rms(py) + 1e-12)
    return {"MAE": mae, "MSE": mse, "CC_temporal": cct, "CC_spectral": ccs,
            "RRMSE_temporal": rr_t, "RRMSE_spectral": rr_s,
            "R2": float(1.0 - np.sum((y - yhat) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))}

def detect_r_peaks(x, fs=128, refractory_s=0.25):
    x = _f(x)
    if len(x) < int(2 * fs):
        return np.array([], int)
    ny = fs / 2.0
    sos = ss.butter(4, [5.0 / ny, min(25.0, ny * 0.95) / ny], btype="band", output="sos")
    b = ss.sosfiltfilt(sos, x)
    e = np.convolve(np.diff(b, prepend=b[0]) ** 2,
                    np.ones(max(1, int(0.10 * fs))) / max(1, int(0.10 * fs)), "same")
    thr = np.percentile(e, 98) * 0.35
    pk, _ = ss.find_peaks(e, height=thr, distance=max(1, int(refractory_s * fs)))
    return pk

def hrv_from_peaks(pk, fs=128):
    out = {"n_peaks": int(len(pk)), "mean_rr_ms": np.nan, "sd_rr_ms": np.nan,
           "mean_hr_bpm": np.nan, "sd_hr_bpm": np.nan, "rmssd_ms": np.nan}
    if len(pk) < 4:
        return out
    rr = np.diff(np.asarray(pk, float)) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3:
        return out
    hr = 60000.0 / rr
    out.update(mean_rr_ms=float(rr.mean()), sd_rr_ms=float(rr.std()),
               mean_hr_bpm=float(hr.mean()), sd_hr_bpm=float(hr.std()),
               rmssd_ms=float(np.sqrt(np.mean(np.diff(rr) ** 2))))
    return out

def peak_detection_scores(y, yhat, fs=128, tol_ms=100.0):
    # Match predicted R peaks to ground-truth peaks within a tolerance window.
    gt = detect_r_peaks(y, fs); pr = detect_r_peaks(yhat, fs)
    tol = tol_ms / 1000.0 * fs
    used = np.zeros(len(pr), bool)
    tp = 0
    errs = []
    for g in gt:
        if len(pr) == 0:
            break
        # float, not the int64 that find_peaks returns -- assigning np.inf into an
        # integer array raises OverflowError even when the mask selects nothing.
        d = np.abs(pr - g).astype(np.float64)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            tp += 1; used[j] = True; errs.append((pr[j] - g) / fs * 1000.0)
    fp = int((~used).sum()); fn = int(len(gt) - tp)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec, "F1": f1,
            "accuracy": tp / max(tp + fp + fn, 1),
            "timing_err_ms_median": float(np.median(np.abs(errs))) if errs else np.nan,
            "timing_err_ms_iqr": float(np.subtract(*np.percentile(np.abs(errs), [75, 25])))
                                  if len(errs) > 3 else np.nan,
            "missed_rate": fn / max(len(gt), 1)}

def aggregate(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    out = {}
    for c in df.columns:
        if df[c].dtype.kind in "fi":
            out[c] = float(df[c].mean()); out[c + "_std"] = float(df[c].std())
    return out

def bland_altman(a, b):
    a, b = _f(a), _f(b)
    m = (a + b) / 2.0; d = a - b
    bias = float(d.mean()); sd = float(d.std())
    return {"mean": m, "diff": d, "bias": bias, "sd": sd,
            "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd}

def wilcoxon_holm(groups, better="higher"):
    # Pairwise Wilcoxon signed-rank across folds, Holm-corrected. groups: {name: [values]}
    import itertools
    names = list(groups)
    raw = []
    for a, b in itertools.combinations(names, 2):
        x, y = np.asarray(groups[a], float), np.asarray(groups[b], float)
        n = min(len(x), len(y))
        if n < 3 or np.allclose(x[:n], y[:n]):
            raw.append((a, b, np.nan)); continue
        try:
            p = float(sstats.wilcoxon(x[:n], y[:n]).pvalue)
        except Exception:
            p = np.nan
        raw.append((a, b, p))
    ps = [r[2] for r in raw]
    order = np.argsort([p if np.isfinite(p) else 1.0 for p in ps])
    m = len(ps); adj = [np.nan] * m; run = 0.0
    for k, i in enumerate(order):
        p = ps[i]
        if not np.isfinite(p):
            continue
        run = max(run, (m - k) * p)
        adj[i] = min(1.0, run)
    return [{"a": raw[i][0], "b": raw[i][1], "p": ps[i], "p_holm": adj[i]} for i in range(m)]
'''
