
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
