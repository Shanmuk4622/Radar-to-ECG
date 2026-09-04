#!/usr/bin/env python3
"""Emit 03_baselines.ipynb"""
import sys
from nb_lib_a import NB, HF_SYNC_SRC, CRVS_DATA_SRC, CRVS_METRICS_SRC
from nb_lib_b import CRVS_MODELS_SRC, CRVS_LOSS_SRC, CRVS_ENGINE_SRC
from nb03_worker import NB03_WORKER_SRC

n = NB(); md, code = n.md, n.code

md(r"""
# NB03 — Baselines: the reproduction gate

**Project:** CardioMamba-Net · **Stage:** 3 of 5
`01_verify` → `02_preprocess` → **`03_baselines`** → `04_cardiomamba_train` → `05_evaluate`

---

## Why this notebook exists

Before we are allowed to claim a new architecture wins, we have to show our pipeline can
**reproduce the numbers we are trying to beat**. This notebook re-implements all four networks
from Chowdhury et al. 2024 — FPN-1D, UNet-1D, LinkNet-1D and MultiResLinkNet — trains them on our
data with their objective, and puts our numbers next to theirs.

If MultiResLinkNet does not land near a temporal correlation of **61.9** on RVA combined, something
in our pipeline is wrong and **nothing after this point is trustworthy**. That is the gate.

One deliberate difference from their setup is declared in the paper:

- **Splits are strictly by subject** and test windows do not overlap (see NB02 §"leakage decision").
  Theirs almost certainly leaked. So our reproduction may land *below* their published figures —
  that is the expected direction, and it is the fair comparison for everything that follows.

Everything else is theirs: **1 input channel** (the arctangent-demodulated displacement `dy`),
ECG mapped to **[0, 1]**, 5 levels, 64 filters doubling, **plain MSE**, Adam at 5e-4,
1024-sample windows at 128 Hz.

---

## ⚠️ Accelerator: **GPU T4 × 2**

*Session options → Accelerator → **GPU T4 x2***, Internet **On**, `HF_TOKEN` secret attached.

Before the first NB03 session, run **`02b_cleanup_baselines_hf_once.ipynb`** on CPU and confirm
`RESET COMPLETE`. NB03 checks its remote receipt before starting any uploader or training work.

Before Run All, please click **+ Add Input → Notebook Output** and attach NB02's saved output.
That read-only Kaggle mount is the preferred corpus source; Hugging Face is only the fallback.

Both GPUs are used through `DataParallel`. AMP (mixed precision) is on, which roughly doubles
throughput on T4s and halves memory.

Every logical run is executed in a short-lived child process, five epochs at a time. When a
child exits, Linux reclaims its entire RAM allocation and CUDA context before the next chunk or
model starts. This is intentionally stronger than `gc.collect()` and prevents the checkpoint
serializer's native-memory growth from accumulating across a long Kaggle kernel.

## The run queue — this is how it survives Kaggle

There are **4 models × 4 experiment settings × 5 folds = 80 runs**. That does not fit in one
12-hour session, and it is not supposed to.

The notebook builds one **canonical queue**, checks which runs are already finished on Hugging
Face, and works through as many as fit in `TIME_BUDGET_H`. Worker processes are restarted
automatically; you do not need to restart the notebook between models. Then it pushes and stops cleanly.
**Start a new session and run it again** — it picks up exactly where it left off. Three or four
sessions completes the matrix. Every experiment/model/fold ID is created once and never
recomputed.

There is deliberately no saved quick-training run in this notebook. The forward/backward smoke
test checks all four models without creating a checkpoint; this prevents a 10-epoch trial from
being mixed into the 120-epoch scientific results.

## Cell-by-cell run guide

| Code cell | What runs | Typical time |
|---:|---|---:|
| 1 | Configuration | < 5 s |
| 2 | Imports, dependency, dual-GPU and disk checks | 1–3 min |
| 3 | Write/import the versioned shared libraries | 10–30 s |
| 4 | HF login; restore run summaries and resume markers | 1–5 min |
| 5 | Mount attached NB02 corpus or download fallback | < 1 min mounted; 3–12 min fallback |
| 6 | Forward/backward smoke test all four baselines | 2–8 min |
| 7 | Build the resumable run queue | < 10 s |
| 8 | Define split, dataset, and recording-safe evaluation helpers | < 10 s |
| 9 | Train/evaluate queued runs; checkpoint every epoch and at least every 5 min | up to 11.25 h/session |
| 10 | Assemble the reproduction table | < 1 min |
| 11 | Draw learning curves and comparison figures | 1–5 min |
| 12 | Write report/card and final upload | 2–15 min |

The training cell prints epoch time and ETA after its first epoch. **Full mode is deliberately a
multi-session queue**; each session uses up to 11.25 hours, pushes, and resumes on the next Run All.
""")

md("---\n# 1 · Configuration")

code(r'''
CFG = {
    "SRC_REPO":  "Shanmuk4622/cr-rvs-radar-ecg-processed-v2", # NB02 output
    "DST_REPO":  "Shanmuk4622/cardiomamba-baselines-v2",       # isolated v2 runs
    "HF_PRIVATE": False,
    "RUN_ID":    "nb03_baselines_v3",
    "PROTOCOL_ID": "baseline-v3-subjectwise-5fold-dy-mse-target01",
    "RESET_RECEIPT": "cleanup_receipts/baselines-v2-canonical-reset-20260902.json",
    "REQUIRE_CLEAN_RESET": True,

    "WORK":    "/kaggle/working/nb03",
    "SCRATCH": "/kaggle/temp/nb03",
    "PUSH_INTERVAL_S": 30 * 60,
    "HF_MAX_UPLOADS_HOUR": 24,

    # ---- faithful reproduction of Chowdhury et al. 2024 ----------------------
    "CHANNELS":   ["dy"],        # THEIR input: one arctangent-demodulated displacement channel
    "BASE":       64,            # section 3.1: "initial layer containing 64 filters"
    "LEVELS":     4,             # 5 levels counting the bottleneck
    "LR":         5e-4,          # section 3.1
    "LOSS":       "mse",         # section 3.1: "As a loss function, the MSE function was used"
    "TARGET_01":  True,          # the paper's [0,1] target convention; now actually applied

    # ---- training ------------------------------------------------------------
    "EPOCHS":     120,
    "PATIENCE":   20,            # section 3.1
    "BATCH":      64,
    # With memory-mapped recordings, worker processes add almost no throughput here but
    # repeatedly retain pinned/shared host pages in long Jupyter sessions.  Keep loading in
    # the main process: measured data time is tiny compared with GPU compute, and RAM stays
    # bounded across the 80-run queue.
    "WORKERS":    0,
    "PIN_MEMORY": False,        # avoids a growing pinned-page pool in a long-lived kernel
    "HOST_RAM_WARN_GB": 18.0,   # warn and keep going; never break the experiment queue
    "WEIGHT_DECAY": 1e-4,
    "AMP":        True,
    "MULTI_GPU":  True,
    "REQUIRE_DUAL_T4": True,
    "SEED":       1337,
    "LOG_EVERY":  25,
    # Every observed epoch is well below five minutes and is checkpointed at its end.
    # Avoid serialising another full model two or three times inside the same short epoch.
    "CHECKPOINT_EVERY_STEPS": 1000,
    "CHECKPOINT_EVERY_S": 300,
    "EPOCHS_PER_PROCESS": 5,    # hard OS-level RAM reset at most every five epochs
    "PROCESS_ISOLATION": True,  # each chunk exits; Linux reclaims all worker RAM/CUDA state

    # ---- the queue -----------------------------------------------------------
    "MODELS":      ["fpn", "unet", "linknet", "multireslinknet"],
    "EXPERIMENTS": ["B_rva", "A_resting", "A_valsalva", "A_apnea"],   # B first: it is the headline
    "N_FOLDS":     5,
    "TIME_BUDGET_H": 11.25,      # use almost the whole 12 h session, then push cleanly
}
import json
print(json.dumps(CFG, indent=2))
''')

code(r'''
import os, sys, gc, json, math, time, warnings, subprocess, platform, shutil, signal
from pathlib import Path
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

def _pip(*p):
    import importlib.util
    miss = [x for x in p if importlib.util.find_spec(x.replace("-", "_")) is None]
    if miss:
        print("installing:", miss)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *miss], check=True)
        for x in miss: __import__(x.replace("-", "_"))
_pip("pyarrow", "huggingface_hub")

import numpy as np, pandas as pd, torch
WORK = Path(CFG["WORK"]); SCRATCH = Path(CFG["SCRATCH"])
for d in (WORK, SCRATCH, WORK / "runs", WORK / "results", WORK / "figures"):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORK))

print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  {p.total_memory/2**30:.1f} GB  sm_{p.major}{p.minor}")
    torch.backends.cudnn.benchmark = True
else:
    print("  !! NO GPU -- set Accelerator to 'GPU T4 x2' in Session options.")
    print("     The notebook will still run on CPU but training will be impractically slow.")
if torch.cuda.device_count() < 2 or not all("T4" in torch.cuda.get_device_name(i)
                                            for i in range(torch.cuda.device_count())):
    raise RuntimeError("Select Kaggle Accelerator: GPU T4 x2, then restart and Run All.")
''')

md(r"""
---
# 2 · Library

Six modules, written to disk and imported. Five come straight from NB02's repo contract
(`crvs_sync`, `crvs_data`, `crvs_metrics`) or are defined here and reused by NB04 and NB05
(`crvs_models`, `crvs_losses`, `crvs_engine`).

`crvs_models.py` is the interesting one — it holds the four baseline architectures, including the
MultiRes block and ResPath that make MultiResLinkNet what it is.
""")

code(r'''
MODULES = {
 "crvs_sync.py":   r"""
__HF_SYNC__
""",
 "crvs_data.py":   r"""
__DATA__
""",
 "crvs_metrics.py": r"""
__METRICS__
""",
 "crvs_models.py": r"""
__MODELS__
""",
 "crvs_losses.py": r"""
__LOSSES__
""",
 "crvs_engine.py": r"""
__ENGINE__
""",
}
for nm, src in MODULES.items():
    (WORK / nm).write_text(src)
    print(f"  {nm:<20} {len(src):>7,} chars")
import hashlib
MODULE_HASHES = {nm: hashlib.sha256(src.encode()).hexdigest() for nm, src in MODULES.items()}
(WORK / "library_hashes.json").write_text(json.dumps(MODULE_HASHES, indent=2))

# Writing a .py and importing it is NOT idempotent inside one kernel: Python caches the
# module object in sys.modules, so re-running this cell after updating the notebook keeps
# the OLD code. That is exactly how a stale .npz loader survived a rebuilt notebook and
# produced a FileNotFoundError deep inside a DataLoader worker. Purge and re-import.
import importlib
for nm in MODULES:
    sys.modules.pop(nm[:-3], None)
importlib.invalidate_caches()

import crvs_data
REQUIRED_LIB = 4
if getattr(crvs_data, "LIB_VERSION", 0) < REQUIRED_LIB:
    raise RuntimeError(
        f"\n{'='*74}\n  Stale crvs_data: version "
        f"{getattr(crvs_data, 'LIB_VERSION', 'missing')}, need >= {REQUIRED_LIB}."
        f"\n  Loaded from {getattr(crvs_data, '__file__', '?')}"
        f"\n  Restart the kernel (Run -> Restart & Run All) and try again.\n{'='*74}")
print(f"\nlibrary written to {WORK}  |  crvs_data v{crvs_data.LIB_VERSION} loaded from "
      f"{crvs_data.__file__}")
''')

code(r'''
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets.")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("\n" + "="*74 +
            "\n  HF_TOKEN not found. Add-ons -> Secrets -> HF_TOKEN (write) -> attach.\n" + "="*74)

# Verify the one-time reset before constructing HFSync. If the guard fails, no uploader,
# signal handler or atexit hook exists yet, so this notebook cannot modify the old repo.
if CFG["REQUIRE_CLEAN_RESET"]:
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(repo_id=CFG["DST_REPO"], repo_type="model",
                        filename=CFG["RESET_RECEIPT"], token=HF_TOKEN,
                        local_dir=str(WORK))
    except Exception as e:
        raise RuntimeError(
            "\n" + "="*74 +
            "\n  One-time baseline cleanup receipt is missing."
            "\n  Run 02b_cleanup_baselines_hf_once.ipynb first, then start a NEW"
            "\n  Kaggle session and Run All here. This guard prevents old smoke-test"
            "\n  checkpoints from being mixed with the canonical baseline queue."
            f"\n  Hub check: {type(e).__name__}: {str(e)[:180]}"
            "\n" + "="*74) from e

from crvs_sync import HFSync
sync = HFSync(repo_id=CFG["DST_REPO"], local_dir=WORK, token=HF_TOKEN, repo_type="model",
              private=CFG["HF_PRIVATE"], run_id=CFG["RUN_ID"],
              push_interval_s=CFG["PUSH_INTERVAL_S"],
              max_upload_calls_hour=CFG["HF_MAX_UPLOADS_HOUR"])
print("\nresults repo:", sync.url, "(public)")

_major_state = {"name": None}
def MAJOR(nm, _state=_major_state):
    _state["name"] = str(nm)
def _nb03_post_run_cell(r=None, _state=_major_state, _sync=sync):
    nm = _state.pop("name", None)
    if nm is not None:
        _sync.stage_done(nm)
_nb03_post_run_cell._crvs_hook_id = "nb03-v3"
try:
    _ip = get_ipython()
    for _old in list(_ip.events.callbacks.get("post_run_cell", [])):
        if str(getattr(_old, "_crvs_hook_id", "")).startswith("nb03-"):
            try: _ip.events.unregister("post_run_cell", _old)
            except Exception: pass
    _ip.events.register("post_run_cell", _nb03_post_run_cell)
    print("post-run-cell push hook registered")
except Exception as e:
    print("hook unavailable:", e)

# Resume: pull back the small artefacts (state, summaries, metrics) but NOT the weights --
# finished runs never need their checkpoints re-downloaded, only their summary.json.
sync.pull(allow_patterns=["*.json", "*.jsonl", "*.csv", "*.md", "runs/**/summary.json",
                          "runs/**/state.json", "runs/B_rva__*/preds_sample.npz",
                          "runs/**/epoch_metrics.csv",
                          "results/*", "cleanup_receipts/*.json"])
STATE = sync.load_state({"completed": [], "sessions": 0, "version": 2})
STATE["version"] = 3
STATE["protocol_id"] = CFG["PROTOCOL_ID"]
STATE["sessions"] = STATE.get("sessions", 0) + 1
sync.save_state(STATE)
print(f"session #{STATE['sessions']}  |  {len(STATE['completed'])} run(s) already complete")
MAJOR("00_setup")
''')

md(r"""
---
# 3 · Fetch the corpus

Recordings are pulled to **scratch**, not to `/kaggle/working`, so the ~400 MB of data never eats
into the 20 GB output budget that the checkpoints need.
""")

code(r'''
from huggingface_hub import snapshot_download

# Prefer NB02's saved Kaggle notebook output: it mounts instantly and costs no working disk.
DATA = None
input_root = Path("/kaggle/input")
if input_root.exists():
    for candidate in input_root.rglob("windows.parquet"):
        if (candidate.parent / "recordings").exists() and (candidate.parent / "norm_stats.json").exists():
            DATA = candidate.parent; break
if DATA is not None:
    print("using attached Kaggle NB02 output:", DATA)
else:
    DATA = SCRATCH / "corpus"; t0 = time.time()
    print("NB02 output not attached; falling back to Hugging Face download.")
    snapshot_download(CFG["SRC_REPO"], repo_type="dataset", token=HF_TOKEN, local_dir=str(DATA),
                      allow_patterns=["recordings/*.npy", "recordings/*.json",
                                      "recordings/*.npz", "windows.parquet", "recordings.csv",
                                      "norm_stats.json", "experiments.json"], max_workers=4)
    print(f"corpus downloaded in {time.time()-t0:.0f}s")

W = pd.read_parquet(DATA / "windows.parquet")
DATA_HASH = hashlib.sha256((DATA / "windows.parquet").read_bytes()).hexdigest()
RECS = pd.read_csv(DATA / "recordings.csv")
NORM = json.loads((DATA / "norm_stats.json").read_text())
EXPINFO = json.loads((DATA / "experiments.json").read_text())
EXPERIMENTS = EXPINFO["experiments"]
REC_DIR = DATA / "recordings"

# Fail here, loudly, rather than inside a DataLoader worker 20 minutes into a run.
_npy = {p.stem for p in (REC_DIR).glob("*.npy")}
_npz = {p.stem for p in (REC_DIR).glob("*.npz")}
_have = _npy | _npz
print(f"recording files: {len(_npy)} .npy (fast path) + {len(_npz)} .npz (legacy, ~85x "
      f"slower per window)")
if not _have:
    raise RuntimeError(
        "\n" + "=" * 74 +
        "\n  No recording files downloaded."
        "\n  Check that NB02 finished and pushed, and that SRC_REPO matches its DST_REPO."
        "\n" + "=" * 74)
_missing = sorted(set(RECS["rec_id"]) - _have)
if _missing:
    print(f"WARNING: {len(_missing)} recording(s) in recordings.csv have no file: "
          f"{_missing[:5]}{' ...' if len(_missing) > 5 else ''}")
    W = W[~W["rec_id"].isin(_missing)].reset_index(drop=True)
    RECS = RECS[~RECS["rec_id"].isin(_missing)].reset_index(drop=True)
    print(f"         dropped their windows; {len(W):,} remain")

print(f"windows      : {len(W):,}   ({int(W['no_overlap'].sum()):,} non-overlapping)")
print(f"recordings   : {len(RECS)}   subjects: {RECS['subject'].nunique()}")
print(f"norm sets    : {len(NORM)}")
print(f"channels     : {EXPINFO['channels']}")
print(f"using        : {CFG['CHANNELS']}  <- the baseline's single-channel input")
print("\nwindows per experiment:")
for e, sc in EXPERIMENTS.items():
    sub = W[W["scenario_canon"].isin(sc)]
    print(f"  {e:<12} {len(sub):>8,} windows  {sub['subject'].nunique():>3} subjects  {sc}")
''')

md(r"""
---
# 4 · Model smoke test

Before spending GPU hours, prove each network actually runs: correct output shape, finite values,
gradients that flow, and a parameter count we can report. The baseline paper reports **no**
parameter or FLOP budget for any of its models — we will, for all of them.
""")

code(r'''
from crvs_models import build_baseline, count_params
from crvs_losses import MSEOnly, CompositeLoss
from crvs_engine import Trainer, seed_all, pick_device
import crvs_engine

# ---------------------------------------------------------------------------
# Long-session host-RAM safety patch
# ---------------------------------------------------------------------------
# This is deliberately a runtime/resource patch, not a change to the embedded model, loss,
# data, or optimiser libraries.  Existing checkpoints therefore keep the same config hash
# and resume safely.  It addresses the hard Kaggle restart seen after RAM climbed across
# epochs/runs, while preserving the exact training mathematics and validation outputs.
MEMORY_RUNTIME_VERSION = "nb03-process-isolated-v1"

def _linux_memory_gb():
    """Current process RSS and machine available RAM (not the monotonic peak RSS)."""
    out = {"process_rss_gb": float("nan"), "host_ram_available_gb": float("nan")}
    try:
        status = Path("/proc/self/status").read_text()
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                out["process_rss_gb"] = float(line.split()[1]) / 2**20
                break
    except Exception:
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                out["host_ram_available_gb"] = float(line.split()[1]) / 2**20
                break
    except Exception:
        pass
    return out

def release_runtime_memory(label=None, aggressive=False):
    """Return unused Python, Arrow and glibc pages to the OS; safe on non-Linux too."""
    if aggressive and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    for _ in range(2 if aggressive else 1):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        if aggressive:
            gc.collect()
    stats = _linux_memory_gb()
    if label:
        print(f"  RAM after {label}: {stats['process_rss_gb']:.2f} GB RSS | "
              f"{stats['host_ram_available_gb']:.2f} GB host available")
    return stats

def _migrate_loss_metric_names(record):
    # Evaluation MSE is `val_MSE`; the lowercase field is the validation loss component.
    # Giving the latter an explicit name makes the CSV usable by case-insensitive readers.
    for old, new in (("train_mse", "train_loss_mse"),
                     ("val_mse", "val_loss_mse")):
        if old in record:
            record.setdefault(new, record[old])
            record.pop(old, None)

if not getattr(Trainer, "_nb03_host_ram_patch", False):
    _base_validation = Trainer._validation
    _base_write_epoch = Trainer._write_epoch
    _base_load = Trainer.load
    _base_loader = Trainer._loader
    _base_payload = Trainer._payload
    _base_save = Trainer.save
    _base_system_stats = crvs_engine._system_stats

    def _cpu_checkpoint_tree(value, cache, path=()):
        # torch.save on live CUDA tensors retained approximately one checkpoint-sized host
        # transfer buffer per epoch (146 MB/epoch for UNet in the observed Hub telemetry).
        # Copy into one reusable pageable-CPU buffer per tensor, then serialize those buffers.
        if torch.is_tensor(value):
            src = value.detach()
            buf = cache.get(path)
            if (buf is None or tuple(buf.shape) != tuple(src.shape) or
                    buf.dtype != src.dtype or buf.layout != src.layout):
                buf = torch.empty_like(src, device="cpu", memory_format=torch.preserve_format)
                cache[path] = buf
            buf.copy_(src, non_blocking=False)
            return buf
        if isinstance(value, dict):
            return {k: _cpu_checkpoint_tree(v, cache, path + (("dict", repr(k)),))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [_cpu_checkpoint_tree(v, cache, path + (("list", i),))
                    for i, v in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(_cpu_checkpoint_tree(v, cache, path + (("tuple", i),))
                         for i, v in enumerate(value))
        return value

    def _bounded_payload(self):
        cache = getattr(self, "_nb03_cpu_checkpoint_buffers", None)
        if cache is None:
            cache = {}
            self._nb03_cpu_checkpoint_buffers = cache
        return _cpu_checkpoint_tree(_base_payload(self), cache)

    def _bounded_save(self, *args, **kwargs):
        try:
            return _base_save(self, *args, **kwargs)
        finally:
            # The reusable buffers stay allocated; transient serialization pages do not.
            release_runtime_memory()

    def _bounded_loader(self, *args, **kwargs):
        loader = _base_loader(self, *args, **kwargs)
        # The base engine enables CUDA pinning.  It is useful for large image batches but
        # retained pinned pages were harmful in this multi-hour, memory-mapped 1-D workload.
        loader.pin_memory = bool(CFG["PIN_MEMORY"])
        return loader

    def _bounded_validation(self, *args, **kwargs):
        try:
            return _base_validation(self, *args, **kwargs)
        finally:
            # The original validation result contains scalars only.  Once it returns, its
            # large waveform/SciPy/Arrow temporaries can be released before the next epoch.
            stats = release_runtime_memory()
            self._nb03_rss_after_validation = stats.get("process_rss_gb", float("nan"))
            if (np.isfinite(self._nb03_rss_after_validation) and
                    self._nb03_rss_after_validation >= CFG["HOST_RAM_WARN_GB"] and
                    not getattr(self, "_nb03_ram_warning_emitted", False)):
                self._nb03_ram_warning_emitted = True
                print(f"  RAM warning: {self._nb03_rss_after_validation:.2f} GB RSS; "
                      "continuing the queue (no automatic RAM stop).")
                if self.sync:
                    self.sync.log("host_memory_warning", run_id=self.run_id,
                                  rss_gb=self._nb03_rss_after_validation)

    def _portable_write_epoch(self, rec):
        for old_rec in self.state.get("history", []):
            _migrate_loss_metric_names(old_rec)
        _migrate_loss_metric_names(rec)
        _base_write_epoch(self, rec)

    def _portable_load(self, *args, **kwargs):
        loaded = _base_load(self, *args, **kwargs)
        for old_rec in self.state.get("history", []):
            _migrate_loss_metric_names(old_rec)
        # A RAM guard is a clean pause, not a model/training failure.  Remove the stale
        # diagnostic once its exact checkpoint has been restored for continuation.
        if self.state.get("stop_reason") == "host_memory_guard":
            self.state.pop("stop_reason", None)
            if str(self.state.get("last_error", "")).startswith("HostMemoryGuard:"):
                self.state.pop("last_error", None)
        return loaded

    def _memory_stats(out_dir):
        rec = _base_system_stats(out_dir)
        rec.update(_linux_memory_gb())
        rec["memory_runtime_version"] = MEMORY_RUNTIME_VERSION
        return rec

    Trainer._validation = _bounded_validation
    Trainer._loader = _bounded_loader
    Trainer._payload = _bounded_payload
    Trainer.save = _bounded_save
    Trainer._write_epoch = _portable_write_epoch
    Trainer.load = _portable_load
    Trainer._nb03_host_ram_patch = True
    crvs_engine._system_stats = _memory_stats

def close_dataset(ds):
    """Explicitly close cached npy/npz handles when a fold finishes or fails."""
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

# Repair already-downloaded CSVs from runs created before the loss-field naming fix.
_csv_repairs = 0
for _metric_csv in (WORK / "runs").glob("*/epoch_metrics.csv"):
    try:
        _df = pd.read_csv(_metric_csv)
        _rename = {k: v for k, v in (("train_mse", "train_loss_mse"),
                                     ("val_mse", "val_loss_mse"))
                   if k in _df.columns and v not in _df.columns}
        _dropped_duplicate = False
        for _old, _new in (("train_mse", "train_loss_mse"),
                           ("val_mse", "val_loss_mse")):
            if _old in _df.columns and _new in _df.columns:
                _df.drop(columns=[_old], inplace=True)
                _dropped_duplicate = True
                _csv_repairs += 1
        if _rename:
            _df.rename(columns=_rename).to_csv(_metric_csv, index=False)
            _csv_repairs += 1
        elif _dropped_duplicate:
            _df.to_csv(_metric_csv, index=False)
    except Exception as _csv_error:
        print(f"  metric CSV repair skipped for {_metric_csv.parent.name}: {_csv_error}")

_ram0 = release_runtime_memory("runtime patch")
print(f"memory policy: {MEMORY_RUNTIME_VERSION} | "
      f"{CFG['EPOCHS_PER_PROCESS']} epochs/child process | workers={CFG['WORKERS']} | "
      f"pin_memory={CFG['PIN_MEMORY']} | repaired CSVs={_csv_repairs}")

class BaselineOutputConvention(torch.nn.Module):
    def __init__(self, base, target_01=False):
        super().__init__(); self.base = base; self.target_01 = bool(target_01)
    def forward(self, x):
        out = self.base(x)
        if self.target_01:
            out["wave"] = (out["wave"] + 1.0) * 0.5
            if "aux" in out:
                out["aux"] = [torch.sigmoid(a) for a in out["aux"]]
        return out

def make_baseline(name):
    base = build_baseline(name, in_ch=len(CFG["CHANNELS"]), out_ch=1,
                          base=CFG["BASE"], levels=CFG["LEVELS"])
    return BaselineOutputConvention(base, target_01=CFG["TARGET_01"])

seed_all(CFG["SEED"])
dev, ngpu, names = pick_device()
print(f"device: {dev}  gpus: {ngpu} {names}\n")

C_IN = len(CFG["CHANNELS"])
x = torch.randn(4, C_IN, 1024, device=dev)
y = (torch.rand(4, 1, 1024, device=dev) if CFG["TARGET_01"] else
     torch.randn(4, 1, 1024, device=dev).clamp(-1, 1))
rows = []
print(f"{'model':<20}{'params':>12}{'MB':>8}{'out shape':>18}{'fwd ms':>9}  grad")
print("-" * 78)
for nm in CFG["MODELS"]:
    m = make_baseline(nm).to(dev)
    m.train()
    t0 = time.time()
    out = m(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000
    loss = torch.nn.functional.mse_loss(out["wave"], y)
    if "aux" in out:
        loss = loss + sum(torch.nn.functional.mse_loss(a, y) for a in out["aux"]) * 0.1
    loss.backward()
    gn = sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
    p = count_params(m)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(x[:1])
        gflops = float(fc.get_total_flops()) / 1e9
    except Exception:
        gflops = float("nan")
    ok = (out["wave"].shape == y.shape and torch.isfinite(out["wave"]).all() and gn > 0)
    rows.append({"model": nm, "params": p, "mb": p * 4 / 2**20,
                 "gflops_per_window": gflops, "forward_ms_batch4": dt, "ok": bool(ok)})
    print(f"{nm:<20}{p:>12,}{p*4/2**20:>8.1f}{str(tuple(out['wave'].shape)):>18}"
          f"{dt:>9.1f}  {'OK' if ok else 'FAIL'}")
    del m, out, loss
    release_runtime_memory()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
assert all(r["ok"] for r in rows), "a baseline failed its smoke test"
pd.DataFrame(rows).to_csv(WORK / "results" / "model_budget.csv", index=False)
BUDGET_BY_MODEL = pd.DataFrame(rows).set_index("model").to_dict("index")
del x, y
release_runtime_memory("smoke tensors", aggressive=True)
print("\nall four baselines forward, backward and produce finite output.")
MAJOR("01_smoke")
''')

md(r"""
---
# 5 · The run queue

Each entry is one (experiment, model, fold). Completed runs are read from HF state and skipped,
so re-running the notebook in a fresh session simply continues.

The IDs below are the only baseline run IDs accepted by the result tables. Trial folders and
artefacts from any other protocol are ignored even if they exist remotely.
""")

code(r'''
QUEUE = []
for exp in CFG["EXPERIMENTS"]:
    for mdl in CFG["MODELS"]:
        for f in range(CFG["N_FOLDS"]):
            QUEUE.append({"run_id": f"{exp}__{mdl}__f{f}", "exp": exp,
                          "model": mdl, "fold": f})

queue_ids = {q["run_id"] for q in QUEUE}
done_all = set(STATE.get("completed", []))
# A matching summary is the completion authority. State alone is only a resume hint: if a
# session died after marking state but before uploading its summary, the run must be revisited
# (Trainer.load then resumes or evaluates it without blindly retraining).
summary_done = set()
for _p in (WORK / "runs").glob("*/summary.json"):
    try:
        _s = json.loads(_p.read_text())
        if (_s.get("run_id") in queue_ids and
                _s.get("protocol_id") == CFG["PROTOCOL_ID"]):
            summary_done.add(_s["run_id"])
    except Exception:
        pass
state_only = (done_all & queue_ids) - summary_done
if state_only:
    print(f"repairing {len(state_only)} completion marker(s) without a canonical summary")
done = summary_done
done_all = (done_all - queue_ids) | done
STATE["completed"] = sorted(done_all)
sync.save_state(STATE)
todo = [q for q in QUEUE if q["run_id"] not in done]
print(f"queue: {len(QUEUE)} run(s) total | {len(done)} done | {len(todo)} remaining")
print(f"epochs per run: {CFG['EPOCHS']}   time budget: {CFG['TIME_BUDGET_H']} h")
print("protocol:", CFG["PROTOCOL_ID"])
print("\nnext up:")
for q in todo[:8]:
    print("   ", q["run_id"])
if len(todo) > 8:
    print(f"    ... and {len(todo)-8} more")
''')

code(r'''
from crvs_data import WindowDataset, WINDOW, FS
from crvs_metrics import seg_metrics, detect_r_peaks, hrv_from_peaks, peak_detection_scores

class TargetConventionDataset:
    # The processed corpus stores ECG in [-1, 1]. The published baseline trained in [0, 1].
    # Keep the conversion at the dataset boundary so training, validation, testing and saved
    # predictions all use one declared convention.
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

def split_for(exp, fold, n_folds=None):
    n_folds = n_folds or CFG["N_FOLDS"]
    sub = W[W["scenario_canon"].isin(EXPERIMENTS[exp])]
    te_g, va_g = fold % n_folds, (fold + 1) % n_folds
    tr = sub[~sub["fold_group"].isin([te_g, va_g])]
    va = sub[(sub["fold_group"] == va_g) & sub["no_overlap"]]
    te = sub[(sub["fold_group"] == te_g) & sub["no_overlap"]]
    assert not (set(tr["subject"]) & set(te["subject"])), "SUBJECT LEAK"
    return tr, va, te

def make_datasets(exp, fold):
    tr, va, te = split_for(exp, fold)
    norm = NORM.get(f"{exp}|{fold}")
    if norm is None:
        raise RuntimeError(f"no normalisation stats for {exp}|{fold} -- re-run NB02")
    idx = [EXPINFO["channels"].index(c) for c in CFG["CHANNELS"]]
    sub_norm = {"mean": [norm["mean"][i] for i in idx],
                "std":  [norm["std"][i] for i in idx]}
    def mk(d, aug):
        base = WindowDataset(REC_DIR, d, sub_norm, CFG["CHANNELS"], augment=aug,
                             seed=CFG["SEED"] + fold)
        return TargetConventionDataset(base, target_01=CFG["TARGET_01"])
    return mk(tr, True), mk(va, False), mk(te, False), (tr, va, te)

def evaluate(Y, P, index, out_dir):
    # Per-window metrics, then per-subject and overall aggregates. Saving per-window rows
    # lets NB05 run the statistics without ever re-running a model.
    rows = []
    idx_frame = index.reset_index(drop=True)
    subs = idx_frame["subject"].to_numpy()
    scen = idx_frame["scenario_canon"].to_numpy()
    for i in range(len(Y)):
        m = seg_metrics(Y[i], P[i], FS)
        m["subject"] = subs[i] if i < len(subs) else "?"
        m["scenario"] = scen[i] if i < len(scen) else "?"
        rows.append(m)
    dfw = pd.DataFrame(rows)
    dfw.to_parquet(out_dir / "metrics_windows.parquet", index=False)
    num = [c for c in dfw.columns if dfw[c].dtype.kind in "fi"]
    agg = {c: float(dfw[c].mean()) for c in num}
    agg.update({c + "_std": float(dfw[c].std()) for c in num})
    # HR/HRV is computed recording-by-recording in chronological window order. Joining
    # different recordings/scenarios would invent an RR interval at every boundary.
    hr_rows = []
    for rid, grp in idx_frame.assign(_row=np.arange(len(idx_frame))).groupby("rec_id"):
        grp = grp.sort_values("start")
        pos = grp["_row"].to_numpy()
        if len(pos) < 2:
            continue
        yg = np.concatenate(Y[pos]); yp = np.concatenate(P[pos])
        g = hrv_from_peaks(detect_r_peaks(yg, FS), FS)
        p = hrv_from_peaks(detect_r_peaks(yp, FS), FS)
        pk = peak_detection_scores(yg, yp, FS)
        hr_rows.append({"rec_id": rid, "subject": str(grp["subject"].iloc[0]),
                        "scenario": str(grp["scenario_canon"].iloc[0]),
                        **{f"gt_{k}": v for k, v in g.items()},
                        **{f"pr_{k}": v for k, v in p.items()}, **pk})
    dfh = pd.DataFrame(hr_rows)
    if len(dfh):
        dfh.to_parquet(out_dir / "metrics_recordings.parquet", index=False)
        numeric = [c for c in dfh.columns if dfh[c].dtype.kind in "fi"]
        dfh.groupby("subject", as_index=False)[numeric].mean().to_parquet(
            out_dir / "metrics_subjects.parquet", index=False)
        for k in ("F1", "precision", "recall", "accuracy", "missed_rate",
                  "timing_err_ms_median"):
            if k in dfh.columns:
                agg["peak_" + k] = float(dfh[k].mean())
        for k in ("mean_hr_bpm", "rmssd_ms"):
            if f"gt_{k}" in dfh and f"pr_{k}" in dfh:
                agg["MAE_" + k] = float((dfh[f"gt_{k}"] - dfh[f"pr_{k}"]).abs().mean())
    return agg, dfw
''')

md(r"""
---
# 6 · Train

The loop below is the whole notebook. For each queued run it builds the split, trains with early
stopping, evaluates on the held-out subjects, writes everything under `runs/<run_id>/`, marks the
run complete and pushes.

Watch the `val` column. If it stops improving in the first few epochs across every model, the
targets are probably misaligned — stop and check NB02's `nb02_fig1_window.png`.

**You can interrupt at any point.** The current epoch's checkpoint is already on disk, the
interrupt handler pushes it, and the next session resumes mid-run.
""")

code(r'''
# Write the isolated worker outside MODULES: it is orchestration infrastructure and therefore
# does not alter the scientific library/config hash of checkpoints that already exist on HF.
WORKER_SRC = r"""
__WORKER__
"""
WORKER_PATH = WORK / "nb03_run_worker.py"
WORKER_PATH.write_text(WORKER_SRC, encoding="utf-8")
WORKER_CONTEXT = WORK / "nb03_worker_context.json"
WORKER_CONTEXT.write_text(json.dumps({
    "cfg": CFG, "work": str(WORK), "data": str(DATA), "data_hash": DATA_HASH,
    "module_hashes": MODULE_HASHES, "budget_by_model": BUDGET_BY_MODEL,
    "memory_runtime_version": MEMORY_RUNTIME_VERSION,
}, indent=2, default=str), encoding="utf-8")

t_start = time.time()
budget = CFG["TIME_BUDGET_H"] * 3600
completed_now = []
failed_now = []
session_stop_reason = None
_active_worker = {"proc": None, "run_id": None}

def _stop_active_worker():
    proc = _active_worker.get("proc")
    if proc is None or proc.poll() is not None:
        return
    print(f"\n  stopping isolated worker {_active_worker.get('run_id')} safely...", flush=True)
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=120)
    except Exception:
        # Do not force-kill here: the parent HF interrupt hook will upload the last atomic
        # epoch checkpoint even if the child needs the platform to terminate it.
        pass

def _pull_active_checkpoint(rid, out):
    if (out / "state.json").exists() and not (out / "state.pt").exists():
        print("  interrupted remote run found; restoring exact checkpoint...")
        return sync.pull(allow_patterns=[f"runs/{rid}/state.pt", f"runs/{rid}/best.pt",
                                         f"runs/{rid}/state.json", f"runs/{rid}/run_config.json",
                                         f"runs/{rid}/environment.json", f"runs/{rid}/*.jsonl",
                                         f"runs/{rid}/*.csv", f"runs/{rid}/validation_windows/*",
                                         f"runs/{rid}/validation_recordings/*"])
    return True

def _launch_chunk(q):
    rid = q["run_id"]
    status_path = WORK / "worker_status.json"
    if status_path.exists():
        status_path.unlink()
    cmd = [sys.executable, "-u", str(WORKER_PATH), "--context", str(WORKER_CONTEXT),
           "--run-id", rid, "--experiment", q["exp"], "--model", q["model"],
           "--fold", str(q["fold"])]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    sync.mark_dirty(f"{rid}:isolated-worker-active")
    proc = subprocess.Popen(cmd, cwd=str(WORK), env=env)
    _active_worker.update(proc=proc, run_id=rid)
    sync.set_before_final_flush(_stop_active_worker)
    try:
        rc = proc.wait()
    finally:
        _active_worker.update(proc=None, run_id=None)
        sync.set_before_final_flush(None)
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        status = {"status": "failed", "run_id": rid,
                  "error": f"worker exited {rc} without a status file"}
    status["return_code"] = rc
    sync.mark_dirty(f"{rid}:isolated-worker-{status.get('status')}")
    parent_ram = release_runtime_memory(f"worker {rid} exited", aggressive=True)
    sync.log("worker_process_reaped", run_id=rid, pid=status.get("pid"),
             worker_status=status.get("status"), epoch=status.get("epoch"),
             worker_rss_gb=status.get("rss_gb"),
             parent_rss_gb=parent_ram.get("process_rss_gb"),
             policy=MEMORY_RUNTIME_VERSION)
    return status

if not CFG.get("PROCESS_ISOLATION", True):
    raise RuntimeError("PROCESS_ISOLATION must stay enabled for this Kaggle workload")

for qi, q in enumerate(list(todo), 1):
    rid = q["run_id"]
    out = WORK / "runs" / rid
    out.mkdir(parents=True, exist_ok=True)
    if not _pull_active_checkpoint(rid, out):
        failed_now.append({"run_id": rid, "error": "remote checkpoint download failed"})
        continue
    print("\n" + "=" * 78)
    print(f"[{qi}/{len(todo)}] {rid} — isolated process chunks of "
          f"{CFG['EPOCHS_PER_PROCESS']} epoch(s)")
    print("=" * 78)
    while True:
        if time.time() - t_start >= budget:
            session_stop_reason = "time_budget"
            print(f"\n=== {CFG['TIME_BUDGET_H']:.2f} h budget reached; pushing and pausing. ===")
            break
        try:
            status = _launch_chunk(q)
        except KeyboardInterrupt:
            _stop_active_worker()
            sync.mark_dirty(f"{rid}:parent-interrupt")
            sync.flush(final=True, force=True, msg=f"{rid} interrupted; isolated checkpoint")
            print("\ninterrupted — the latest atomic child checkpoint was pushed.")
            raise
        state = status.get("status")
        if state == "chunk_complete":
            print(f"  worker exited cleanly at epoch {status.get('epoch')}; "
                  "OS RAM reclaimed; launching the next chunk.")
            continue
        if state == "run_complete" and (out / "summary.json").exists():
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            metrics = summary.get("metrics", {})
            done.add(rid); done_all.add(rid); completed_now.append(rid)
            STATE["completed"] = sorted(done_all); sync.save_state(STATE)
            pushed = sync.flush(force=True, msg=(
                f"{rid} complete CCt={metrics.get('CC_temporal', float('nan')):.2f}"))
            if pushed:
                for name in ("state.pt", "best.pt"):
                    checkpoint = out / name
                    if checkpoint.exists():
                        checkpoint.unlink()
                sync.log("local_checkpoints_pruned", run_id=rid, remote_copy=True)
            break
        error = status.get("error", f"worker returned {status.get('return_code')}")
        print(f"  !! isolated worker failed: {error}")
        failed_now.append({"run_id": rid, "error": error})
        sync.flush(force=True, msg=f"{rid} worker failure checkpoint")
        break
    if session_stop_reason:
        break

print(f"\ncompleted this session: {len(completed_now)}   total: {len(done)}/{len(QUEUE)}")
print("Each worker process has exited; its RAM and CUDA context were reclaimed by Linux.")

# The legacy in-kernel loop remains below only as a source-readable fallback.  Emptying its
# input guarantees it cannot execute in this process-isolated notebook.
todo = []
MAJOR("02_training_isolated")
''')

code(r'''
t_start = globals().get("t_start", time.time())
budget = CFG["TIME_BUDGET_H"] * 3600
completed_now = globals().get("completed_now", [])
failed_now = globals().get("failed_now", [])
session_stop_reason = globals().get("session_stop_reason")

for qi, q in enumerate(todo, 1):
    el = time.time() - t_start
    if el > budget:
        session_stop_reason = "time_budget"
        print(f"\n=== time budget reached ({el/3600:.2f} h). Stopping cleanly. ===")
        print(f"    {len(todo)-qi+1} run(s) left -- start a new session and re-run this notebook.")
        break
    rid, exp, mdl, fold = q["run_id"], q["exp"], q["model"], q["fold"]
    out = WORK / "runs" / rid
    out.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 78)
    print(f"[{qi}/{len(todo)}]  {rid}   ({el/3600:.2f} h elapsed)")
    print("=" * 78)
    tr = model = loss_fn = tr_ds = va_ds = te_ds = None
    tri = vai = tei = Y = P = agg = dfw = sel = None
    try:
        # The lightweight startup pull tells us whether an interrupted checkpoint exists.
        # Restore its weights/optimizer just in time; completed runs are never downloaded.
        if (out / "state.json").exists() and not (out / "state.pt").exists():
            print("  interrupted remote run found; restoring exact checkpoint...")
            sync.pull(allow_patterns=[f"runs/{rid}/state.pt", f"runs/{rid}/best.pt",
                                      f"runs/{rid}/state.json", f"runs/{rid}/run_config.json",
                                      f"runs/{rid}/environment.json", f"runs/{rid}/*.jsonl",
                                      f"runs/{rid}/*.csv", f"runs/{rid}/validation_windows/*",
                                      f"runs/{rid}/validation_recordings/*"])
        tr_ds, va_ds, te_ds, (tri, vai, tei) = make_datasets(exp, fold)
        print(f"  train {len(tr_ds):,} | val {len(va_ds):,} | test {len(te_ds):,} windows   "
              f"test subjects: {sorted(tei['subject'].unique())}")
        seed_all(CFG["SEED"] + fold)
        model = make_baseline(mdl)
        loss_fn = MSEOnly() if CFG["LOSS"] == "mse" else CompositeLoss()
        tr = Trainer(model, loss_fn, out, rid, sync=sync, lr=CFG["LR"],
                     weight_decay=CFG["WEIGHT_DECAY"], epochs=CFG["EPOCHS"],
                     patience=CFG["PATIENCE"], batch_size=CFG["BATCH"],
                     num_workers=CFG["WORKERS"], amp=CFG["AMP"],
                     multi_gpu=CFG["MULTI_GPU"], log_every=CFG["LOG_EVERY"],
                     checkpoint_every_steps=CFG["CHECKPOINT_EVERY_STEPS"],
                     checkpoint_every_s=CFG["CHECKPOINT_EVERY_S"], seed=CFG["SEED"] + fold,
                     require_dual_gpu=CFG["REQUIRE_DUAL_T4"],
                     run_config={"protocol_id": CFG["PROTOCOL_ID"],
                                 "experiment": exp, "model": mdl, "fold": fold,
                                 "epochs": CFG["EPOCHS"], "channels": CFG["CHANNELS"],
                                 "target_01": CFG["TARGET_01"],
                                 "base": CFG["BASE"], "levels": CFG["LEVELS"],
                                 "loss": CFG["LOSS"], "lr": CFG["LR"],
                                 "batch": CFG["BATCH"], "seed": CFG["SEED"] + fold,
                                 "data_index_sha256": DATA_HASH, "library_sha256": MODULE_HASHES,
                                 "train_subjects": sorted(map(str, tri["subject"].unique())),
                                 "val_subjects": sorted(map(str, vai["subject"].unique())),
                                 "test_subjects": sorted(map(str, tei["subject"].unique()))})
        tr.load()
        tr.fit(tr_ds, va_ds)
        Y, P = tr.predict(te_ds)
        agg, dfw = evaluate(Y, P, tei, out)

        keep = min(200, len(Y))
        sel = np.linspace(0, len(Y) - 1, keep).astype(int)
        np.savez_compressed(out / "preds_sample.npz", y=Y[sel].astype(np.float32),
                            p=P[sel].astype(np.float32),
                            subject=tei["subject"].to_numpy()[sel].astype(str))
        summary = {"run_id": rid, "protocol_id": CFG["PROTOCOL_ID"],
                   "config_hash": tr.config_hash,
                   "memory_runtime_version": MEMORY_RUNTIME_VERSION,
                   "data_loader_workers": CFG["WORKERS"],
                   "pin_memory": CFG["PIN_MEMORY"],
                   "experiment": exp, "model": mdl, "fold": fold,
                   "channels": CFG["CHANNELS"], "target_01": CFG["TARGET_01"],
                   "loss": CFG["LOSS"], "epochs_run": tr.state["epoch"],
                   "best_epoch": tr.state["best_epoch"], "best_val": tr.state["best"],
                   "params": count_params(tr.raw_model),
                   "gflops_per_window": BUDGET_BY_MODEL.get(mdl, {}).get("gflops_per_window"),
                   "forward_ms_batch4": BUDGET_BY_MODEL.get(mdl, {}).get("forward_ms_batch4"),
                   "n_train": len(tr_ds), "n_val": len(va_ds), "n_test": len(te_ds),
                   "test_subjects": sorted(map(str, tei["subject"].unique())),
                   "metrics": agg,
                   "finished_utc": datetime.now(timezone.utc).isoformat()}
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"  --> CC_t {agg['CC_temporal']:.2f}  CC_s {agg['CC_spectral']:.2f}  "
              f"MAE {agg['MAE']:.5f}  MSE {agg['MSE']:.5f}  "
              f"RRMSE_t {agg['RRMSE_temporal']:.4f}  F1 {agg.get('peak_F1', float('nan')):.3f}")
        done.add(rid); done_all.add(rid); completed_now.append(rid)
        STATE["completed"] = sorted(done_all); sync.save_state(STATE)
        pushed = sync.flush(force=True, msg=f"{rid} complete CCt={agg['CC_temporal']:.2f}")
        if pushed:
            # The remote commit now contains both recovery and best checkpoints. Remove the
            # local copies so a long queue cannot exhaust Kaggle's 20 GB working volume.
            for name in ("state.pt", "best.pt"):
                p = out / name
                if p.exists(): p.unlink()
            sync.log("local_checkpoints_pruned", run_id=rid, remote_copy=True)
    except KeyboardInterrupt:
        print("\ninterrupted -- checkpoint saved and pushed; re-run to resume this exact run.")
        raise
    except Exception as e:
        import traceback
        print(f"  !! {type(e).__name__}: {e}")
        (out / "error.txt").write_text(traceback.format_exc())
        sync.log("run_failed", run=rid, err=f"{type(e).__name__}: {e}")
        failed_now.append({"run_id": rid, "error": f"{type(e).__name__}: {e}"})
    finally:
        _ram_before_cleanup = _linux_memory_gb().get("process_rss_gb", float("nan"))
        for _ds in (tr_ds, va_ds, te_ds):
            if _ds is not None:
                close_dataset(_ds)
        if tr is not None:
            # Drop the reusable CPU checkpoint staging area explicitly at every run boundary.
            getattr(tr, "_nb03_cpu_checkpoint_buffers", {}).clear()
        tr = model = loss_fn = tr_ds = va_ds = te_ds = None
        tri = vai = tei = Y = P = agg = dfw = sel = None
        _ram_after_cleanup = release_runtime_memory(rid, aggressive=True).get(
            "process_rss_gb", float("nan"))
        sync.log("run_memory_cleanup", run_id=rid,
                 rss_before_gb=_ram_before_cleanup, rss_after_gb=_ram_after_cleanup,
                 reclaimed_gb=(_ram_before_cleanup - _ram_after_cleanup
                               if np.isfinite(_ram_before_cleanup) and
                                  np.isfinite(_ram_after_cleanup) else None),
                 policy=MEMORY_RUNTIME_VERSION)

print(f"\ncompleted this session: {len(completed_now)}   total: {len(done)}/{len(QUEUE)}")
MAJOR("02_training")
''')

md(r"""
---
# 7 · Results, next to the published table

The comparison that decides whether we may proceed. Our MultiResLinkNet row on `B_rva` is the one
that matters: the baseline reports **CC_temporal 61.86, CC_spectral 79.96, MAE 0.14841,
RRMSE_t 0.44618**.

Read the gap in the right direction. Because our splits are subject-wise with non-overlapping test
windows, landing **somewhat below** their figures is the expected and honest outcome — it means we
removed the leakage, not that our implementation is worse. What must hold is the **ordering**:
MultiResLinkNet should still beat FPN, UNet and LinkNet on correlation. If the ordering inverts,
the reimplementation is wrong.
""")

code(r'''
rows = []
ignored_summaries = []
for p in sorted((WORK / "runs").glob("*/summary.json")):
    try:
        s = json.loads(p.read_text())
        if (s.get("run_id") not in queue_ids or
                s.get("protocol_id") != CFG["PROTOCOL_ID"]):
            ignored_summaries.append(s.get("run_id", p.parent.name))
            continue
        rows.append({"experiment": s["experiment"], "model": s["model"], "fold": s["fold"],
                     "run_id": s["run_id"],
                     "params": s.get("params"), "best_epoch": s.get("best_epoch"),
                     **{k: v for k, v in s["metrics"].items() if not k.endswith("_std")}})
    except Exception:
        pass
R = pd.DataFrame(rows)
flat = pd.DataFrame(); CMP = pd.DataFrame()
GATE = {"protocol_id": CFG["PROTOCOL_ID"], "complete": False, "passed": False,
        "expected_b_rva_runs": len(CFG["MODELS"]) * CFG["N_FOLDS"],
        "completed_b_rva_runs": 0,
        "updated_utc": datetime.now(timezone.utc).isoformat()}
if ignored_summaries:
    print(f"ignored {len(ignored_summaries)} non-canonical summary file(s): "
          f"{ignored_summaries[:5]}{' ...' if len(ignored_summaries) > 5 else ''}\n")
if not len(R):
    print("no canonical completed runs yet -- run the training cell.")
else:
    R.to_csv(WORK / "results" / "runs_raw.csv", index=False)
    cols = ["MAE", "MSE", "CC_temporal", "CC_spectral", "RRMSE_temporal", "RRMSE_spectral"]
    agg = (R.groupby(["experiment", "model"])[cols + ["params"]]
             .agg(["mean", "std"]).round(5))
    print("=" * 100); print("OUR RUNS  (mean +/- std across folds)"); print("=" * 100)
    flat = R.groupby(["experiment", "model"])[cols].mean().round(5).reset_index()
    nfold = R.groupby(["experiment", "model"]).size().rename("folds").reset_index()
    flat = flat.merge(nfold, on=["experiment", "model"])
    print(flat.to_string(index=False))
    flat.to_csv(WORK / "results" / "runs_summary.csv", index=False)

    PAPER = {
      ("B_rva","fpn"):            dict(MAE=.14316, MSE=.03422, CC_temporal=59.63, CC_spectral=69.53, RRMSE_temporal=.44694, RRMSE_spectral=.83026),
      ("B_rva","unet"):           dict(MAE=.14798, MSE=.03741, CC_temporal=57.65, CC_spectral=68.39, RRMSE_temporal=.45315, RRMSE_spectral=.94118),
      ("B_rva","linknet"):        dict(MAE=.14780, MSE=.03723, CC_temporal=58.69, CC_spectral=70.91, RRMSE_temporal=.45487, RRMSE_spectral=.86909),
      ("B_rva","multireslinknet"):dict(MAE=.14841, MSE=.03793, CC_temporal=61.86, CC_spectral=79.96, RRMSE_temporal=.44618, RRMSE_spectral=.73269),
      ("A_resting","fpn"):            dict(CC_temporal=58.37, CC_spectral=71.38),
      ("A_resting","unet"):           dict(CC_temporal=63.10, CC_spectral=74.68),
      ("A_resting","linknet"):        dict(CC_temporal=64.35, CC_spectral=74.37),
      ("A_resting","multireslinknet"):dict(CC_temporal=66.10, CC_spectral=82.44),
      ("A_valsalva","fpn"):            dict(CC_temporal=57.53, CC_spectral=65.97),
      ("A_valsalva","unet"):           dict(CC_temporal=58.38, CC_spectral=68.79),
      ("A_valsalva","linknet"):        dict(CC_temporal=56.63, CC_spectral=66.87),
      ("A_valsalva","multireslinknet"):dict(CC_temporal=60.14, CC_spectral=77.05),
      ("A_apnea","fpn"):            dict(CC_temporal=39.12, CC_spectral=51.26),
      ("A_apnea","unet"):           dict(CC_temporal=56.14, CC_spectral=69.97),
      ("A_apnea","linknet"):        dict(CC_temporal=56.22, CC_spectral=70.35),
      ("A_apnea","multireslinknet"):dict(CC_temporal=55.33, CC_spectral=74.66),
    }
    cmp_rows = []
    for _, r in flat.iterrows():
        p = PAPER.get((r["experiment"], r["model"]), {})
        cmp_rows.append({"experiment": r["experiment"], "model": r["model"], "folds": r["folds"],
                         "CC_t_ours": round(r["CC_temporal"], 2),
                         "CC_t_paper": p.get("CC_temporal"),
                         "CC_t_delta": (round(r["CC_temporal"] - p["CC_temporal"], 2)
                                        if "CC_temporal" in p else None),
                         "CC_s_ours": round(r["CC_spectral"], 2),
                         "CC_s_paper": p.get("CC_spectral"),
                         "MAE_ours": round(r["MAE"], 5), "MAE_paper": p.get("MAE")})
    CMP = pd.DataFrame(cmp_rows)
    print("\n" + "=" * 100); print("OURS vs. PUBLISHED"); print("=" * 100)
    print(CMP.to_string(index=False))
    CMP.to_csv(WORK / "results" / "vs_paper.csv", index=False)

    b_raw = R[R["experiment"] == "B_rva"]
    expected_b = {(m, f) for m in CFG["MODELS"] for f in range(CFG["N_FOLDS"])}
    got_b = set(zip(b_raw["model"], b_raw["fold"]))
    missing_b = sorted(expected_b - got_b)
    GATE["completed_b_rva_runs"] = len(got_b & expected_b)
    GATE["missing"] = [{"model": m, "fold": int(f)} for m, f in missing_b]
    GATE["complete"] = not missing_b
    print("\n" + "-" * 100)
    print("REPRODUCTION GATE  (Experiment B, RVA combined)")
    print("-" * 100)
    if missing_b:
        print(f"  GATE PENDING -- {len(got_b & expected_b)}/{len(expected_b)} canonical runs complete.")
        print("  Missing:", ", ".join(f"{m}/f{f}" for m, f in missing_b))
        print("  A partial or single-model result can never pass this gate.")
    else:
        b = flat[flat["experiment"] == "B_rva"].sort_values("CC_temporal", ascending=False)
        print(f"  ranking by CC_temporal: {' > '.join(b['model'].tolist())}")
        top = b.iloc[0]["model"]
        ok = top == "multireslinknet"
        GATE["passed"] = bool(ok)
        GATE["ranking"] = b["model"].tolist()
        print(f"  MultiResLinkNet ranks first: {ok}")
        mr = b[b["model"] == "multireslinknet"]
        if len(mr):
            v = float(mr.iloc[0]["CC_temporal"])
            GATE["multireslinknet_cc_temporal"] = v
            GATE["published_cc_temporal"] = 61.86
            print(f"  our CC_temporal {v:.2f}  vs published 61.86  (delta {v-61.86:+.2f})")
        print("\n  " + ("GATE PASSED -- ordering reproduced, proceed to NB04."
                        if ok else
                        "GATE NOT PASSED -- MultiResLinkNet is not top after all 20 canonical"
                        "\n  B_rva runs. Check the reimplementation before trusting downstream results."))
(WORK / "results" / "reproduction_gate.json").write_text(
    json.dumps(GATE, indent=2, default=str))
MAJOR("03_results")
''')

code(r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = {"radar": "#0F7C82", "ecg": "#AF3A2C", "muted": "#5C6B71", "ink": "#10171B",
     "grid": "#D3DADB", "amber": "#8A6212"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.color": S["grid"], "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 8.5, "axes.titlesize": 10, "axes.titleweight": "bold"})
FIG = WORK / "figures"

if len(R):
    b = R[R["experiment"] == "B_rva"]
    if len(b):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        order = ["fpn", "unet", "linknet", "multireslinknet"]
        order = [o for o in order if o in set(b["model"])]
        for ax, met, ref in [(axes[0], "CC_temporal",
                              {"fpn":59.63,"unet":57.65,"linknet":58.69,"multireslinknet":61.86}),
                             (axes[1], "CC_spectral",
                              {"fpn":69.53,"unet":68.39,"linknet":70.91,"multireslinknet":79.96})]:
            vals = [b[b["model"] == o][met].mean() for o in order]
            ax.bar(order, vals, color=[S["radar"]] * (len(order) - 1) + [S["ecg"]],
                   edgecolor="white", label="ours")
            ax.plot(order, [ref[o] for o in order], "o--", color=S["ink"], ms=5, lw=1.2,
                    label="published")
            ax.set_title(met + "   (Experiment B, RVA)")
            ax.tick_params(axis="x", rotation=18)
            ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / "nb03_fig1_vs_paper.png"); plt.close(fig)
        print("  wrote nb03_fig1_vs_paper.png")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    for m in sorted(set(R["model"])):
        hs = []
        for p in (WORK / "runs").glob(f"*__{m}__*/state.json"):
            try:
                hs += [(h["epoch"], h["val"]) for h in json.loads(p.read_text())["history"]]
            except Exception:
                pass
        if hs:
            d = pd.DataFrame(hs, columns=["epoch", "val"]).groupby("epoch")["val"].mean()
            ax.plot(d.index, d.values, lw=1.3, label=m)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("validation loss")
    ax.set_title("Baseline training curves (mean over runs)", loc="left")
    ax.legend(frameon=False, ncol=4, fontsize=7)
    fig.savefig(FIG / "nb03_fig2_curves.png"); plt.close(fig)
    print("  wrote nb03_fig2_curves.png")

    samp = sorted((WORK / "runs").glob("B_rva__*/preds_sample.npz"))
    if samp:
        fig, axes = plt.subplots(len(samp), 1, figsize=(11, 1.9 * len(samp)), sharex=True)
        axes = np.atleast_1d(axes)
        tt = np.arange(1024) / 128.0
        for ax, sp in zip(axes, samp):
            z = np.load(sp)
            k = min(3, len(z["y"]) - 1)
            ax.plot(tt, z["y"][k], lw=1.0, color=S["ink"], label="ground truth")
            ax.plot(tt, z["p"][k], lw=1.0, color=S["ecg"], label="reconstructed", alpha=.85)
            ax.set_ylabel(sp.parent.name.split("__")[1], rotation=0, ha="right",
                          va="center", fontsize=7.5)
            ax.tick_params(labelleft=False)
        axes[0].legend(frameon=False, ncol=2, fontsize=7)
        axes[0].set_title("Reconstructed vs. true ECG — one held-out window per baseline", loc="left")
        axes[-1].set_xlabel("seconds")
        fig.tight_layout(); fig.savefig(FIG / "nb03_fig3_qualitative.png"); plt.close(fig)
        print("  wrote nb03_fig3_qualitative.png")
MAJOR("04_figures")
''')

code(r'''
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
card = f"""---
license: cc-by-4.0
tags: [radar, ecg, biosignals, contactless-monitoring, time-series]
pipeline_tag: audio-to-audio
---

# CardioMamba-Net — checkpoints and results

Radar-to-ECG reconstruction on the CR-RVS dataset. Produced by `03_baselines.ipynb` and
`04_cardiomamba_train.ipynb`. Corpus: [`{CFG['SRC_REPO']}`](https://huggingface.co/datasets/{CFG['SRC_REPO']}).

## Layout

```
runs/<experiment>__<model>__f<fold>/
    best.pt                  weights at the best validation epoch
    state.pt                 full resumable state (model, optimiser, scheduler, scaler, RNG)
    summary.json             config + aggregated test metrics
    metrics_windows.parquet  per-window metrics on held-out subjects
    metrics_subjects.parquet per-subject HR / HRV / peak-detection scores
    preds_sample.npz         200 held-out reconstructions for qualitative figures
results/                     cross-run tables, including ours vs. the published numbers
figures/                     comparison and training-curve figures
```

## Protocol

128 Hz, 1024-sample (8 s) windows. Splits **by subject**; test windows do not overlap.
Normalisation statistics are per fold from **training windows only**. Baselines use the paper's
single-channel input (`dy`), [0,1] ECG target, 5 levels, 64 base filters, plain MSE, Adam 5e-4.
Canonical protocol ID: `{CFG['PROTOCOL_ID']}`. Non-canonical/quick summaries are excluded from
all aggregate tables and from the reproduction gate.

Updated {now}.
"""
(WORK / "README.md").write_text(card)
ok = sync.flush(final=True, msg=f"{CFG['RUN_ID']} — {len(done)}/{len(QUEUE)} runs complete")
_reason = globals().get("session_stop_reason")
_failures = globals().get("failed_now", [])
if len(done) == len(QUEUE):
    _session_label = "CANONICAL QUEUE COMPLETE"
elif _reason == "time_budget":
    _session_label = "SESSION PAUSED — TIME BUDGET"
elif _failures:
    _session_label = "SESSION ENDED WITH RUN ERRORS"
else:
    _session_label = "SESSION PAUSED — RUNS REMAIN"
if not ok:
    _session_label += " (FINAL PUSH HAD A PROBLEM)"
print("\n" + "=" * 76)
print("  " + _session_label)
print("=" * 76)
print(f"  repo      : {sync.url}")
print(f"  runs done : {len(done)}/{len(QUEUE)}")
print(f"  this run  : {len(completed_now)} new")
print(f"  failures  : {len(_failures)}")
print(f"  elapsed   : {(time.time()-t_start)/3600:.2f} h")
print("=" * 76)
if len(done) < len(QUEUE):
    print(f"\n  {len(QUEUE)-len(done)} run(s) remain. Start a NEW session and run this notebook")
    print("  again -- it resumes from Hugging Face and skips everything already finished.")
else:
    if GATE.get("passed"):
        print("\n  Canonical queue complete and reproduction gate PASSED.")
        print("  Next: 04_cardiomamba_train.ipynb")
    else:
        print("\n  Canonical queue complete, but the reproduction gate did NOT pass.")
        print("  Do not proceed to NB04 until the result has been reviewed.")
''')

md(r"""
---
# 8 · Troubleshooting

**No GPU / very slow epochs** — *Session options → Accelerator → GPU T4 x2*. On CPU an epoch takes
minutes instead of seconds; the notebook runs but the queue will not finish.

**CUDA out of memory** — lower `CFG["BATCH"]` to 32 or 16. MultiResLinkNet with `BASE=64` is the
heaviest of the four. Batch 64 across two T4s is comfortable; batch 128 is not.

**`no normalisation stats for <exp>|<fold>`** — NB02 did not emit that combination, usually because
the split was empty after the quality gate. Re-run NB02, or drop that experiment from
`CFG["EXPERIMENTS"]`.

**`SUBJECT LEAK` assertion** — a genuine bug in fold assignment. Do not bypass it.

**Validation loss flat from epoch 1** — check `nb02_fig1_window.png`. If the ECG and `peak_map`
rows are not aligned, the targets are wrong and no architecture will help.

**Notebook says it allocated too much memory / kernel restarted** — this version defaults to
`CFG["WORKERS"] = 0` and runs at most five epochs in each isolated child process. The child exits
after writing an atomic checkpoint, so Linux—not Python's garbage collector—reclaims all of its
RAM and CUDA state. The parent then launches the next chunk automatically. `worker_process_reaped`
events record worker and parent RSS in HF logs. Do not disable `PROCESS_ISOLATION`, raise
`EPOCHS_PER_PROCESS`, raise `WORKERS`, or enable `PIN_MEMORY` in a long Kaggle session.

**Session ended mid-queue** — expected. Start a new session, run the notebook again, and it
resumes. Progress is per run *and* per epoch.

**Cleanup receipt missing** — run `02b_cleanup_baselines_hf_once.ipynb` once. It removes the old
quick/full baseline artefacts in one audited Hub commit and leaves the receipt required here.

**The gate says pending** — expected until all four `B_rva` models have all five folds. A partial
queue cannot pass the gate and cannot authorize NB04.
""")

out = sys.argv[1] if len(sys.argv) > 1 else "03_baselines.ipynb"
# The former in-kernel training cell is retained in this generator temporarily for readable
# migration history, but must never be emitted: process isolation is the only supported path.
n.cells = [c for c in n.cells if not (
    c["cell_type"] == "code" and
    't_start = globals().get("t_start", time.time())' in "".join(c["source"]))]
for i, c in enumerate(n.cells):
    if c["cell_type"] == "code":
        s = "".join(c["source"])
        if "__HF_SYNC__" in s:
            s = (s.replace("__HF_SYNC__", HF_SYNC_SRC.strip("\n"))
                  .replace("__DATA__", CRVS_DATA_SRC.strip("\n"))
                  .replace("__METRICS__", CRVS_METRICS_SRC.strip("\n"))
                  .replace("__MODELS__", CRVS_MODELS_SRC.strip("\n"))
                  .replace("__LOSSES__", CRVS_LOSS_SRC.strip("\n"))
                  .replace("__ENGINE__", CRVS_ENGINE_SRC.strip("\n")))
            n.cells[i]["source"] = n._src(s)
        elif "__WORKER__" in s:
            n.cells[i]["source"] = n._src(
                s.replace("__WORKER__", NB03_WORKER_SRC.strip("\n")))
n.write(out, accelerator="nvidiaTeslaT4")
