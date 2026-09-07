#!/usr/bin/env python3
"""Emit 04_cardiomamba_train.ipynb"""
import sys
from nb_lib_a import NB, HF_SYNC_SRC, CRVS_DATA_SRC, CRVS_METRICS_SRC
from nb_lib_b import CRVS_MODELS_SRC, CRVS_CMNET_SRC, CRVS_LOSS_SRC, CRVS_ENGINE_SRC
from nb04_worker import NB04_WORKER_SRC

n = NB(); md, code = n.md, n.code

md(r"""
# NB04 — CardioMamba-Net

**Project:** CardioMamba-Net · **Stage:** 4 of 5
`01_verify` → `02_preprocess` → `03_baselines` → **`04_cardiomamba_train`** → `05_evaluate`

---

## The model, and why each piece is there

Every component answers a specific weakness identified in `00_admin/PLAN.md` §4. Nothing is here
because it is fashionable.

| | Contribution | Answers | What it does |
|---|---|---|---|
| **C1** | Physics-informed 8-channel input | G2 | The baseline collapses (I, Q) into one demodulated channel by hand. We feed all eight — I, Q, phase, displacement, velocity, **acceleration**, envelope, cardiac residual — through a learnable 1×1 mix, so the network can rediscover arctangent demodulation if that is genuinely optimal. Acceleration is the SCG analogue where the QRS-synchronous mechanical event actually lives. |
| **C2** | Dual-domain encoder | G3 | MultiRes convolution blocks (strong local morphology) running **alongside** a learnable lifting-wavelet branch (adaptive multi-resolution), fused per scale through channel attention. The cardiac signature is narrowband and non-stationary, buried under a much larger respiratory component — one view is not enough. |
| **C3** | Bidirectional SSM bottleneck | G4 | An 8 s window holds 8–10 cardiac cycles, and a convolutional decoder treats each independently. A bidirectional S4D state-space stack gives a global receptive field in linear time, so beat *n* informs beat *n+1*. **First SSM applied to radar→ECG waveform synthesis.** |
| **C4** | Multi-task decoder + peak-conditioned FiLM | G5, G1 | Three heads — waveform, R-peak heatmap, instantaneous RR. Then the novel part: peak logits feed back through FiLM to modulate the *final* decoder block of the waveform head, so the model is told "a QRS belongs here" **before** it draws one. |
| **C5** | Morphology-aware composite loss | G1 | Huber + multi-resolution STFT + focal BCE on peaks + L1 on RR + peak-weighted L1 + (1 − Pearson). MSE is the conditional mean, so it flattens the R peak — which is exactly why the baseline over-estimates RMSSD by ~2×. |

## ⚠️ Accelerator: **GPU T4 × 2**, Internet **On**, `HF_TOKEN` attached

Please also attach NB02's saved output with **+ Add Input → Notebook Output** before Run All.
The notebook verifies the mounted corpus and only downloads from HF when that input is absent.

## The ablation ladder

This notebook does not just train one model. It trains the **ladder** from `PLAN.md` §7, so the
paper can attribute every point of improvement to a specific component rather than waving at "our
architecture":

1. MultiResLinkNet + MSE — the baseline, our run *(from NB03)*
2. \+ composite loss only *(same backbone, C5)*
3. \+ 8-channel input only *(same backbone, C1)*
4. \+ C1 + C5 together
5. Full CardioMamba-Net, no wavelet branch *(ablates C2)*
6. Full, no SSM bottleneck *(ablates C3)*
7. Full, single-task *(ablates C4)*
8. Full, multi-task but no FiLM *(ablates the refinement specifically)*
9. **Full CardioMamba-Net** *(C1–C5)*
10. Full, Transformer bottleneck instead of SSM — the fair-fight control for C3

Same queue machinery as NB03: completed runs are skipped, the session stops cleanly at the time
budget, and re-running continues. **Canonical mode is now the default (`QUICK = False`)**. It
trains the five full-model headline folds first, then the remaining ablations and generalisation
experiments. `QUICK = True` remains available only as a 10-epoch architecture smoke test.

## Queue mode — one worker by default

The safe default is one Kaggle notebook: keep `QUEUE_WORKERS = 1` and `WORKER_ID = 0`. That worker
owns the entire queue, restores the current Hugging Face summaries/checkpoints, skips completed
runs, and continues every unfinished run. This is the mode to use when only one Kaggle session is
available.

Four external Kaggle copies are still optional: set `QUEUE_WORKERS = 4` in every copy and use a
distinct `WORKER_ID` from 0 through 3. Do not mix the one-worker and four-worker modes at the same
time, because they intentionally describe different ownership of the same canonical run IDs. Each
notebook uses both of its local T4 GPUs for one model; worker IDs partition experiments, not GPUs.

Each run may train for 150 epochs, cannot early-stop before epoch 40, and saves `best.pt` using
mean per-window validation temporal correlation — the same primary quantity reported at test
time. Test data is never used for checkpoint selection. These choices remove the old 10-epoch
under-training and loss/test-metric mismatch; they improve the validity of the comparison but do
not predetermine which architecture wins.

### Numerical recovery and legacy checkpoint repair (engine v7)

The earlier run left valid `best.pt` files but 58 current checkpoints stopped on NaN/Inf. This
version detects those v5 error states, rolls back to the last finite best checkpoint, and resumes
with a learning-rate reduction. A second recovery disables AMP; engine v7 also correctly restores
those older disabled-AMP checkpoints whose scaler state is intentionally empty. Non-finite
gradients are skipped before the optimizer can corrupt weights. Recovery is bounded and logged in
`recovery_events.jsonl` and the final summary—after the configured limit the run stops visibly
rather than being silently accepted. Existing runs with `summary.json` are never retrained.

## On `mamba-ssm`

The SSM here is **pure PyTorch** (S4D-Lin as an FFT convolution). It needs no `nvcc`, no custom
CUDA kernel, and it always builds on Kaggle — which the official `mamba-ssm` package frequently
does not. Set `CFG["BOTTLENECK"] = "transformer"` for the attention control. Both are in the ladder.

## Cell-by-cell run guide

| Code cell | What runs | Typical time |
|---:|---|---:|
| 1 | Configuration | < 5 s |
| 2 | Imports, dependency, dual-GPU and disk checks | 1–3 min |
| 3 | Write/import all versioned model/training libraries | 10–30 s |
| 4 | HF login and restore lightweight resume metadata | 1–5 min |
| 5 | Mount attached NB02 corpus or download fallback | < 1 min mounted; 3–12 min fallback |
| 6 | Smoke every architecture; parameters, memory and gradients | 5–15 min |
| 7 | Probe each composite-loss term and head gradient | 2–8 min |
| 8 | Build ablation plus A–F experiment queue | < 10 s |
| 9 | Define split/dataset/recording-safe evaluation helpers | < 10 s |
| 10 | Train/evaluate queue; exact mid-epoch checkpointing | quick: 30–75 min; full: many 10.5 h sessions |
| 11 | Merge NB03 baseline and produce ablation tables | 1–5 min |
| 12 | Draw curves and ablation figures | 2–8 min |
| 13 | Final blocking HF upload and run summary | 2–15 min |

The training cell prints per-epoch ETA. Full mode includes five folds, LOSO subjects, and held-out
scenarios, so it is intentionally resumed across multiple Kaggle sessions.
""")

md("---\n# 1 · Configuration")

code(r'''
CFG = {
    "SRC_REPO":  "Shanmuk4622/cr-rvs-radar-ecg-processed-v2",
    "BASELINE_REPO": "Shanmuk4622/cardiomamba-baselines-v2",
    "DST_REPO":  "Shanmuk4622/cardiomamba-net-v2",
    "HF_PRIVATE": False,
    "RUN_ID":    "nb04_cardiomamba_v2",

    "WORK":    "/kaggle/working/nb04",
    "SCRATCH": "/kaggle/temp/nb04",
    "PUSH_INTERVAL_S": 30 * 60,
    # Four simultaneous notebooks => at most 48 upload-folder calls/hour account-wide.
    "HF_MAX_UPLOADS_HOUR": 12,

    # ---- architecture (target: < 5 M params, < 1.5 GFLOPs per 8 s window) ----
    "CHANNELS_FULL": ["I", "Q", "phi", "dy", "vel", "acc", "amp", "cardiac"],   # C1
    "CHANNELS_BASE": ["dy"],                                                    # the baseline's input
    # BASE=32 gives ~4.1 M parameters. The 5 M budget is crossed at BASE>=40 (5.57 M),
    # so do not raise this without re-checking the budget cell.
    "BASE":       32,
    "LEVELS":     4,
    "D_SSM":      256,
    "SSM_BLOCKS": 3,
    "D_STATE":    64,
    "DROPOUT":    0.1,

    # ---- composite loss weights (C5). Tune w_stft and w_peakw first. --------
    "W": {"huber": 1.0, "stft": 0.5, "peak": 0.3, "rr": 0.1, "peakw": 0.5, "corr": 0.3},
    "HUBER_DELTA": 0.1,
    "PEAK_WEIGHT": 4.0,

    # ---- training ------------------------------------------------------------
    "EPOCHS":   150,
    "PATIENCE": 25,
    # Select the checkpoint on the same per-window temporal correlation used by the
    # headline evaluation. Do not allow patience to stop a full run before epoch 40.
    "MONITOR": "val_window_CC_temporal_mean",
    "MONITOR_MODE": "max",
    "MIN_EPOCHS": 40,
    "BATCH":    48,
    "WORKERS":  0,
    "PIN_MEMORY": False,
    "LR":       8e-4,
    "WEIGHT_DECAY": 1e-4,
    "AMP":      True,
    "MULTI_GPU": True,
    "REQUIRE_DUAL_T4": True,
    "SEED":     1337,
    "LOG_EVERY": 25,
    "CHECKPOINT_EVERY_STEPS": 1000,
    "CHECKPOINT_EVERY_S": 300,
    "EPOCHS_PER_PROCESS": 5,    # hard OS-level RAM/CUDA reset every five epochs
    "PROCESS_ISOLATION": True,  # required for long Kaggle queues
    # Numerical fail-safe: restore best.pt, reduce LR, and eventually disable AMP.
    "RECOVERY_LR_FACTOR": 0.25,
    "MAX_NUMERICAL_RECOVERIES": 3,
    "DISABLE_AMP_AFTER_RECOVERIES": 2,
    "MAX_NONFINITE_GRAD_BATCHES": 8,
    "SAME_SESSION_RECOVERY_RETRIES": 3,

    # ---- queue ---------------------------------------------------------------
    "EXPERIMENT":  "B_rva",      # the ablation ladder runs on the headline experiment
    "EXTRA_EXPERIMENTS": ["A_resting", "A_valsalva", "A_apnea", "C_all5"],  # full model only
    "RUN_LOSO": True,                 # Experiment D: one full-model run per retained subject
    "RUN_CROSS_SCENARIO": True,       # Experiment F: one held-out scenario per run
    "N_FOLDS":     5,
    "TIME_BUDGET_H": 10.5,
    "QUEUE_WORKERS": 1,   # default: one notebook owns every unfinished canonical run
    "WORKER_ID": 0,       # keep 0 in one-worker mode; use unique IDs 0..3 only with four copies
    # Canonical training is the default. QUICK remains available only for architecture smoke tests.
    "QUICK": False,
    "QUICK_EPOCHS": 10,
    "QUICK_FOLDS": 1,
    # NB03 is a scientific gate, not just a dependency. QUICK validation is still allowed
    # when it fails; the expensive full queue requires an explicit, visible override.
    # NB03's 80 runs completed, but its published-value reproduction gate did not pass.
    # Proceed knowingly so the canonical CardioMamba queue can be trained on the same splits.
    "ALLOW_FAILED_BASELINE_GATE": True,
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
QUEUE_WORKERS = int(CFG["QUEUE_WORKERS"])
WORKER_ID = int(CFG["WORKER_ID"])
if QUEUE_WORKERS < 1 or not 0 <= WORKER_ID < QUEUE_WORKERS:
    raise ValueError(f"WORKER_ID must be in 0..{QUEUE_WORKERS-1}; got {WORKER_ID}")
if not CFG["QUICK"] and QUEUE_WORKERS not in (1, 4):
    raise RuntimeError("Canonical NB04 supports QUEUE_WORKERS=1 (one notebook, WORKER_ID=0) "
                       "or QUEUE_WORKERS=4 (four copies, IDs 0..3). Do not mix modes.")
if QUEUE_WORKERS == 1 and WORKER_ID != 0:
    raise ValueError("One-worker mode requires WORKER_ID=0.")
SHARD_TAG = f"worker_{WORKER_ID}_of_{QUEUE_WORKERS}"
SHARD_META = WORK / "queue_workers" / SHARD_TAG
SHARD_RESULTS = WORK / "results" / "workers" / SHARD_TAG
SHARD_FIGURES = WORK / "figures" / "workers" / SHARD_TAG
for d in (WORK, SCRATCH, WORK / "runs", WORK / "results", WORK / "figures",
          SHARD_META, SHARD_RESULTS, SHARD_FIGURES):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORK))
print(f"queue worker {WORKER_ID}/{QUEUE_WORKERS-1} | namespace {SHARD_TAG}")

print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  {p.total_memory/2**30:.1f} GB")
    torch.backends.cudnn.benchmark = True
else:
    print("  !! NO GPU -- set Accelerator to 'GPU T4 x2'. Training on CPU is impractical here.")
if torch.cuda.device_count() < 2 or not all("T4" in torch.cuda.get_device_name(i)
                                            for i in range(torch.cuda.device_count())):
    raise RuntimeError("Select Kaggle Accelerator: GPU T4 x2, then restart and Run All.")
''')

md(r"""
---
# 2 · Library

Seven modules. `crvs_cmnet.py` is the new one — it holds the lifting wavelet, the S4D state-space
layer, the FiLM conditioning and the assembled network. The rest are byte-identical to NB03, so
the two notebooks train under exactly the same engine and evaluate with exactly the same metrics.
""")

code(r'''
MODULES = {
 "crvs_sync.py":    r"""
__HF_SYNC__
""",
 "crvs_data.py":    r"""
__DATA__
""",
 "crvs_metrics.py": r"""
__METRICS__
""",
 "crvs_models.py":  r"""
__MODELS__
""",
 "crvs_cmnet.py":   r"""
__CMNET__
""",
 "crvs_losses.py":  r"""
__LOSSES__
""",
 "crvs_engine.py":  r"""
__ENGINE__
""",
}
for nm, src in MODULES.items():
    (WORK / nm).write_text(src)
    print(f"  {nm:<20} {len(src):>7,} chars")
import hashlib
MODULE_HASHES = {nm: hashlib.sha256(src.encode()).hexdigest() for nm, src in MODULES.items()}
(WORK / "library_hashes.json").write_text(json.dumps(MODULE_HASHES, indent=2))

# Purge before importing: Python caches modules in sys.modules, so re-running this cell
# after updating the notebook would silently keep the previous version of the library.
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
        f"\n  Restart the kernel (Run -> Restart & Run All).\n{'='*74}")
print(f"\ncrvs_data v{crvs_data.LIB_VERSION} loaded from {crvs_data.__file__}")
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

from crvs_sync import HFSync
sync = HFSync(repo_id=CFG["DST_REPO"], local_dir=WORK, token=HF_TOKEN, repo_type="model",
              private=CFG["HF_PRIVATE"], run_id=f"{CFG['RUN_ID']}__{SHARD_TAG}",
              push_interval_s=CFG["PUSH_INTERVAL_S"],
              max_upload_calls_hour=CFG["HF_MAX_UPLOADS_HOUR"])
# The uploader owns the whole local folder, but each parallel queue worker keeps independent
# durable bookkeeping. Only immutable/non-overlapping run directories are shared on HF.
sync.history = SHARD_META / "sync_history.jsonl"
sync.state_path = SHARD_META / "sync_state.json"
sync.log("queue_worker_ready", worker_id=WORKER_ID, queue_workers=QUEUE_WORKERS,
         namespace=SHARD_TAG)
print("\nresults repo:", sync.url, "(public)")

_M = {"f": False, "n": ""}
def MAJOR(nm):
    _M["f"] = True; _M["n"] = nm
def _hook(r=None):
    if _M["f"]:
        nm = _M["n"]; _M["f"] = False; _M["n"] = ""; sync.stage_done(nm)
try:
    get_ipython().events.register("post_run_cell", _hook)
    print("post-run-cell push hook registered")
except Exception as e:
    print("hook unavailable:", e)

sync.pull(allow_patterns=["*.json", "*.jsonl", "*.csv", "*.md",
                          "runs/**/summary.json", "runs/**/state.json",
                          f"queue_workers/{SHARD_TAG}/*", "results/*"])
STATE = sync.load_state({"completed": [], "sessions": 0, "version": 3,
                         "worker_id": WORKER_ID, "queue_workers": QUEUE_WORKERS})
STATE["sessions"] = STATE.get("sessions", 0) + 1
STATE["worker_id"] = WORKER_ID
STATE["queue_workers"] = QUEUE_WORKERS
sync.save_state(STATE)
print(f"{SHARD_TAG} session #{STATE['sessions']} | "
      f"{len(STATE['completed'])} recorded run(s) complete in this shard")

from huggingface_hub import hf_hub_download
try:
    gate_path = hf_hub_download(
        CFG["BASELINE_REPO"], "results/reproduction_gate.json", repo_type="model",
        token=HF_TOKEN, local_dir=str(SCRATCH / "baseline_gate"))
    BASELINE_GATE = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    print("baseline reproduction gate:",
          f"complete={BASELINE_GATE.get('complete')} passed={BASELINE_GATE.get('passed')}")
    print("  ranking:", " > ".join(BASELINE_GATE.get("ranking", [])))
    print("  MultiResLinkNet CC_t:", BASELINE_GATE.get("multireslinknet_cc_temporal"),
          "| published:", BASELINE_GATE.get("published_cc_temporal"))
except Exception as e:
    BASELINE_GATE = {"complete": False, "passed": False,
                     "read_error": f"{type(e).__name__}: {e}"}
    print("WARNING: could not verify NB03 reproduction gate:", BASELINE_GATE["read_error"])
MAJOR("00_setup")
''')

code(r'''
from huggingface_hub import snapshot_download
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
_npy = {p.stem for p in REC_DIR.glob("*.npy")}
_npz = {p.stem for p in REC_DIR.glob("*.npz")}
_have = _npy | _npz
print(f"recording files: {len(_npy)} .npy (fast) + {len(_npz)} .npz (legacy)")
if not _have:
    raise RuntimeError("No recording files downloaded -- check NB02 finished and that "
                       "SRC_REPO matches its DST_REPO.")
_missing = sorted(set(RECS["rec_id"]) - _have)
if _missing:
    print(f"WARNING: {len(_missing)} recording(s) have no file; dropping their windows")
    W = W[~W["rec_id"].isin(_missing)].reset_index(drop=True)
    RECS = RECS[~RECS["rec_id"].isin(_missing)].reset_index(drop=True)
print(f"windows {len(W):,} | recordings {len(RECS)} | subjects {RECS['subject'].nunique()}")
print("channel order on disk:", EXPINFO["channels"])
''')

md(r"""
---
# 3 · Build the network and check the budget

Before training we prove the model runs, count its parameters, and confirm every head produces a
finite output with gradients. The parameter budget matters for the paper: the baseline reports
**no** parameter or FLOP count, so "we win *and* we are smaller" is a free column in our results
table — provided we actually are smaller. Target is **under 5 M**.
""")

code(r'''
import torch.nn.functional as F
from crvs_cmnet import build_cmnet
from crvs_models import build_baseline, count_params
from crvs_losses import CompositeLoss, MSEOnly
from crvs_engine import Trainer, seed_all, pick_device

seed_all(CFG["SEED"])
dev, ngpu, gnames = pick_device()
print(f"device {dev} | gpus {ngpu} {gnames}\n")

def make_model(spec):
    if spec["kind"] == "baseline":
        return build_baseline(spec["model"], in_ch=len(spec["channels"]), out_ch=1,
                              base=64, levels=CFG["LEVELS"])
    return build_cmnet(in_ch=len(spec["channels"]), base=CFG["BASE"], levels=CFG["LEVELS"],
                       d_ssm=CFG["D_SSM"], ssm_blocks=CFG["SSM_BLOCKS"],
                       d_state=CFG["D_STATE"], bottleneck=spec.get("bottleneck", "ssm"),
                       use_wavelet=spec.get("wavelet", True),
                       multitask=spec.get("multitask", True),
                       use_film=spec.get("film", True), dropout=CFG["DROPOUT"])

VARIANTS = {
  # kind      channels                  bottleneck   wavelet multitask film  loss
  "L2_loss_only":  dict(kind="baseline", model="multireslinknet", channels=CFG["CHANNELS_BASE"], loss="composite"),
  "L3_c1_only":    dict(kind="baseline", model="multireslinknet", channels=CFG["CHANNELS_FULL"], loss="mse"),
  "L4_c1_c5":      dict(kind="baseline", model="multireslinknet", channels=CFG["CHANNELS_FULL"], loss="composite"),
  "L5_no_wavelet": dict(kind="cmnet", channels=CFG["CHANNELS_FULL"], wavelet=False, loss="composite"),
  "L6_no_ssm":     dict(kind="cmnet", channels=CFG["CHANNELS_FULL"], bottleneck="conv", loss="composite"),
  "L7_singletask": dict(kind="cmnet", channels=CFG["CHANNELS_FULL"], multitask=False, loss="composite"),
  "L8_no_film":    dict(kind="cmnet", channels=CFG["CHANNELS_FULL"], film=False, loss="composite"),
  "L9_full":       dict(kind="cmnet", channels=CFG["CHANNELS_FULL"], loss="composite"),
  "L10_transformer": dict(kind="cmnet", channels=CFG["CHANNELS_FULL"], bottleneck="transformer", loss="composite"),
}

print(f"{'variant':<18}{'in':>4}{'params':>12}{'MB':>7}{'fwd ms':>9}  heads")
print("-" * 78)
budget = []
for nm, spec in VARIANTS.items():
    m = make_model(spec).to(dev); m.train()
    x = torch.randn(2, len(spec["channels"]), 1024, device=dev)
    t0 = time.time(); out = m(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000
    y = torch.randn(2, 1, 1024, device=dev).clamp(-1, 1)
    pk = torch.rand(2, 1, 1024, device=dev)
    rr = torch.rand(2, 1, 1024, device=dev) + 0.5
    lf = CompositeLoss(**{f"w_{k}": v for k, v in CFG["W"].items()},
                       huber_delta=CFG["HUBER_DELTA"], peak_weight=CFG["PEAK_WEIGHT"])
    loss, parts = lf(out, y, pk, rr)
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
    heads = "+".join(k for k in ("wave", "peak", "rr") if k in out)
    ok = torch.isfinite(out["wave"]).all() and gn > 0 and math.isfinite(float(loss))
    budget.append({"variant": nm, "in_ch": len(spec["channels"]), "params": p,
                   "mb": p * 4 / 2**20, "fwd_ms_batch2": dt,
                   "gflops_per_window": gflops, "ok": bool(ok)})
    print(f"{nm:<18}{len(spec['channels']):>4}{p:>12,}{p*4/2**20:>7.1f}{dt:>9.1f}  "
          f"{heads}  {'OK' if ok else 'FAIL'}")
    del m, out, loss
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

B = pd.DataFrame(budget)
BUDGET_BY_VARIANT = B.set_index("variant").to_dict("index")
B.to_csv(SHARD_RESULTS / "variant_budget.csv", index=False)
if not B["ok"].all():
    raise RuntimeError("a variant failed its smoke test:\n" + B.to_string(index=False))

# ---- structural self-checks -------------------------------------------------
# Two invariants that are invisible if they break: a decoder skip whose channels do not
# line up gets dropped silently (that once cost 59 % of MultiResLinkNet's gradient), and a
# wavelet branch one octave off its conv counterpart makes C2 fuse mismatched resolutions
# while still training fine. Both are checked here, before any GPU time is spent.
print("\nstructural self-checks")
print("-" * 66)
m = make_model(VARIANTS["L9_full"]).to(dev); m.train()
xx = torch.randn(2, 8, 1024, device=dev)
_interp = F.interpolate
_calls = {"n": 0}
def _counting(*a, **k):
    _calls["n"] += 1
    return _interp(*a, **k)
F.interpolate = _counting
try:
    oo = m(xx)
finally:
    F.interpolate = _interp
_verdict = ("aligned octave for octave" if _calls["n"] == 0
            else "MISALIGNED -- C2 would be fusing mismatched resolutions")
print(f"  fusion upsampling calls : {_calls['n']}   ({_verdict})")

with torch.no_grad():
    h0 = m.stem(m.mix(xx))
    wf = m.wave(h0)
    hh = h0; lens = []
    for i, e in enumerate(m.encs):
        hh = e(hh)
        lens.append((i, hh.shape[-1], wf[i].shape[-1], hh.shape[1], wf[i].shape[1]))
        hh = F.max_pool1d(hh, 2)
print(f"  {'level':<7}{'conv len':>10}{'wavelet len':>13}{'conv ch':>9}{'wav ch':>8}   match")
for i, cl, wl, cc, wc in lens:
    print(f"  {i:<7}{cl:>10}{wl:>13}{cc:>9}{wc:>8}   {'yes' if cl == wl else 'NO'}")
if any(cl != wl for _, cl, wl, _, _ in lens):
    raise RuntimeError("wavelet branch is not octave-aligned with the convolution branch")

# Reachability must be tested through the loss used for training. Backpropagating only
# `oo["wave"]` correctly leaves the independent RR head untouched and used to produce a
# false failure here. Positive synthetic RR targets exercise all three task heads.
yy = torch.randn(2, 1, 1024, device=dev).clamp(-1, 1)
pp = torch.rand(2, 1, 1024, device=dev)
rr_target = torch.rand(2, 1, 1024, device=dev) + 0.5
reach_loss, _ = CompositeLoss(
    **{f"w_{k}": v for k, v in CFG["W"].items()},
    huber_delta=CFG["HUBER_DELTA"], peak_weight=CFG["PEAK_WEIGHT"]
)(oo, yy, pp, rr_target)
reach_loss.backward()
dead = [n_ for n_, p_ in m.named_parameters() if p_.grad is None]
nonfinite = [n_ for n_, p_ in m.named_parameters()
             if p_.grad is not None and not torch.isfinite(p_.grad).all()]
n_dead = sum(p_.numel() for n_, p_ in m.named_parameters() if n_ in set(dead))
print(f"\n  parameters unreachable from composite loss: {len(dead)} tensors / {n_dead:,} values")
if dead or nonfinite:
    from collections import Counter
    if dead:
        print("  unreachable by submodule:", dict(Counter(d.split(".")[0] for d in dead)))
    if nonfinite:
        print("  non-finite gradients:", nonfinite[:12])
    raise RuntimeError(
        "some parameters are unreachable from the real composite loss, or have non-finite "
        "gradients. Inspect the named submodules above.")
print("  every parameter is reachable through the real composite loss.")
del m, oo, reach_loss, yy, pp, rr_target
gc.collect()
if dev.type == "cuda":
    torch.cuda.empty_cache()
full = B[B["variant"] == "L9_full"].iloc[0]
print(f"\nCardioMamba-Net (full): {full['params']:,} params ({full['mb']:.1f} MB)")
print(f"budget target < 5 M   -> {'WITHIN BUDGET' if full['params'] < 5e6 else 'OVER BUDGET'}")
print("\nloss terms on the smoke batch:", {k: round(v, 4) for k, v in parts.items()})
MAJOR("01_smoke")
''')

md(r"""
---
# 4 · Verify the composite loss actually does what it claims

A loss with six terms is easy to get wrong in a way that still trains. Two checks:

1. **Every term responds to the right error.** Corrupt a signal in a specific way and confirm the
   corresponding term rises while the others stay put — smoothing the QRS should spike the
   peak-weighted and STFT terms far more than plain Huber. That difference *is* the whole argument
   for C5, so we measure it rather than asserting it.
2. **The gradient reaches every head.** If the RR head has no gradient path, the multi-task claim
   is empty.
""")

code(r'''
import torch.nn.functional as Fnn
from scipy import signal as ss_

fs = 128; L = 1024
t = np.arange(L) / fs
beat = int(fs * 60 / 68)
clean = np.zeros(L); clean[::beat] = 1.0
clean = ss_.convolve(clean, ss_.windows.gaussian(21, 2.2), "same")
clean = clean / (np.abs(clean).max() + 1e-9)
pkmap = np.zeros(L)
for p in range(0, L, beat):
    a, b = max(0, p - 9), min(L, p + 10)
    g = np.exp(-0.5 * ((np.arange(a, b) - p) / 3.0) ** 2)
    pkmap[a:b] = np.maximum(pkmap[a:b], g)

def T(a):
    return torch.tensor(a, dtype=torch.float32, device=dev).reshape(1, 1, -1)

y = T(clean); pk = T(pkmap); rr = T(np.full(L, beat / fs))
cases = {
    "perfect":            clean.copy(),
    "smoothed QRS":       ss_.convolve(clean, ss_.windows.gaussian(25, 5), "same"),
    "amplitude x0.5":     clean * 0.5,
    "shifted +6 samples": np.roll(clean, 6),
    "white noise added":  clean + 0.10 * np.random.RandomState(0).randn(L),
}
lf = CompositeLoss(**{f"w_{k}": v for k, v in CFG["W"].items()},
                   huber_delta=CFG["HUBER_DELTA"], peak_weight=CFG["PEAK_WEIGHT"])
rows = []
for nm, sig in cases.items():
    sig = sig / (np.abs(sig).max() + 1e-9)
    pred = {"wave": T(sig), "peak": T(pkmap * 4 - 2), "rr": T(np.full(L, beat / fs))}
    tot, parts = lf(pred, y, pk, rr)
    mse = float(Fnn.mse_loss(T(sig), y))
    rows.append({"case": nm, "MSE": mse, **{k: round(v, 5) for k, v in parts.items()},
                 "total": round(float(tot), 5)})
D = pd.DataFrame(rows)
print(D.to_string(index=False))

base = D[D["case"] == "perfect"].iloc[0]
sm = D[D["case"] == "smoothed QRS"].iloc[0]
print("\nsmoothed QRS vs perfect -- relative rise per term:")
for term in ("MSE", "huber", "stft", "peakw", "corr"):
    if term in D.columns:
        b = max(float(base[term]), 1e-6)
        print(f"  {term:<8} x{float(sm[term])/b:>8.1f}")
print("\nThis is the argument for C5 in one table: smoothing the QRS barely moves MSE,")
print("but the peak-weighted and spectral terms react strongly. A model trained on MSE")
print("alone has almost no incentive to keep the R peak sharp.")
D.to_csv(SHARD_RESULTS / "loss_probe.csv", index=False)

m = make_model(VARIANTS["L9_full"]).to(dev); m.train()
x = torch.randn(2, 8, 1024, device=dev)
out = m(x)
loss, _ = lf(out, torch.randn(2,1,1024,device=dev).clamp(-1,1),
             torch.rand(2,1,1024,device=dev), torch.rand(2,1,1024,device=dev)+.5)
loss.backward()
heads = {"head_wave": 0.0, "head_peak": 0.0, "head_rr": 0.0, "film": 0.0, "wave": 0.0, "bott": 0.0}
for nmp, p in m.named_parameters():
    if p.grad is None:
        continue
    for h in heads:
        if nmp.startswith(h):
            heads[h] += float(p.grad.norm())
print("\ngradient norm reaching each component:")
for h, v in heads.items():
    print(f"  {h:<12} {v:>12.5f}  {'OK' if v > 0 else 'NO GRADIENT'}")
assert heads["head_peak"] > 0 and heads["head_rr"] > 0, "a multi-task head is not learning"
del m
gc.collect()
if dev.type == "cuda":
    torch.cuda.empty_cache()
MAJOR("02_loss_probe")
''')

md(r"""
---
# 5 · The ablation ladder queue

Rungs 2–10 on the headline experiment (`B_rva`), plus the full model on every other experiment so
NB05 has the per-scenario and all-five-scenario numbers. Rung 1 (MultiResLinkNet + MSE) comes from
NB03 — it is the same run, so there is no reason to train it twice.
""")

code(r'''
folds = list(range(CFG["QUICK_FOLDS"] if CFG["QUICK"] else CFG["N_FOLDS"]))
EPOCHS = CFG["QUICK_EPOCHS"] if CFG["QUICK"] else CFG["EPOCHS"]

QUEUE = []
queue_variants = (["L9_full"] if CFG["QUICK"] else
                  ["L9_full"] + [v for v in VARIANTS if v != "L9_full"])
for vname in queue_variants:                            # ladder on B_rva
    for f in folds:
        prefix = "quick__" if CFG["QUICK"] else ""
        QUEUE.append({"run_id": f"{prefix}{CFG['EXPERIMENT']}__{vname}__f{f}",
                      "exp": CFG["EXPERIMENT"], "variant": vname, "fold": f})
if not CFG["QUICK"]:                                    # full model everywhere else
    for exp in CFG["EXTRA_EXPERIMENTS"]:
        for f in folds:
            QUEUE.append({"run_id": f"{exp}__L9_full__f{f}", "exp": exp,
                          "variant": "L9_full", "fold": f})
    if CFG["RUN_LOSO"]:
        for lid in EXPINFO.get("loso_values", sorted(map(int, W["loso_id"].unique()))):
            QUEUE.append({"run_id": f"D_loso__L9_full__s{lid}", "exp": "D_loso",
                          "variant": "L9_full", "fold": int(lid)})
    if CFG["RUN_CROSS_SCENARIO"]:
        for sc in EXPINFO.get("cross_scenarios", sorted(W["scenario_canon"].unique())):
            safe = str(sc).lower().replace("-", "_").replace(" ", "_")
            QUEUE.append({"run_id": f"F_cross_{safe}__L9_full", "exp": f"F_cross:{sc}",
                          "variant": "L9_full", "fold": 0})

GLOBAL_QUEUE = list(QUEUE)
QUEUE = [q for queue_index, q in enumerate(GLOBAL_QUEUE)
         if queue_index % QUEUE_WORKERS == WORKER_ID]
completed_from_summaries = {
    p.parent.name for p in (WORK / "runs").glob("*/summary.json")
}
global_queue_ids = {q["run_id"] for q in GLOBAL_QUEUE}
global_done_at_start = completed_from_summaries & global_queue_ids
queue_ids = {q["run_id"] for q in QUEUE}
# A final summary is the completion authority. Old worker state is advisory only: if it says a
# run completed but summary.json is absent, that run must be resumed instead of silently skipped.
recorded_completed = set(STATE.get("completed", [])) & queue_ids
stale_completed = recorded_completed - completed_from_summaries
if stale_completed:
    print(f"ignoring {len(stale_completed)} stale completion marker(s) without summary.json")
completed_all = completed_from_summaries & queue_ids
done = set(completed_all)
todo = [q for q in QUEUE if q["run_id"] not in done]
print(f"global queue: {len(GLOBAL_QUEUE)} run(s) | at least {len(global_done_at_start)} already done")
print(f"{SHARD_TAG}: {len(QUEUE)} assigned | {len(done)} done | {len(todo)} remaining")
print(f"epochs {EPOCHS} | folds {folds} | budget {CFG['TIME_BUDGET_H']} h")
if CFG["QUICK"]:
    print("\n>>> QUICK MODE: full model only, 1 fold, 10 epochs. Then set QUICK=False.")
if not BASELINE_GATE.get("passed", False):
    print("\n>>> BASELINE REPRODUCTION GATE HAS NOT PASSED.")
    if CFG["QUICK"]:
        print("    QUICK architecture validation may run, but full training is blocked.")
    elif not CFG["ALLOW_FAILED_BASELINE_GATE"]:
        raise RuntimeError(
            "NB03 is complete but its reproduction gate failed. Review/fix NB03 before spending "
            "GPU time on the full NB04 queue. To proceed knowingly, set "
            "CFG['ALLOW_FAILED_BASELINE_GATE']=True.")
    else:
        print("    WARNING: explicit override enabled; downstream comparisons are not validated.")
print("\nnext up:")
for q in todo[:10]:
    print("   ", q["run_id"])
if len(todo) > 10:
    print(f"    ... and {len(todo)-10} more")
''')

code(r'''
from crvs_data import WindowDataset, FS
from crvs_metrics import seg_metrics, detect_r_peaks, hrv_from_peaks, peak_detection_scores

def split_for(exp, fold, n_folds=None):
    n_folds = n_folds or CFG["N_FOLDS"]
    if exp == "D_loso":
        ids = EXPINFO.get("loso_values", sorted(map(int, W["loso_id"].unique())))
        fold = int(fold); vid = ids[(ids.index(fold) + 1) % len(ids)]
        sub = W[W["scenario_canon"].isin(EXPERIMENTS["C_all5"])]
        tr = sub[~sub["loso_id"].isin([fold, vid])]
        va = sub[(sub["loso_id"] == vid) & sub["no_overlap"]]
        te = sub[(sub["loso_id"] == fold) & sub["no_overlap"]]
        return tr, va, te
    if exp.startswith("F_cross:"):
        test_sc = exp.split(":", 1)[1]
        scenarios = EXPINFO.get("cross_scenarios", list(EXPERIMENTS["C_all5"]))
        val_sc = scenarios[(scenarios.index(test_sc) + 1) % len(scenarios)]
        tr = W[~W["scenario_canon"].isin([test_sc, val_sc])]
        va = W[(W["scenario_canon"] == val_sc) & W["no_overlap"]]
        te = W[(W["scenario_canon"] == test_sc) & W["no_overlap"]]
        return tr, va, te
    sub = W[W["scenario_canon"].isin(EXPERIMENTS[exp])]
    te_g, va_g = fold % n_folds, (fold + 1) % n_folds
    tr = sub[~sub["fold_group"].isin([te_g, va_g])]
    va = sub[(sub["fold_group"] == va_g) & sub["no_overlap"]]
    te = sub[(sub["fold_group"] == te_g) & sub["no_overlap"]]
    assert not (set(tr["subject"]) & set(te["subject"])), "SUBJECT LEAK"
    return tr, va, te

def make_datasets(exp, fold, channels):
    tr, va, te = split_for(exp, fold)
    if not exp.startswith("F_cross:"):
        assert not (set(tr["subject"]) & set(te["subject"])), "SUBJECT LEAK train/test"
        assert not (set(va["subject"]) & set(te["subject"])), "SUBJECT LEAK val/test"
    norm_key = f"{exp}|{fold}" if not exp.startswith("F_cross:") else f"{exp}|0"
    norm = NORM.get(norm_key)
    if norm is None:
        raise RuntimeError(f"no normalisation stats for {norm_key} -- re-run NB02 v2")
    idx = [EXPINFO["channels"].index(c) for c in channels]
    sn = {"mean": [norm["mean"][i] for i in idx], "std": [norm["std"][i] for i in idx]}
    mk = lambda d, aug: WindowDataset(REC_DIR, d, sn, channels, augment=aug,
                                      seed=CFG["SEED"] + fold)
    return mk(tr, True), mk(va, False), mk(te, False), (tr, va, te)

def evaluate(Y, P, index, out_dir):
    idx_frame = index.reset_index(drop=True)
    subs = idx_frame["subject"].to_numpy(); scen = idx_frame["scenario_canon"].to_numpy()
    rows = []
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
    hr = []
    for rid, grp in idx_frame.assign(_row=np.arange(len(idx_frame))).groupby("rec_id"):
        grp = grp.sort_values("start"); pos = grp["_row"].to_numpy()
        if len(pos) < 2:
            continue
        yg = np.concatenate(Y[pos]); yp = np.concatenate(P[pos])
        g = hrv_from_peaks(detect_r_peaks(yg, FS), FS)
        pr = hrv_from_peaks(detect_r_peaks(yp, FS), FS)
        hr.append({"rec_id": rid, "subject": str(grp["subject"].iloc[0]),
                   "scenario": str(grp["scenario_canon"].iloc[0]),
                   **{f"gt_{k}": v for k, v in g.items()},
                   **{f"pr_{k}": v for k, v in pr.items()},
                   **peak_detection_scores(yg, yp, FS)})
    dfh = pd.DataFrame(hr)
    if len(dfh):
        dfh.to_parquet(out_dir / "metrics_recordings.parquet", index=False)
        numeric = [c for c in dfh.columns if dfh[c].dtype.kind in "fi"]
        dfh.groupby("subject", as_index=False)[numeric].mean().to_parquet(
            out_dir / "metrics_subjects.parquet", index=False)
        for k in ("F1", "precision", "recall", "accuracy", "missed_rate", "timing_err_ms_median"):
            if k in dfh.columns:
                agg["peak_" + k] = float(dfh[k].mean())
        for k in ("mean_hr_bpm", "rmssd_ms"):
            if f"gt_{k}" in dfh and f"pr_{k}" in dfh:
                agg["MAE_" + k] = float((dfh[f"gt_{k}"] - dfh[f"pr_{k}"]).abs().mean())
    return agg, dfw
''')

md(r"""
---
# 6 · Train the canonical queue

Interrupt-safe and resumable inside an isolated worker process, exactly as in the repaired NB03.
Every five epochs the worker exits and Linux reclaims all of its RAM and CUDA state; the parent
immediately launches the next chunk. The five headline folds for `L9_full` run first, so a
scientifically usable full-model result is produced as early as possible. Checkpoint selection and
early stopping use `val_window_CC_temporal_mean`, with at least 40 epochs of training. Watch both
`CCt-win` and `CCt-global` in each epoch line. If another rung ultimately beats rung 9, the
ablation is evidence and must be reported honestly; the notebook never tunes on the test set.
""")

code(r'''
# The parent notebook never constructs training datasets or models. Each child handles at most
# EPOCHS_PER_PROCESS epochs and exits, so Linux reclaims every Python/Arrow/CUDA allocation.
WORKER_SRC = r"""
__WORKER__
"""
WORKER_PATH = WORK / "nb04_run_worker.py"
WORKER_PATH.write_text(WORKER_SRC, encoding="utf-8")
MEMORY_RUNTIME_VERSION = "nb04-process-isolated-numerical-recovery-v3"
worker_cfg = dict(CFG)
worker_cfg["ACTIVE_EPOCHS"] = EPOCHS
WORKER_CONTEXT = SHARD_META / "worker_context.json"
WORKER_STATUS = SHARD_META / "worker_status.json"
WORKER_CONTEXT.write_text(json.dumps({
    "cfg": worker_cfg, "variants": VARIANTS, "work": str(WORK), "data": str(DATA),
    "status_path": str(WORKER_STATUS),
    "data_hash": DATA_HASH, "module_hashes": MODULE_HASHES,
    "budget_by_variant": BUDGET_BY_VARIANT,
    "memory_runtime_version": MEMORY_RUNTIME_VERSION,
}, indent=2, default=str), encoding="utf-8")

def _linux_memory_gb():
    result = {"process_rss_gb": float("nan"), "host_ram_available_gb": float("nan")}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["process_rss_gb"] = float(line.split()[1]) / 2**20
                break
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                result["host_ram_available_gb"] = float(line.split()[1]) / 2**20
                break
    except Exception:
        pass
    return result

def release_runtime_memory(label=None):
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
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    stats = _linux_memory_gb()
    if label:
        print(f"  RAM after {label}: {stats['process_rss_gb']:.2f} GB RSS | "
              f"{stats['host_ram_available_gb']:.2f} GB host available")
    return stats

t_start = time.time()
budget_s = CFG["TIME_BUDGET_H"] * 3600
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
        # Never force-kill while a checkpoint may be moving into place. The parent interrupt
        # hook uploads the latest complete atomic checkpoint that exists.
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
    status_path = WORKER_STATUS
    if status_path.exists():
        status_path.unlink()
    cmd = [sys.executable, "-u", str(WORKER_PATH), "--context", str(WORKER_CONTEXT),
           "--run-id", rid, "--experiment", q["exp"], "--variant", q["variant"],
           "--fold", str(q["fold"])]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    sync.mark_dirty(f"{rid}:isolated-worker-active")
    proc = subprocess.Popen(cmd, cwd=str(WORK), env=env)
    _active_worker.update(proc=proc, run_id=rid)
    sync.set_before_final_flush(_stop_active_worker)
    try:
        return_code = proc.wait()
    finally:
        _active_worker.update(proc=None, run_id=None)
        sync.set_before_final_flush(None)
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        status = {"status": "failed", "run_id": rid,
                  "error": f"worker exited {return_code} without a status file"}
    status["return_code"] = return_code
    sync.mark_dirty(f"{rid}:isolated-worker-{status.get('status')}")
    parent_ram = release_runtime_memory(f"worker {rid} exited")
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
    numerical_retry_count = 0
    while True:
        if time.time() - t_start >= budget_s:
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
        worker_state = status.get("status")
        if worker_state == "chunk_complete":
            print(f"  worker exited cleanly at epoch {status.get('epoch')}; "
                  "OS RAM/CUDA state reclaimed; launching the next chunk.")
            continue
        if worker_state == "run_complete" and (out / "summary.json").exists():
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            metrics = summary.get("metrics", {})
            done.add(rid)
            completed_all.add(rid)
            completed_now.append(rid)
            STATE["completed"] = sorted(completed_all)
            sync.save_state(STATE)
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
        sync.flush(force=True, msg=f"{rid} worker failure checkpoint")
        is_numerical = ("non-finite" in str(error).lower() or
                        "floatingpointerror" in str(error).lower())
        if (is_numerical and
                numerical_retry_count < int(CFG["SAME_SESSION_RECOVERY_RETRIES"])):
            numerical_retry_count += 1
            print(f"  automatic numerical recovery {numerical_retry_count}/"
                  f"{CFG['SAME_SESSION_RECOVERY_RETRIES']}: launching a fresh process; "
                  "it will roll back to best.pt and lower the learning rate")
            continue
        failed_now.append({"run_id": rid, "error": error,
                           "recovery_retries": numerical_retry_count})
        break
    if session_stop_reason:
        break

print(f"\ncompleted this session: {len(completed_now)}  |  "
      f"{SHARD_TAG}: {len(done)}/{len(QUEUE)}")
print(f"worker failures this session: {len(failed_now)}")
print("Every worker exited; Linux reclaimed its full RAM and CUDA context.")
MAJOR("03_training_isolated")
''')

md(r"""
---
# 7 · The ablation table

The table the paper's Discussion is built on. Each rung adds one component; the `Δ CC_t` column is
what that component is worth. This is what turns "our architecture is better" into "the SSM
bottleneck contributes 3.2 points of temporal correlation".

Rung 1 is pulled from NB03's separate baseline v2 repo.
""")

code(r'''
rows = []
for p in sorted((WORK / "runs").glob("*/summary.json")):
    try:
        s = json.loads(p.read_text())
        is_quick = str(s.get("run_id", p.parent.name)).startswith("quick__")
        if is_quick != bool(CFG["QUICK"]):
            continue
        rows.append({"experiment": s["experiment"], "variant": s.get("variant", s.get("model")),
                     "fold": s["fold"], "params": s.get("params"),
                     **{k: v for k, v in s["metrics"].items() if not k.endswith("_std")}})
    except Exception:
        pass
# Rung 1 comes from NB03's separate v2 repo. Pull summaries only (not checkpoints).
baseline_cache = SCRATCH / "baseline_compare"
try:
    snapshot_download(CFG["BASELINE_REPO"], repo_type="model", token=HF_TOKEN,
                      local_dir=str(baseline_cache),
                      allow_patterns=["runs/B_rva__multireslinknet__f*/summary.json"], max_workers=4)
except Exception as e:
    print("baseline summaries unavailable:", type(e).__name__, e)
for p in sorted((baseline_cache / "runs").glob("B_rva__multireslinknet__f*/summary.json")):
    try:
        s = json.loads(p.read_text())
        rows.append({"experiment": s["experiment"], "variant": "L1_baseline_mse",
                     "fold": s["fold"], "params": s.get("params"),
                     **{k: v for k, v in s["metrics"].items() if not k.endswith("_std")}})
    except Exception:
        pass

R = pd.DataFrame(rows)
if not len(R):
    print("no completed runs yet -- run the training cell.")
else:
    R.to_csv(SHARD_RESULTS / "runs_raw.csv", index=False)
    LADDER = ["L1_baseline_mse", "L2_loss_only", "L3_c1_only", "L4_c1_c5", "L5_no_wavelet",
              "L6_no_ssm", "L7_singletask", "L8_no_film", "L9_full", "L10_transformer"]
    LABEL = {
      "L1_baseline_mse": "1. MultiResLinkNet + MSE  (baseline)",
      "L2_loss_only":    "2. + composite loss (C5)",
      "L3_c1_only":      "3. + 8-channel input (C1)",
      "L4_c1_c5":        "4. + C1 + C5",
      "L5_no_wavelet":   "5. CardioMamba, no wavelet (-C2)",
      "L6_no_ssm":       "6. CardioMamba, no SSM (-C3)",
      "L7_singletask":   "7. CardioMamba, single-task (-C4)",
      "L8_no_film":      "8. CardioMamba, no FiLM refinement",
      "L9_full":         "9. CardioMamba-Net (full, C1-C5)",
      "L10_transformer": "10. Transformer bottleneck (control)",
    }
    cols = ["MAE", "MSE", "CC_temporal", "CC_spectral", "RRMSE_temporal", "RRMSE_spectral"]
    extra = [c for c in ("peak_F1", "MAE_mean_hr_bpm", "MAE_rmssd_ms") if c in R.columns]
    b = R[R["experiment"] == CFG["EXPERIMENT"]]
    A = (b.groupby("variant")[cols + extra + ["params"]].mean()
          .reindex([v for v in LADDER if v in set(b["variant"])]))
    A["folds"] = b.groupby("variant").size().reindex(A.index)
    ref = A.loc["L1_baseline_mse", "CC_temporal"] if "L1_baseline_mse" in A.index else np.nan
    A["dCC_t_vs_baseline"] = (A["CC_temporal"] - ref).round(2)
    A.index = [LABEL.get(i, i) for i in A.index]
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("=" * 118)
    print(f"ABLATION LADDER  —  experiment {CFG['EXPERIMENT']}  (mean over folds)")
    print("=" * 118)
    print(A.round(5).to_string())
    A.to_csv(SHARD_RESULTS / "ablation.csv")

    print("\n" + "-" * 118)
    print("TARGETS FROM PLAN.md")
    print("-" * 118)
    if "9. CardioMamba-Net (full, C1-C5)" in A.index:
        f = A.loc["9. CardioMamba-Net (full, C1-C5)"]
        checks = [("CC_temporal >= 80", f["CC_temporal"], 80, "ge"),
                  ("CC_spectral >= 88", f["CC_spectral"], 88, "ge"),
                  ("params < 5 M", f["params"], 5e6, "lt")]
        if "MAE_mean_hr_bpm" in A.columns:
            checks.append(("HR MAE < 2 bpm", f["MAE_mean_hr_bpm"], 2, "lt"))
        if "MAE_rmssd_ms" in A.columns:
            checks.append(("RMSSD MAE < 8 ms", f["MAE_rmssd_ms"], 8, "lt"))
        for nm, v, thr, op in checks:
            hit = (v >= thr) if op == "ge" else (v < thr)
            print(f"  {nm:<22} actual {v:>12,.3f}   {'MET' if hit else 'not yet'}")
        print(f"\n  vs the published MultiResLinkNet (CC_t 61.86, CC_s 79.96):")
        print(f"    CC_temporal {f['CC_temporal']:.2f}  ({f['CC_temporal']-61.86:+.2f})")
        print(f"    CC_spectral {f['CC_spectral']:.2f}  ({f['CC_spectral']-79.96:+.2f})")
MAJOR("04_ablation")
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
FIG = SHARD_FIGURES

if len(R):
    b = R[R["experiment"] == CFG["EXPERIMENT"]]
    order = [v for v in LADDER if v in set(b["variant"])]
    if order:
        vals = [b[b["variant"] == v]["CC_temporal"].mean() for v in order]
        cols_ = [S["ecg"] if v == "L9_full" else
                 S["muted"] if v == "L1_baseline_mse" else S["radar"] for v in order]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.barh(range(len(order)), vals, color=cols_, edgecolor="white")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABEL.get(v, v) for v in order], fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline(61.86, color=S["ink"], ls="--", lw=1.2, label="published MultiResLinkNet")
        for i, v in enumerate(vals):
            ax.text(v + .3, i, f"{v:.1f}", va="center", fontsize=7)
        ax.set_xlabel("temporal correlation (x100)")
        ax.set_title(f"Ablation ladder — {CFG['EXPERIMENT']}", loc="left")
        ax.legend(frameon=False, fontsize=7)
        fig.savefig(FIG / "nb04_fig1_ablation.png"); plt.close(fig)
        print("  wrote nb04_fig1_ablation.png")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    for v in order:
        hs = []
        for p in (WORK / "runs").glob(f"*__{v}__*/state.json"):
            try:
                hs += [(h["epoch"], h["val"]) for h in json.loads(p.read_text())["history"]]
            except Exception:
                pass
        if hs:
            d = pd.DataFrame(hs, columns=["e", "v"]).groupby("e")["v"].mean()
            ax.plot(d.index, d.values, lw=1.2,
                    label=v, color=S["ecg"] if v == "L9_full" else None,
                    zorder=3 if v == "L9_full" else 2)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("validation loss")
    ax.set_title("Ablation training curves", loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=6.5)
    fig.savefig(FIG / "nb04_fig2_curves.png"); plt.close(fig)
    print("  wrote nb04_fig2_curves.png")

    full_prefix = "quick__" if CFG["QUICK"] else ""
    fullp = sorted((WORK / "runs").glob(
        f"{full_prefix}{CFG['EXPERIMENT']}__L9_full__f*/preds_sample.npz"))
    basep = sorted(Path(CFG["WORK"]).parent.glob(
        "nb03/runs/B_rva__multireslinknet__f*/preds_sample.npz"))
    if fullp:
        zf = np.load(fullp[0]); zb = np.load(basep[0]) if basep else None
        k = min(4, len(zf["y"]) - 1); tt = np.arange(1024) / 128.0
        rows_ = 3 if zb is not None else 2
        fig, axes = plt.subplots(rows_, 1, figsize=(11, 2.0 * rows_), sharex=True)
        axes[0].plot(tt, zf["y"][k], lw=1.1, color=S["ink"])
        axes[0].set_ylabel("ground truth", rotation=0, ha="right", va="center", fontsize=8)
        axes[1].plot(tt, zf["p"][k], lw=1.1, color=S["ecg"])
        axes[1].set_ylabel("CardioMamba-Net", rotation=0, ha="right", va="center",
                           fontsize=8, color=S["ecg"])
        if zb is not None:
            kb = min(k, len(zb["y"]) - 1)
            axes[2].plot(tt, zb["p"][kb], lw=1.1, color=S["muted"])
            axes[2].set_ylabel("MultiResLinkNet", rotation=0, ha="right", va="center", fontsize=8)
        for a in axes:
            a.tick_params(labelleft=False)
        axes[-1].set_xlabel("seconds")
        axes[0].set_title("Held-out reconstruction — is the QRS sharp, or smeared?", loc="left")
        fig.tight_layout(); fig.savefig(FIG / "nb04_fig3_qualitative.png"); plt.close(fig)
        print("  wrote nb04_fig3_qualitative.png")
MAJOR("05_figures")
''')

code(r'''
known_global_done = global_done_at_start | set(completed_now)
ok = sync.flush(final=True, msg=(
    f"{CFG['RUN_ID']} {SHARD_TAG} — {len(done)}/{len(QUEUE)} assigned runs complete"))
print("\n" + "=" * 76)
print("  SESSION COMPLETE" if ok else "  SESSION COMPLETE (final push had a problem)")
print("=" * 76)
print(f"  repo      : {sync.url}")
print(f"  worker    : {WORKER_ID}/{QUEUE_WORKERS-1} ({SHARD_TAG})")
print(f"  shard done: {len(done)}/{len(QUEUE)}   this session: {len(completed_now)}")
print(f"  global    : at least {len(known_global_done)}/{len(GLOBAL_QUEUE)} visible to this session")
print(f"  elapsed   : {(time.time()-t_start)/3600:.2f} h")
print("=" * 76)
if CFG["QUICK"] and len(QUEUE) and len(done) == len(QUEUE):
    print("\n  QUICK validation complete.")
    if BASELINE_GATE.get("passed", False):
        print("  Set CFG['QUICK'] = False, restart, and Run All for the full queue.")
    else:
        print("  NB03's reproduction gate is still failed; review it before full training.")
        print("  Full mode remains blocked unless ALLOW_FAILED_BASELINE_GATE=True is set knowingly.")
elif len(done) < len(QUEUE):
    print(f"\n  {len(QUEUE)-len(done)} run(s) remain for {SHARD_TAG}.")
    print("  Start a NEW session with the SAME WORKER_ID; it resumes this exact shard.")
else:
    print(f"\n  {SHARD_TAG} is complete. Do not reuse this WORKER_ID.")
    if QUEUE_WORKERS == 1 or len(known_global_done) == len(GLOBAL_QUEUE):
        print("  All canonical runs are visible. Next: 05_evaluate_and_figures.ipynb")
    else:
        print("  Continue workers with the other IDs. Run NB05 only after all four shards finish.")
''')

md(r"""
---
# 8 · Troubleshooting

**Notebook says it allocated too much memory / kernel restarted** — this version uses
`WORKERS=0`, disables pinned host pages, and trains only five epochs in each isolated child
process. The child exits after an atomic checkpoint, so Linux reclaims all RAM and CUDA state.
Do not disable `PROCESS_ISOLATION`, raise `EPOCHS_PER_PROCESS`, raise `WORKERS`, or enable
`PIN_MEMORY` in a long Kaggle session. If a *single batch* runs out of GPU memory, lower
`CFG["BATCH"]` to 32 or 24; if necessary, lower `CFG["D_SSM"]` to 192 and use a new run ID.

**S4D produces NaN** — the kernel is computed in float32 outside autocast on purpose. If NaNs
appear, engine v7 automatically reloads the last finite `best.pt`, reduces the effective learning
rate to 25%, and retries in a fresh process. On the second recovery it disables AMP. Keep the
original `CFG["LR"]` unchanged so existing checkpoints remain scientifically traceable.

**Old run fails with `source state dict is empty`** — upload this revised notebook. Engine v7
recognizes the intentionally empty scaler state written after AMP was disabled and resumes that
checkpoint in float32. It does not discard the trained weights or restart the run.

**Old run stops at the same NaN batch** — this version recognizes the prior `last_error`, performs
a compatible engine-only migration, and restarts from `best_epoch` instead of replaying the
poisoned/failing batch. Look for a `RECOVERED ...` line before training resumes.

**`L10_transformer` much slower than `L9_full`** — expected. Attention is quadratic in sequence
length; the SSM is linear. That gap is itself a result worth reporting.

**The full model is not the best rung** — do not quietly retune until it wins. Report what the
ablation says. If `L6_no_ssm` beats `L9_full`, the SSM is not earning its place on this dataset
and the paper is more interesting for saying so.

**Param count over 5 M** — reduce `CFG["BASE"]` to 24 or `CFG["SSM_BLOCKS"]` to 2, then re-run the
smoke cell before training.

**Session ended mid-queue** — expected and handled. New session, run again, it resumes.

**One-worker launch (default)** — keep `QUEUE_WORKERS=1`, `WORKER_ID=0`, and run one notebook. It
will see all 100 canonical IDs, skip every ID with a remote `summary.json`, and resume the rest.

**Optional four-worker launch** — make four Kaggle copies. Keep `QUEUE_WORKERS=4` everywhere and
set exactly one `WORKER_ID` per copy: 0, 1, 2, 3. On restart, keep that copy's same ID. Never run two
copies with the same ID, and never run a one-worker copy concurrently with these four. Each copy
uses both local T4 GPUs for one assigned run; `WORKER_ID` partitions the queue, not the GPUs.

**`NB03 ... reproduction gate failed`** — NB03 completed all 80 runs, but its scientific check did
not reproduce the paper's published value. This notebook visibly proceeds because
`ALLOW_FAILED_BASELINE_GATE=True`; comparisons against our NB03 re-runs still use the same split,
but must not be described as a validated reproduction of the published baseline.
""")

out = sys.argv[1] if len(sys.argv) > 1 else "04_cardiomamba_train.ipynb"
for i, c in enumerate(n.cells):
    if c["cell_type"] == "code":
        s = "".join(c["source"])
        if "__HF_SYNC__" in s:
            s = (s.replace("__HF_SYNC__", HF_SYNC_SRC.strip("\n"))
                  .replace("__DATA__", CRVS_DATA_SRC.strip("\n"))
                  .replace("__METRICS__", CRVS_METRICS_SRC.strip("\n"))
                  .replace("__MODELS__", CRVS_MODELS_SRC.strip("\n"))
                  .replace("__CMNET__", CRVS_CMNET_SRC.strip("\n"))
                  .replace("__LOSSES__", CRVS_LOSS_SRC.strip("\n"))
                  .replace("__ENGINE__", CRVS_ENGINE_SRC.strip("\n")))
            n.cells[i]["source"] = n._src(s)
        if "__WORKER__" in s:
            n.cells[i]["source"] = n._src(s.replace("__WORKER__", NB04_WORKER_SRC.strip("\n")))
n.write(out, accelerator="nvidiaTeslaT4")
