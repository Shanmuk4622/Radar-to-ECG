#!/usr/bin/env python3
"""Emit 04_cardiomamba_train.ipynb"""
import sys
from nb_lib_a import NB, HF_SYNC_SRC, CRVS_DATA_SRC, CRVS_METRICS_SRC
from nb_lib_b import CRVS_MODELS_SRC, CRVS_CMNET_SRC, CRVS_LOSS_SRC, CRVS_ENGINE_SRC

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
budget, and re-running continues. **`QUICK = True` first** — one fold, 25 epochs, ~1 h, and it
proves the whole ladder runs.

## On `mamba-ssm`

The SSM here is **pure PyTorch** (S4D-Lin as an FFT convolution). It needs no `nvcc`, no custom
CUDA kernel, and it always builds on Kaggle — which the official `mamba-ssm` package frequently
does not. Set `CFG["BOTTLENECK"] = "transformer"` for the attention control. Both are in the ladder.
""")

md("---\n# 1 · Configuration")

code(r'''
CFG = {
    "SRC_REPO":  "Shanmuk4622/cr-rvs-radar-ecg-processed",
    "DST_REPO":  "Shanmuk4622/cardiomamba-net",
    "HF_PRIVATE": False,
    "RUN_ID":    "nb04_cardiomamba_v1",

    "WORK":    "/kaggle/working/nb04",
    "SCRATCH": "/kaggle/temp/nb04",
    "PUSH_INTERVAL_S": 30 * 60,
    "HF_MAX_REQ_HOUR": 120,

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
    "BATCH":    48,
    "WORKERS":  2,
    "LR":       8e-4,
    "WEIGHT_DECAY": 1e-4,
    "AMP":      True,
    "MULTI_GPU": True,
    "SEED":     1337,

    # ---- queue ---------------------------------------------------------------
    "EXPERIMENT":  "B_rva",      # the ablation ladder runs on the headline experiment
    "EXTRA_EXPERIMENTS": ["A_resting", "A_valsalva", "A_apnea", "C_all5"],  # full model only
    "N_FOLDS":     5,
    "TIME_BUDGET_H": 10.5,
    "QUICK": True,
    "QUICK_EPOCHS": 25,
    "QUICK_FOLDS": 1,
}
import json
print(json.dumps(CFG, indent=2))
''')

code(r'''
import os, sys, gc, json, math, time, warnings, subprocess, platform
from pathlib import Path
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

def _pip(*p):
    import importlib.util
    miss = [x for x in p if importlib.util.find_spec(x.replace("-", "_")) is None]
    if miss:
        print("installing:", miss)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *miss], check=False)
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
        print(f"  GPU{i}: {p.name}  {p.total_memory/2**30:.1f} GB")
    torch.backends.cudnn.benchmark = True
else:
    print("  !! NO GPU -- set Accelerator to 'GPU T4 x2'. Training on CPU is impractical here.")
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

# Purge before importing: Python caches modules in sys.modules, so re-running this cell
# after updating the notebook would silently keep the previous version of the library.
import importlib
for nm in MODULES:
    sys.modules.pop(nm[:-3], None)
importlib.invalidate_caches()
import crvs_data
REQUIRED_LIB = 3
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
              private=CFG["HF_PRIVATE"], run_id=CFG["RUN_ID"],
              push_interval_s=CFG["PUSH_INTERVAL_S"], max_req_hour=CFG["HF_MAX_REQ_HOUR"])
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
                          "runs/**/summary.json", "runs/**/state.json", "results/*"])
STATE = sync.load_state({"completed": [], "sessions": 0})
STATE["sessions"] = STATE.get("sessions", 0) + 1
sync.save_state(STATE)
print(f"session #{STATE['sessions']}  |  {len(STATE['completed'])} run(s) already complete")
MAJOR("00_setup")
''')

code(r'''
from huggingface_hub import snapshot_download
DATA = SCRATCH / "corpus"
t0 = time.time()
snapshot_download(CFG["SRC_REPO"], repo_type="dataset", token=HF_TOKEN, local_dir=str(DATA),
                  allow_patterns=["recordings/*.npy", "recordings/*.json",
                                  "recordings/*.npz",   # legacy corpus still works
                                  "windows.parquet", "recordings.csv",
                                  "norm_stats.json", "experiments.json"])
print(f"corpus downloaded in {time.time()-t0:.0f}s")
W = pd.read_parquet(DATA / "windows.parquet")
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
    heads = "+".join(k for k in ("wave", "peak", "rr") if k in out)
    ok = torch.isfinite(out["wave"]).all() and gn > 0 and math.isfinite(float(loss))
    budget.append({"variant": nm, "in_ch": len(spec["channels"]), "params": p,
                   "mb": p * 4 / 2**20, "fwd_ms": dt, "ok": bool(ok)})
    print(f"{nm:<18}{len(spec['channels']):>4}{p:>12,}{p*4/2**20:>7.1f}{dt:>9.1f}  "
          f"{heads}  {'OK' if ok else 'FAIL'}")
    del m, out, loss
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

B = pd.DataFrame(budget)
B.to_csv(WORK / "results" / "variant_budget.csv", index=False)
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

oo["wave"].sum().backward()
dead = [n_ for n_, p_ in m.named_parameters()
        if p_.grad is None or float(p_.grad.abs().sum()) == 0.0]
n_dead = sum(p_.numel() for n_, p_ in m.named_parameters() if n_ in set(dead))
print(f"\n  parameters receiving no gradient: {len(dead)} tensors / {n_dead:,} values")
if dead:
    from collections import Counter
    print("  by submodule:", dict(Counter(d.split(".")[0] for d in dead)))
    raise RuntimeError(
        "some parameters get no gradient — they cost compute and learn nothing. "
        "A dropped skip connection is the usual cause.")
print("  every parameter is reachable by the loss.")
del m, oo
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
D.to_csv(WORK / "results" / "loss_probe.csv", index=False)

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
for vname in VARIANTS:                                  # ladder on B_rva
    for f in folds:
        QUEUE.append({"run_id": f"{CFG['EXPERIMENT']}__{vname}__f{f}",
                      "exp": CFG["EXPERIMENT"], "variant": vname, "fold": f})
if not CFG["QUICK"]:                                    # full model everywhere else
    for exp in CFG["EXTRA_EXPERIMENTS"]:
        for f in folds:
            QUEUE.append({"run_id": f"{exp}__L9_full__f{f}", "exp": exp,
                          "variant": "L9_full", "fold": f})

done = set(STATE.get("completed", []))
todo = [q for q in QUEUE if q["run_id"] not in done]
print(f"queue: {len(QUEUE)} run(s) | {len(done)} done | {len(todo)} remaining")
print(f"epochs {EPOCHS} | folds {folds} | budget {CFG['TIME_BUDGET_H']} h")
if CFG["QUICK"]:
    print("\n>>> QUICK MODE: ladder only, 1 fold, 25 epochs. Then set QUICK=False.")
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
    sub = W[W["scenario_canon"].isin(EXPERIMENTS[exp])]
    te_g, va_g = fold % n_folds, (fold + 1) % n_folds
    tr = sub[~sub["fold_group"].isin([te_g, va_g])]
    va = sub[(sub["fold_group"] == va_g) & sub["no_overlap"]]
    te = sub[(sub["fold_group"] == te_g) & sub["no_overlap"]]
    assert not (set(tr["subject"]) & set(te["subject"])), "SUBJECT LEAK"
    return tr, va, te

def make_datasets(exp, fold, channels):
    tr, va, te = split_for(exp, fold)
    norm = NORM.get(f"{exp}|{fold}")
    if norm is None:
        raise RuntimeError(f"no normalisation stats for {exp}|{fold} -- re-run NB02")
    idx = [EXPINFO["channels"].index(c) for c in channels]
    sn = {"mean": [norm["mean"][i] for i in idx], "std": [norm["std"][i] for i in idx]}
    mk = lambda d, aug: WindowDataset(REC_DIR, d, sn, channels, augment=aug,
                                      seed=CFG["SEED"] + fold)
    return mk(tr, True), mk(va, False), mk(te, False), (tr, va, te)

def evaluate(Y, P, index, out_dir):
    subs = index["subject"].to_numpy(); scen = index["scenario_canon"].to_numpy()
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
    for s in pd.unique(subs):
        sel = subs == s
        if sel.sum() < 2:
            continue
        yg = np.concatenate(Y[sel]); yp = np.concatenate(P[sel])
        g = hrv_from_peaks(detect_r_peaks(yg, FS), FS)
        pr = hrv_from_peaks(detect_r_peaks(yp, FS), FS)
        hr.append({"subject": s, **{f"gt_{k}": v for k, v in g.items()},
                   **{f"pr_{k}": v for k, v in pr.items()},
                   **peak_detection_scores(yg, yp, FS)})
    dfh = pd.DataFrame(hr)
    if len(dfh):
        dfh.to_parquet(out_dir / "metrics_subjects.parquet", index=False)
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
# 6 · Train the ladder

Interrupt-safe and resumable at the epoch level, exactly as in NB03. Watch `CC_t` in the summary
line after each run — the rungs should climb: rung 2 (loss only) should already beat the NB03
baseline, and rung 9 (full) should be the best of the ladder. If rung 9 is *not* the best, the
ablation is telling you something real and the paper should report it honestly rather than the
architecture being quietly retuned until it wins.
""")

code(r'''
t_start = time.time(); budget_s = CFG["TIME_BUDGET_H"] * 3600
completed_now = []

for qi, q in enumerate(todo, 1):
    el = time.time() - t_start
    if el > budget_s:
        print(f"\n=== time budget reached ({el/3600:.2f} h). Stopping cleanly. ===")
        print(f"    {len(todo)-qi+1} run(s) left -- start a new session and re-run.")
        break
    rid, exp, vname, fold = q["run_id"], q["exp"], q["variant"], q["fold"]
    spec = VARIANTS[vname]
    out = WORK / "runs" / rid; out.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 78)
    print(f"[{qi}/{len(todo)}]  {rid}   ({el/3600:.2f} h elapsed)")
    print("=" * 78)
    try:
        tr_ds, va_ds, te_ds, (tri, vai, tei) = make_datasets(exp, fold, spec["channels"])
        print(f"  in_ch {len(spec['channels'])} | train {len(tr_ds):,} val {len(va_ds):,} "
              f"test {len(te_ds):,} | test subjects {sorted(tei['subject'].unique())}")
        seed_all(CFG["SEED"] + fold)
        model = make_model(spec)
        loss_fn = (CompositeLoss(**{f"w_{k}": v for k, v in CFG["W"].items()},
                                 huber_delta=CFG["HUBER_DELTA"],
                                 peak_weight=CFG["PEAK_WEIGHT"])
                   if spec["loss"] == "composite" else MSEOnly())
        tr = Trainer(model, loss_fn, out, rid, sync=sync, lr=CFG["LR"],
                     weight_decay=CFG["WEIGHT_DECAY"], epochs=EPOCHS,
                     patience=CFG["PATIENCE"], batch_size=CFG["BATCH"],
                     num_workers=CFG["WORKERS"], amp=CFG["AMP"], multi_gpu=CFG["MULTI_GPU"])
        tr.load(); tr.fit(tr_ds, va_ds)
        Y, P = tr.predict(te_ds)
        agg, dfw = evaluate(Y, P, tei, out)
        keep = min(200, len(Y)); sel = np.linspace(0, len(Y) - 1, keep).astype(int)
        np.savez_compressed(out / "preds_sample.npz", y=Y[sel].astype(np.float32),
                            p=P[sel].astype(np.float32),
                            subject=tei["subject"].to_numpy()[sel].astype(str))
        (out / "summary.json").write_text(json.dumps({
            "run_id": rid, "experiment": exp, "variant": vname, "fold": fold,
            "spec": {k: v for k, v in spec.items()},
            "params": count_params(tr.raw_model), "epochs_run": tr.state["epoch"],
            "best_epoch": tr.state["best_epoch"], "best_val": tr.state["best"],
            "n_train": len(tr_ds), "n_val": len(va_ds), "n_test": len(te_ds),
            "test_subjects": sorted(map(str, tei["subject"].unique())),
            "metrics": agg, "finished_utc": datetime.now(timezone.utc).isoformat()},
            indent=2, default=str))
        print(f"  --> CC_t {agg['CC_temporal']:.2f}  CC_s {agg['CC_spectral']:.2f}  "
              f"MAE {agg['MAE']:.5f}  RRMSE_t {agg['RRMSE_temporal']:.4f}  "
              f"F1 {agg.get('peak_F1', float('nan')):.3f}  "
              f"dRMSSD {agg.get('MAE_rmssd_ms', float('nan')):.1f} ms")
        done.add(rid); completed_now.append(rid)
        STATE["completed"] = sorted(done); sync.save_state(STATE)
        sync.stage_done(f"run:{rid}", cc_t=round(agg["CC_temporal"], 2))
        del tr, model, tr_ds, va_ds, te_ds, Y, P
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("\ninterrupted -- checkpoint saved and pushed; re-run to resume this run.")
        raise
    except Exception as e:
        import traceback
        print(f"  !! {type(e).__name__}: {e}")
        (out / "error.txt").write_text(traceback.format_exc())
        sync.log("run_failed", run=rid, err=f"{type(e).__name__}: {e}")

print(f"\ncompleted this session: {len(completed_now)}  |  total {len(done)}/{len(QUEUE)}")
MAJOR("03_training")
''')

md(r"""
---
# 7 · The ablation table

The table the paper's Discussion is built on. Each rung adds one component; the `Δ CC_t` column is
what that component is worth. This is what turns "our architecture is better" into "the SSM
bottleneck contributes 3.2 points of temporal correlation".

Rung 1 is pulled from NB03's runs if they are present in the same repo.
""")

code(r'''
rows = []
for p in sorted((WORK / "runs").glob("*/summary.json")):
    try:
        s = json.loads(p.read_text())
        rows.append({"experiment": s["experiment"], "variant": s.get("variant", s.get("model")),
                     "fold": s["fold"], "params": s.get("params"),
                     **{k: v for k, v in s["metrics"].items() if not k.endswith("_std")}})
    except Exception:
        pass
# rung 1 comes from NB03
for p in sorted(Path(CFG["WORK"]).parent.glob("nb03/runs/B_rva__multireslinknet__f*/summary.json")):
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
    R.to_csv(WORK / "results" / "runs_raw.csv", index=False)
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
    A.to_csv(WORK / "results" / "ablation.csv")

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
FIG = WORK / "figures"

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

    fullp = sorted((WORK / "runs").glob(f"{CFG['EXPERIMENT']}__L9_full__f*/preds_sample.npz"))
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
ok = sync.flush(final=True, msg=f"{CFG['RUN_ID']} — {len(done)}/{len(QUEUE)} runs complete")
print("\n" + "=" * 76)
print("  SESSION COMPLETE" if ok else "  SESSION COMPLETE (final push had a problem)")
print("=" * 76)
print(f"  repo      : {sync.url}")
print(f"  runs done : {len(done)}/{len(QUEUE)}   this session: {len(completed_now)}")
print(f"  elapsed   : {(time.time()-t_start)/3600:.2f} h")
print("=" * 76)
if len(done) < len(QUEUE):
    print(f"\n  {len(QUEUE)-len(done)} run(s) remain. Start a NEW session and re-run this")
    print("  notebook -- it resumes from Hugging Face and skips finished runs.")
else:
    print("\n  Ladder complete. Next: 05_evaluate_and_figures.ipynb")
''')

md(r"""
---
# 8 · Troubleshooting

**CUDA out of memory** — lower `CFG["BATCH"]` to 32 or 24. The wavelet branch roughly doubles
encoder activations. If it persists, drop `CFG["D_SSM"]` to 192.

**S4D produces NaN** — the kernel is computed in float32 outside autocast on purpose. If NaNs
still appear, lower `CFG["LR"]` to 5e-4 and check `nb04_fig2_curves.png` for the epoch it began.

**`L10_transformer` much slower than `L9_full`** — expected. Attention is quadratic in sequence
length; the SSM is linear. That gap is itself a result worth reporting.

**The full model is not the best rung** — do not quietly retune until it wins. Report what the
ablation says. If `L6_no_ssm` beats `L9_full`, the SSM is not earning its place on this dataset
and the paper is more interesting for saying so.

**Param count over 5 M** — reduce `CFG["BASE"]` to 24 or `CFG["SSM_BLOCKS"]` to 2, then re-run the
smoke cell before training.

**Session ended mid-queue** — expected and handled. New session, run again, it resumes.
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
n.write(out, accelerator="nvidiaTeslaT4")
