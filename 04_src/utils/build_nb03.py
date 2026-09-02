#!/usr/bin/env python3
"""Emit 03_baselines.ipynb"""
import sys
from nb_lib_a import NB, HF_SYNC_SRC, CRVS_DATA_SRC, CRVS_METRICS_SRC
from nb_lib_b import CRVS_MODELS_SRC, CRVS_LOSS_SRC, CRVS_ENGINE_SRC

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

Two honest differences from their setup, both declared in the paper:

- **Splits are strictly by subject** and test windows do not overlap (see NB02 §"leakage decision").
  Theirs almost certainly leaked. So our reproduction may land *below* their published figures —
  that is the expected direction, and it is the fair comparison for everything that follows.
- **ECG target is [−1, 1]**, not [0, 1]. Set `CFG["TARGET_01"] = True` to run their convention.

Everything else is theirs: **1 input channel** (the arctangent-demodulated displacement `dy`),
5 levels, 64 filters doubling, **plain MSE**, Adam at 5e-4, 1024-sample windows at 128 Hz.

---

## ⚠️ Accelerator: **GPU T4 × 2**

*Session options → Accelerator → **GPU T4 x2***, Internet **On**, `HF_TOKEN` secret attached.

Both GPUs are used through `DataParallel`. AMP (mixed precision) is on, which roughly doubles
throughput on T4s and halves memory.

## The run queue — this is how it survives Kaggle

There are **4 models × 4 experiment settings × 5 folds = 80 runs**. That does not fit in one
12-hour session, and it is not supposed to.

The notebook builds a **queue**, checks which runs are already finished on Hugging Face, and works
through as many as fit in `TIME_BUDGET_H`. Then it pushes and stops cleanly. **Start a new session
and run it again** — it picks up exactly where it left off. Three or four sessions completes the
matrix. Nothing is ever recomputed.

Start with `QUICK = True`: one fold, 25 epochs, ~40 minutes, and it proves the whole path end to
end. Then set it `False` and let the queue run.
""")

md("---\n# 1 · Configuration")

code(r'''
CFG = {
    "SRC_REPO":  "Shanmuk4622/cr-rvs-radar-ecg-processed",   # NB02 output
    "DST_REPO":  "Shanmuk4622/cardiomamba-net",              # models + results (public)
    "HF_PRIVATE": False,
    "RUN_ID":    "nb03_baselines_v1",

    "WORK":    "/kaggle/working/nb03",
    "SCRATCH": "/kaggle/temp/nb03",
    "PUSH_INTERVAL_S": 30 * 60,
    "HF_MAX_REQ_HOUR": 120,

    # ---- faithful reproduction of Chowdhury et al. 2024 ----------------------
    "CHANNELS":   ["dy"],        # THEIR input: one arctangent-demodulated displacement channel
    "BASE":       64,            # section 3.1: "initial layer containing 64 filters"
    "LEVELS":     4,             # 5 levels counting the bottleneck
    "LR":         5e-4,          # section 3.1
    "LOSS":       "mse",         # section 3.1: "As a loss function, the MSE function was used"
    "TARGET_01":  False,         # True = their [0,1] ECG convention

    # ---- training ------------------------------------------------------------
    "EPOCHS":     120,
    "PATIENCE":   20,            # section 3.1
    "BATCH":      64,
    "WORKERS":    2,
    "WEIGHT_DECAY": 1e-4,
    "AMP":        True,
    "MULTI_GPU":  True,
    "SEED":       1337,

    # ---- the queue -----------------------------------------------------------
    "MODELS":      ["fpn", "unet", "linknet", "multireslinknet"],
    "EXPERIMENTS": ["B_rva", "A_resting", "A_valsalva", "A_apnea"],   # B first: it is the headline
    "N_FOLDS":     5,
    "TIME_BUDGET_H": 10.5,       # stop cleanly before Kaggle's 12 h wall
    "QUICK":       True,         # <-- first run: 1 fold, 25 epochs. Then set False.
    "QUICK_EPOCHS": 25,
    "QUICK_FOLDS":  1,
}
import json
print(json.dumps(CFG, indent=2))
''')

code(r'''
import os, sys, gc, json, math, time, warnings, subprocess, platform, shutil
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
        print(f"  GPU{i}: {p.name}  {p.total_memory/2**30:.1f} GB  sm_{p.major}{p.minor}")
    torch.backends.cudnn.benchmark = True
else:
    print("  !! NO GPU -- set Accelerator to 'GPU T4 x2' in Session options.")
    print("     The notebook will still run on CPU but training will be impractically slow.")
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

# Writing a .py and importing it is NOT idempotent inside one kernel: Python caches the
# module object in sys.modules, so re-running this cell after updating the notebook keeps
# the OLD code. That is exactly how a stale .npz loader survived a rebuilt notebook and
# produced a FileNotFoundError deep inside a DataLoader worker. Purge and re-import.
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

# Resume: pull back the small artefacts (state, summaries, metrics) but NOT the weights --
# finished runs never need their checkpoints re-downloaded, only their summary.json.
sync.pull(allow_patterns=["*.json", "*.jsonl", "*.csv", "*.md", "runs/**/summary.json",
                          "runs/**/state.json", "results/*"])
STATE = sync.load_state({"completed": [], "sessions": 0})
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

seed_all(CFG["SEED"])
dev, ngpu, names = pick_device()
print(f"device: {dev}  gpus: {ngpu} {names}\n")

C_IN = len(CFG["CHANNELS"])
x = torch.randn(4, C_IN, 1024, device=dev)
y = torch.randn(4, 1, 1024, device=dev).clamp(-1, 1)
rows = []
print(f"{'model':<20}{'params':>12}{'MB':>8}{'out shape':>18}{'fwd ms':>9}  grad")
print("-" * 78)
for nm in CFG["MODELS"]:
    m = build_baseline(nm, in_ch=C_IN, out_ch=1, base=CFG["BASE"], levels=CFG["LEVELS"]).to(dev)
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
    ok = (out["wave"].shape == y.shape and torch.isfinite(out["wave"]).all() and gn > 0)
    rows.append({"model": nm, "params": p, "mb": p * 4 / 2**20, "ok": bool(ok)})
    print(f"{nm:<20}{p:>12,}{p*4/2**20:>8.1f}{str(tuple(out['wave'].shape)):>18}"
          f"{dt:>9.1f}  {'OK' if ok else 'FAIL'}")
    del m, out, loss
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
assert all(r["ok"] for r in rows), "a baseline failed its smoke test"
pd.DataFrame(rows).to_csv(WORK / "results" / "model_budget.csv", index=False)
print("\nall four baselines forward, backward and produce finite output.")
MAJOR("01_smoke")
''')

md(r"""
---
# 5 · The run queue

Each entry is one (experiment, model, fold). Completed runs are read from HF state and skipped,
so re-running the notebook in a fresh session simply continues.

`QUICK = True` collapses this to one fold and 25 epochs — enough to confirm the whole path works
and to see whether the losses look sane, without committing a full session.
""")

code(r'''
folds = range(CFG["QUICK_FOLDS"] if CFG["QUICK"] else CFG["N_FOLDS"])
EPOCHS = CFG["QUICK_EPOCHS"] if CFG["QUICK"] else CFG["EPOCHS"]

QUEUE = []
for exp in CFG["EXPERIMENTS"]:
    for mdl in CFG["MODELS"]:
        for f in folds:
            QUEUE.append({"run_id": f"{exp}__{mdl}__f{f}", "exp": exp, "model": mdl, "fold": f})

done = set(STATE.get("completed", []))
todo = [q for q in QUEUE if q["run_id"] not in done]
print(f"queue: {len(QUEUE)} run(s) total | {len(done)} done | {len(todo)} remaining")
print(f"epochs per run: {EPOCHS}   time budget: {CFG['TIME_BUDGET_H']} h")
if CFG["QUICK"]:
    print("\n>>> QUICK MODE. Confirm this completes, then set CFG['QUICK']=False and re-run.")
print("\nnext up:")
for q in todo[:8]:
    print("   ", q["run_id"])
if len(todo) > 8:
    print(f"    ... and {len(todo)-8} more")
''')

code(r'''
from crvs_data import WindowDataset, WINDOW, FS
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

def make_datasets(exp, fold):
    tr, va, te = split_for(exp, fold)
    norm = NORM.get(f"{exp}|{fold}")
    if norm is None:
        raise RuntimeError(f"no normalisation stats for {exp}|{fold} -- re-run NB02")
    idx = [EXPINFO["channels"].index(c) for c in CFG["CHANNELS"]]
    sub_norm = {"mean": [norm["mean"][i] for i in idx],
                "std":  [norm["std"][i] for i in idx]}
    mk = lambda d, aug: WindowDataset(REC_DIR, d, sub_norm, CFG["CHANNELS"], augment=aug,
                                      seed=CFG["SEED"] + fold)
    return mk(tr, True), mk(va, False), mk(te, False), (tr, va, te)

def evaluate(Y, P, index, out_dir):
    # Per-window metrics, then per-subject and overall aggregates. Saving per-window rows
    # lets NB05 run the statistics without ever re-running a model.
    rows = []
    subs = index["subject"].to_numpy()
    scen = index["scenario_canon"].to_numpy()
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
    # continuous-signal HR / HRV, computed on the concatenated test signal per subject
    hr_rows = []
    for s in pd.unique(subs):
        sel = subs == s
        if sel.sum() < 2:
            continue
        yg = np.concatenate(Y[sel]); yp = np.concatenate(P[sel])
        g = hrv_from_peaks(detect_r_peaks(yg, FS), FS)
        p = hrv_from_peaks(detect_r_peaks(yp, FS), FS)
        pk = peak_detection_scores(yg, yp, FS)
        hr_rows.append({"subject": s, **{f"gt_{k}": v for k, v in g.items()},
                        **{f"pr_{k}": v for k, v in p.items()}, **pk})
    dfh = pd.DataFrame(hr_rows)
    if len(dfh):
        dfh.to_parquet(out_dir / "metrics_subjects.parquet", index=False)
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
t_start = time.time()
budget = CFG["TIME_BUDGET_H"] * 3600
completed_now = []

for qi, q in enumerate(todo, 1):
    el = time.time() - t_start
    if el > budget:
        print(f"\n=== time budget reached ({el/3600:.2f} h). Stopping cleanly. ===")
        print(f"    {len(todo)-qi+1} run(s) left -- start a new session and re-run this notebook.")
        break
    rid, exp, mdl, fold = q["run_id"], q["exp"], q["model"], q["fold"]
    out = WORK / "runs" / rid
    out.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 78)
    print(f"[{qi}/{len(todo)}]  {rid}   ({el/3600:.2f} h elapsed)")
    print("=" * 78)
    try:
        tr_ds, va_ds, te_ds, (tri, vai, tei) = make_datasets(exp, fold)
        print(f"  train {len(tr_ds):,} | val {len(va_ds):,} | test {len(te_ds):,} windows   "
              f"test subjects: {sorted(tei['subject'].unique())}")
        seed_all(CFG["SEED"] + fold)
        model = build_baseline(mdl, in_ch=len(CFG["CHANNELS"]), out_ch=1,
                               base=CFG["BASE"], levels=CFG["LEVELS"])
        loss_fn = MSEOnly() if CFG["LOSS"] == "mse" else CompositeLoss()
        tr = Trainer(model, loss_fn, out, rid, sync=sync, lr=CFG["LR"],
                     weight_decay=CFG["WEIGHT_DECAY"], epochs=EPOCHS,
                     patience=CFG["PATIENCE"], batch_size=CFG["BATCH"],
                     num_workers=CFG["WORKERS"], amp=CFG["AMP"],
                     multi_gpu=CFG["MULTI_GPU"])
        tr.load()
        tr.fit(tr_ds, va_ds)
        Y, P = tr.predict(te_ds)
        agg, dfw = evaluate(Y, P, tei, out)

        keep = min(200, len(Y))
        sel = np.linspace(0, len(Y) - 1, keep).astype(int)
        np.savez_compressed(out / "preds_sample.npz", y=Y[sel].astype(np.float32),
                            p=P[sel].astype(np.float32),
                            subject=tei["subject"].to_numpy()[sel].astype(str))
        summary = {"run_id": rid, "experiment": exp, "model": mdl, "fold": fold,
                   "channels": CFG["CHANNELS"], "loss": CFG["LOSS"], "epochs_run": tr.state["epoch"],
                   "best_epoch": tr.state["best_epoch"], "best_val": tr.state["best"],
                   "params": count_params(tr.raw_model),
                   "n_train": len(tr_ds), "n_val": len(va_ds), "n_test": len(te_ds),
                   "test_subjects": sorted(map(str, tei["subject"].unique())),
                   "metrics": agg,
                   "finished_utc": datetime.now(timezone.utc).isoformat()}
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"  --> CC_t {agg['CC_temporal']:.2f}  CC_s {agg['CC_spectral']:.2f}  "
              f"MAE {agg['MAE']:.5f}  MSE {agg['MSE']:.5f}  "
              f"RRMSE_t {agg['RRMSE_temporal']:.4f}  F1 {agg.get('peak_F1', float('nan')):.3f}")
        done.add(rid); completed_now.append(rid)
        STATE["completed"] = sorted(done); sync.save_state(STATE)
        sync.stage_done(f"run:{rid}", cc_t=round(agg["CC_temporal"], 2))
        del tr, model, tr_ds, va_ds, te_ds, Y, P
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("\ninterrupted -- checkpoint saved and pushed; re-run to resume this exact run.")
        raise
    except Exception as e:
        import traceback
        print(f"  !! {type(e).__name__}: {e}")
        (out / "error.txt").write_text(traceback.format_exc())
        sync.log("run_failed", run=rid, err=f"{type(e).__name__}: {e}")

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
for p in sorted((WORK / "runs").glob("*/summary.json")):
    try:
        s = json.loads(p.read_text())
        rows.append({"experiment": s["experiment"], "model": s["model"], "fold": s["fold"],
                     "params": s.get("params"), "best_epoch": s.get("best_epoch"),
                     **{k: v for k, v in s["metrics"].items() if not k.endswith("_std")}})
    except Exception:
        pass
R = pd.DataFrame(rows)
if not len(R):
    print("no completed runs yet -- run the training cell.")
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

    b = flat[flat["experiment"] == "B_rva"].sort_values("CC_temporal", ascending=False)
    if len(b):
        print("\n" + "-" * 100)
        print("REPRODUCTION GATE  (Experiment B, RVA combined)")
        print("-" * 100)
        print(f"  ranking by CC_temporal: {' > '.join(b['model'].tolist())}")
        top = b.iloc[0]["model"]
        ok = top == "multireslinknet"
        print(f"  MultiResLinkNet ranks first: {ok}")
        mr = b[b["model"] == "multireslinknet"]
        if len(mr):
            v = float(mr.iloc[0]["CC_temporal"])
            print(f"  our CC_temporal {v:.2f}  vs published 61.86  (delta {v-61.86:+.2f})")
        print("\n  " + ("GATE PASSED -- ordering reproduced, proceed to NB04."
                        if ok else
                        "GATE NOT PASSED -- MultiResLinkNet is not top. Check the reimplementation"
                        "\n  before trusting anything downstream. (With QUICK=True and one fold this"
                        "\n  is common; re-check with the full queue.)"))
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
single-channel input (`dy`), 5 levels, 64 base filters, plain MSE, Adam 5e-4.

Updated {now}.
"""
(WORK / "README.md").write_text(card)
ok = sync.flush(final=True, msg=f"{CFG['RUN_ID']} — {len(done)}/{len(QUEUE)} runs complete")
print("\n" + "=" * 76)
print("  SESSION COMPLETE" if ok else "  SESSION COMPLETE (final push had a problem)")
print("=" * 76)
print(f"  repo      : {sync.url}")
print(f"  runs done : {len(done)}/{len(QUEUE)}")
print(f"  this run  : {len(completed_now)} new")
print(f"  elapsed   : {(time.time()-t_start)/3600:.2f} h")
print("=" * 76)
if len(done) < len(QUEUE):
    print(f"\n  {len(QUEUE)-len(done)} run(s) remain. Start a NEW session and run this notebook")
    print("  again -- it resumes from Hugging Face and skips everything already finished.")
else:
    print("\n  Queue complete. Next: 04_cardiomamba_train.ipynb")
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

**DataLoader worker crashes** — set `CFG["WORKERS"] = 0`. Kaggle occasionally struggles with
worker processes plus memory-mapped `.npy` files.

**Session ended mid-queue** — expected. Start a new session, run the notebook again, and it
resumes. Progress is per run *and* per epoch.
""")

out = sys.argv[1] if len(sys.argv) > 1 else "03_baselines.ipynb"
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
n.write(out, accelerator="nvidiaTeslaT4")
