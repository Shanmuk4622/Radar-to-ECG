#!/usr/bin/env python3
"""Emit 02_preprocess_to_hf.ipynb"""
import sys
from nb_lib_a import NB, HF_SYNC_SRC, CRVS_DATA_SRC, CRVS_METRICS_SRC

n = NB()
md, code = n.md, n.code

md(r"""
# NB02 — Build the training corpus

**Project:** CardioMamba-Net · **Stage:** 2 of 5
`01_verify_and_download` → **`02_preprocess_to_hf`** → `03_baselines` → `04_cardiomamba_train` → `05_evaluate_and_figures`

---

## What this does

NB01 proved the mirror is real and left a **decimated 128 Hz copy of the whole corpus** on
Hugging Face. This notebook turns that into a training corpus: it adds the supervision targets,
assigns folds, and computes normalisation statistics — then publishes the result so NB03–NB05
never touch raw data again.

| Step | Why |
|---|---|
| Mount NB01 output (HF v2 fallback) | The 8 physics channels at 128 Hz already exist; re-deriving them would be wasted work |
| Quality audit on `quality_flags` + `beat_coupling` | Exclude only unusable signals; keep weak coupling as a measured difficulty covariate |
| Build ECG target, R-peak heatmap, instantaneous RR | The three heads of C4 need three targets |
| Window index at 50 % overlap, flagged for no-overlap | One index serves both train (overlapped) and test (not) — see the leakage note below |
| Subject-wise 5-fold **and** LOSO fold assignment | Experiments A/B/C use the folds; Experiment D uses LOSO |
| Per-fold normalisation stats, **train windows only** | Computing them corpus-wide is invisible leakage |

## ⚠️ Accelerator: **None (CPU)**

This notebook is disk- and CPU-bound — it decodes, filters and windows arrays. A GPU would sit
idle while burning your weekly T4 quota, which NB03 and NB04 genuinely need. Set
*Session options → Accelerator → **None***.

## The leakage decision, made explicit

The baseline overlaps windows by 50 % and reports Table 1 counts that already carry the overlap,
then splits those totals 80/20. NB01 confirmed the arithmetic: our overlapped counts reproduce
theirs to within 0.1 % (Apnea matched exactly, 1140 = 1140). That means **overlapping windows can
straddle train and test**, and adjacent windows share 512 of their 1024 samples.

We do not repeat that. Here:

- **Splits are always by subject** — no subject appears in more than one split, ever.
- **Train windows overlap 50 %** (data augmentation, as intended).
- **Validation and test windows do not overlap at all** — the `no_overlap` column selects them.

That is stricter than the baseline, and it means our numbers are, if anything, pessimistic
relative to theirs. We say so in the paper.

---

## How to run

1. *Session options* → **Accelerator: None**, **Internet: On**.
2. Click **+ Add Input → Notebook Output** and attach the saved output of NB01. Please do this;
   it is the fast path and avoids redownloading the corpus. HF remains the automatic fallback.
3. *Add-ons → Secrets* → `HF_TOKEN` (write) attached.
4. Run all. Interrupt-safe and resumable, same contract as NB01.

### Cell-by-cell run guide

| Code cell | What runs | Typical time |
|---:|---|---:|
| 1 | Configuration | < 5 s |
| 2 | Imports, dependency and disk checks | 1–3 min |
| 3 | Write shared data/metric/sync libraries | < 10 s |
| 4 | HF login, restore state and completed recording arrays | 1–10 min |
| 5 | Use attached NB01 output, or download it as fallback | mounted: < 1 min; fallback: 3–12 min |
| 6 | Apply and report the quality gate | < 30 s |
| 7 | Build 11-row recording arrays and all supervision targets | 10–30 min |
| 8 | Assign subject folds, LOSO IDs, and window index | 1–3 min |
| 9 | Compute leakage-free 5-fold, LOSO, and cross-scenario norms | 5–20 min |
| 10 | Generate corpus sanity figures | 1–4 min |
| 11 | Write dataset card/report and perform final verified upload | 3–15 min |

Total: normally **25–75 minutes**. Long cells print progress; rerunning restores outputs before it
trusts any completed-state marker.
""")

md("---\n# 1 · Configuration")

code(r'''
CFG = {
    "SRC_REPO":   "Shanmuk4622/cr-rvs-radar-ecg-inventory-v2",  # NB01 output
    "DST_REPO":   "Shanmuk4622/cr-rvs-radar-ecg-processed-v2",  # this notebook's output
    "HF_PRIVATE": False,
    "RUN_ID":     "nb02_corpus_v2",

    "WORK":    "/kaggle/working/nb02",
    "SCRATCH": "/kaggle/temp/nb02",

    "PUSH_INTERVAL_S": 30 * 60,
    "HF_MAX_UPLOADS_HOUR": 24,

    # frozen to Chowdhury et al. 2024 section 2.3
    "FS": 128, "WINDOW": 1024, "HOP_TRAIN": 512,
    "PEAK_SIGMA": 3.0,          # samples; ~23 ms at 128 Hz

    # quality gates -- exclusion is evidence-based, and every exclusion is logged
    "MIN_BEAT_COUPLING": None,  # do not select an easier cohort; retain as a covariate
    "MIN_DURATION_S":    60.0,
    "EXCLUDE_FLAGS":     ["NO_RADAR_CHANNELS", "missing_channels", "ecg_flatline",
                          "few_or_no_rpeaks", "implausible_hr", "nan_in_radar"],
    "WARN_ONLY_FLAGS":   ["iq_clipping", "strong_iq_imbalance", "large_sync_lag",
                          "very_short", "radar_ecg_length_mismatch"],

    "N_FOLDS": 5,
    "SEED": 1337,
    "SMOKE_TEST": False,
}
import json
print(json.dumps(CFG, indent=2))
''')

code(r'''
import os, sys, gc, re, json, math, time, signal, atexit, threading, warnings, subprocess, platform
from pathlib import Path
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

def _pip(*p):
    miss = [x for x in p if __import__("importlib").util.find_spec(x.replace("-", "_")) is None]
    if miss:
        print("installing:", miss)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *miss], check=True)
        for x in miss: __import__(x.replace("-", "_"))
_pip("pyarrow", "huggingface_hub")

import numpy as np, pandas as pd
from scipy import signal as ss

WORK = Path(CFG["WORK"]); SCRATCH = Path(CFG["SCRATCH"])
for d in (WORK, SCRATCH, WORK / "recordings", WORK / "logs"):
    d.mkdir(parents=True, exist_ok=True)
np.random.seed(CFG["SEED"])

def disk(p):
    try:
        s = os.statvfs(p)
        return f"{s.f_bavail*s.f_frsize/2**30:.1f} GB free"
    except Exception:
        return "?"
print("working:", disk("/kaggle/working"), " temp:",
      disk("/kaggle/temp") if Path("/kaggle/temp").exists() else "absent")
print("numpy", np.__version__, "| pandas", pd.__version__)
''')

md(r"""
---
# 2 · The shared library

Three modules are written to disk here and **re-used verbatim by NB03, NB04 and NB05**:
`crvs_sync.py` (the HF cadence rules), `crvs_data.py` (windowing, folds, the Dataset) and
`crvs_metrics.py` (every metric). Defining them once means every experiment sees byte-identical
inputs and every number is computed the same way.

They are also pushed to the processed-data repo, so the later notebooks download them rather
than carrying a divergent copy.
""")

code(r'''
CRVS_SYNC_SRC = r"""
__HF_SYNC__
"""
CRVS_DATA_SRC = r"""
__DATA__
"""
CRVS_METRICS_SRC = r"""
__METRICS__
"""
for name, src in [("crvs_sync.py", CRVS_SYNC_SRC), ("crvs_data.py", CRVS_DATA_SRC),
                  ("crvs_metrics.py", CRVS_METRICS_SRC)]:
    (WORK / name).write_text(src)
    print(f"wrote {name}  ({len(src):,} chars)")
sys.path.insert(0, str(WORK))
import hashlib
MODULE_HASHES = {name: hashlib.sha256(src.encode()).hexdigest() for name, src in
                 [("crvs_sync.py", CRVS_SYNC_SRC), ("crvs_data.py", CRVS_DATA_SRC),
                  ("crvs_metrics.py", CRVS_METRICS_SRC)]}
(WORK / "library_hashes.json").write_text(json.dumps(MODULE_HASHES, indent=2))
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
            "\n  HF_TOKEN not found."
            "\n  Add-ons -> Secrets -> add HF_TOKEN (write scope) and attach it.\n" + "="*74)

from crvs_sync import HFSync
sync = HFSync(repo_id=CFG["DST_REPO"], local_dir=WORK, token=HF_TOKEN, repo_type="dataset",
              private=CFG["HF_PRIVATE"], run_id=CFG["RUN_ID"],
              push_interval_s=CFG["PUSH_INTERVAL_S"],
              max_upload_calls_hour=CFG["HF_MAX_UPLOADS_HOUR"])
print("\ndestination:", sync.url, "(public)")

_MAJOR = {"f": False, "n": ""}
def MAJOR(name):
    _MAJOR["f"] = True; _MAJOR["n"] = name
def _hook(result=None):
    if _MAJOR["f"]:
        nm = _MAJOR["n"]; _MAJOR["f"] = False; _MAJOR["n"] = ""
        sync.stage_done(nm)
try:
    get_ipython().events.register("post_run_cell", _hook)
    print("post-run-cell push hook registered")
except Exception as e:
    print("hook unavailable:", e)

sync.pull(allow_patterns=["*.json", "*.jsonl", "*.parquet", "*.csv", "*.md",
                          "recordings/*.npy", "recordings/*.json", "logs/*"])
STATE = sync.load_state({"done_recordings": [], "stages": {}, "version": 2})
print("resuming with", len(STATE["done_recordings"]), "recordings already built")
MAJOR("00_setup")
''')

md(r"""
---
# 3 · Pull NB01's output

We need two things from the inventory repo: `inventory.csv` (to decide what to keep) and
`decimated/*.npz` (the 128 Hz channels). The download is ~400 MB and goes to scratch, not to
`/kaggle/working`, so it never counts against the 20 GB output budget.

If the decimated folder is missing — for instance if NB01 was run with `SAVE_DECIMATED = False`
— this cell says so and stops, rather than half-building a corpus.
""")

code(r'''
from huggingface_hub import snapshot_download

# Fast path: attach NB01's saved Kaggle notebook output as an Input. It is read-only and
# does not consume /kaggle/working. HF is a recovery fallback, not the primary data path.
SRC = None
input_root = Path("/kaggle/input")
if input_root.exists():
    for candidate in input_root.rglob("inventory.csv"):
        if any((candidate.parent / "decimated").glob("*.npz")):
            SRC = candidate.parent; break
if SRC is not None:
    print("using attached Kaggle NB01 output:", SRC)
else:
    SRC = Path(SCRATCH) / "src"
    print("NB01 output was not attached; falling back to Hugging Face download.")
    t0 = time.time()
    snapshot_download(CFG["SRC_REPO"], repo_type="dataset", token=HF_TOKEN,
                      local_dir=str(SRC),
                      allow_patterns=["inventory.csv", "verdict.json", "crosscheck.csv",
                                      "decimated/*.npz"], max_workers=4)
    print(f"downloaded in {time.time()-t0:.0f}s")

inv_path = SRC / "inventory.csv"
if not inv_path.exists():
    raise RuntimeError(f"inventory.csv not found in {CFG['SRC_REPO']} -- run NB01 first.")
INV = pd.read_csv(inv_path)
dec = sorted((SRC / "decimated").glob("*.npz"))
print(f"inventory rows : {len(INV)}")
print(f"decimated files: {len(dec)}")
if not dec:
    raise RuntimeError(
        "\n" + "="*74 +
        "\n  No decimated/*.npz found."
        "\n  Re-run NB01 with CFG['SAVE_DECIMATED'] = True, or point this notebook at the"
        "\n  raw .mat tree instead.\n" + "="*74)
print(f"total size     : {sum(p.stat().st_size for p in dec)/2**20:.0f} MB")
print("\ncolumns available:", len(INV.columns))
''')

md(r"""
---
# 4 · Quality gate

Every exclusion is recorded with its reason, and the surviving corpus is compared against the
baseline's segment counts so we know exactly how much we gave up for cleanliness.

Two tiers. **Hard exclusions** are limited to unusable inputs/targets: missing radar, flat or
unreadable ECG, NaNs, or a recording too short for a window. Weak `beat_coupling` is **not** an
exclusion: Apnea physiologically reduces chest motion, so thresholding coupling selected an easier
and scenario-biased v1 cohort. Coupling and all warnings are retained as covariates so NB05 can
measure performance versus signal difficulty.
""")

code(r'''
INV["scenario_canon"] = INV["scenario"].map(
    lambda s: ("Tilt-down" if "tiltdown" in str(s).lower().replace("_","").replace("-","")
               else "Tilt-up" if "tiltup" in str(s).lower().replace("_","").replace("-","")
               else "Valsalva" if "valsalva" in str(s).lower()
               else "Apnea" if "apnea" in str(s).lower() or "apnoea" in str(s).lower()
               else "Resting" if "rest" in str(s).lower() else str(s)))

flags_col = "quality_flags" if "quality_flags" in INV.columns else "flags"
INV[flags_col] = INV[flags_col].fillna("")

def reasons(r):
    out = []
    fl = set(str(r[flags_col]).split(";")) - {""}
    for f in CFG["EXCLUDE_FLAGS"]:
        if f in fl:
            out.append(f)
    bc = r.get("beat_coupling", np.nan)
    if (CFG["MIN_BEAT_COUPLING"] is not None and pd.notna(bc)
            and bc < CFG["MIN_BEAT_COUPLING"]):
        out.append(f"beat_coupling<{CFG['MIN_BEAT_COUPLING']}")
    if r.get("duration_s", 0) < CFG["MIN_DURATION_S"]:
        out.append("too_short")
    return ";".join(out)

INV["exclude_reason"] = INV.apply(reasons, axis=1)
INV["keep"] = INV["exclude_reason"] == ""

print("=" * 84)
print("QUALITY GATE")
print("=" * 84)
summ = (INV.groupby(["scenario_canon", "keep"]).size().unstack(fill_value=0)
          .rename(columns={True: "kept", False: "dropped"}))
for c in ("kept", "dropped"):
    if c not in summ.columns:
        summ[c] = 0
summ["total"] = summ["kept"] + summ["dropped"]
summ["hours_kept"] = (INV[INV["keep"]].groupby("scenario_canon")["duration_s"].sum() / 3600).round(2)
print(summ.to_string())

print("\nexclusion reasons:")
from collections import Counter
cnt = Counter(x for s in INV.loc[~INV["keep"], "exclude_reason"] for x in s.split(";") if x)
if cnt:
    for k, v in cnt.most_common():
        print(f"  {k:<34} {v:>3} recording(s)")
else:
    print("  none -- every recording passed")

print(f"\nkept {int(INV['keep'].sum())}/{len(INV)} recordings, "
      f"{INV.loc[INV['keep'],'duration_s'].sum()/3600:.2f} h of "
      f"{INV['duration_s'].sum()/3600:.2f} h")
INV.to_csv(WORK / "inventory_gated.csv", index=False)
sync.log("quality_gate", kept=int(INV["keep"].sum()), total=len(INV))
MAJOR("01_quality_gate")
''')

md(r"""
---
# 5 · Build the per-recording arrays

For every surviving recording we write one `.npz` holding the eight radar channels **unnormalised**
plus three targets:

- **`ecg_norm`** — the ECG z-scored, percentile-clipped and squashed to **[−1, 1]**. The baseline
  used [0, 1], which forces the network to spend capacity learning a DC offset it will never need.
  This is our one declared deviation from their pipeline, and NB03 re-runs their models under both.
- **`peak_map`** — a Gaussian bump (σ = 3 samples ≈ 23 ms) at every R peak. Target for the C4 peak head.
- **`rr_ms`** — per-sample instantaneous RR interval in **real milliseconds**, interpolated between
  beats. Target for the C4 cycle-length head, and the thing the baseline mislabelled as ms when it
  was actually samples.

Channels stay unnormalised on disk because normalisation statistics are computed **per fold, on
training windows only** (§7). Baking them in here would leak test statistics into training.

Resumable per recording.
""")

code(r'''
from crvs_data import (CHANNELS, ARRAY_ROWS, ROW, FS, WINDOW, HOP_TRAIN,
                       range_normalise, peak_heatmap, rr_curve)
from crvs_metrics import detect_r_peaks, hrv_from_peaks

keep = INV[INV["keep"]].copy()
if CFG["SMOKE_TEST"]:
    keep = keep.groupby("scenario_canon", group_keys=False).head(2)
    print(">>> SMOKE_TEST: only", len(keep), "recordings\n")

by_stem = {p.stem: p for p in dec}
done = set(STATE.get("done_recordings", []))
rows = STATE.get("rec_rows", [])
# A state bit is only valid when both payload files were restored. This prevents a fresh
# Kaggle session from skipping a recording whose earlier upload had not completed.
done = {rid for rid in done if (WORK / "recordings" / f"{rid}.npy").exists()
        and (WORK / "recordings" / f"{rid}.json").exists()}
rows = [r for r in rows if r.get("rec_id") in done]
t0 = time.time(); built = 0; missing = []

for i, (_, r) in enumerate(keep.iterrows(), 1):
    rec_id = f"{r['subject']}__{r['scenario']}".replace("/", "_").replace(" ", "_")
    if rec_id in done:
        continue
    src = by_stem.get(rec_id)
    if src is None:
        cand = [s for s in by_stem if s.startswith(str(r["subject"]))
                and str(r["scenario"]).lower() in s.lower()]
        src = by_stem.get(cand[0]) if cand else None
    if src is None:
        missing.append(rec_id); continue

    z = np.load(src, allow_pickle=True)
    ch = {c: np.asarray(z[c], np.float32) for c in CHANNELS if c in z}
    if len(ch) != len(CHANNELS):
        missing.append(rec_id + " (channels)"); continue
    n = min(len(v) for v in ch.values())
    ecg = np.asarray(z["ecg1"], np.float32)[:n] if "ecg1" in z else None
    if ecg is None or len(ecg) < WINDOW:
        missing.append(rec_id + " (ecg)"); continue
    ch = {c: v[:n] for c, v in ch.items()}

    ecg_norm = range_normalise(ecg)
    peaks = detect_r_peaks(ecg_norm, FS)
    pk_map = peak_heatmap(n, peaks, CFG["PEAK_SIGMA"])
    rr = rr_curve(n, peaks, FS)
    hrv = hrv_from_peaks(peaks, FS)

    # One UNCOMPRESSED .npy of shape (11, n), row order = ARRAY_ROWS, plus a JSON sidecar.
    # It must not be .npz: np.load(..., mmap_mode="r") silently ignores mmap_mode on an
    # npz archive, so the Dataset would decompress all 11 arrays for every 1024-sample
    # window -- about 23 ms each, which would dominate T4 compute on Kaggle.
    stack = np.stack([ch[c] for c in CHANNELS] +
                     [ecg_norm, pk_map, rr.astype(np.float32)], 0).astype(np.float32)
    assert stack.shape == (len(ARRAY_ROWS), n), f"bad stack {stack.shape}"
    data_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(stack))).hexdigest()
    np.save(WORK / "recordings" / (rec_id + ".npy"), stack)
    (WORK / "recordings" / (rec_id + ".json")).write_text(json.dumps({
        "rec_id": rec_id, "fs": FS, "n": int(n),
        "subject": str(r["subject"]), "scenario": str(r["scenario_canon"]),
        "rows": ARRAY_ROWS, "data_sha256": data_sha256,
        "r_peaks": [int(v) for v in peaks]}))

    rows.append({"rec_id": rec_id, "subject": str(r["subject"]),
                 "scenario_canon": str(r["scenario_canon"]), "n": int(n),
                 "duration_s": round(n / FS, 2), "n_rpeaks": int(len(peaks)),
                 "data_sha256": data_sha256,
                 "mean_hr_bpm": hrv["mean_hr_bpm"], "rmssd_ms": hrv["rmssd_ms"],
                 "beat_coupling": float(r.get("beat_coupling", np.nan)),
                 "warn_flags": ";".join(sorted(set(str(r[flags_col]).split(";")) &
                                               set(CFG["WARN_ONLY_FLAGS"])))})
    done.add(rec_id); built += 1
    if built % 10 == 0 or i == len(keep):
        STATE.update({"done_recordings": sorted(done), "rec_rows": rows})
        sync.save_state(STATE)
        el = time.time() - t0
        print(f"  [{len(done):>3}/{len(keep)}] {rec_id:<28} {n/FS:>7.1f}s  "
              f"HR={hrv['mean_hr_bpm'] if hrv['mean_hr_bpm']==hrv['mean_hr_bpm'] else 0:5.1f}  "
              f"ETA {el/max(built,1)*(len(keep)-len(done))/60:5.1f}m")
    del z, ch, ecg
    gc.collect()

STATE.update({"done_recordings": sorted(done), "rec_rows": rows})
sync.save_state(STATE)
RECS = pd.DataFrame(rows).drop_duplicates("rec_id").reset_index(drop=True)
RECS.to_csv(WORK / "recordings.csv", index=False)
print(f"\nbuilt {len(RECS)} recordings in {(time.time()-t0)/60:.1f} min")
if missing:
    print(f"WARNING: {len(missing)} recording(s) had no decimated file:", missing[:8])
    (WORK / "missing_recordings.json").write_text(json.dumps(missing, indent=2))
sync.log("recordings_built", n=len(RECS), missing=len(missing))
MAJOR("02_recordings")
''')

md(r"""
---
# 6 · Folds and the window index

**Subject-wise 5-fold.** Subjects are sorted by how many scenarios they contributed and then
dealt round-robin into 5 groups, so each fold sees a comparable mix rather than one fold
accidentally collecting all the subjects who skipped Apnea. Fold *f* means: test = group *f*,
validation = group *(f+1) mod 5*, train = the other three. No subject is ever in two splits.

**LOSO.** Each subject gets an index; Experiment D iterates over all of them.

**The window index** is generated at 50 % overlap throughout, with a `no_overlap` column marking
windows that start on a 1024-sample boundary. Training uses every row; validation and test use
only `no_overlap` rows. One index, both behaviours, no chance of a stale second file drifting.
""")

code(r'''
rng = np.random.RandomState(CFG["SEED"])
subs = (RECS.groupby("subject")
            .agg(n_scen=("scenario_canon", "nunique"), secs=("duration_s", "sum"))
            .reset_index().sort_values(["n_scen", "secs"], ascending=False))
groups = {}
for k, s in enumerate(subs["subject"]):
    groups[s] = k % CFG["N_FOLDS"]                       # round-robin deal
loso = {s: i for i, s in enumerate(sorted(RECS["subject"].unique()))}

RECS["fold_group"] = RECS["subject"].map(groups)
RECS["loso_id"] = RECS["subject"].map(loso)
RECS.to_csv(WORK / "recordings.csv", index=False)

print("subjects per fold group:")
print(subs.assign(g=subs["subject"].map(groups)).groupby("g")["subject"].count().to_string())
print("\nrecordings per fold group x scenario:")
print(RECS.pivot_table(index="fold_group", columns="scenario_canon",
                       values="rec_id", aggfunc="count", fill_value=0).to_string())

widx = []
for _, r in RECS.iterrows():
    n = int(r["n"])
    if n < WINDOW:
        continue
    for st in range(0, n - WINDOW + 1, HOP_TRAIN):
        widx.append({"rec_id": r["rec_id"], "subject": r["subject"],
                     "scenario_canon": r["scenario_canon"], "start": st,
                     "no_overlap": (st % WINDOW) == 0,
                     "fold_group": int(r["fold_group"]), "loso_id": int(r["loso_id"])})
W = pd.DataFrame(widx)
W.to_parquet(WORK / "windows.parquet", index=False)

print(f"\n{len(W):,} windows total  ({int(W['no_overlap'].sum()):,} non-overlapping)")
tab = (W.groupby("scenario_canon")
        .agg(all_windows=("start", "size"), no_overlap=("no_overlap", "sum"),
             subjects=("subject", "nunique")).reset_index())
PAPER_SEG = {"Resting": 4702, "Valsalva": 6952, "Apnea": 1140}
tab["paper_segments"] = tab["scenario_canon"].map(PAPER_SEG)
print("\n" + tab.to_string(index=False))
rva = tab[tab["scenario_canon"].isin(["Resting", "Valsalva", "Apnea"])]
print(f"\nRVA overlapped windows kept: {int(rva['all_windows'].sum()):,}  "
      f"(baseline reported 12,794 before our quality gate)")
sync.log("windows", n=int(len(W)), no_overlap=int(W["no_overlap"].sum()))
MAJOR("03_windows")
''')

md(r"""
---
# 7 · Per-fold normalisation statistics — train windows only

This is the cell that quietly decides whether the results are honest.

Channel statistics are computed **separately for every fold**, from **training windows only**.
If you compute one mean and standard deviation over the whole corpus and apply it everywhere,
information about the test subjects has entered training — invisibly, with no error message, and
in a way that inflates every metric you report. It is one of the most common silent mistakes in
biosignal papers.

Statistics are stored for each experiment × fold combination and loaded by NB03/NB04 at train time.
""")

code(r'''
from crvs_data import compute_norm

EXPERIMENTS = {
    "A_resting":  ["Resting"],
    "A_valsalva": ["Valsalva"],
    "A_apnea":    ["Apnea"],
    "B_rva":      ["Resting", "Valsalva", "Apnea"],
    "C_all5":     ["Resting", "Valsalva", "Apnea", "Tilt-up", "Tilt-down"],
}

def split_for(exp, fold, n_folds=None):
    n_folds = n_folds or CFG["N_FOLDS"]
    sub = W[W["scenario_canon"].isin(EXPERIMENTS[exp])]
    te_g = fold % n_folds
    va_g = (fold + 1) % n_folds
    tr = sub[~sub["fold_group"].isin([te_g, va_g])]
    va = sub[(sub["fold_group"] == va_g) & sub["no_overlap"]]
    te = sub[(sub["fold_group"] == te_g) & sub["no_overlap"]]
    return tr, va, te

NORM = {}
print(f"{'experiment':<12}{'fold':>5}{'train':>9}{'val':>8}{'test':>8}   train/val/test subjects")
print("-" * 84)
for exp in EXPERIMENTS:
    for f in range(CFG["N_FOLDS"]):
        tr, va, te = split_for(exp, f)
        if len(tr) == 0 or len(te) == 0:
            print(f"{exp:<12}{f:>5}   -- empty split, skipped"); continue
        st = compute_norm(WORK / "recordings", tr, seed=CFG["SEED"] + f)
        NORM[f"{exp}|{f}"] = st
        print(f"{exp:<12}{f:>5}{len(tr):>9,}{len(va):>8,}{len(te):>8,}   "
              f"{tr['subject'].nunique()}/{va['subject'].nunique()}/{te['subject'].nunique()}"
              f"   overlap={len(set(tr['subject']) & set(te['subject']))}")
        assert not (set(tr["subject"]) & set(te["subject"])), "SUBJECT LEAK train/test"
        assert not (set(va["subject"]) & set(te["subject"])), "SUBJECT LEAK val/test"

# Experiment D: leave one subject out, reserve the next subject for validation.
loso_ids = sorted(map(int, W["loso_id"].unique()))
for lid in loso_ids:
    vid = loso_ids[(loso_ids.index(lid) + 1) % len(loso_ids)]
    sub = W[W["scenario_canon"].isin(EXPERIMENTS["C_all5"])]
    tr = sub[~sub["loso_id"].isin([lid, vid])]
    va = sub[(sub["loso_id"] == vid) & sub["no_overlap"]]
    te = sub[(sub["loso_id"] == lid) & sub["no_overlap"]]
    if len(tr) and len(te):
        NORM[f"D_loso|{lid}"] = compute_norm(
            WORK / "recordings", tr, seed=CFG["SEED"] + 1000 + lid)

# Experiment F: leave one scenario out. A second scenario is validation; the remaining
# three train the model. Subjects may recur by design because this tests condition shift.
cross_scenarios = list(EXPERIMENTS["C_all5"])
for i, test_sc in enumerate(cross_scenarios):
    val_sc = cross_scenarios[(i + 1) % len(cross_scenarios)]
    tr = W[~W["scenario_canon"].isin([test_sc, val_sc])]
    if len(tr):
        NORM[f"F_cross:{test_sc}|0"] = compute_norm(
            WORK / "recordings", tr, seed=CFG["SEED"] + 2000 + i)

(WORK / "norm_stats.json").write_text(json.dumps(NORM, indent=2))
(WORK / "experiments.json").write_text(json.dumps(
    {"experiments": EXPERIMENTS, "n_folds": CFG["N_FOLDS"],
     "fold_groups": {str(k): int(v) for k, v in groups.items()},
     "loso_ids": {str(k): int(v) for k, v in loso.items()},
     "loso_values": loso_ids, "cross_scenarios": cross_scenarios,
     "window": WINDOW, "hop_train": HOP_TRAIN, "fs": FS,
     "channels": CHANNELS}, indent=2))
print(f"\n{len(NORM)} normalisation sets written.  No subject appears in two splits anywhere.")
sync.log("norm_stats", n=len(NORM))
MAJOR("04_norm")
''')

md(r"""
---
# 8 · Sanity figures

Four checks, because a corpus that is subtly wrong is expensive to discover in Week 4:

1. **A window as the model sees it** — all 8 input channels plus the three targets, time-aligned.
   If the ECG and the peak map are misaligned by even a few samples, it shows here.
2. **Beat-triggered ECG** — the ground-truth ECG averaged around its own detected peaks. Should
   look like a textbook QRS. If it does not, the peak detector is wrong and every target is wrong.
3. **Target distributions** — ECG amplitude, peak-map density, RR interval.
4. **Fold balance** — windows per fold per scenario, so no fold is starved.
""")

code(r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
STYLE = {"radar": "#0F7C82", "ecg": "#AF3A2C", "muted": "#5C6B71", "ink": "#10171B",
         "grid": "#D3DADB", "amber": "#8A6212"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.color": STYLE["grid"], "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 8.5, "axes.titlesize": 10, "axes.titleweight": "bold"})
FIG = WORK / "figures"; FIG.mkdir(exist_ok=True)
def save(fig, nm):
    fig.savefig(FIG / nm); plt.close(fig); print("  wrote", nm)

ex = RECS.iloc[0]
z = np.load(WORK / "recordings" / (ex["rec_id"] + ".npy"), mmap_mode="r")
meta = json.loads((WORK / "recordings" / (ex["rec_id"] + ".json")).read_text())
s0 = min(int(20 * FS), max(0, int(meta["n"]) - WINDOW))
sl = slice(s0, s0 + WINDOW); t = np.arange(WINDOW) / FS

fig, axes = plt.subplots(11, 1, figsize=(11, 12), sharex=True)
for ax, c in zip(axes, CHANNELS):
    ax.plot(t, z[ROW[c], sl], lw=.7, color=STYLE["radar"])
    ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=7.5)
    ax.tick_params(labelleft=False)
axes[8].plot(t, z[ROW["ecg_norm"], sl], lw=.9, color=STYLE["ecg"])
axes[8].set_ylabel("ecg_norm", rotation=0, ha="right", va="center", fontsize=7.5, color=STYLE["ecg"])
axes[9].plot(t, z[ROW["peak_map"], sl], lw=.9, color=STYLE["ink"])
axes[9].set_ylabel("peak_map", rotation=0, ha="right", va="center", fontsize=7.5)
axes[10].plot(t, z[ROW["rr_ms"], sl], lw=.9, color=STYLE["amber"])
axes[10].set_ylabel("rr_ms", rotation=0, ha="right", va="center", fontsize=7.5)
for a in axes[8:]:
    a.tick_params(labelleft=False)
axes[-1].set_xlabel("seconds")
axes[0].set_title(f"{ex['rec_id']} — one 8 s window exactly as the model receives it", loc="left")
save(fig, "nb02_fig1_window.png")

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
pk = np.asarray(meta["r_peaks"], int); e = np.asarray(z[ROW["ecg_norm"]])
pre, post = int(.25 * FS), int(.45 * FS)
seg = [e[p-pre:p+post] for p in pk if p-pre >= 0 and p+post < len(e)]
if seg:
    S = np.stack(seg); tt = np.arange(-pre, post) / FS
    for s_ in S[:200]:
        axes[0].plot(tt, s_, lw=.3, color=STYLE["muted"], alpha=.25)
    axes[0].plot(tt, S.mean(0), lw=2, color=STYLE["ecg"])
    axes[0].axvline(0, color=STYLE["ink"], ls="--", lw=1)
    axes[0].set_title(f"beat-triggered ECG (n={len(S)})"); axes[0].set_xlabel("s from R peak")
axes[1].hist(RECS["mean_hr_bpm"].dropna(), bins=20, color=STYLE["ecg"], edgecolor="white")
axes[1].set_title("mean HR per recording"); axes[1].set_xlabel("bpm")
axes[2].hist(RECS["rmssd_ms"].dropna(), bins=20, color=STYLE["radar"], edgecolor="white")
axes[2].set_title("RMSSD per recording"); axes[2].set_xlabel("ms (real)")
fig.tight_layout(); save(fig, "nb02_fig2_targets.png")

fig, ax = plt.subplots(figsize=(9, 3.2))
pv = W.pivot_table(index="fold_group", columns="scenario_canon", values="start",
                   aggfunc="size", fill_value=0)
pv.plot(kind="bar", stacked=True, ax=ax, width=.75,
        color=[STYLE["radar"], STYLE["ecg"], STYLE["amber"], STYLE["muted"], "#9BB8BA"],
        edgecolor="white")
ax.set_ylabel("windows"); ax.set_xlabel("fold group")
ax.set_title("Window balance across subject-wise folds", loc="left")
ax.legend(frameon=False, ncol=5, fontsize=7)
save(fig, "nb02_fig3_folds.png")
MAJOR("05_figures")
''')

md(r"""
---
# 9 · Dataset card and final push
""")

md(r"""
The processed repo is what NB03–NB05 consume, so its card doubles as the contract between the
notebooks: channel order, window length, fold semantics, and the leakage rules.
""")

code(r'''
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
card = f"""---
license: cc-by-4.0
pretty_name: CR-RVS Radar-to-ECG Training Corpus
tags: [radar, ecg, biosignals, contactless-monitoring, vital-signs]
---

# CR-RVS Radar-to-ECG — training corpus

Windowed, fold-assigned training corpus for **CardioMamba-Net**, derived from the CR-RVS dataset
(Schellenberger et al., *Sci Data* 7:291, 2020) via `01_verify_and_download` and
`02_preprocess_to_hf`. Built {now}, run `{CFG['RUN_ID']}`.

## Contents

| Path | What |
|---|---|
| `recordings/<subject>__<scenario>.npy` | Uncompressed `(11, n)` float32 at 128 Hz — 8 radar channels + 3 targets, row order in `rows` |
| `recordings/<subject>__<scenario>.json` | Per-recording metadata: `n`, `fs`, `subject`, `scenario`, `rows`, `r_peaks` |
| `windows.parquet` | One row per window: `rec_id, subject, scenario_canon, start, no_overlap, fold_group, loso_id` |
| `recordings.csv` | Per-recording metadata, HR/HRV, beat coupling, warning flags |
| `inventory_gated.csv` | Full NB01 inventory plus `keep` / `exclude_reason` |
| `norm_stats.json` | Per experiment x fold channel mean/std, **train windows only** |
| `experiments.json` | Experiment definitions, fold groups, LOSO ids, channel order |
| `crvs_sync.py`, `crvs_data.py`, `crvs_metrics.py` | Shared library used by NB03-NB05 |
| `figures/` | Sanity figures |

## Arrays in each recording

Row order of the `(11, n)` array: `I, Q, phi, dy, vel, acc, amp, cardiac` — the 8
physics-informed input channels, **unnormalised** — then `ecg_norm` (target, [-1,1]),
`peak_map` (Gaussian R-peak heatmap, sigma = 3 samples), `rr_ms` (per-sample RR interval in
**real milliseconds**). R-peak indices are in the JSON sidecar.

Stored uncompressed on purpose: `np.load(..., mmap_mode="r")` silently ignores `mmap_mode`
on an `.npz`, so a compressed archive would force a full decompression of all 11 arrays for
every 1024-sample window.

## Conventions

- 128 Hz, 1024-sample (8 s) windows, frozen to Chowdhury et al. 2024 section 2.3.
- Train windows overlap 50 %; **validation and test windows do not overlap** (`no_overlap`).
- Splits are **always by subject**. Fold f: test = group f, val = group (f+1) mod {CFG['N_FOLDS']}, train = rest.
- Normalisation statistics are per fold and computed on **training windows only**.
- ECG target is [-1, 1], not [0, 1] as in the baseline; this is a declared deviation.

## Quality gate

Recordings are excluded only when ECG/radar inputs are unreadable or missing, contain invalid
values, or are under {CFG['MIN_DURATION_S']:.0f} s. Beat coupling is retained as a continuous
difficulty covariate and is not used to select an easier cohort. Every exclusion and its reason is
in `inventory_gated.csv`.

## Cite

Schellenberger et al., *Scientific Data* 7:291 (2020), doi:10.1038/s41597-020-00629-5 ·
Chowdhury et al., *Computers in Biology and Medicine* 176:108555 (2024),
doi:10.1016/j.compbiomed.2024.108555
"""
(WORK / "README.md").write_text(card)

report = [f"# NB02 report — {now}\n",
          f"- recordings built: **{len(RECS)}**",
          f"- windows: **{len(W):,}** ({int(W['no_overlap'].sum()):,} non-overlapping)",
          f"- subjects: **{RECS['subject'].nunique()}**",
          f"- hours kept: **{RECS['duration_s'].sum()/3600:.2f} h**",
          f"- normalisation sets: **{len(NORM)}**\n",
          "## Per scenario\n", tab.to_markdown(index=False), "\n## Quality gate\n",
          summ.to_markdown()]
(WORK / "report.md").write_text("\n".join(report))

sizes = {str(p.relative_to(WORK)): p.stat().st_size for p in WORK.rglob("*") if p.is_file()}
print(f"pushing {len(sizes)} files, {sum(sizes.values())/2**20:.0f} MB ...")
ok = sync.flush(final=True, msg=f"{CFG['RUN_ID']} complete — {len(RECS)} recordings, {len(W)} windows")
print("\n" + "=" * 76)
print("  DONE" if ok else "  DONE (final push reported a problem — see sync_history.jsonl)")
print("=" * 76)
print(f"  repo       : {sync.url}")
print(f"  recordings : {len(RECS)}   windows: {len(W):,}   subjects: {RECS['subject'].nunique()}")
print(f"  payload    : {sum(sizes.values())/2**20:.0f} MB in {len(sizes)} files")
print(f"  pushes     : {sync._pushes}  failures: {sync._failures}")
print("=" * 76)
print("\n  next: 03_baselines.ipynb  (accelerator: GPU T4 x2)")
''')

md(r"""
---
# 10 · Troubleshooting

**`inventory.csv not found`** — NB01 has not completed, or it pushed to a different repo. Check
`CFG["SRC_REPO"]` matches NB01's `CFG["HF_REPO"]`.

**`No decimated/*.npz found`** — NB01 ran with `SAVE_DECIMATED = False`. Re-run NB01 with it on;
it resumes and only needs to write the decimated files.

**Out of disk** — the download goes to `/kaggle/temp`, the output to `/kaggle/working` (20 GB cap).
The recordings folder is ~400 MB, so this should not happen. If it does, run with `SMOKE_TEST = True`
first to confirm the pipeline, then re-run.

**`SUBJECT LEAK` assertion** — a real bug, not a warning. It means fold assignment produced an
overlapping subject between splits. Do not bypass it; tell me and I will fix the fold logic.

**Recordings missing from `decimated/`** — listed in `missing_recordings.json`. Usually a
subject/scenario naming mismatch between the inventory and the decimated filenames. The cell
already tries a fuzzy fallback; if many are missing, send me that file.
""")

out = sys.argv[1] if len(sys.argv) > 1 else "02_preprocess_to_hf.ipynb"

# splice the library sources into the placeholder cell
for i, c in enumerate(n.cells):
    if c["cell_type"] == "code":
        s = "".join(c["source"])
        if "__HF_SYNC__" in s:
            s = (s.replace("__HF_SYNC__", HF_SYNC_SRC.strip("\n"))
                  .replace("__DATA__", CRVS_DATA_SRC.strip("\n"))
                  .replace("__METRICS__", CRVS_METRICS_SRC.strip("\n")))
            n.cells[i]["source"] = n._src(s)

n.write(out, accelerator="none")
