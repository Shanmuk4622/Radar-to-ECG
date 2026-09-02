#!/usr/bin/env python3
"""Emit 05_evaluate_and_figures.ipynb"""
import sys
from nb_lib_a import NB, HF_SYNC_SRC, CRVS_DATA_SRC, CRVS_METRICS_SRC
from nb_lib_b import CRVS_MODELS_SRC, CRVS_CMNET_SRC, CRVS_LOSS_SRC, CRVS_ENGINE_SRC

n = NB(); md, code = n.md, n.code

md(r"""
# NB05 — Evaluation, statistics and paper figures

**Project:** CardioMamba-Net · **Stage:** 5 of 5
`01_verify` → `02_preprocess` → `03_baselines` → `04_cardiomamba_train` → **`05_evaluate`**

---

## What this produces

Everything the manuscript needs, generated from the per-window metrics that NB03 and NB04 already
wrote — so **no model is retrained here**. Only the robustness section (§7) runs inference, and it
is optional.

| Output | Corresponds to |
|---|---|
| `table2_per_scenario.csv` | Their Table 2 — per-scenario, all models |
| `table3_rva_combined.csv` | Their Table 3 — the headline comparison |
| `table3b/3c/3d_*.csv` | All-five, LOSO, and held-out-scenario generalisation |
| `table4_peak_detection.csv` | Their Table 4 — R-peak accuracy/precision/recall/F1 |
| `table5_hrv.csv` | Their Table 5 — μRR, σRR, μHR, σHR, RMSSD, **in real milliseconds** |
| `table6_ablation.csv` | Ours — the ablation ladder |
| `table7_significance.csv` | Ours — Wilcoxon signed-rank with Holm correction |
| `table8_budget.csv` | Ours — parameters, and correlation per million parameters |
| 10 figures | Bland–Altman ×2, per-subject box plots, qualitative grid, ablation, robustness, budget scatter |

## What the statistics are for

The baseline reports means and standard deviations and stops. That is not enough to claim a win.
Here the full model is compared with every ablation using a **subject-paired Wilcoxon signed-rank
test**, and the p-values are **Holm-corrected**. Five fold averages are too few for a two-sided
Wilcoxon test to reach 0.05 even when every fold moves in the same direction; subjects are the
independent experimental units and provide the defensible paired analysis.

Bland–Altman with limits of agreement is the standard way to report agreement between two
measurement methods in clinical work, and it is what a reviewer from a medical journal will look
for on heart rate and HRV. A correlation coefficient alone does not tell them whether the method
can be trusted on an individual patient.

## ⚠️ Accelerator: **GPU T4 × 2** *(only needed for §7)*

Sections 1–6 and 8 run fine on CPU. If you only want the tables and figures, set
`CFG["RUN_ROBUSTNESS"] = False` and use **Accelerator: None**.

When robustness is enabled, please attach NB02's saved output with **+ Add Input → Notebook
Output**. Tables use the HF run repositories; the attached corpus makes inference faster and
keeps it outside the 20 GB working area.

## Cell-by-cell run guide

| Code cell | What runs | Typical time |
|---:|---|---:|
| 1 | Configuration | < 5 s |
| 2 | Imports/dependencies/output folders | 1–3 min |
| 3 | Write libraries, HF login, start results-repo sync | 1–3 min |
| 4 | Download summaries/metrics from baseline and model repos | 2–15 min |
| 5 | Build waveform comparison tables | < 1 min |
| 6 | Build peak, HR and HRV tables | < 2 min |
| 7 | Ablation, subject-paired Wilcoxon-Holm, compute budget | 1–5 min |
| 8 | Generate manuscript figures | 2–10 min |
| 9 | Optional GPU robustness inference | 20–90 min |
| 10 | Write manuscript summary and final HF upload | 2–15 min |

Total without robustness: **10–45 minutes**. With robustness: typically **30–120 minutes**, mainly
depending on whether inputs/checkpoints are already cached.
""")

md("---\n# 1 · Configuration")

code(r'''
CFG = {
    "DATA_REPO":     "Shanmuk4622/cr-rvs-radar-ecg-processed-v2",
    "BASELINE_REPO": "Shanmuk4622/cardiomamba-baselines-v2",
    "MODEL_REPO":    "Shanmuk4622/cardiomamba-net-v2",
    "RESULT_REPO":   "Shanmuk4622/cardiomamba-results-v2",
    "HF_PRIVATE": False,
    "RUN_ID":     "nb05_evaluation_v2",

    "WORK":    "/kaggle/working/nb05",
    "SCRATCH": "/kaggle/temp/nb05",
    "PUSH_INTERVAL_S": 30 * 60,
    "HF_MAX_UPLOADS_HOUR": 24,

    "RUN_ROBUSTNESS": True,          # needs GPU + checkpoints; set False for tables only
    "SNR_DB": [12, 6, 3, 0, -3],
    "MOTION_AMPLITUDE": [0.25, 0.50],   # normalised slow-drift amplitudes
    "TEST_CHANNEL_DROPOUT": True,
    "ROBUST_MODELS": ["L9_full", "multireslinknet"],
    "ROBUST_MAX_WINDOWS": 800,

    "HEADLINE_EXP": "B_rva",
    "ALPHA": 0.05,
    "SEED": 1337,
}
import json
print(json.dumps(CFG, indent=2))
''')

code(r'''
import os, sys, gc, json, math, time, warnings, subprocess, itertools
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

import numpy as np, pandas as pd
WORK = Path(CFG["WORK"]); SCRATCH = Path(CFG["SCRATCH"])
for d in (WORK, SCRATCH, WORK / "tables", WORK / "figures"):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORK))
pd.set_option("display.width", 240, "display.max_columns", 60)
print("pandas", pd.__version__, "| numpy", np.__version__)
''')

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
import importlib
for nm in MODULES:
    sys.modules.pop(nm[:-3], None)
importlib.invalidate_caches()
import crvs_data
REQUIRED_LIB = 4
if getattr(crvs_data, "LIB_VERSION", 0) < REQUIRED_LIB:
    raise RuntimeError(f"stale crvs_data v{getattr(crvs_data,'LIB_VERSION','missing')}, "
                       f"need >= {REQUIRED_LIB}. Restart the kernel.")
print(f"library written ({len(MODULES)} modules), crvs_data v{crvs_data.LIB_VERSION}")

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
sync = HFSync(repo_id=CFG["RESULT_REPO"], local_dir=WORK, token=HF_TOKEN, repo_type="model",
              private=CFG["HF_PRIVATE"], run_id=CFG["RUN_ID"],
              push_interval_s=CFG["PUSH_INTERVAL_S"],
              max_upload_calls_hour=CFG["HF_MAX_UPLOADS_HOUR"])
print("\nresults repo:", sync.url)
sync.pull(allow_patterns=["*.json", "*.jsonl", "*.md", "tables/*", "figures/*"])
_M = {"f": False, "n": ""}
def MAJOR(nm):
    _M["f"] = True; _M["n"] = nm
def _hook(r=None):
    if _M["f"]:
        nm = _M["n"]; _M["f"] = False; _M["n"] = ""; sync.stage_done(nm)
try:
    get_ipython().events.register("post_run_cell", _hook)
except Exception:
    pass
MAJOR("00_setup")
''')

md(r"""
---
# 2 · Collect every run

Pull the per-run summaries, per-window metrics and per-subject HRV from the model repo.
Checkpoints (`*.pt`) are **not** downloaded unless §7 runs — they are the bulk of the repo and the
tables do not need them.
""")

code(r'''
from huggingface_hub import snapshot_download
pats = ["runs/**/summary.json", "runs/**/metrics_windows.parquet",
        "runs/**/metrics_subjects.parquet", "runs/**/metrics_recordings.parquet",
        "runs/**/preds_sample.npz",
        "runs/**/state.json", "runs/**/run_config.json", "results/*", "README.md"]
if CFG["RUN_ROBUSTNESS"]:
    pats.append("runs/**/best.pt")
t0 = time.time()
RUN_ROOTS = []
for label, repo in (("baselines", CFG["BASELINE_REPO"]), ("cardiomamba", CFG["MODEL_REPO"])):
    root = SCRATCH / label
    snapshot_download(repo, repo_type="model", token=HF_TOKEN,
                      local_dir=str(root), allow_patterns=pats, max_workers=4)
    RUN_ROOTS.append(root)
print(f"downloaded in {time.time()-t0:.0f}s")

def norm_variant(s):
    v = s.get("variant") or s.get("model")
    if v == "multireslinknet" and s.get("loss", "mse") == "mse":
        return "multireslinknet"
    return v

rows, wrows, srows = [], [], []
summary_paths = []
for root in RUN_ROOTS:
    summary_paths.extend((root / "runs").glob("*/summary.json"))
for p in sorted(summary_paths):
    try:
        s = json.loads(p.read_text())
    except Exception:
        continue
    v = norm_variant(s)
    base = {"run_id": s["run_id"], "experiment": s["experiment"], "variant": v,
            "fold": s["fold"], "params": s.get("params"), "best_epoch": s.get("best_epoch"),
            "gflops_per_window": s.get("gflops_per_window"),
            "forward_ms": s.get("forward_ms_batch2", s.get("forward_ms_batch4"))}
    rows.append({**base, **{k: val for k, val in s["metrics"].items() if not k.endswith("_std")}})
    mw = p.parent / "metrics_windows.parquet"
    if mw.exists():
        d = pd.read_parquet(mw); d["variant"] = v; d["experiment"] = s["experiment"]
        d["fold"] = s["fold"]; wrows.append(d)
    msj = p.parent / "metrics_subjects.parquet"
    if msj.exists():
        d = pd.read_parquet(msj); d["variant"] = v; d["experiment"] = s["experiment"]
        d["fold"] = s["fold"]; srows.append(d)

R  = pd.DataFrame(rows)
WD = pd.concat(wrows, ignore_index=True) if wrows else pd.DataFrame()
SD = pd.concat(srows, ignore_index=True) if srows else pd.DataFrame()
if not len(R):
    raise RuntimeError("No runs found. Run NB03 and NB04 first.")
R.to_csv(WORK / "tables" / "all_runs.csv", index=False)
print(f"runs: {len(R)}   window rows: {len(WD):,}   subject rows: {len(SD):,}")
print("\nruns per experiment x variant:")
print(R.pivot_table(index="variant", columns="experiment", values="fold",
                    aggfunc="count", fill_value=0).to_string())
''')

md(r"""
---
# 3 · Tables 2 and 3 — the direct comparison

Same layout as the baseline's tables so a reader can put them side by side. The `_paper` columns
are their published figures; `Δ` is ours minus theirs.

A reminder on how to read the sign. Our splits are strictly subject-wise with non-overlapping test
windows, which theirs almost certainly were not. So a **baseline** row landing below its published
value is expected — that is leakage being removed. The claim we are making is about
**CardioMamba-Net versus our own re-run baselines**, under identical conditions. The published
column is context, not the yardstick.
""")

code(r'''
COLS  = ["MAE", "MSE", "CC_temporal", "CC_spectral", "RRMSE_temporal", "RRMSE_spectral"]
ORDER = ["fpn", "unet", "linknet", "multireslinknet",
         "L2_loss_only", "L3_c1_only", "L4_c1_c5", "L5_no_wavelet", "L6_no_ssm",
         "L7_singletask", "L8_no_film", "L9_full", "L10_transformer"]
NICE = {"fpn": "FPN", "unet": "UNet", "linknet": "LinkNet",
        "multireslinknet": "MultiResLinkNet", "L9_full": "CardioMamba-Net (ours)",
        "L10_transformer": "CardioMamba-Net (Transformer)"}

PAPER_A = {
 ("A_resting","fpn"):(0.14204,0.03170,58.37,71.38,0.46940,0.73374),
 ("A_resting","unet"):(0.13872,0.03219,63.10,74.68,0.45760,0.86096),
 ("A_resting","linknet"):(0.13588,0.03034,64.35,74.37,0.45116,0.81111),
 ("A_resting","multireslinknet"):(0.13258,0.03066,66.10,82.44,0.43682,0.71412),
 ("A_valsalva","fpn"):(0.14985,0.03679,57.53,65.97,0.46395,0.87990),
 ("A_valsalva","unet"):(0.15249,0.03928,58.38,68.79,0.46553,0.99554),
 ("A_valsalva","linknet"):(0.15087,0.03869,56.63,66.87,0.46195,0.99095),
 ("A_valsalva","multireslinknet"):(0.15286,0.04012,60.14,77.05,0.46083,0.80660),
 ("A_apnea","fpn"):(0.15310,0.03853,39.12,51.26,0.51017,1.00889),
 ("A_apnea","unet"):(0.14406,0.03495,56.14,69.97,0.47825,0.92034),
 ("A_apnea","linknet"):(0.14572,0.03526,56.22,70.35,0.47944,0.91749),
 ("A_apnea","multireslinknet"):(0.14474,0.03474,55.33,74.66,0.47692,0.82392),
 ("B_rva","fpn"):(0.14316,0.03422,59.63,69.53,0.44694,0.83026),
 ("B_rva","unet"):(0.14798,0.03741,57.65,68.39,0.45315,0.94118),
 ("B_rva","linknet"):(0.14780,0.03723,58.69,70.91,0.45487,0.86909),
 ("B_rva","multireslinknet"):(0.14841,0.03793,61.86,79.96,0.44618,0.73269),
}

def table_for(exps, fname, title):
    sub = R[R["experiment"].isin(exps)]
    if not len(sub):
        print(f"(no runs for {exps})"); return None
    g = sub.groupby(["experiment", "variant"])
    T = g[COLS].mean().round(5)
    Tsd = g[COLS].std().round(5)
    T["folds"] = g.size()
    T = T.reset_index()
    for i, r in T.iterrows():
        key = (r["experiment"], r["variant"])
        if key in PAPER_A:
            p = PAPER_A[key]
            T.loc[i, "CC_t_paper"] = p[2]; T.loc[i, "CC_s_paper"] = p[3]
            T.loc[i, "MAE_paper"] = p[0]
            T.loc[i, "dCC_t"] = round(r["CC_temporal"] - p[2], 2)
    T["order"] = T["variant"].map(lambda v: ORDER.index(v) if v in ORDER else 99)
    T = T.sort_values(["experiment", "order"]).drop(columns="order")
    T["variant"] = T["variant"].map(lambda v: NICE.get(v, v))
    print("=" * 130); print(title); print("=" * 130)
    print(T.to_string(index=False))
    T.to_csv(WORK / "tables" / fname, index=False)
    return T

T2 = table_for(["A_resting", "A_valsalva", "A_apnea"], "table2_per_scenario.csv",
               "TABLE 2  —  per scenario  (their Table 2)")
print()
T3 = table_for(["B_rva"], "table3_rva_combined.csv",
               "TABLE 3  —  Resting + Valsalva + Apnea combined  (their Table 3)")
print()
TC = table_for(["C_all5"], "table3b_all_five.csv",
               "TABLE 3b  —  ALL FIVE SCENARIOS  (new: the baseline never evaluated Tilt)")
print()
TD = table_for(["D_loso"], "table3c_loso.csv",
               "TABLE 3c  —  LEAVE-ONE-SUBJECT-OUT GENERALISATION")
print()
cross_exps = sorted(x for x in R["experiment"].unique() if str(x).startswith("F_cross:"))
TF = table_for(cross_exps, "table3d_cross_scenario.csv",
               "TABLE 3d  —  HELD-OUT-SCENARIO GENERALISATION")
MAJOR("01_tables_2_3")
''')

md(r"""
---
# 4 · Tables 4 and 5 — beats and rhythm

Table 4 is R-peak detection on the reconstructed ECG. We add two columns the baseline does not
report: **median timing error in milliseconds** and **missed-detection rate**. Precision and recall
alone hide whether a "detected" beat is 10 ms or 90 ms off, and for HRV that difference is
everything.

Table 5 is the HRV comparison, and it is where the baseline's weakness is most visible. Their
predicted RMSSD is roughly **double** ground truth in every scenario (12.95 → 23.87 ms resting;
13.17 → 31.50 ms apnea) — the fingerprint of a smeared, jittery QRS. Our RMSSD error column is
the direct test of whether C5 fixed it.

We also report μRR in **genuine milliseconds**. Theirs is 126 ms alongside a heart rate of 62 bpm,
which is arithmetically impossible — 126 samples at 128 Hz is 0.98 s, so their column is samples
mislabelled as milliseconds.
""")

code(r'''
if len(SD):
    keep = [v for v in ORDER if v in set(SD["variant"])]
    g = SD[SD["experiment"] == CFG["HEADLINE_EXP"]].groupby("variant")
    T4 = g.agg(accuracy=("accuracy", "mean"), F1=("F1", "mean"),
               precision=("precision", "mean"), recall=("recall", "mean"),
               TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"),
               timing_err_ms=("timing_err_ms_median", "mean"),
               missed_rate=("missed_rate", "mean")).round(4)
    T4 = T4.reindex([v for v in keep if v in T4.index])
    T4.index = [NICE.get(i, i) for i in T4.index]
    print("=" * 118)
    print(f"TABLE 4  —  R-peak detection on the reconstructed ECG  ({CFG['HEADLINE_EXP']})")
    print("=" * 118)
    print(T4.to_string())
    print("\npublished (MultiResLinkNet, RVA): accuracy 0.886  F1 0.939  precision 0.973  recall 0.908")
    T4.to_csv(WORK / "tables" / "table4_peak_detection.csv")

    rows5 = []
    for v in keep:
        d = SD[(SD["variant"] == v) & (SD["experiment"] == CFG["HEADLINE_EXP"])]
        if not len(d):
            continue
        rows5.append({"variant": NICE.get(v, v), "signal": "ground truth",
                      "mu_RR_ms": d["gt_mean_rr_ms"].mean(), "sd_RR_ms": d["gt_sd_rr_ms"].mean(),
                      "mu_HR_bpm": d["gt_mean_hr_bpm"].mean(), "sd_HR_bpm": d["gt_sd_hr_bpm"].mean(),
                      "RMSSD_ms": d["gt_rmssd_ms"].mean()})
        rows5.append({"variant": NICE.get(v, v), "signal": "predicted",
                      "mu_RR_ms": d["pr_mean_rr_ms"].mean(), "sd_RR_ms": d["pr_sd_rr_ms"].mean(),
                      "mu_HR_bpm": d["pr_mean_hr_bpm"].mean(), "sd_HR_bpm": d["pr_sd_hr_bpm"].mean(),
                      "RMSSD_ms": d["pr_rmssd_ms"].mean()})
        rows5.append({"variant": NICE.get(v, v), "signal": "|error|",
                      "mu_RR_ms": (d["gt_mean_rr_ms"] - d["pr_mean_rr_ms"]).abs().mean(),
                      "sd_RR_ms": np.nan,
                      "mu_HR_bpm": (d["gt_mean_hr_bpm"] - d["pr_mean_hr_bpm"]).abs().mean(),
                      "sd_HR_bpm": np.nan,
                      "RMSSD_ms": (d["gt_rmssd_ms"] - d["pr_rmssd_ms"]).abs().mean()})
    T5 = pd.DataFrame(rows5).round(2)
    print("\n" + "=" * 118)
    print(f"TABLE 5  —  HR and HRV, in REAL milliseconds  ({CFG['HEADLINE_EXP']})")
    print("=" * 118)
    print(T5.to_string(index=False))
    T5.to_csv(WORK / "tables" / "table5_hrv.csv", index=False)
    print("\npublished (MultiResLinkNet, resting): RMSSD ground truth 12.95 ms -> predicted 23.87 ms")
    print("i.e. an 84 % over-estimate. The |error| rows above are the direct comparison.")
else:
    print("no per-subject metrics found -- NB03/NB04 write these; re-run them.")
MAJOR("02_tables_4_5")
''')

md(r"""
---
# 5 · Table 6 — the ablation, and Table 7 — significance

Table 7 is what lets us write "significantly better" when the data support it. Wilcoxon
signed-rank is paired and non-parametric: each subject contributes one average temporal
correlation per model, regardless of how many windows that subject has.

The planned comparisons are the full model against each baseline/ablation. Holm correction
controls the family-wise error rate without pretending that all 45 possible pairs were hypotheses
we intended to test.
""")

code(r'''
from scipy import stats as sstats

b = R[R["experiment"] == CFG["HEADLINE_EXP"]]
LAB6 = {"multireslinknet": "1. MultiResLinkNet + MSE (baseline)",
        "L2_loss_only": "2. + composite loss (C5)",
        "L3_c1_only": "3. + 8-channel input (C1)",
        "L4_c1_c5": "4. + C1 + C5",
        "L5_no_wavelet": "5. CardioMamba, no wavelet (-C2)",
        "L6_no_ssm": "6. CardioMamba, no SSM (-C3)",
        "L7_singletask": "7. CardioMamba, single-task (-C4)",
        "L8_no_film": "8. CardioMamba, no FiLM",
        "L9_full": "9. CardioMamba-Net (full)",
        "L10_transformer": "10. Transformer bottleneck (control)"}
lad = [v for v in LAB6 if v in set(b["variant"])]
if lad:
    extra = [c for c in ("peak_F1", "MAE_mean_hr_bpm", "MAE_rmssd_ms") if c in b.columns]
    T6 = b[b["variant"].isin(lad)].groupby("variant")[COLS + extra + ["params"]].mean()
    T6 = T6.reindex(lad)
    T6["folds"] = b.groupby("variant").size().reindex(lad)
    ref = T6.loc["multireslinknet", "CC_temporal"] if "multireslinknet" in T6.index else np.nan
    T6["dCC_t"] = (T6["CC_temporal"] - ref).round(2)
    T6.index = [LAB6[i] for i in T6.index]
    print("=" * 132); print("TABLE 6  —  ablation ladder"); print("=" * 132)
    print(T6.round(5).to_string())
    T6.to_csv(WORK / "tables" / "table6_ablation.csv")

    # Subject is the independent unit. First average windows within subject/model, then
    # align the exact same subjects for every full-vs-comparator test.
    subject_cc = (WD[WD["experiment"] == CFG["HEADLINE_EXP"]]
                  .groupby(["subject", "variant"], as_index=False)["CC_temporal"].mean())
    full = "L9_full"; raw = []
    if full in set(subject_cc["variant"]):
        for v in [x for x in lad if x != full]:
            pair = subject_cc[subject_cc["variant"].isin([full, v])].pivot(
                index="subject", columns="variant", values="CC_temporal").dropna()
            if len(pair) < 6 or np.allclose(pair[full], pair[v]):
                p = np.nan
            else:
                p = float(sstats.wilcoxon(pair[full], pair[v], alternative="two-sided").pvalue)
            raw.append({"a": full, "b": v, "n_subjects": len(pair), "p": p,
                        "median_delta_cc_t": float(np.median(pair[full]-pair[v])) if len(pair) else np.nan})
    if raw:
        finite = sorted([i for i, r in enumerate(raw) if np.isfinite(r["p"])], key=lambda i: raw[i]["p"])
        running = 0.0; mtests = len(finite)
        for rank, i in enumerate(finite):
            running = max(running, (mtests-rank)*raw[i]["p"])
            raw[i]["p_holm"] = min(1.0, running)
        for r in raw: r.setdefault("p_holm", np.nan)
        T7 = pd.DataFrame(raw)
        T7["a"] = T7["a"].map(LAB6); T7["b"] = T7["b"].map(LAB6)
        T7["significant"] = T7["p_holm"] < CFG["ALPHA"]
        T7 = T7.sort_values("p_holm", na_position="last")
        print("\n" + "=" * 118)
        print("TABLE 7  —  subject-paired Wilcoxon on CC_temporal, Holm-corrected")
        print("=" * 118)
        print(T7.to_string(index=False))
        T7.to_csv(WORK / "tables" / "table7_significance.csv", index=False)
        vs = T7[(T7["a"].str.contains("full")) | (T7["b"].str.contains("full"))]
        if len(vs):
            print(f"\n  comparisons involving the full model: "
                  f"{int(vs['significant'].sum())}/{len(vs)} significant at alpha={CFG['ALPHA']}")
    else:
        print("\nTABLE 7 skipped: full-model and comparator per-subject rows are incomplete.")
        print("Run NB03/NB04 with QUICK=False so all subject-held-out folds complete.")

if "params" in R.columns and R["params"].notna().any():
    T8 = (R[R["experiment"] == CFG["HEADLINE_EXP"]]
          .groupby("variant").agg(params=("params", "mean"),
                                  gflops=("gflops_per_window", "mean"),
                                  forward_ms=("forward_ms", "mean"),
                                  CC_temporal=("CC_temporal", "mean")).dropna(subset=["params", "CC_temporal"]))
    T8["M_params"] = (T8["params"] / 1e6).round(3)
    T8["CC_per_Mparam"] = (T8["CC_temporal"] / T8["M_params"]).round(2)
    T8 = T8.sort_values("CC_temporal", ascending=False)
    T8.index = [NICE.get(i, LAB6.get(i, i)) for i in T8.index]
    print("\n" + "=" * 96)
    print("TABLE 8  —  budget.  The baseline paper reports neither parameters nor FLOPs.")
    print("=" * 96)
    print(T8[["M_params", "gflops", "forward_ms", "CC_temporal", "CC_per_Mparam"]].to_string())
    T8.to_csv(WORK / "tables" / "table8_budget.csv")
MAJOR("03_tables_6_7_8")
''')

md(r"""
---
# 6 · Figures

Ten figures, all written to `figures/` at 160 dpi and pushed. The house palette matches NB01–NB04
so the whole paper reads as one piece of work: **teal for radar (input), red for ECG (output)** —
the two accents mean something rather than decorating.
""")

code(r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from crvs_metrics import bland_altman

S = {"radar": "#0F7C82", "ecg": "#AF3A2C", "muted": "#5C6B71", "ink": "#10171B",
     "grid": "#D3DADB", "amber": "#8A6212", "soft": "#9BB8BA"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.color": S["grid"], "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.fontsize": 7.5})
FIG = WORK / "figures"
def save(f, nm):
    f.savefig(FIG / nm); plt.close(f); print("  wrote", nm)

# --- F1 ours vs published --------------------------------------------------
if T3 is not None and len(T3):
    d = T3.dropna(subset=["CC_t_paper"])
    if len(d):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        for ax, ours, paper, ttl in [(axes[0], "CC_temporal", "CC_t_paper", "temporal correlation"),
                                     (axes[1], "CC_spectral", "CC_s_paper", "spectral correlation")]:
            xs = np.arange(len(d))
            ax.bar(xs - .2, d[ours], .4, color=S["radar"], edgecolor="white", label="our run")
            ax.bar(xs + .2, d[paper], .4, color=S["muted"], edgecolor="white", label="published")
            ax.set_xticks(xs); ax.set_xticklabels(d["variant"], rotation=20, ha="right", fontsize=7)
            ax.set_title(ttl + "  (RVA combined)")
            ax.legend(frameon=False)
        fig.tight_layout(); save(fig, "fig01_ours_vs_published.png")

# --- F2 ablation ladder ----------------------------------------------------
if lad:
    vals = [b[b["variant"] == v]["CC_temporal"].mean() for v in lad]
    cols = [S["ecg"] if v == "L9_full" else S["muted"] if v == "multireslinknet"
            else S["radar"] for v in lad]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.barh(range(len(lad)), vals, color=cols, edgecolor="white")
    ax.set_yticks(range(len(lad))); ax.set_yticklabels([LAB6[v] for v in lad], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(61.86, color=S["ink"], ls="--", lw=1.2, label="published MultiResLinkNet")
    for i, v in enumerate(vals):
        ax.text(v + .3, i, f"{v:.1f}", va="center", fontsize=7)
    ax.set_xlabel("temporal correlation (x100)")
    ax.set_title("Ablation ladder — what each contribution is worth", loc="left")
    ax.legend(frameon=False)
    save(fig, "fig02_ablation.png")

# --- F3/F4 Bland-Altman ----------------------------------------------------
if len(SD):
    for met, unit, fn in [("mean_hr_bpm", "bpm", "fig03_bland_altman_hr.png"),
                          ("rmssd_ms", "ms", "fig04_bland_altman_rmssd.png")]:
        picks = [v for v in ("multireslinknet", "L9_full") if v in set(SD["variant"])]
        if not picks or f"gt_{met}" not in SD.columns:
            continue
        fig, axes = plt.subplots(1, len(picks), figsize=(5.4 * len(picks), 3.6), squeeze=False)
        for ax, v in zip(axes[0], picks):
            d = SD[(SD["variant"] == v) & (SD["experiment"] == CFG["HEADLINE_EXP"])].dropna(
                subset=[f"gt_{met}", f"pr_{met}"])
            if not len(d):
                continue
            ba = bland_altman(d[f"pr_{met}"].to_numpy(), d[f"gt_{met}"].to_numpy())
            c = S["ecg"] if v == "L9_full" else S["muted"]
            ax.scatter(ba["mean"], ba["diff"], s=22, color=c, alpha=.75, edgecolor="white", lw=.5)
            ax.axhline(ba["bias"], color=S["ink"], lw=1.2)
            ax.axhline(ba["loa_hi"], color=S["ink"], ls="--", lw=1)
            ax.axhline(ba["loa_lo"], color=S["ink"], ls="--", lw=1)
            ax.axhline(0, color=S["grid"], lw=.8)
            ax.set_title(f"{NICE.get(v, LAB6.get(v, v))}\nbias {ba['bias']:+.2f}  "
                         f"LoA [{ba['loa_lo']:+.1f}, {ba['loa_hi']:+.1f}] {unit}", fontsize=8.5)
            ax.set_xlabel(f"mean of methods ({unit})")
            ax.set_ylabel(f"predicted - true ({unit})")
        fig.suptitle(f"Bland–Altman agreement — {met.replace('_',' ')}", y=1.02,
                     fontsize=10, fontweight="bold")
        fig.tight_layout(); save(fig, fn)

# --- F5 per-subject box plots ---------------------------------------------
if len(WD):
    picks = [v for v in ORDER if v in set(WD["variant"])][:6]
    d = WD[(WD["variant"].isin(picks)) & (WD["experiment"] == CFG["HEADLINE_EXP"])]
    if len(d):
        fig, ax = plt.subplots(figsize=(11, 3.8))
        data = [d[d["variant"] == v]["CC_temporal"].dropna().to_numpy() for v in picks]
        bp = ax.boxplot(data, labels=[NICE.get(v, LAB6.get(v, v)) for v in picks],
                        patch_artist=True, showfliers=False, widths=.6)
        for patch, v in zip(bp["boxes"], picks):
            patch.set_facecolor(S["ecg"] if v == "L9_full" else S["radar"])
            patch.set_alpha(.75); patch.set_edgecolor("white")
        for m in bp["medians"]:
            m.set_color(S["ink"]); m.set_linewidth(1.4)
        ax.set_ylabel("temporal correlation per window (x100)")
        ax.set_title("Distribution across held-out windows — means hide the tail", loc="left")
        ax.tick_params(axis="x", rotation=14, labelsize=7)
        save(fig, "fig05_distribution.png")

    dsub = d.groupby(["variant", "subject"])["CC_temporal"].mean().reset_index()
    if len(dsub):
        fig, ax = plt.subplots(figsize=(11, 3.6))
        for k, v in enumerate(picks):
            s = dsub[dsub["variant"] == v]
            ax.scatter(np.full(len(s), k) + np.random.uniform(-.14, .14, len(s)),
                       s["CC_temporal"], s=26,
                       color=S["ecg"] if v == "L9_full" else S["radar"],
                       alpha=.8, edgecolor="white", lw=.5)
        ax.set_xticks(range(len(picks)))
        ax.set_xticklabels([NICE.get(v, LAB6.get(v, v)) for v in picks], rotation=14, fontsize=7)
        ax.set_ylabel("per-subject mean CC_temporal (x100)")
        ax.set_title("Per-subject performance — one dot per held-out subject", loc="left")
        save(fig, "fig06_per_subject.png")

# --- F7 qualitative grid ---------------------------------------------------
sp = {}
for p in sorted((RUNS / "runs").glob("*/preds_sample.npz")):
    nm = p.parent.name
    for v in ORDER:
        if f"__{v}__" in nm and nm.startswith(CFG["HEADLINE_EXP"]) and v not in sp:
            sp[v] = p
picks = [v for v in ("multireslinknet", "L6_no_ssm", "L9_full") if v in sp]
if picks:
    z0 = np.load(sp[picks[0]])
    k = min(5, len(z0["y"]) - 1); tt = np.arange(1024) / 128.0
    fig, axes = plt.subplots(len(picks) + 1, 1, figsize=(11, 1.9 * (len(picks) + 1)), sharex=True)
    axes[0].plot(tt, z0["y"][k], lw=1.2, color=S["ink"])
    axes[0].set_ylabel("ground truth", rotation=0, ha="right", va="center", fontsize=8)
    for ax, v in zip(axes[1:], picks):
        z = np.load(sp[v]); kk = min(k, len(z["p"]) - 1)
        ax.plot(tt, z["p"][kk], lw=1.2,
                color=S["ecg"] if v == "L9_full" else S["muted"])
        ax.set_ylabel(NICE.get(v, LAB6.get(v, v)), rotation=0, ha="right", va="center", fontsize=7.5)
    for a in axes:
        a.tick_params(labelleft=False)
    axes[-1].set_xlabel("seconds")
    axes[0].set_title("Held-out reconstruction — the question is whether the QRS stays sharp",
                      loc="left")
    fig.tight_layout(); save(fig, "fig07_qualitative.png")

# --- F8 budget scatter -----------------------------------------------------
if "params" in R.columns and R["params"].notna().any():
    d = (R[R["experiment"] == CFG["HEADLINE_EXP"]]
         .groupby("variant").agg(p=("params", "mean"), c=("CC_temporal", "mean")).dropna())
    if len(d):
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        for v, r in d.iterrows():
            col = S["ecg"] if v == "L9_full" else S["muted"] if v in NICE else S["radar"]
            ax.scatter(r["p"] / 1e6, r["c"], s=110, color=col, edgecolor="white", lw=1, zorder=3)
            ax.annotate(NICE.get(v, LAB6.get(v, v)).split(".")[-1].strip(),
                        (r["p"] / 1e6, r["c"]), fontsize=6.8, xytext=(5, 4),
                        textcoords="offset points")
        ax.axhline(61.86, color=S["ink"], ls="--", lw=1, label="published MultiResLinkNet")
        ax.set_xlabel("parameters (millions)"); ax.set_ylabel("CC_temporal (x100)")
        ax.set_title("Accuracy against model size — smaller and better is the claim", loc="left")
        ax.legend(frameon=False)
        save(fig, "fig08_budget.png")
MAJOR("04_figures")
''')

md(r"""
---
# 7 · Experiment E — robustness *(optional, needs GPU)*

The baseline never tests robustness. radarODE-MTL set the precedent that it matters, and a reviewer
will ask: what happens when the radar signal is noisier than a clinical recording room?

We test white noise across SNRs, low-frequency motion drift, and every single-channel dropout, then
retain all waveform metrics for every condition. No retraining — this measures how gracefully
each model degrades and which physics channel it relies on. Set
`CFG["RUN_ROBUSTNESS"] = False` to skip.
""")

code(r'''
if not CFG["RUN_ROBUSTNESS"]:
    print("robustness skipped (CFG['RUN_ROBUSTNESS'] = False)")
    ROB = pd.DataFrame()
else:
    import torch
    from crvs_data import WindowDataset, FS
    from crvs_models import build_baseline
    from crvs_cmnet import build_cmnet
    from crvs_metrics import seg_metrics
    from crvs_engine import pick_device, seed_all

    dev, ngpu, _ = pick_device()
    print("device:", dev, "| gpus:", ngpu)
    if dev.type != "cuda":
        print("  (CPU -- this will be slow; consider setting RUN_ROBUSTNESS=False)")

    DATA = None
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for candidate in input_root.rglob("windows.parquet"):
            if (candidate.parent / "recordings").exists() and (candidate.parent / "norm_stats.json").exists():
                DATA = candidate.parent; break
    if DATA is None:
        DATA = SCRATCH / "corpus"
        snapshot_download(CFG["DATA_REPO"], repo_type="dataset", token=HF_TOKEN,
                          local_dir=str(DATA),
                          allow_patterns=["recordings/*.npy", "recordings/*.json",
                                          "recordings/*.npz", "windows.parquet", "norm_stats.json",
                                          "experiments.json"], max_workers=4)
    else:
        print("using attached Kaggle NB02 output:", DATA)
    Wn = pd.read_parquet(DATA / "windows.parquet")
    NORM = json.loads((DATA / "norm_stats.json").read_text())
    EXPINFO = json.loads((DATA / "experiments.json").read_text())
    EXPERIMENTS = EXPINFO["experiments"]
    REC_DIR = DATA / "recordings"

    def load_run(run_dir):
        s = json.loads((run_dir / "summary.json").read_text())
        rc_path = run_dir / "run_config.json"
        rc = json.loads(rc_path.read_text()) if rc_path.exists() else {}
        v = norm_variant(s)
        spec = s.get("spec", {})
        ch = spec.get("channels") or s.get("channels") or ["dy"]
        if spec.get("kind") == "cmnet" or (v or "").startswith("L") and spec.get("kind") != "baseline":
            m = build_cmnet(in_ch=len(ch), base=rc.get("base", 32),
                            levels=rc.get("levels", 4), d_ssm=rc.get("d_ssm", 256),
                            ssm_blocks=rc.get("ssm_blocks", 3), d_state=rc.get("d_state", 64),
                            bottleneck=spec.get("bottleneck", "ssm"),
                            use_wavelet=spec.get("wavelet", True),
                            multitask=spec.get("multitask", True),
                            use_film=spec.get("film", True), dropout=rc.get("dropout", 0.1))
        else:
            m = build_baseline(spec.get("model", s.get("model", "multireslinknet")),
                               in_ch=len(ch), out_ch=1, base=rc.get("base", 64),
                               levels=rc.get("levels", 4))
        sd = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["model"]
        m.load_state_dict(sd, strict=True)
        return m.to(dev).eval(), ch, s, v

    rob_path = WORK / "tables" / "table9_robustness.csv"
    rob = pd.read_csv(rob_path).to_dict("records") if rob_path.exists() else []
    finished = {(str(r["variant"]), str(r["corruption"]), f"{float(r.get('level', 0)):g}",
                 "" if pd.isna(r.get("channel", "")) else str(r.get("channel", ""))) for r in rob}
    seed_all(CFG["SEED"])
    for v in CFG["ROBUST_MODELS"]:
        cand = []
        for root in RUN_ROOTS:
            cand.extend(p for p in (root / "runs").glob(f"{CFG['HEADLINE_EXP']}__{v}__f*")
                        if (p / "best.pt").exists() and (p / "summary.json").exists())
        if not cand:
            print(f"  no checkpoint for {v} -- skipped"); continue
        rd = sorted(cand)[0]
        try:
            model, ch, s, vv = load_run(rd)
        except Exception as e:
            print(f"  could not load {rd.name}: {type(e).__name__}: {e}"); continue
        fold = s["fold"]
        sub = Wn[Wn["scenario_canon"].isin(EXPERIMENTS[CFG["HEADLINE_EXP"]])]
        te = sub[(sub["fold_group"] == fold % 5) & sub["no_overlap"]]
        te = te.iloc[:CFG["ROBUST_MAX_WINDOWS"]]
        norm = NORM[f"{CFG['HEADLINE_EXP']}|{fold}"]
        idx = [EXPINFO["channels"].index(c) for c in ch]
        sn = {"mean": [norm["mean"][i] for i in idx], "std": [norm["std"][i] for i in idx]}
        ds = WindowDataset(REC_DIR, te, sn, ch, augment=False)
        print(f"\n  {v}: {len(ds)} test windows, {len(ch)} channel(s)")
        cases = [("clean", 0, "")] + [("awgn", x, "") for x in CFG["SNR_DB"]]
        cases += [("motion_drift", x, "") for x in CFG["MOTION_AMPLITUDE"]]
        if CFG["TEST_CHANNEL_DROPOUT"]:
            cases += [("channel_dropout", 1, c) for c in ch]
        for case_i, (corruption, level, channel) in enumerate(cases):
            key = (v, corruption, f"{float(level):g}", str(channel))
            if key in finished:
                print(f"    {corruption} {level} {channel} restored"); continue
            seed_all(CFG["SEED"] + 1000 * CFG["ROBUST_MODELS"].index(v) + case_i)
            mets = []
            with torch.no_grad():
                for i in range(0, len(ds), 32):
                    xb, yb = [], []
                    for j in range(i, min(i + 32, len(ds))):
                        x, y, _, _ = ds[j]; xb.append(x); yb.append(y)
                    X = torch.stack(xb).to(dev); Y = torch.stack(yb)
                    if corruption == "awgn":
                        p_sig = X.pow(2).mean(dim=(1, 2), keepdim=True)
                        X = X + torch.randn_like(X) * (p_sig / (10 ** (float(level)/10))).sqrt()
                    elif corruption == "motion_drift":
                        tt = torch.arange(X.shape[-1], device=dev) / FS
                        drift = float(level) * torch.sin(2*math.pi*0.30*tt)[None, None, :]
                        X = X + drift
                    elif corruption == "channel_dropout":
                        X[:, ch.index(channel), :] = 0
                    out = model(X)["wave"].float().cpu().numpy()[:, 0]
                    Yn = Y.numpy()[:, 0]
                    for a, bb in zip(Yn, out):
                        mets.append(seg_metrics(a, bb, FS))
            row = {"variant": v, "corruption": corruption, "level": level,
                   "channel": channel, "n": len(mets)}
            for metric in mets[0] if mets else []:
                vals = np.asarray([m[metric] for m in mets], float)
                row[metric] = float(np.nanmean(vals)); row[metric+"_std"] = float(np.nanstd(vals))
            rob.append(row); finished.add(key)
            pd.DataFrame(rob).to_csv(rob_path, index=False)
            sync.mark_dirty(f"robustness:{v}:{corruption}:{level}:{channel}")
            print(f"    {corruption:<16} {str(level):>5} {channel:<8} "
                  f"CC_t {row.get('CC_temporal', float('nan')):6.2f}")
        del model
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    ROB = pd.DataFrame(rob)
    if len(ROB):
        ROB.to_csv(WORK / "tables" / "table9_robustness.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        for v in ROB["variant"].unique():
            d = ROB[(ROB["variant"] == v) & (ROB["corruption"] == "awgn")].copy()
            d["level"] = pd.to_numeric(d["level"]); d = d.sort_values("level", ascending=False)
            ax.plot(d["level"], d["CC_temporal"], "o-", lw=1.6, ms=5,
                    color=S["ecg"] if v == "L9_full" else S["muted"],
                    label=NICE.get(v, LAB6.get(v, v)))
        ax.invert_xaxis()
        ax.set_xlabel("input SNR (dB) — noisier to the right")
        ax.set_ylabel("CC_temporal (x100)")
        ax.set_title("Experiment E — graceful degradation under input noise", loc="left")
        ax.legend(frameon=False)
        save(fig, "fig09_robustness.png")
MAJOR("05_robustness")
''')

md(r"""
---
# 8 · Manuscript-ready summary and final push

A single `RESULTS.md` collecting every table, so drafting Section 6 of the paper is transcription
rather than archaeology.
""")

code(r'''
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
L = []
A = L.append
A(f"# CardioMamba-Net — results\n\nGenerated {now} from `{CFG['BASELINE_REPO']}` and `{CFG['MODEL_REPO']}`.\n")
A(f"- runs analysed: **{len(R)}**")
A(f"- experiments: {sorted(R['experiment'].unique())}")
A(f"- variants: {sorted(R['variant'].unique())}\n")

if T3 is not None and len(T3):
    A("## Table 3 — RVA combined (headline)\n")
    A(T3.round(5).to_markdown(index=False)); A("")
if T2 is not None and len(T2):
    A("## Table 2 — per scenario\n"); A(T2.round(5).to_markdown(index=False)); A("")
if TC is not None and len(TC):
    A("## Table 3b — all five scenarios (new)\n"); A(TC.round(5).to_markdown(index=False)); A("")
if TD is not None and len(TD):
    A("## Table 3c — leave-one-subject-out\n"); A(TD.round(5).to_markdown(index=False)); A("")
if TF is not None and len(TF):
    A("## Table 3d — held-out scenario\n"); A(TF.round(5).to_markdown(index=False)); A("")
try:
    A("## Table 4 — R-peak detection\n"); A(T4.to_markdown()); A("")
    A("## Table 5 — HR and HRV (real ms)\n"); A(T5.to_markdown(index=False)); A("")
except Exception:
    pass
try:
    A("## Table 6 — ablation ladder\n"); A(T6.round(5).to_markdown()); A("")
    A("## Table 7 — subject-paired Wilcoxon, Holm-corrected\n"); A(T7.to_markdown(index=False)); A("")
    A("## Table 8 — budget\n"); A(T8[["M_params","CC_temporal","CC_per_Mparam"]].to_markdown()); A("")
except Exception:
    pass
if CFG["RUN_ROBUSTNESS"] and len(ROB):
    A("## Table 9 — robustness\n"); A(ROB.to_markdown(index=False)); A("")

A("## Figures\n")
for p in sorted(FIG.glob("*.png")):
    A(f"- `figures/{p.name}`")
A("\n## Reading notes for the manuscript\n")
A("- Our splits are strictly subject-wise with non-overlapping test windows. The baseline's "
  "Table 1 counts carry the 50 % overlap and are split 80/20, which permits overlapping windows "
  "across train and test. Baseline rows landing below their published values is the expected "
  "consequence of removing that, not a weaker implementation.")
A("- Correlations are reported x100 throughout, matching the baseline's tables.")
A("- mu_RR is in genuine milliseconds. The baseline's Table 5 reports 126 ms alongside 62 bpm, "
  "which is arithmetically impossible; 126 samples at 128 Hz is 0.98 s.")
A("- Significance uses subject-paired Wilcoxon tests for predeclared full-model comparisons "
  "with Holm correction.")
(WORK / "RESULTS.md").write_text("\n".join(L))

sizes = {str(p.relative_to(WORK)): p.stat().st_size for p in WORK.rglob("*") if p.is_file()}
ok = sync.flush(final=True, msg=f"{CFG['RUN_ID']} — evaluation complete, {len(R)} runs")
print("\n" + "=" * 76)
print("  EVALUATION COMPLETE" if ok else "  COMPLETE (final push had a problem)")
print("=" * 76)
print(f"  repo    : {sync.url}")
print(f"  runs    : {len(R)}")
print(f"  tables  : {len(list((WORK/'tables').glob('*.csv')))}")
print(f"  figures : {len(list(FIG.glob('*.png')))}")
print(f"  payload : {sum(sizes.values())/2**20:.1f} MB")
print("=" * 76)
print("\n  RESULTS.md holds every table in markdown, ready to paste into the manuscript.")
''')

md(r"""
---
# 9 · Troubleshooting

**`No runs found`** — NB03 and NB04 push to separate `BASELINE_REPO` and `MODEL_REPO`.
Check both names and that at least one run finished.

**Table 7 skipped** — expected while quick runs are incomplete. Run NB03/NB04 with
`QUICK = False` so every held-out subject has paired results.

**Bland–Altman plots empty** — the per-subject metrics come from `metrics_subjects.parquet`, which
is only written when a test split has at least two windows per subject. Re-run with the full folds.

**Robustness section slow or out of memory** — lower `CFG["ROBUST_MAX_WINDOWS"]`, or set
`CFG["RUN_ROBUSTNESS"] = False` and run it in its own session.

**A checkpoint fails to load** — architecture config drifted between training and evaluation. The
loader uses `strict=True` and stops on any mismatch. If you changed `CFG["BASE"]` or `D_SSM`
after training, restore the recorded configuration rather than forcing a partial load.
""")

out = sys.argv[1] if len(sys.argv) > 1 else "05_evaluate_and_figures.ipynb"
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
