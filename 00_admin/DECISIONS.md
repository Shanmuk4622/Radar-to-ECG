# Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-01 | **Architecture tier: Recommended — full CardioMamba-Net (C1–C5).** C6 (diffusion + uncertainty) is parked as an optional stretch, revisited only after Stage 1 clears the targets. | Q1-journal novelty, and the contributions stack in stages so there is a publishable result even if the last stage disappoints. |
| 2026-09-01 | **Reimplement all four baselines** (FPN-1D, UNet-1D, LinkNet-1D, MultiResLinkNet) on identical data, folds and seeds. Report both "as published" and "our run". | Reviewer expectation, and it doubles as the correctness gate: failure to reproduce their Table 2 means our pipeline has a bug. |
| 2026-09-01 | Preprocessing frozen to the baseline's (128 Hz, 1024-sample windows, 50 % train overlap, 0.5–40 Hz, 50 Hz notch, order-5 polynomial detrend). One declared deviation: ECG target normalised to [-1,1]. | Any gain must be attributable to the model, not to a better pipeline. |
| 2026-09-01 | Working model name **CardioMamba-Net**; final name to be picked before submission. | Placeholder. |

## Open items
- [ ] Step 0: verify the Kaggle mirror `pedababugaddala/datasets-file` really contains the raw per-subject `.mat` tree.
- [ ] Confirm HF repo names: `Shanmuk4622/cr-rvs-radar-ecg-processed` (dataset), `Shanmuk4622/cardiomamba-net` (checkpoints).
- [ ] Decide whether Experiment C (all five scenarios) uses one model or per-scenario fine-tuning.

## Notebook 01 — build notes (2026-09-01)

- Built and validated: 35 cells (12 markdown, 23 code). Every code cell `ast.parse`d; the embedded
  `hf_sync.py` source compiled; all four DSP unit tests executed and passed outside Kaggle.
- **Bug caught during validation:** the first `fit_ellipse` used anisotropic normalisation and the
  usual closed-form conic axis formulas, which mis-paired the semi-axes with the rotation angle and
  returned `None` on a perfect ellipse. Replaced with an SVD algebraic fit plus a quadratic-form
  eigendecomposition, which pairs axes to angle correctly by construction.
- **Second bug:** I/Q correction was a scalar gain hack (`Q *= a/b`). Replaced with a proper
  translate → de-rotate → per-axis-normalise map (`iq_correct`). On a synthetic receiver with 1.61x
  gain imbalance and 0.42 rad rotation this recovers the true displacement exactly, versus 0.475 mm
  peak error uncorrected. This matters: the baseline paper cites "ellipse fitting" but never
  publishes the method, and cardiac chest motion is sub-millimetre — 0.475 mm of demodulation error
  is larger than the signal we are trying to recover.
- HF repos for the project (all **public**): `cr-rvs-radar-ecg-inventory` (NB01),
  `cr-rvs-radar-ecg-processed` (NB02), `cardiomamba-net` (NB03–04).

## NB01 first real run — 2026-09-01

**Gate result: `RAW_MAT_TREE`, proceed = True.** The Kaggle mirror is the genuine
per-subject `.mat` tree: 135 files, 5.597 GB, 30 subject folders, zero derivative files.
`PLAN.md` holds unchanged. Details in `02_data/DATASET_FACTS.md`.

The HF sync layer worked on the first run — repo created public, resume pull succeeded,
stage-boundary pushes fired for `00_setup`, `01_verdict` and `02_schema`.

### Four bugs found and fixed

1. **Lowercase `radar_i` / `radar_q`** (blocking, and the dangerous one). The dataset paper
   documents `radar_I` / `radar_Q`; the files use lowercase. The exact-match lookup returned
   `None`, so `n_radar` was 0 and `duration_s` was 0.0 — and *the census would have completed
   normally*, producing an inventory of zeros. Fixed with case-insensitive lookup plus a hard
   guard that aborts if I, Q or ECG are missing from the probe file. A silent wrong answer is
   worse than a crash.
2. **Ellipse fit returned `None` on a perfect ellipse.** Anisotropic normalisation plus the
   standard closed-form conic axis formulas, which mis-pair semi-axes with rotation. Replaced
   with an SVD algebraic fit and a quadratic-form eigendecomposition.
3. **`filtfilt(method="gust")` produced NaN.** At 2000 Hz a 0.8–20 Hz band is 0.0008–0.02
   normalised; in transfer-function form that filter is ill-conditioned and Gustafsson's method
   solves a singular system. Rewritten on second-order sections.
4. **Three bad quality-flag heuristics.** Clipping was detected as "within 0.05 % of the
   extreme", which flags every clean sinusoid — now counts exact repeated extreme values.
   The longest-flat-run computation was wrong. The sync estimator searched ±60 s when the
   published data is already hardware-synchronised, so it found noise.

### Added: `beat_coupling`

Replaced the fragile envelope cross-correlation with a **beat-triggered average** of radar
acceleration around every R peak, scored against ensembles built from random trigger times.
The ratio is calibrated per file: ~1.0 means no coupling, >1.5 means the radar demonstrably
sees the heartbeat. It answers the precondition for the whole project per subject, gives NB02
an evidence-based exclusion criterion, and the ensemble itself is a paper figure
(`fig09_beat_triggered.png`) — the same thing the baseline paper shows once, in a supplementary
figure, without quantifying it.

Verified end to end against a synthetic `.mat` built to the exact schema above: healthy file
0 flags with receiver parameters recovered exactly and coupling 12.3× above chance; a
deliberately clipped-and-flatlined file raised all 5 expected flags.

## NB02–NB05 built — 2026-09-01

Four notebooks added: `02_preprocess_to_hf` (CPU), `03_baselines` (2×T4), `04_cardiomamba_train`
(2×T4), `05_evaluate_and_figures` (2×T4, GPU only needed for the robustness section).

### Architecture of the notebook set

- **One shared library, seven modules** (`crvs_sync`, `crvs_data`, `crvs_metrics`, `crvs_models`,
  `crvs_cmnet`, `crvs_losses`, `crvs_engine`), embedded byte-identically in every notebook and
  verified identical by md5 at build time. No cross-notebook download dependency, one place to fix
  a bug, and every experiment provably sees the same inputs and the same metric code.
- **Run queues, not monolithic scripts.** NB03 has 80 runs (4 models × 4 experiments × 5 folds),
  NB04 has the 10-rung ablation ladder plus the full model on every experiment. Each notebook
  checks Hugging Face for completed runs, works through as many as fit in `TIME_BUDGET_H`, then
  stops cleanly. Re-running in a new session continues. Resumption is per run *and* per epoch.
- **`QUICK = True` by default** in NB03/NB04: one fold, 25 epochs. Proves the whole path in about
  an hour before committing a full session.

### Two protocol decisions, both declared in the paper

1. **Splits are strictly by subject, and validation/test windows do not overlap.** The baseline's
   Table 1 counts carry the 50 % overlap and are then split 80/20, which permits overlapping
   windows across train and test. Our reproduction may therefore land *below* their published
   numbers — that is leakage being removed, not a weaker implementation. The comparison that
   carries the claim is CardioMamba-Net against our own re-run baselines.
2. **Normalisation statistics are per fold, from training windows only.** Corpus-wide statistics
   are invisible leakage.

### Nine bugs found and fixed before any GPU time was spent

Validated by executing the model code, not by reading it. Two were severe:

1. **Skip connections silently discarded** (`MultiResLinkNet1D` and `CardioMambaNet`). The decoder
   at step *k* emerged with `rev[k+1]` channels while `skips[-1-k]` carried `rev[k]`, so the
   `if h.shape[1] == s.shape[1]` guard was never true and the skip was dropped every time. Effect:
   **58.7 % of MultiResLinkNet's parameters and 31.9 % of CardioMamba-Net's received zero
   gradient** in every configuration, while still costing full compute. Both networks were
   effectively plain encoder–decoders — and the ResPath skip is the entire point of MultiResUNet.
   This would have failed NB03's reproduction gate for completely the wrong reason. Fixed by
   targeting `rev[k]`; the silent guard is now a `RuntimeError`. Verified three ways: gradient
   coverage went 0 → 100 %, zeroing the ResPath weights now changes the output, and the decoder
   stopped needing any interpolation to paper over misalignment.
2. **`peak_detection_scores` crashed unconditionally** — `d[used] = np.inf` on the int64 array
   `find_peaks` returns. It runs *after* training in both NB03 and NB04, so a full GPU session
   would have completed and then been lost in post-processing.
3. **Empty train loader reported `train 0.00000`** rather than failing — `drop_last=True` on a
   split smaller than one batch yields zero batches, the optimiser never steps, and a checkpoint
   is written that looks trained and is not. Now raises.
4. **A sticky `done` flag** made raising `EPOCHS` mid-project a silent no-op.
5. **Recordings were `.npz`**, and `np.load(mmap_mode="r")` silently ignores `mmap_mode` on an
   npz archive — so every window decompressed all 11 arrays to slice out 1024 samples: **23.4 ms
   per window**, which would have dominated T4 compute. Now one uncompressed `.npy` of shape
   `(11, n)` plus a JSON sidecar: **0.27 ms, an 85× speed-up.**
6. **Read-only tensors** aliasing the memmap (`np.asarray` on a basic-indexed view does not copy).
7. **`MultiResBlock` generated zero-width stages** below `base = 4`.
8. **Wavelet branch was one octave below the conv branch**, so the "dual-domain encoder" was
   fusing mismatched resolutions and upsampling at every level. Level 0 is now taken at the input
   resolution. This one mattered for the *scientific* claim in C2, not just for correctness.
9. Deprecated `torch.cuda.amp` API, `TransformerBottleneck` max length, `range_normalise` on a
   constant input.

### Budget

CardioMamba-Net at `BASE=32` is **~4.1 M parameters**, inside the < 5 M target. The wider (correct)
decoder moved the crossover down: `BASE >= 40` now exceeds 5 M. Do not raise `BASE` without
re-running NB04's budget cell.

## NB02 complete, NB03 first attempt — 2026-09-01

### NB02 succeeded

`Shanmuk4622/cr-rvs-radar-ecg-processed` is live and public.

| | |
|---|---|
| Recordings kept | **76** of 135 |
| Subjects | **27** of 30 |
| Windows | **12,425** (6,233 non-overlapping) |
| Normalisation sets | 25 (5 experiments x 5 folds) |

Per experiment:

| Experiment | Windows | Subjects |
|---|---|---|
| A_resting | 3,407 | 22 |
| A_valsalva | 4,727 | 19 |
| A_apnea | **548** | **10** |
| B_rva | 8,682 | 25 |
| C_all5 | 12,425 | 27 |

### ⚠️ Open question: the quality gate is aggressive, and not uniformly so

59 of 135 recordings were dropped. That is defensible in itself — every exclusion is recorded
with its reason in `inventory_gated.csv` — but the loss is **not evenly distributed**:

- Apnea fell from 24 subjects to **10**, and from the baseline's 1,140 segments to **548**.
- Resting kept 22 of 30, Valsalva 19 of 27.

Apnea is breath-holding: the chest is deliberately still, so the beat-triggered coupling ratio is
*expected* to be lower there for physiological reasons rather than because the recording is bad.
`MIN_BEAT_COUPLING = 1.30` is therefore likely to be **systematically biased against Apnea** — and
Apnea is precisely the scenario where the baseline performs worst (CC_t 55.3), so a filtered,
easier Apnea subset would make our number look better without being comparable.

**Decision needed before the paper's Table 2 is final.** Options, in order of preference:

1. Make the threshold scenario-aware — a lower bar for Apnea, justified physiologically and stated
   in the Methods.
2. Report Apnea both gated and ungated, and let the gap be part of the result.
3. Lower `MIN_BEAT_COUPLING` globally to ~1.1 and rely on the other flags.

Whatever we choose has to be declared, because a reviewer comparing our Apnea row against theirs
will otherwise be comparing different populations.

### NB03 first attempt failed — two causes, one non-obvious

All 16 queued runs failed at the first batch with
`FileNotFoundError: .../recordings/GDN0002__Resting.npz` inside a DataLoader worker.

**Cause 1 — the corpus is `.npz`.** NB02 was run from the build that predates the `.npy` change,
so the HF corpus holds compressed `.npz` recordings while NB03 downloaded `recordings/*.npy` and
found nothing.

**Cause 2 — a stale module, which is the interesting one.** The traceback quoted a `.npz` path
from *inside* `crvs_data.py`, but the notebook that ran already contained the `.npy` loader. The
notebook writes `crvs_data.py` to disk and then imports it — and **that is not idempotent inside a
kernel**. Python caches the module object in `sys.modules`, so once an older version has been
imported in a session, re-running the cell with new source silently keeps the old code. Every
symptom pointed at the data; the fault was in the import.

### Fixes

- **`_Rec`, a dual-format reader.** `crvs_data` now reads `.npy` (preferred) or `.npz`
  transparently, so the existing corpus trains without a 400 MB re-upload. Verified by execution:
  the two formats return **numerically identical** tensors (max difference exactly 0.0 on all four
  outputs), `compute_norm` agrees to 0.0, single-channel selection works on both, the `.npy` is
  preferred when both are present, and a missing recording now raises a message naming both
  extensions and the directory instead of a bare `[Errno 2]` from a worker process.
- **Measured cost of the legacy format: 11.33 ms vs 0.07 ms per window — 155x.** Worse than the
  85x estimated earlier. Training will work on `.npz`, but re-running NB02 is worth doing.
- **`LIB_VERSION = 3` plus a stale-module purge.** Every notebook now pops the modules from
  `sys.modules`, calls `importlib.invalidate_caches()`, and asserts the version after import,
  printing the file it actually loaded. A stale library is now impossible to run past.
- **Early corpus check.** NB03/NB04 count the recording files right after download, drop windows
  whose file is missing, and fail immediately with a clear message if none arrived — instead of
  surfacing 20 minutes later inside a worker.

### Note on hardware

The session ran on a **single** T4, not two. `DataParallel` handles that transparently; expect
roughly double the wall-clock of the two-GPU estimate.
