# CR-RVS dataset — verified facts

Confirmed by running NB01 against the Kaggle mirror `pedababugaddala/datasets-file`
on 2026-09-01. **These override the dataset paper where they disagree.**

## Verdict: `RAW_MAT_TREE` — proceed

| | |
|---|---|
| Files | 135 `.mat`, 5.597 GB, zero derivative files |
| Subject folders | 30 (`GDN0001` … `GDN0030`), 1–5 files each |
| Layout | `<root>/pedababugaddala/datasets-file/datasets_subject_01_to_10_scidata/GDN0004/GDN0004_3_Apnea.mat` |
| Max depth | 4 |
| Mount | `/kaggle/input/datasets`, added as a Kaggle Input (instant, no disk cost) |

135 = 30 Resting + 27 Valsalva + 24 Apnea + 27 TiltUp + 27 TiltDown, which matches the
subject counts the baseline paper reports for the three scenarios it used.

## ⚠️ The variable names are LOWERCASE

The dataset paper documents `radar_I` / `radar_Q`. The files actually contain
**`radar_i` / `radar_q`**. An exact-match lookup returns `None` and a census will run to
completion having recorded **zero radar samples** — silently. Always match case-insensitively.

## Full schema (17 variables, MATLAB v5)

| variable | type | notes |
|---|---|---|
| `radar_i`, `radar_q` | float64 vector | **lowercase**; 2000 Hz |
| `tfm_ecg1`, `tfm_ecg2` | float64 vector | 2000 Hz — the reconstruction target |
| `tfm_icg` | float64 vector | 1000 Hz |
| `tfm_bp` | float64 vector | 200 Hz |
| `tfm_z0` | float64 vector | 100 Hz |
| `tfm_intervention` | float64 vector | 2000 Hz |
| `fs_radar`, `fs_ecg`, `fs_intervention` | scalar | 2000.0 |
| `fs_icg` | scalar | 1000.0 |
| `fs_bp` | scalar | 200.0 |
| `fs_z0` | scalar | 100.0 |
| `measurement_info` | object array [3] | `[timestamp, scenario, subject_id]` — parses cleanly |
| `tfm_param`, `tfm_param_time` | array | **empty** in the files checked |

**Every channel has its own sampling rate.** Assuming one global rate corrupts every derived
duration. Example (GDN0001 Resting, 607.6 s): radar 1,215,200 · ecg1 1,215,200 · icg 607,600 ·
bp 121,520 · z0 60,760 — all consistent at 607.6 s.

## Scenario naming

`Resting`, `Valsalva`, `Apnea`, `TiltUp`, `TiltDown` — in the filename suffix.

**The scenario index is not stable across subjects.** GDN0001 has no Apnea, so its `_3_` is
TiltUp; GDN0004's `_3_` is Apnea. Never parse the number — only the trailing name.
Subject ID comes from the `GDN\d+` folder name (also present in `measurement_info`).

## Implications for NB02

- Case-insensitive variable access, always.
- Resample per channel using that channel's own rate.
- Derive subject from the path, scenario from the filename suffix, cross-checked against
  `measurement_info`.
- `tfm_param` is empty — do not plan on the TFM-derived parameters.
- Files are ~40 MB each on average; 135 files at once is fine on Kaggle, one at a time in memory.


---

# Historical processed corpus (NB02 v1 output)

Public: **`Shanmuk4622/cr-rvs-radar-ecg-processed`**

| | |
|---|---|
| Recordings kept | 76 of 135 |
| Subjects | 27 of 30 |
| Windows | 12,425 (6,233 non-overlapping) |
| Normalisation sets | 25 (5 experiments x 5 folds) |
| Storage format | compressed `.npz` (a `.npy` rebuild is available and 155x faster to read) |

| Experiment | Windows | Subjects | Paper's subjects |
|---|---|---|---|
| A_resting | 3,407 | 22 | 30 |
| A_valsalva | 4,727 | 19 | 27 |
| A_apnea | 548 | 10 | 24 |
| B_rva | 8,682 | 25 | — |
| C_all5 | 12,425 | 27 | not evaluated by the baseline |

**The Apnea shortfall is the number to watch.** 10 subjects against the paper's 24. Apnea is
breath-holding, so the chest is deliberately still and the beat-coupling ratio is *expected* to
fall for physiological rather than quality reasons — which means `MIN_BEAT_COUPLING = 1.30` is
probably biased against exactly the scenario where the baseline is weakest. See
`00_admin/DECISIONS.md` for the resolution.

## V2 rebuild (2026-09-02)

The clean-run destination is **`Shanmuk4622/cr-rvs-radar-ecg-processed-v2`**. It rebuilds
uncompressed mmap-ready `.npy` arrays, does not exclude on beat coupling, and adds normalisation
sets for five-fold experiments, subject-level LOSO, and held-out-scenario testing. Counts are
intentionally left pending until NB02 v2 is run on Kaggle; no v1 number is copied forward as a v2
result.
