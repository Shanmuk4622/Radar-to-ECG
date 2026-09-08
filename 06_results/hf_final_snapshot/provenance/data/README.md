---
license: cc-by-4.0
pretty_name: CR-RVS Radar-to-ECG Training Corpus
tags: [radar, ecg, biosignals, contactless-monitoring, vital-signs]
---

# CR-RVS Radar-to-ECG — training corpus

Windowed, fold-assigned training corpus for **CardioMamba-Net**, derived from the CR-RVS dataset
(Schellenberger et al., *Sci Data* 7:291, 2020) via `01_verify_and_download` and
`02_preprocess_to_hf`. Built 2026-09-02 13:11 UTC, run `nb02_corpus_v2`.

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
- Splits are **always by subject**. Fold f: test = group f, val = group (f+1) mod 5, train = rest.
- Normalisation statistics are per fold and computed on **training windows only**.
- ECG target is [-1, 1], not [0, 1] as in the baseline; this is a declared deviation.

## Quality gate

Recordings are excluded only when ECG/radar inputs are unreadable or missing, contain invalid
values, or are under 60 s. Beat coupling is retained as a continuous
difficulty covariate and is not used to select an easier cohort. Every exclusion and its reason is
in `inventory_gated.csv`.

## Cite

Schellenberger et al., *Scientific Data* 7:291 (2020), doi:10.1038/s41597-020-00629-5 ·
Chowdhury et al., *Computers in Biology and Medicine* 176:108555 (2024),
doi:10.1016/j.compbiomed.2024.108555
