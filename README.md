# Radar -> ECG : Contactless ECG Reconstruction from 24 GHz CW Radar

Research project aiming to beat **MultiResLinkNet** (Chowdhury et al., *Computers in Biology and
Medicine* 176:108555, 2024) on the Schellenberger **CR-RVS** dataset with a new architecture.

**Final experimental status (2026-09-07):** all 80 baseline and 100 CardioMamba canonical runs
were completed and NB05 generated its full evaluation package. The experiment is scientifically
useful, but the original full-model/SOTA hypothesis was not confirmed: full L9 improves strongly
over the strict baseline rerun, while no-FiLM and single-task variants obtain the best correlation.

**Start with the final evidence:** [`06_results/RESULTS_AND_INSIGHTS.md`](06_results/RESULTS_AND_INSIGHTS.md)
and [`07_paper/PAPER_DIRECTION.md`](07_paper/PAPER_DIRECTION.md).

The pinned Hugging Face download and integrity manifest are documented in
[`06_results/hf_final_snapshot/README.md`](06_results/hf_final_snapshot/README.md).

The original plan and audit trail remain in [`00_admin/PLAN.md`](00_admin/PLAN.md).

**Complete manuscript:** [`07_paper/README.md`](07_paper/README.md) links the LaTeX, compiled PDF,
and source package. A deeper manuscript audit found a 125-Hz stream labelled 128 Hz; physiological
reevaluation is needed before submission. See [`07_paper/MANUSCRIPT_AUDIT.md`](07_paper/MANUSCRIPT_AUDIT.md).

**Run the rewritten Kaggle workflow:** [`03_notebooks/README.md`](03_notebooks/README.md).
The five-stage v2 workflow plus its one-time baseline cleanup notebook use chained Kaggle Notebook
Outputs, dual T4 where training is required, atomic mid-epoch recovery, conservative Hugging Face
uploads, and comprehensive epoch telemetry.
The file-by-file audit and v1 failure analysis are in
[`00_admin/REPOSITORY_AUDIT_V2.md`](00_admin/REPOSITORY_AUDIT_V2.md).

## Folder structure

```
.
├── 00_admin/                  plan, decisions, meeting/working notes
│   └── PLAN.md                <- the full research plan
├── 01_literature/
│   ├── papers/                source PDFs (renamed to year_author_topic)
│   ├── extracted_text/        machine-readable text of each PDF
│   └── notes/                 per-paper notes
│       └── baseline_results_MultiResLinkNet.md   <- every number we must beat
├── 02_data/
│   ├── raw/                   .mat files from Kaggle/figshare  (NOT versioned)
│   ├── interim/               per-subject parsed + resampled arrays
│   └── processed/             windowed train-ready shards / HDF5
├── 03_notebooks/              Kaggle-ready .ipynb, numbered in run order
├── 04_src/
│   ├── data/                  loaders, windowing, augmentation
│   ├── models/                baselines + CardioMamba-Net
│   ├── losses/                composite loss components
│   ├── training/              train loop, resumable HF checkpointing
│   ├── evaluation/            metrics, peak detection, statistics
│   └── utils/                 config, seeding, logging
├── 05_experiments/
│   ├── configs/               one YAML per run
│   ├── logs/                  history.jsonl per run
│   └── checkpoints/           local mirror of the HF checkpoints
├── 06_results/
│   ├── tables/                CSV/markdown results tables
│   ├── figures/               publication figures
│   └── metrics/               raw per-fold metric dumps
└── 07_paper/
    ├── draft/                 manuscript
    ├── figures/               final figure files
    └── references/            .bib
```

## Key references

- **Baseline** — F. A. Chowdhury et al., "ECG waveform generation from radar signals: A deep
  learning perspective," *Comput. Biol. Med.* 176:108555, 2024. doi:10.1016/j.compbiomed.2024.108555
- **Dataset** — S. Schellenberger et al., "A dataset of clinically recorded radar vital signs with
  synchronised reference sensor signals," *Sci. Data* 7:291, 2020. doi:10.1038/s41597-020-00629-5
  (data: doi:10.6084/m9.figshare.12186516)
- Kaggle mirror used for fast download: `pedababugaddala/datasets-file`

## Conventions

- Sampling rate after preprocessing: **128 Hz**. Window: **1024 samples (8 s)**, 50 % overlap on train only.
- V2 data, checkpoints and results live in five stage-specific Hugging Face repositories under
  `Shanmuk4622/`; exact names and run instructions are in `03_notebooks/README.md`.
- `02_data/raw/` and `05_experiments/checkpoints/` should never be committed to git.
