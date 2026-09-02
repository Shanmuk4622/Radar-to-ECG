# Kaggle notebooks — v2 clean run

Run these notebooks in order. They are self-contained, Kaggle-compatible, output-conscious, and
start in smoke-test mode where training is involved. The earlier v1 run remains useful evidence,
but v2 uses new Hugging Face repositories and does not treat any v1 model as complete.

| # | Notebook | Purpose | Accelerator | Typical runtime |
|---|---|---|---|---|
| 01 | `01_verify_and_download.ipynb` | Verify raw mirror; census; 128 Hz derivative | CPU | 25–70 min |
| 02 | `02_preprocess_to_hf.ipynb` | Targets, quality audit, folds, LOSO/cross-scenario norms | CPU | 25–75 min |
| 03 | `03_baselines.ipynb` | Four published baselines and reproduction gate | **T4 x2** | quick 20–45 min; full queue spans sessions |
| 04 | `04_cardiomamba_train.ipynb` | C1–C5, ablations, experiments A–F | **T4 x2** | quick 30–75 min; full queue spans sessions |
| 05 | `05_evaluate_and_figures.ipynb` | Tables, subject-paired statistics, figures, robustness | CPU or **T4 x2** | 10–45 min; +20–90 min robustness |

Every notebook contains its own cell-by-cell description and time estimate.

## Inputs to attach

1. NB01: add Kaggle dataset `pedababugaddala/datasets-file`.
2. After NB01 completes, **Save Version**. Attach that Notebook Output to NB02.
3. After NB02 completes, **Save Version**. Attach that Notebook Output to NB03, NB04, and NB05.
4. Attach the Kaggle secret `HF_TOKEN` with write access to all five notebooks.

The notebook-output chain is the primary, fast path. Hugging Face is the durable backup and
automatic recovery path.

## Hugging Face v2 repositories

| Repository | Produced by | Contents |
|---|---|---|
| `Shanmuk4622/cr-rvs-radar-ecg-inventory-v2` | NB01 | inventory, previews, 128 Hz corpus, figures |
| `Shanmuk4622/cr-rvs-radar-ecg-processed-v2` | NB02 | mmap-ready recordings, targets, indexes, all normalisation sets |
| `Shanmuk4622/cardiomamba-baselines-v2` | NB03 | baseline checkpoints, metrics and logs |
| `Shanmuk4622/cardiomamba-net-v2` | NB04 | CardioMamba checkpoints, ablations, A–F results |
| `Shanmuk4622/cardiomamba-results-v2` | NB05 | final tables, statistics, figures, report |

All are configured public. Change `HF_PRIVATE` before the first run if the dataset license or
release plan requires private storage.

## Recovery contract

- Local checkpoints are atomic and include model, optimizer, scheduler, AMP scaler, Python/NumPy/
  Torch/CUDA random states, early-stopping counter, active epoch, batch cursor and partial epoch
  aggregates.
- Training data order and augmentation are deterministic functions of seed, epoch and sample.
- State is saved every 50 optimizer steps or five minutes, at every epoch, and at every run end.
- A normal Stop/SIGINT/SIGTERM first creates an emergency checkpoint and then performs a blocking
  HF upload. A hard machine loss cannot run cleanup, so the maximum remote recovery gap is the last
  successful scheduled upload (normally 30 minutes).
- Major notebook stages request an immediate upload. Ordinary best epochs do not; this avoids the
  v1 failure mode that created dozens of commits in one hour.
- The scheduler permits only 24 `upload_folder` calls per hour. This is deliberately far below
  the approximate API ceiling because one folder upload may make multiple HTTP requests/commits.
- On restart, an interrupted run's `state.pt` and `best.pt` are restored before `Trainer.load()`.
  A config-hash mismatch or corrupt checkpoint stops loudly instead of silently restarting.

## Telemetry retained

Each run keeps loss components, train/validation totals, MAE/MSE, temporal and spectral
correlation, temporal/spectral RRMSE, R², waveform peak scores, peak-head precision/recall/F1,
RR-head MAE/RMSE, recording-safe HR/HRV errors, learning rate, AMP scale, gradient norm/clipping,
throughput, data/compute time, GPU memory/utilisation/temperature/power, disk usage, per-batch
JSONL, per-epoch JSONL/CSV, and per-window/per-recording validation Parquet files.

## Training sequence

Keep `QUICK=True` for the first successful run:

- NB03 trains one MultiResLinkNet/RVA fold for 10 epochs.
- NB04 trains one full CardioMamba/RVA fold for 10 epochs; the smoke cells exercise all variants.

Then set `QUICK=False` and Run All in fresh Kaggle sessions. Each queue works for at most 10.5
hours, uploads, exits cleanly, and continues next session.

## Source of truth

Edit the generators and shared libraries in `04_src/utils`, not notebook JSON by hand. Rebuild:

```powershell
python 04_src/utils/build_nb01.py 03_notebooks/01_verify_and_download.ipynb
python 04_src/utils/build_nb02.py 03_notebooks/02_preprocess_to_hf.ipynb
python 04_src/utils/build_nb03.py 03_notebooks/03_baselines.ipynb
python 04_src/utils/build_nb04.py 03_notebooks/04_cardiomamba_train.ipynb
python 04_src/utils/build_nb05.py 03_notebooks/05_evaluate_and_figures.ipynb
```

Every builder parses each generated code cell and round-trips the notebook JSON.

## Historical v1 evidence

The 2026-09-01 NB01/NB02 run found 135 raw MATLAB files (30 subjects) and produced 76 retained
recordings, 12,425 windows and 25 original fold-normalisation sets. The compressed v1 arrays were
measured much slower per window than mmap-ready `.npy`. V2 intentionally rebuilds them and also
adds LOSO and cross-scenario normalisation sets. See `00_admin/DECISIONS.md` and
`02_data/DATASET_FACTS.md` for the audit trail.
