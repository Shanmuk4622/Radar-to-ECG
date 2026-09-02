# Repository audit and v2 rewrite — 2026-09-02

## Scope read

Every tracked/source artifact in the workspace was inventoried. Both papers were read from the
extracted text and visually checked across all rendered PDF pages. Generated Python bytecode was
treated as cache, and every `.gitkeep` was treated as an intentional empty-directory marker.

## File map

| Path | Role and current interpretation |
|---|---|
| `README.md` | Project entry point and folder map; now links to the v2 runbook. |
| `00_admin/PLAN.md` | Scientific hypothesis, C1–C5 architecture, targets, experiments and compute plan; recovery/statistics/storage rules updated to the implemented v2 contract. |
| `00_admin/DECISIONS.md` | Historical decisions and v1 run findings; v2 restart, coupling, recovery, HRV and statistical decisions appended without erasing history. |
| `00_admin/notes/.gitkeep` | Reserved working-notes directory. |
| `01_literature/papers/2020_schellenberger_*.pdf` | Primary CR-RVS dataset paper: acquisition hardware, synchronized references, scenarios, subject/data records and signal schema. |
| `01_literature/papers/2024_chowdhury_*.pdf` | Primary baseline paper: preprocessing, 1024/128 Hz windows, four 1-D architectures, MSE training and reported waveform/peak/HRV tables. |
| `01_literature/extracted_text/*.txt` | Searchable text counterparts of the two papers; consistent with the rendered PDFs. |
| `01_literature/notes/baseline_results_MultiResLinkNet.md` | Local transcription of baseline metrics/claims used for reproduction tables and gates. |
| `02_data/DATASET_FACTS.md` | Verified raw-mirror schema and historical v1 corpus counts; v2 destination and pending-rebuild status added. |
| `02_data/raw/.gitkeep` | Placeholder only; raw MATLAB data belongs on Kaggle, not Git. |
| `02_data/interim/.gitkeep` | Placeholder for non-versioned intermediate data. |
| `02_data/processed/.gitkeep` | Placeholder for non-versioned processed data. Durable v2 data goes to Kaggle output/HF. |
| `03_notebooks/README.md` | Rewritten v2 runbook: exact inputs, order, accelerators, time estimates, five HF repos, recovery and telemetry. |
| `03_notebooks/01_verify_and_download.ipynb` | CPU inventory/gate notebook, raw Kaggle Input first, v2 inventory repo. |
| `03_notebooks/02_preprocess_to_hf.ipynb` | CPU corpus builder, attached NB01 output first, mmap `.npy`, unbiased quality gate, 5-fold/LOSO/cross-scenario norms. |
| `03_notebooks/03_baselines.ipynb` | Dual-T4 baseline queue, isolated v2 repo, exact active-run restore, rich epoch metrics. |
| `03_notebooks/04_cardiomamba_train.ipynb` | Dual-T4 C1–C5/ablation queue plus A–F, isolated v2 repo, exact active-run restore. |
| `03_notebooks/05_evaluate_and_figures.ipynb` | Two-source model collection, subject-paired statistics, recording-safe HRV, noise/motion/channel-drop robustness, separate results repo. |
| `04_src/utils/nb_lib_a.py` | Notebook generator helper plus shared sync, data and metric module sources. Sync/data are v2/v4. |
| `04_src/utils/nb_lib_b.py` | Shared baseline/CardioMamba/loss sources and v4 training engine. |
| `04_src/utils/build_nb01.py` … `build_nb05.py` | Source-of-truth notebook generators. They splice shared modules, parse code cells, clear outputs and round-trip notebook JSON. |
| `04_src/utils/__pycache__/*` | Generated local Python cache; not a source of truth and not edited. |
| `04_src/{data,models,losses,training,evaluation}/*.gitkeep` | Reserved future package layout; live reusable code currently remains embedded for Kaggle portability. |
| `05_experiments/{configs,logs,checkpoints}/.gitkeep` | Reserved local mirrors; training truth is in stage-specific HF repos. |
| `06_results/{tables,figures,metrics}/.gitkeep` | Reserved local result mirrors; NB05 v2 publishes to its results repo. |
| `07_paper/{draft,figures,references}/.gitkeep` | Reserved manuscript structure. |

## Blocking v1 findings corrected

1. Training startup restored summaries but excluded `state.pt`; an interrupted run could silently
   restart. V2 restores the active weights/optimizer before loading.
2. Preprocessing trusted completed markers without restoring completed arrays. V2 restores payloads
   and revalidates both files before skipping.
3. Every new best epoch requested an HF upload; one observed run produced dozens of pushes. V2
   saves every best locally, uploads every 30 minutes/major stage/stop, and caps upload calls.
4. Checkpoints were non-atomic and omitted Python/CUDA RNG, batch cursor, partial aggregates and
   early-stop state. All are now atomic/persisted.
5. Worker augmentation used mutable RNG state. V2 augmentation and shuffled order are deterministic
   by seed/epoch/index, enabling exact mid-epoch replay.
6. Epoch history held only total train/validation loss. V2 retains loss components, biomedical
   metrics, optimisation/system telemetry and fine-grained Parquet/JSONL records.
7. HRV concatenated unrelated recordings and invented boundary RR intervals. V2 groups by recording.
8. Five fold means cannot produce a two-sided Wilcoxon p below 0.0625. V2 uses subjects as paired
   independent units and limits tests to predeclared full-model comparisons.
9. The coupling gate selected an easier, Apnea-poor cohort. V2 retains coupling as a covariate.
10. Quick runs reused full IDs and could be mistaken for completed training. V2 prefixes quick IDs.
11. NB03–NB05 could collide in one repository-level state file. V2 uses stage-specific repositories.
12. The plan named LOSO and cross-scenario experiments but the queue did not execute them. V2 does.

## Verification performed

- All five builders and shared Python files compile.
- All five generated notebooks are valid notebook v4 JSON, have no outputs/execution counts, and
  every code cell parses.
- All embedded shared modules compile and are byte-identical to their source strings.
- A runtime smoke test wrote and restored the complete training telemetry/checkpoint.
- A forced failure after batch 1 resumed at batch 2 and produced weights bit-for-bit identical to
  uninterrupted training.
- `git diff --check` passes.
