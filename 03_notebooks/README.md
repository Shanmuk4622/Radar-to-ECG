# Notebooks

Run in order. Each is self-contained and Kaggle-ready.

| # | Notebook | Purpose | Accel | Status |
|---|---|---|---|---|
| 01 | `01_verify_and_download.ipynb` | Verify the mirror; per-file census; cross-check against both papers | **None (CPU)** | **DONE** — verdict `RAW_MAT_TREE`, 135 files, cross-check within 0.1 % |
| 02 | `02_preprocess_to_hf.ipynb` | Windowed corpus, targets, folds; push to HF | **None (CPU)** | **DONE** — 76 recordings, 27 subjects, 12,425 windows |
| 03 | `03_baselines.ipynb` | FPN-1D, UNet-1D, LinkNet-1D, MultiResLinkNet — the reproduction gate | **2×T4** | ready (v2 — dual-format corpus reader) |
| 04 | `04_cardiomamba_train.ipynb` | CardioMamba-Net (C1–C5) + the 10-rung ablation ladder | **2×T4** | ready |
| 05 | `05_evaluate_and_figures.ipynb` | All 8 tables, Bland–Altman, Wilcoxon+Holm, 10 figures, robustness | **2×T4** (CPU if robustness off) | ready |

## Shared conventions

- `hf_sync.py` is written by NB01 into its HF repo and imported by NB02–05. One implementation of
  the 30-minute cadence, the stage-boundary push, the interrupt push and the resume logic.
- Every notebook: `HF_TOKEN` from Kaggle Secrets; **public** HF repos; `state.json` + `history.jsonl`
  pushed with every flush; restart-safe by design.
- Signal constants (128 Hz, 1024-sample windows, 50 % overlap, 0.5–40 Hz ECG band) are frozen to the
  baseline paper so results are directly comparable. See `00_admin/PLAN.md` §6.

## HF repos

| Repo | Contents | Made by |
|---|---|---|
| `Shanmuk4622/cr-rvs-radar-ecg-inventory` | inventory, previews, decimated corpus, figures | NB01 |
| `Shanmuk4622/cr-rvs-radar-ecg-processed` | windowed training corpus + fold assignments | NB02 |
| `Shanmuk4622/cardiomamba-net` | checkpoints, logs, metrics | NB03–04 |

All public.

## Generators

`04_src/utils/build_nb01.py` emits `01_verify_and_download.ipynb`. Edit the generator, not the
`.ipynb` — it validates every code cell with `ast.parse` and round-trips the JSON on write.


## Run order and expected wall-clock

| Notebook | Accelerator | Time | Sessions |
|---|---|---|---|
| 01 | None | 15–40 min | 1 |
| 02 | None | 20–45 min | 1 |
| 03 | GPU T4 x2 | ~1 h in QUICK, then 3–4 full sessions | queue, resumable |
| 04 | GPU T4 x2 | ~1 h in QUICK, then 3–4 full sessions | queue, resumable |
| 05 | GPU T4 x2 | 10–25 min | 1 |

Set `CFG["QUICK"] = True` on the first run of NB03 and NB04 — one fold, 25 epochs, enough to prove
the path end to end. Then set it `False` and let the queue work through sessions.

## Shared library

`crvs_sync, crvs_data, crvs_metrics, crvs_models, crvs_cmnet, crvs_losses, crvs_engine` are
embedded byte-identically in every notebook (verified by md5 at build time). Edit them in
`04_src/utils/nb_lib_a.py` / `nb_lib_b.py` and re-run the generators — never edit an `.ipynb`
by hand.

## Generators

```
cd 03_notebooks
PYTHONPATH=../04_src/utils python3 ../04_src/utils/build_nb02.py 02_preprocess_to_hf.ipynb
PYTHONPATH=../04_src/utils python3 ../04_src/utils/build_nb03.py 03_baselines.ipynb
PYTHONPATH=../04_src/utils python3 ../04_src/utils/build_nb04.py 04_cardiomamba_train.ipynb
PYTHONPATH=../04_src/utils python3 ../04_src/utils/build_nb05.py 05_evaluate_and_figures.ipynb
```

Every generator `ast.parse`s each code cell and round-trips the JSON before writing.


## Progress

| Stage | Status | Result |
|---|---|---|
| NB01 | complete | `RAW_MAT_TREE`. 135 `.mat`, 5.6 GB, 30 subjects. Durations match the papers to **0.00 %**; Apnea segment count matched **exactly** (1140 = 1140) |
| NB02 | complete | [`cr-rvs-radar-ecg-processed`](https://huggingface.co/datasets/Shanmuk4622/cr-rvs-radar-ecg-processed) — 76 recordings, 27 subjects, 12,425 windows, 25 normalisation sets |
| NB03 | ready to run | first attempt failed on a corpus-format mismatch plus a stale imported module; both fixed |
| NB04 | ready | |
| NB05 | ready | |

### Corpus formats

The live corpus was written as compressed `.npz`. `crvs_data._Rec` reads **either** `.npz` or the
newer uncompressed `.npy`, so nothing needs re-uploading — but the legacy format costs
**11.3 ms per window against 0.07 ms**, a 155x difference that will dominate GPU time. Re-running
NB02 regenerates the corpus as `.npy` and is worth the 20 minutes.

### If a notebook behaves as though your edits did not take

Restart the kernel. The notebooks write `crvs_*.py` to disk and import them, and Python caches
modules in `sys.modules` — so a second run in the same kernel keeps the first version. Each
notebook now purges the cache and asserts `crvs_data.LIB_VERSION`, printing the file it loaded, so
this fails loudly instead of silently.
