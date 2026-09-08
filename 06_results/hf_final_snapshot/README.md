# Pinned Hugging Face final snapshot

Downloaded and verified on **2026-09-07** after NB05 completed. This directory keeps the compact,
publication-relevant outputs and their provenance in one place. It is a snapshot, not a mutable
training workspace.

## Contents

| Directory | Contents |
|---|---|
| `evaluation/` | NB05 `RESULTS.md`, audit/generation metadata, 11 tables and 9 figures |
| `provenance/cardiomamba/` | All 100 canonical summaries plus the excluded quick summary, configs, environments and result CSVs |
| `provenance/baselines/` | All 80 baseline summaries, configs, environments and result artifacts |
| `provenance/data/` | Final processed-data report, inventory, recording/window indexes, normalization statistics and library hashes |
| `checkpoints/cardiomamba_l9/` | The five primary B_rva L9 `best.pt` checkpoints |
| `SNAPSHOT_MANIFEST.json` | Relative path, byte size and SHA-256 for every downloaded artifact |

The manifest covers **612 downloaded artifacts** totaling **250,981,240 bytes (239.35 MiB)**. It
does not include itself or this local explanatory README. Checkpoints and Hugging Face cache
metadata are intentionally ignored by Git; they remain present locally.

## Pinned revisions

| Hugging Face repository | Revision |
|---|---|
| `Shanmuk4622/cardiomamba-results-v2` | `bb25560143d1044dc8671ef487160cc0049f7282` |
| `Shanmuk4622/cardiomamba-net-v2` | `dd5764c51031f5d9415ecddbe64ff10ab0d02348` |
| `Shanmuk4622/cardiomamba-baselines-v2` | `84bf3e7992a33aecacfb30ab860872c620ca7699` |
| `Shanmuk4622/cr-rvs-radar-ecg-processed-v2` | `be14c2a90442d38235c5095d7d4eac07dfa9adcd` |

NB05 evaluation generation: `ca226baa6c1f1ae1`, generated at `2026-09-07T13:31:45Z`.

## Integrity and interpretation

`SNAPSHOT_MANIFEST.json` is the mechanical integrity record for the downloaded artifacts. The NB05 generation manifest declares
23 outputs; all 23 were present and matched their recorded byte sizes and SHA-256 hashes when this
snapshot was built.

The generation status is `inputs-complete-gate-failed`. This records a **scientific gate failure**:
our strict MultiResLinkNet reproduction did not reach the published temporal-correlation target.
It is not an incomplete download or notebook failure. All **80 baseline runs** and **100 canonical
CardioMamba runs** were consumed, and the non-canonical quick run was excluded.

For the interpreted findings, read `../RESULTS_AND_INSIGHTS.md`. The original machine-generated
NB05 report remains in `evaluation/RESULTS.md`.
