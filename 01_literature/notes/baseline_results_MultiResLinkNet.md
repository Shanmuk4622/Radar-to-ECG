# Baseline reference numbers — Chowdhury et al. 2024 (Comput Biol Med 176:108555)

Everything our paper must beat. Copied verbatim from Tables 1-5 of the PDF.

## Setup we must replicate for a fair comparison
- Dataset: Schellenberger et al. 2020 CR-RVS (24 GHz six-port CW radar + Task Force Monitor).
- Input: **single-channel** radar displacement obtained from I/Q by ellipse fitting + arctangent demodulation.
- Resample 2000 Hz -> **128 Hz** (linear interpolation).
- 50/60 Hz notch; ECG bandpass ("Halter" filter) 0.5-40 Hz.
- Baseline drift removal: order-5 polynomial fit.
- Windows: **1024 samples = 8.0 s**, 50 % overlap on the TRAIN set only.
- Normalisation: z-score then range-normalise to [0,1].
- Models: FPN, UNet, LinkNet, MultiResLinkNet. 5 levels, 64 filters doubling per level.
- Loss: **MSE only**. Adam, lr = 5e-4 constant, 300 epochs, patience 20.
- Split 80/20, 5-fold CV, claimed subject-wise stratification. Google Colab Pro.
- Peak detector used for HR metrics: "Two Average" (best of Hamilton, Pan-Tompkins, SWT, Engzee, Threshold, Christov, WQRS, Two Average).

## Segment counts (Table 1)
| Scenario | Subjects | Segments | Train (80%) | Test (20%) |
|---|---|---|---|---|
| Resting  | 30 | 4702 | 3762 | 940 |
| Valsalva | 27 | 6952 | 5562 | 1390 |
| Apnea    | 24 | 1140 |  912 | 228 |
| RVA combined | - | 12794 | 10236 | 2558 |

## Experiment A — per scenario (Table 2)
| Scenario | Model | MAE | MSE | CC_temporal | CC_spectral | RRMSE_t | RRMSE_s |
|---|---|---|---|---|---|---|---|
| Resting | FPN | 0.14204 | 0.03170 | 58.374 | 71.376 | 0.46940 | 0.73374 |
| Resting | UNet | 0.13872 | 0.03219 | 63.103 | 74.681 | 0.45760 | 0.86096 |
| Resting | LinkNet | 0.13588 | 0.03034 | 64.347 | 74.374 | 0.45116 | 0.81111 |
| Resting | **MultiResLinkNet** | **0.13258** | 0.03066 | **66.095** | **82.439** | **0.43682** | **0.71412** |
| Valsalva | FPN | 0.14985 | 0.03679 | 57.530 | 65.968 | 0.46395 | 0.87990 |
| Valsalva | UNet | 0.15249 | 0.03928 | 58.381 | 68.789 | 0.46553 | 0.99554 |
| Valsalva | LinkNet | 0.15087 | 0.03869 | 56.627 | 66.866 | 0.46195 | 0.99095 |
| Valsalva | **MultiResLinkNet** | 0.15286 | 0.04012 | **60.136** | **77.052** | **0.46083** | **0.80660** |
| Apnea | FPN | 0.15310 | 0.03853 | 39.119 | 51.261 | 0.51017 | 1.00889 |
| Apnea | UNet | 0.14406 | 0.03495 | 56.141 | 69.975 | 0.47825 | 0.92034 |
| Apnea | LinkNet | 0.14572 | 0.03526 | **56.223** | 70.350 | 0.47944 | 0.91749 |
| Apnea | **MultiResLinkNet** | 0.14474 | **0.03474** | 55.331 | **74.658** | **0.47692** | **0.82392** |

## Experiment B — RVA combined (Table 3)
| Model | MAE | MSE | CC_temporal | CC_spectral | RRMSE_t | RRMSE_s |
|---|---|---|---|---|---|---|
| FPN | 0.14316 | 0.03422 | 59.632 | 69.534 | 0.44694 | 0.83026 |
| UNet | 0.14798 | 0.03741 | 57.647 | 68.388 | 0.45315 | 0.94118 |
| LinkNet | 0.14780 | 0.03723 | 58.686 | 70.913 | 0.45487 | 0.86909 |
| **MultiResLinkNet** | 0.14841 | 0.03793 | **61.863** | **79.962** | **0.44618** | **0.73269** |

## Table 4 — R-peak detection on synthesised ECG (MultiResLinkNet + "Two Average")
| Metric | Resting | Valsalva | Apnea | RVA |
|---|---|---|---|---|
| Accuracy | 0.951 | 0.895 | 0.799 | 0.886 |
| F1 | 0.955 | 0.944 | 0.888 | 0.939 |
| Precision | 0.977 | 0.970 | 0.939 | 0.973 |
| Recall | 0.934 | 0.920 | 0.842 | 0.908 |
| TP | 17787 | 26039 | 4156 | 17512 |
| FP | 409 | 799 | 266 | 482 |
| FN | 1243 | 2240 | 779 | 1764 |

## Table 5 — HRV comparison (GT vs prediction)
| Scenario | Signal | mu_RR (ms) | sd_RR (ms) | mu_HR (bpm) | sd_HR (bpm) | RMSSD (ms) |
|---|---|---|---|---|---|---|
| Resting | GT | 126.49 | 19.26 | 62.12 | 9.638 | 12.95 |
| Resting | Pred | 125.71 | 22.41 | 63.34 | 13.72 | 23.87 |
| Valsalva | GT | 125.79 | 20.41 | 62.70 | 10.56 | 13.58 |
| Valsalva | Pred | 125.42 | 23.34 | 63.64 | 13.96 | 24.08 |
| Apnea | GT | 118.25 | 19.60 | 66.72 | 11.17 | 13.17 |
| Apnea | Pred | 117.81 | 25.49 | 68.57 | 17.05 | 31.50 |

## Errors / weak points spotted in the baseline paper (ammunition for our Discussion)
1. **mu_RR is physically impossible.** 126 ms of R-R interval implies ~476 bpm, yet they report 62 bpm in the same row. Their mu_RR is almost certainly reported in *samples at 128 Hz* (126 samples / 128 Hz = 0.984 s = 61 bpm), mislabelled as ms. We must report RR in real ms.
2. **RMSSD is nearly 2x over-estimated** in every scenario (12.95 -> 23.87 ms resting; 13.17 -> 31.50 ms apnea). This is the signature of a jittery, over-smoothed QRS - exactly what pure-MSE training produces. HRV fidelity is the weakest link and is our clearest target.
3. **Recall < precision everywhere** (0.842 apnea) - the model misses beats rather than inventing them. Under-detection, again consistent with amplitude smoothing.
4. **Contradictory split description.** Section 3.1 says "80 % / 20 % of the total data" and Section 3.2 says "subject-wise stratification ... no data leakage", but the segment counts in Table 1 are an exact 80/20 of the segment totals, which is not achievable with clean subject-wise splitting on 30/27/24 subjects. Leakage is plausible. We will run strict **LOSO** and report both.
5. **Tilt-up and Tilt-down scenarios were dropped entirely** - 2 of the 5 protocol scenarios never evaluated.
6. **Loss is plain MSE**, no spectral / morphological / peak-aware term.
7. **The I/Q pair is collapsed to one channel** before the network; amplitude and higher-harmonic information is discarded.
8. **No statistical significance testing, no Bland-Altman, no ablation study, no parameter/FLOP budget.**
