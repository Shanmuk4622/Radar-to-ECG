# Final results and scientific insights

> **Manuscript audit update (2026-09-07):** a deeper source-and-artifact audit confirmed that
> preprocessing produced 125-Hz signals labelled 128 Hz (134/134 saved lengths match). HR/HRV
> values below are archived nominal measurements and require reevaluation. The model is
> bidirectional S4D, not selective Mamba; stored FLOP estimates have unverified FFT/complex
> operator coverage. These qualifications supersede any stronger wording below. See
> [the manuscript audit](../07_paper/MANUSCRIPT_AUDIT.md) and the complete LaTeX manuscript.

Verified from the public Hugging Face repositories on **2026-09-07** after NB05 completed.
The evaluation snapshot is pinned to generation `ca226baa6c1f1ae1` and results-repository commit
`bb25560143d1044dc8671ef487160cc0049f7282`.

## 1. Executive verdict

The experiment is **operationally complete and scientifically useful**, but it is **not a clean
state-of-the-art success**.

- NB03 contributed all **80/80** canonical baseline runs and NB04 contributed all **100/100**
  canonical CardioMamba runs. The quick-test run was correctly excluded.
- The full L9 CardioMamba model improves substantially over our strict MultiResLinkNet rerun on
  temporal correlation, spectral correlation, peak detection, HR error, RMSSD error, parameter
  count and FLOPs.
- The full model does not meet the preregistered physiological targets and does not reproduce the
  published baseline correlation. The baseline reproduction gate therefore correctly failed.
- The strongest result is an ablation finding: **L8 without FiLM** has the best mean temporal and
  spectral correlations, while **L7 single-task** is significantly better than the full L9 model
  in the subject-paired confirmatory test. The complete architecture is over-engineered for this
  dataset as presently trained.
- The most defensible paper is consequently a rigorous benchmarking, ablation, robustness and
  negative-results paper—not a paper claiming that the full model beats the published state of
  the art.

## 2. Completeness and provenance

| Item | Verified result |
|---|---:|
| Baseline queue | 80/80 complete |
| CardioMamba queue | 100/100 complete |
| Core ablations | 9 variants x 5 folds |
| NB05 input status | complete |
| NB05 validation status | failed only because the declared reproduction gate failed |
| Quick runs included | 0 |
| Declared NB05 artifacts | 23/23 present and hash-valid |
| Downloaded snapshot artifacts | 612 |
| Downloaded artifact size | 239.35 MiB |

All 100 NB04 summaries are finite and contain the required 13 artifacts. The final report contains
11 tables and 9 figures. `inputs-complete-gate-failed` means that evaluation completed but the
baseline reproduction requirement was not met; it does not mean that NB05 crashed or omitted
runs.

Pinned source revisions:

| Repository | Commit |
|---|---|
| `cardiomamba-results-v2` | `bb25560143d1044dc8671ef487160cc0049f7282` |
| `cardiomamba-net-v2` | `dd5764c51031f5d9415ecddbe64ff10ab0d02348` |
| `cardiomamba-baselines-v2` | `84bf3e7992a33aecacfb30ab860872c620ca7699` |
| `cr-rvs-radar-ecg-processed-v2` | `be14c2a90442d38235c5095d7d4eac07dfa9adcd` |

## 3. Final dataset and protocol

V2 inventories 135 recordings from 30 subjects and retains **134 recordings** after excluding
only `GDN0030_4_TiltUp.mat` for an ECG flatline. It contains **20,757 overlapping windows** and
**10,411 non-overlapping windows**.

| Scenario | Recordings | Windows | Non-overlapping windows |
|---|---:|---:|---:|
| Apnea | 24 | 1,115 | 563 |
| Resting | 30 | 4,609 | 2,311 |
| Tilt-down | 27 | 4,198 | 2,106 |
| Tilt-up | 26 | 4,044 | 2,027 |
| Valsalva | 27 | 6,791 | 3,404 |

The primary B_rva comparison uses strict subject-wise five-fold validation. Subject-level paired
statistics, LOSO, cross-scenario tests, computational cost, clinical metrics and noise/channel
robustness are also reported.

## 4. Primary B_rva results

| Model | MAE | MSE | CC temporal (%) | CC spectral (%) | Peak F1 | HR MAE (bpm) | RMSSD MAE (ms) | Params (M) | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Published MultiResLinkNet | 0.14841 | 0.03793 | 61.863 | 79.962 | 0.939 | — | — | — | — |
| Strict MultiResLinkNet rerun | 0.165734 | 0.042689 | 44.611 | 73.702 | 0.621 | 14.282 | 263.951 | 9.219 | 3.478 |
| L9 full CardioMamba | **0.164874** | **0.042436** | 57.464 | 87.442 | 0.808 | 3.498 | 200.830 | 4.091 | 1.245 |
| L8 no FiLM | 0.171548 | 0.044040 | **61.985** | **89.159** | **0.840** | **3.117** | **175.108** | 4.088 | — |
| L7 single-task | — | — | 61.739 | 88.814 | 0.839 | 3.370 | 184.076 | 4.081 | — |

Relative to the strict MultiResLinkNet rerun, the full L9 model provides:

- **+12.853 percentage points** temporal correlation and **+13.740 points** spectral correlation.
- 0.52% lower MAE and 0.59% lower MSE.
- **75.51% lower HR MAE** and **23.91% lower RMSSD MAE**.
- **55.63% fewer parameters** and **64.21% fewer GFLOPs**.

The subject-paired full-versus-baseline comparison has raw `p=0.014538`, but Holm-adjusted
`p=0.101766`; it is therefore not significant at 0.05 after multiplicity correction.

### Preregistered target check for the full model

| Target | Full L9 | Result |
|---|---:|---|
| Temporal CC >= 80% | 57.464% | Fail |
| Spectral CC >= 88% | 87.442% | Narrow fail |
| HR MAE < 2 bpm | 3.498 bpm | Fail |
| RMSSD MAE < 8 ms | 200.830 ms | Fail |
| Parameters < 5 M | 4.091 M | Pass |
| GFLOPs < 1.5 | 1.245 | Pass |

L8 no-FiLM passes the spectral target but still falls well short of the temporal and physiological
targets.

## 5. What the ablations say

- L8 no-FiLM is the best core variant by mean temporal CC (61.985%) and spectral CC (89.159%).
- L7 single-task is close behind (61.739%, 88.814%) and is the only comparison that remains
  significant after Holm correction against full L9: median temporal-CC difference `-2.263`
  points for full versus single-task, raw `p=0.002188`, adjusted `p=0.01969`. The direction favors
  the simpler model.
- Removing the SSM reduces mean temporal CC from 57.464% to 51.523%, while replacing it with a
  Transformer gives 50.887%. These trends support the SSM, but neither comparison survives Holm
  correction (`p=0.2746` and `p=0.0964`, respectively).
- Removing wavelets gives 56.908%, very close to the full model, with no corrected significance.
- C1 alone reaches 59.623%, whereas C1+C5 reaches 56.436%; the contribution stack is not additive.
- The C5 composite-loss model on the baseline backbone reaches 52.821% temporal and 87.836%
  spectral CC, with peak F1 0.845. The loss helps peak-related behavior but does not establish that
  every full-model component is necessary.

**Interpretation:** the SSM is promising, but FiLM and the auxiliary multi-task arrangement do not
earn their complexity in this dataset. The paper should report this explicitly rather than select
the full architecture after seeing the results. L8 is a post-hoc winner and needs a predeclared
confirmation run before being promoted as the primary model.

## 6. Fold variability and stability

Temporal-CC fold means expose substantial subject-composition sensitivity:

| Model | Fold values (%) | SD |
|---|---|---:|
| Strict MultiResLinkNet | 51.98, 43.19, 63.70, 8.69, 55.50 | 21.39 |
| L9 full | 58.01, 50.70, 68.81, 64.99, 44.82 | 9.89 |
| L8 no FiLM | 63.62, 50.50, 71.16, 61.81, 62.83 | 7.41 |

Full L9 beats the strict baseline in four of five folds, but part of its mean advantage comes from
the baseline collapse in fold 3. No-FiLM beats the baseline in all five folds and is more stable.

Training completed, but numerical recovery was frequent: **58/100** NB04 runs recovered at least
once, with 78 recovery events in total; 20 runs ended with AMP disabled. All 30 LOSO runs recovered
at least once. Final outputs are valid, but selected models should be confirmed with FP32 or a
lower learning rate before publication.

## 7. Scenario and generalization results

For the full model, mean temporal/spectral CC is 45.071/82.556% on Apnea,
56.658/87.525% on Resting and 51.473/86.692% on Valsalva. The combined five-scenario model gives
56.024/86.710%, and LOSO gives 54.940/86.905% across 30 held-out subjects.

Cross-scenario temporal/spectral CC is:

| Held-out scenario | Temporal CC (%) | Spectral CC (%) |
|---|---:|---:|
| Apnea | 67.989 | 91.069 |
| Resting | 75.192 | 94.059 |
| Tilt-down | 73.349 | 93.101 |
| Tilt-up | **28.702** | 81.687 |
| Valsalva | 77.576 | 94.589 |

Tilt-up is the clear domain-shift failure. These are one held-out-scenario run each, so they are
valuable diagnostic evidence but not a replicated five-fold estimate.

## 8. Peak, HR and HRV findings

At recording level, full L9 improves peak F1 from 0.627 for strict MultiResLinkNet to **0.813**,
with precision 0.894, recall 0.764 and a missed-beat rate of 0.236. It remains below the published
peak F1 of 0.939.

Full L9 reduces aggregate absolute HR error from 13.61 to **3.20 bpm** and RMSSD error from
271.28 to **201.42 ms**. Bland-Altman HR bias improves from +5.73 bpm with limits
`[-36.2, +47.6]` to -1.39 bpm with limits `[-11.9, +9.1]`. RMSSD remains unsuitable: full-model
bias is approximately +198.73 ms with very wide limits `[-74.1, +471.6]`.

The evaluation's ground-truth aggregate RMSSD is about 85 ms, while the published table reports
about 13 ms. That mismatch, together with the published paper's apparent RR sample/ms labeling
error, requires a detector, unit and aggregation audit before making any clinical HRV claim.

## 9. Robustness findings

The robustness experiment uses one selected fold and 800 windows, so it is exploratory rather
than confirmatory. Still, the contrast is large:

| Input condition | Full L9 temporal CC (%) | Strict baseline temporal CC (%) |
|---|---:|---:|
| Clean | 63.008 | 57.486 |
| 12 dB SNR | 60.515 | 8.925 |
| 6 dB SNR | 55.121 | 3.686 |
| 3 dB SNR | 49.224 | 2.003 |
| 0 dB SNR | 40.827 | 1.547 |
| -3 dB SNR | 29.447 | 1.150 |
| Motion, scale 0.5 | 59.019 | 47.550 |

At 12 dB, full L9 retains 96.04% of its clean correlation versus 15.52% for the strict baseline.
This is one of the strongest publishable findings and should be repeated over all folds.

Inference-only channel dropout indicates the greatest sensitivity to the derived cardiac channel
(56.33%) and acceleration (58.71%). Dropping Q or dy does not hurt on this selected fold. Because
the network was not retrained, these are sensitivity tests—not causal feature ablations.

## 10. Claims supported by the evidence

Supported:

1. The evaluated CardioMamba family is more parameter- and compute-efficient than the strict
   MultiResLinkNet rerun.
2. Full L9 substantially improves mean correlation, peak reconstruction and HR/HRV errors over
   that strict rerun, although the corrected confirmatory test is not significant.
3. The SSM variants trend above no-SSM and Transformer variants.
4. The full model is dramatically more robust to injected noise in the selected-fold test.
5. FiLM and auxiliary multi-task learning do not improve the primary correlation outcome here.
6. Strict subject-wise validation exposes much lower and more variable baseline performance than
   the published number.

Not supported:

1. That full L9 beats the published MultiResLinkNet result or is state of the art.
2. That it meets the preregistered waveform or physiological targets.
3. That accurate clinical HRV is recovered.
4. That every proposed component is beneficial.
5. That the exploratory robustness and cross-scenario numbers generalize without replication.

## Bottom line

This is a **successful, complete experiment with a publishable negative/diagnostic result**. It
does not validate the original “full CardioMamba beats all targets” hypothesis. Its strongest
story is that strict subject-wise evaluation changes the apparent baseline, a compact SSM model
improves efficiency and robustness, and careful ablation rejects unnecessary FiLM/multi-task
complexity while exposing HRV and Tilt-up as unresolved failures.
