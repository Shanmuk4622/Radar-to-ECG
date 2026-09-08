# Recommended paper direction

> **Update:** the complete manuscript is now in `latex/main.tex`. The preparation audit identified
> a confirmed 125-Hz stream labelled 128 Hz, the S4D-versus-Mamba naming mismatch and incomplete
> computational-profiling evidence. Read `MANUSCRIPT_AUDIT.md` before using the earlier advice
> below. Physiological reevaluation is required before submission.

## Recommendation

Write a **strict benchmarking, ablation, robustness and negative-results paper** on radar-to-ECG
reconstruction. Do not frame the paper as “the full CardioMamba model beats the published state of
the art.” The completed evidence does not support that claim. It does support a more credible and
useful story about validation rigor, architectural simplicity, efficiency, robustness and the gap
between waveform correlation and clinical HRV fidelity.

### Recommended title

**Re-evaluating Radar-to-ECG Reconstruction Under Strict Subject-Wise Validation: Ablations,
Robustness, and Cross-Scenario Generalisation**

Alternatives:

- **When More Architecture Is Not Better: A Subject-Wise Evaluation of State-Space Models for
  Contactless ECG Reconstruction**
- **Beyond Mean Waveform Correlation: Reproducibility, Robustness, and HRV Failure Modes in
  Radar-to-ECG Reconstruction**

## Central research question

How do modern state-space radar-to-ECG models behave under leakage-resistant subject-wise
validation, clinically relevant peak/HRV measurements, systematic component ablation and
physically motivated corruption tests?

## Defensible thesis

Strict subject-wise evaluation produces a much weaker and more variable MultiResLinkNet baseline
than the published headline result. Compact state-space variants improve efficiency, stability,
peak reconstruction and noise robustness relative to this strict rerun, but the full proposed
multi-task/FiLM design is not the best variant and none of the tested models achieves reliable HRV
reconstruction. This demonstrates why architecture claims in contactless biosignal reconstruction
need subject-level statistics, component ablations and physiological endpoints in addition to
average waveform correlation.

## Main contributions

1. A reproducible 134-recording, 30-subject CR-RVS pipeline with one documented signal-quality
   exclusion, subject-wise folds, LOSO and held-out-scenario protocols.
2. A controlled rerun of four baseline families and a transparent reproduction gate.
3. A 100-run state-space-model study covering architectural, objective, LOSO, cross-scenario and
   robustness experiments with resumable provenance.
4. Evidence that the compact no-FiLM and single-task variants outperform the proposed full model
   on correlation; the single-task advantage remains significant after Holm correction.
5. Evidence of strong noise robustness relative to the strict MultiResLinkNet rerun, together with
   a clear Tilt-up generalization failure.
6. A physiological audit showing that improved waveform/peak metrics do not yet translate into
   clinically credible RMSSD estimation.

## Results to put in the abstract

- All 180 canonical training runs completed and were evaluated.
- Full L9 improves temporal/spectral correlation from 44.61/73.70% for strict MultiResLinkNet to
  57.46/87.44%, while reducing parameters from 9.22 M to 4.09 M and GFLOPs from 3.48 to 1.24.
- No-FiLM reaches the best mean temporal/spectral correlation at 61.98/89.16%.
- Single-task reaches 61.74/88.81% and significantly outperforms full L9 in the subject-paired,
  Holm-corrected comparison (`p=0.01969`).
- At 12 dB injected noise, full L9 retains 96.04% of clean temporal correlation, versus 15.52% for
  the strict baseline in the selected-fold robustness test.
- Full L9 reduces HR MAE substantially but RMSSD MAE remains about 201 ms, preventing a clinical
  HRV claim.

Label the robustness number as exploratory unless it is repeated across all five folds.

## Claims to avoid

- “CardioMamba beats the published state of the art.”
- “The full C1-C5 architecture is validated.”
- “Multi-task learning or FiLM improves reconstruction.”
- “The model recovers clinically accurate HRV.”
- “Noise robustness is proven across subjects” before the all-fold rerun.
- “Cross-scenario generalization is solved”; Tilt-up temporal CC is only 28.70%.

## Suggested manuscript structure

1. **Introduction** — contactless ECG value; why subject leakage and waveform-only metrics can
   overstate performance; concise contribution list.
2. **Related work** — radar vital signs, radar-to-ECG networks, state-space sequence models,
   subject-independent biosignal validation and physiological evaluation.
3. **Dataset and reproducible protocol** — all 135 files, single flatline exclusion, final scenario
   counts, preprocessing, subject-wise folds, LOSO and cross-scenario design.
4. **Models and hypotheses** — strict baselines; C1-C5; explain that each ablation tests a declared
   question rather than merely shrinking a network.
5. **Evaluation** — waveform, spectral, peak, HR/HRV, subject-paired Wilcoxon with Holm correction,
   parameters/FLOPs, robustness and Bland-Altman.
6. **Results** — reproduction gate first, primary comparison second, ablations third, then
   generalization, physiology and robustness.
7. **Discussion** — why the simpler variants win; baseline fold collapse; AMP recoveries; Tilt-up
   domain shift; detector/unit/aggregation concerns for RMSSD; limitations.
8. **Conclusion** — rigorous evaluation narrows the justified claim and identifies what must be
   fixed before clinical translation.

## Recommended tables and figures

- Dataset/scenario census and validation design.
- Primary B_rva table with published baseline, strict rerun, full, no-FiLM and single-task.
- Subject-level paired tests with raw and Holm-adjusted p-values.
- Full ablation table with fold dispersion, parameters and FLOPs.
- Scenario, LOSO and held-out-scenario table highlighting Tilt-up.
- Peak/HR/HRV table with explicit units and aggregation level.
- Architecture-versus-efficiency plot.
- Fold distribution/paired-subject plot rather than only mean bars.
- HR and RMSSD Bland-Altman plots.
- Noise robustness curves, clearly labeled exploratory if not rerun across folds.
- Representative successful and failed waveform reconstructions.

## Follow-up experiments required before submission

### Essential

1. **Confirm L8 no-FiLM and L7 single-task prospectively.** Freeze the choice and hyperparameters,
   rerun with fresh seeds, and do not select the winner again after seeing results.
2. **Audit peak detection and HRV units.** Manually inspect a stratified set of recordings, verify
   sampling-rate conversion and boundary handling, and reconcile the roughly 85 ms ground-truth
   RMSSD with the published roughly 13 ms value.
3. **Repeat robustness over all five folds** with confidence intervals and subject-level pairing.
4. **Stabilize selected training.** Confirm the leading models in FP32 or with a lower learning
   rate because 58/100 runs required numerical recovery and 20 ended with AMP disabled.
5. **Warm up and repeat latency measurements.** The current first-call outliers make parameter and
   FLOP comparisons reliable but latency comparisons preliminary.

### Strongly recommended

6. Investigate Tilt-up domain shift using signal-quality covariates and per-subject errors without
   changing the held-out test set.
7. Report bootstrap confidence intervals for all primary effect sizes.
8. Separate exploratory model selection from the final confirmatory comparison.
9. Add error analyses by scenario, subject, recording duration and beat-coupling score.
10. If resources allow, validate the fixed winning model on an external radar-ECG dataset.

## Publication positioning

The work is not ready as a “new SOTA architecture” paper. It is close to a credible methods and
reproducibility paper after the essential follow-ups above. Appropriate venues are journals and
conferences receptive to biomedical signal-processing validation, negative results and
reproducibility. Choose a venue only after checking its current scope, format and data/code policy.

## Paper-readiness verdict

- **Enough evidence to start writing:** yes.
- **Enough evidence to submit the current SOTA claim:** no.
- **Best present contribution:** rigorous evidence that strict validation and physiological metrics
  materially change the conclusion, plus a compact and robust state-space alternative whose
  simpler variants outperform its full design.
