# Audit findings discovered during manuscript preparation

Verified on 2026-09-07 from the source code, pinned HF modules and additional per-window artifacts.
These findings qualify the earlier `RESULTS_AND_INSIGHTS.md` and paper-direction memo.

## 1. Confirmed physical sampling rate: 125 Hz, labelled 128 Hz

NB01 selects `round(source_fs / 128)` as an integer decimation factor. All 135 raw recordings
have radar and ECG sampling rates of 2,000 Hz, yielding decimation by 16 and an actual 125-Hz
stream. The metadata nevertheless records 128 Hz. All **134/134** retained array lengths match
`ceil(raw_sample_count / 16)`. The supporting rows are in
`latex/generated/sampling_rate_audit.csv`.

Consequences:

- A 1,024-sample window lasts 8.192 seconds, not 8 seconds.
- Sample-wise MAE, MSE and temporal correlation are unchanged by rate relabelling.
- The implemented spectral correlation uses the same Welch-bin arrays and is unchanged by a
  common rate relabelling, but its frequency labels are wrong.
- Peak detection, smoothing, refractory intervals, event matching, interval rejection and HR/HRV
  conversion use the wrong nominal rate. Those metrics cannot be called physically validated.
- Multiplying unchanged interval values by 128/125 gives a 2.4% unit correction, but rerunning the
  rate-dependent detector may change the events themselves. Algebraic scaling is not reevaluation.

The manuscript retains archived values as nominal and explains the issue in Methods and Appendix B.
This task does not rewrite preprocessing or training notebooks: that needs a separate corrected
data/evaluation revision to preserve the completed experiment.

## 2. The implemented model is S4D, not selective Mamba

The archived `crvs_cmnet.py` hash matches all 45 primary experimental configurations. It constructs
diagonal, input-independent state-space kernels and applies FFT convolution in both directions.
There are no input-dependent selective state-space parameters or Mamba scan. The paper describes
the model as a bidirectional S4D network and records CardioMamba as a repository name only.

FFT convolution has O(L log L) sequence-dependent convolution cost plus kernel formation.
Neither a linear-time streaming claim nor a causal real-time claim is supported.

## 3. Computational estimates are incomplete evidence

Stored GFLOP estimates come from `torch.utils.flop_counter.FlopCounterMode`. Coverage of complex
kernel formation and FFT operations has not been established. The manuscript reports counted
operations with this caveat. Parameter counts remain directly verifiable. Large first-call
latency outliers prevent a validated speedup claim.

## 4. Architectural and training comparisons have different strengths

Baseline and NB04 runs differ in target scaling, initial learning rate, epoch budget, patience,
checkpoint selection and possibly engine revision. L2 is therefore not a pure loss ablation.
L5--L10 are more closely matched architectural comparisons. L7 retains peak-weighted waveform
supervision even though it has only one output head.

The archived current NB04 engine matches only 1/45 primary experimental run configurations;
the other shared experimental modules match 45/45. All 20 primary baseline runs match the
archived baseline engine. Exact replay requires the per-run engine revision, not only the
repository HEAD. This is disclosed rather than silently treating the runs as identical builds.

## 5. Statistics and aggregation checked independently

All 13 primary variants have 6,278 held-out window results covering the same 30 subjects.
The nine archived participant-paired signed-rank and Holm-adjusted probabilities were reproduced
within 1e-12. Added 20,000-replicate bootstrap intervals are explicitly retrospective, unadjusted
and conditional on the existing fitted models. They do not override the adjusted tests.

NB05 Table 4 aggregates event counts from per-subject recording averages; those columns are not
pooled integer event counts. The paper omits them. It uses equal-participant physiological
averages and distinguishes them from equal-fold recording-error summaries.

## 6. Robustness is a feature-space stress test

The perturbations occur after derivation and normalisation, with independent noise on derived
channels and different noise seeds for the two models. Only the first 800 windows of fold 0
are evaluated. This is not a matched raw-sensor corruption experiment or an all-fold result.
Zeroing Q improving correlation and the deterministic phase/displacement redundancy support
further feature analysis, not a causal claim that Q should be removed.

## Submission decision

The complete manuscript is suitable for author review and further development. Before submission:

1. Resolve the physical sampling-rate contract and recompute peak/HRV outputs with checked events.
2. Confirm leading models with fixed selection criteria and additional seeds.
3. Match baseline optimisation settings where making an architecture-only claim.
4. Repeat robustness across folds with shared raw-I/Q corruptions.
5. Resolve per-run engine versions and audit operation counts/latency.
6. Fill the intentionally blank author and declaration fields.

These are limitations of the evidence, not missing prose that can be repaired through stronger wording.
