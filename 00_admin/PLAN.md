# Research Plan — Contactless ECG Reconstruction from CW-Radar
### Beating MultiResLinkNet (Chowdhury et al., Comput Biol Med 2024) with a new architecture

Author: Shanmukesh (SCOPE, VIT-AP)
Status: **plan / pre-implementation**. Last updated: 2026-09-01.

---

## 1. What we are doing in one paragraph

Chowdhury et al. (2024) reconstruct an ECG waveform from a 24 GHz continuous-wave radar signal
using **MultiResLinkNet**, a 1-D LinkNet whose convolution blocks are replaced by MultiRes blocks
and whose skip connections are ResPaths. They train with plain MSE on a single-channel radar
displacement signal downsampled to 128 Hz, in 8-second (1024-sample) windows, on the
Schellenberger CR-RVS dataset. Their best numbers are a **temporal correlation of 0.62–0.66** and a
**spectral correlation of 0.75–0.82**, with **RMSSD roughly 2x over-estimated**. We keep their
dataset, their windowing and their metrics so the comparison is apples-to-apples, and we replace
the model, the input representation and the objective. Target: **temporal CC ≥ 0.80, spectral CC
≥ 0.88, HR MAE < 2 bpm, RMSSD error < 8 ms**, plus two scenarios (Tilt-up / Tilt-down) the baseline
never touched and a strict leave-one-subject-out protocol they never ran.

---

## 2. The data

**Schellenberger et al., Sci Data 7:291 (2020)** — "A dataset of clinically recorded radar vital signs
with synchronised reference sensor signals."

| Property | Value |
|---|---|
| Radar | 24 GHz six-port CW interferometry, 1 Tx + 1 Rx antenna |
| Radar sampling | 2000 Hz, 24-bit ADC (ADS1298), signals stored in mV |
| Reference | Task Force Monitor (TFM); ECG lead I + II, ICG, Z0, BP, intervention |
| ECG in file | `tfm_ecg1`, `tfm_ecg2`, stored resampled (TFM 1000 Hz -> ~2000 Hz to match radar) |
| Subjects | 30 healthy (14 M / 16 F), age 30.7 ± 9.9 y, BMI 23.2 ± 3.3 |
| Scenarios | Resting, Valsalva, Apnea, Tilt-up, Tilt-down |
| Total | ~86 459 s ≈ 24 h; resting alone averages 2882 s/subject |
| Format | One folder per subject ID, one `.mat` per scenario; plus `additional_data.xlsx` |
| Key variables | `radar_I`, `radar_Q`, `fs_radar`, `tfm_ecg1`, `tfm_ecg2`, `tfm_icg`, `tfm_z0`, `tfm_bp`, `tfm_intervention`, `tfm_param`, `measurement_info` |
| Sync | Gold-code sequence cross-correlated between radar ADC and TFM external input |

Physics that matters for the model design: displacement `dy = (dphi/2pi)*(lambda/2)`, `lambda ≈ 12.5 mm`
at 24 GHz, so **one full 2*pi phase wrap corresponds to only 6.25 mm** of chest motion. Cardiac
chest-wall motion is sub-millimetre and rides on a respiratory excursion of several millimetres.
This is why baseline drift removal matters and why the *derivative* of displacement (velocity /
acceleration, i.e. an SCG-like signal) carries the beat information — the baseline paper says so
themselves in their Limitations: *"the radar signal has picked up some mechanical signal related to
the heart's electrical activity at the moment of the QRS complex... the DL model is learning to
convert the mechanical signal to the electrical signal."*

**Download route (VERIFIED 2026-09-01):** Kaggle mirror `pedababugaddala/datasets-file`, added as a Kaggle Input
inside the notebook. Original: figshare DOI `10.6084/m9.figshare.12186516`.
*Step 0 is DONE: verdict `RAW_MAT_TREE`. 135 .mat files, 5.597 GB, 30 subject folders. See
`02_data/DATASET_FACTS.md` — note the variable names are LOWERCASE `radar_i`/`radar_q`.*

Storage arithmetic (this project is small, which is good news for Kaggle):
raw `.mat` for 24 h at 2000 Hz across ~8 channels ≈ **7–9 GB** (fits the 20 GB Kaggle disk).
After decimation to 128 Hz and keeping 7 derived channels + ECG in `float32`:
`86 459 s x 128 Hz x 8 ch x 4 B ≈ 354 MB` — the whole processed corpus fits in RAM.

---

## 3. Why the baseline underperforms (the gap analysis)

This is the heart of the paper. Six specific, defensible weaknesses:

**G1 — Pure MSE regression produces an over-smoothed ECG.**
MSE is the conditional mean. The R peak is the highest-variance, lowest-duration feature of the ECG,
so MSE systematically flattens it. The fingerprint is visible in their own Table 5: **RMSSD nearly
doubles** (12.95 -> 23.87 ms resting, 13.17 -> 31.50 ms apnea) and **recall (0.842–0.934) is always
below precision (0.939–0.977)** — beats are missed, not invented. Fixing the objective is the single
highest-leverage change available.

**G2 — The I/Q pair is destroyed before the network ever sees it.**
They apply ellipse fitting + arctangent demodulation to collapse (I, Q) into one displacement
channel and feed that. Amplitude information, residual quadrature imbalance and higher harmonics of
chest motion are thrown away. The network should be allowed to learn its own demodulation.

**G3 — No explicit time-frequency modelling.**
A 1-D CNN with 5 levels has a limited, fixed receptive field. Concurrent 2025 work
(radarODE, LifWavNet) shows that a time-frequency view — synchrosqueezed spectrograms or learnable
lifting wavelets — is what unlocks this task, because the cardiac component is a narrowband
non-stationary signature buried under a much larger respiratory component.

**G4 — No long-range / quasi-periodic modelling.**
An 8-second window contains ~8–10 cardiac cycles. A convolutional decoder treats each one
independently. Nothing in the architecture exploits the fact that beat *n+1* looks like beat *n*.

**G5 — Single-task supervision.**
radarODE-MTL (2024/25) showed that decomposing the problem into *waveform + R-peak location +
cycle length* and training them jointly lifts PCC from 90.1 % to 92.7 % on mmWave data. The baseline
regresses the waveform and nothing else, then bolts a classical peak detector on afterwards.

**G6 — Evaluation is thin and the protocol is questionable.**
Table 1's segment counts are an exact 80/20 split of *segments*, which is not reachable with the
subject-wise splitting the text claims for 30/27/24 subjects — **leakage is plausible**. Tilt-up and
Tilt-down are dropped. There is no ablation, no significance test, no Bland-Altman, no parameter or
FLOP budget, and `mu_RR` is reported in samples but labelled milliseconds.

---

## 4. Where the field is now (so our novelty claim survives review)

| Work | Modality / data | Idea | Why we are still novel |
|---|---|---|---|
| Chowdhury 2024, **MultiResLinkNet** | 24 GHz CW, CR-RVS | MultiRes blocks + ResPaths in a 1-D LinkNet, MSE | Our direct baseline |
| **radarODE** (2024) | mmWave, MMECG | Synchrosqueezed spectrogram backbone + ODE morphological prior per cardiac cycle | Different dataset/radar; we do not use an ODE prior, we use a learned SSM |
| **radarODE-MTL** (2024/25) | mmWave, MMECG | 3 decoders (morphology / anchor / cycle length) + Eccentric Gradient Alignment | We port the multi-task idea to CW radar for the first time and combine it with SSM + wavelet |
| **LifWavNet** (2025) | CW + Doppler, CR-RVS + Med-Radar | Learnable lifting-wavelet MRAS, temporal L1 + multi-resolution STFT loss | Closest competitor. It has **no sequence model, no multi-task head, no SSL pretraining, no uncertainty**. Our contribution is orthogonal and we should cite and beat it |
| Mamba/SSM for ECG (2025) | 12-lead ECG | Bi-directional SSMs for classification / delineation | Nobody has used an SSM for **radar->ECG waveform synthesis** |

**Conclusion: the open lane is a dual-domain (time + learnable wavelet) encoder, an SSM bottleneck
for quasi-periodicity, multi-task decoding, and a morphology-aware composite loss — evaluated on all
five CR-RVS scenarios under LOSO.** No published paper combines these.

---

## 5. Proposed method — **CardioMamba-Net** (working name)

> Alternative names to pick from later: *MSW-MambaNet*, *RadarECG-DMNet*, *CardioLift-SSM*.

Five contributions, each of which is a row in the ablation table and a sentence in the abstract.

### C1 — Physics-informed multi-channel input stem (replaces their arctangent demodulation)

Instead of one channel, feed an **8-channel tensor** of shape `(8, 1024)`:

| # | Channel | Rationale |
|---|---|---|
| 1 | `I` (DC-removed, ellipse-corrected) | raw quadrature, lets the net learn its own demodulation |
| 2 | `Q` (DC-removed, ellipse-corrected) | same |
| 3 | `phi = unwrap(arctan2(Q, I))` | the baseline's only channel — keeps parity |
| 4 | `dy` = displacement in mm = `phi*lambda/(4*pi)` | physically scaled |
| 5 | `d(dy)/dt` — velocity | first mechanical derivative |
| 6 | `d2(dy)/dt2` — acceleration | **SCG analogue; the QRS-synchronous mechanical event lives here** |
| 7 | `A = sqrt(I^2+Q^2)` amplitude envelope | motion/posture/quality proxy, discarded by the baseline |
| 8 | band-limited cardiac residual (0.8–20 Hz, respiration removed) | separates cardiac from respiratory excursion |

Derivatives via Savitzky-Golay (order 3, window 9 at 128 Hz) so we differentiate without amplifying
noise. A learnable `1x1` mixing conv follows, so the network can rediscover arctangent demodulation
if that really is optimal — the ablation "channels 3 only" reproduces the baseline input exactly.

### C2 — Dual-domain encoder: MultiRes conv branch || learnable lifting-wavelet branch

- **Branch A (time):** the baseline's MultiRes blocks (3 parallel 3x3/5x5/7x7-equivalent 1-D convs,
  concatenated, plus a residual 1x1) — we keep this because it is a strong local morphology
  extractor, and keeping it makes the comparison honest.
- **Branch B (time-frequency):** a **learnable lifting-wavelet decomposition** (split -> predict ->
  update, 4 levels, CSConv predictors) in the spirit of LifWavNet, but used as a *parallel encoder*
  rather than the whole network. Gives adaptive multi-resolution analysis without a fixed basis.
- **Fusion:** at each of the 4/5 scales, concatenate A and B features and pass through a
  **cross-domain gated fusion** (SE / ECA channel attention producing per-channel gates), so the
  network chooses time vs. time-frequency evidence per scale and per scenario.

### C3 — Bi-directional Mamba (SSM) bottleneck for quasi-periodicity

At the bottleneck (sequence length 1024/2^5 = 32–64, channels 512–1024), stack **2–4 bidirectional
Mamba-2 blocks** (or a 4-layer pre-norm Transformer with RoPE as the fallback if `mamba-ssm` will
not build on Kaggle — see §8 troubleshooting). Linear-time, gives a global receptive field over the
whole 8-second window, and lets beat *n* inform beat *n+1*. **This is the first SSM applied to
radar->ECG waveform synthesis.**

### C4 — Multi-task decoder with peak-conditioned refinement

One shared LinkNet-style decoder (additive skips via ResPaths) with **three heads**:

1. **Waveform head** — the ECG, 1024 samples. `tanh` output on [-1,1].
2. **R-peak head** — a 1024-length probability map, supervised with a Gaussian-smoothed
   (sigma ≈ 3 samples) target around each ground-truth R peak. Focal BCE.
3. **Cycle-length head** — per-sample instantaneous RR interval (regression, in ms), which
   regularises the model toward physiological rhythm.

Then the novel bit: **peak-conditioned refinement.** The R-peak logits are fed back through a FiLM
layer (`gamma`, `beta` per channel) that modulates the *final* decoder block of the waveform head.
The model is explicitly told "a QRS belongs here" before it draws the waveform, which is exactly the
failure mode (missed / smeared beats) that G1 identified. Cheap, interpretable, and a clean ablation.

### C5 — Morphology-aware composite loss (kills the over-smoothing)

```
L = w1*Huber(y, y_hat)                       # temporal fidelity, robust to outliers
  + w2*MultiResSTFT(y, y_hat)                # spectral convergence + log-magnitude,
                                             #   windows 256 / 128 / 64 at 128 Hz
  + w3*FocalBCE(peak_map, peak_hat)          # R-peak localisation
  + w4*L1(rr, rr_hat)                        # cycle length
  + w5*PeakWeighted L1                       # per-sample weight 1 + 4*peak_map -> QRS costs 5x
  + w6*(1 - PearsonCC(y, y_hat))             # optimise the reported metric directly
  [+ w7*SoftDTW(gamma=0.1, band=+/-8 samples)]   # optional, absorbs electromechanical delay jitter
  [+ w8*AdversarialHinge(D(y_hat))]              # optional Stage-2 patch discriminator
```
Start with `w = [1.0, 0.5, 0.3, 0.1, 0.5, 0.3]`; tune `w2` and `w5` first — they are the ones that
sharpen the QRS. The adversarial term is a **stretch goal only**; add it after the deterministic
model is beating the baseline, never before.

### Optional C6 (stretch, high novelty / high risk) — few-step diffusion refinement + uncertainty
Stage 1 CardioMamba-Net gives a deterministic coarse ECG. Stage 2 is a small conditional
denoiser (rectified-flow / 4-step DDIM) conditioned on `[coarse ECG, bottleneck features, peak map]`
that restores sharp morphology. Sampling it *K = 20* times yields a **per-sample confidence band** —
"contactless ECG with calibrated uncertainty" is a genuinely new claim and is exactly what a
clinical reviewer wants. Do this **only if Stage 1 has already cleared the targets**, because
diffusion can hallucinate morphology and that is an ethical problem in a diagnostic signal.

### Budget
Target **< 5 M parameters** and **< 1.5 GFLOPs** per 8-second window, and report both — the baseline
paper reports neither, so a "we win *and* we are smaller" table is free credit.

---

## 6. Experiment matrix

Preprocessing is **frozen to the baseline's** (128 Hz, 1024-sample windows, 50 % train overlap,
0.5–40 Hz ECG band, 50 Hz notch, order-5 polynomial detrend, z-score then [0,1] range) so that any
gain is attributable to the model and not to a better pipeline. One deviation, declared in the
paper: we normalise the ECG target to **[-1, 1]** rather than [0, 1] so the network is not forced to
learn a DC offset (and we re-report the baseline under both, to be fair).

| ID | Experiment | Purpose | Baseline to beat |
|---|---|---|---|
| **A** | Per-scenario: Resting / Valsalva / Apnea, 5-fold | Direct comparison with their Table 2 | CC_t 66.1 / 60.1 / 55.3 |
| **B** | RVA combined, 5-fold | Direct comparison with their Table 3 | CC_t 61.86, CC_s 79.96 |
| **C** | **All five scenarios** (+ Tilt-up, Tilt-down) | New claim; nobody has done this | none — we define it |
| **D** | **LOSO** (30 folds, resting; 24–27 for the rest) | Honest cross-subject generalisation; tests the leakage suspicion | none — we define it |
| **E** | Robustness: AWGN at +6/+3/0/-3 dB, random body-motion injection, 10 % sample dropout | Deployment realism (radarODE-MTL set this precedent) | none |
| **F** | Cross-scenario transfer: train Resting -> test Valsalva/Apnea/Tilt | Shows the model is not memorising a posture | none |

### Baselines to re-implement and run ourselves (do not just quote their table)
FPN-1D, UNet-1D, LinkNet-1D, **MultiResLinkNet** (faithful reimplementation), Attention-UNet-1D,
and if time permits a LifWavNet-style lifting network. Identical data, identical folds, identical
seeds. **Quoting their numbers alone is the fastest way to get rejected** — a reviewer will ask why
our reimplementation differs. Report both "as published" and "our run".

### Ablation table (one row each, all on Experiment B)
1. MultiResLinkNet + MSE (= the baseline, our run)
2. + composite loss (C5) only
3. + multi-channel input (C1) only
4. + wavelet branch (C2)
5. + Mamba bottleneck (C3)
6. + multi-task heads (C4), no FiLM
7. + peak-conditioned FiLM refinement (full model)
8. full model + SSL pretraining
9. full model + diffusion refinement (if C6 is attempted)
10. Transformer bottleneck instead of Mamba (fair-fight control)

### Metrics — everything the baseline reported, plus what it should have
- **Theirs (mandatory for comparability):** MAE, MSE, CC_temporal, CC_spectral, RRMSE_temporal,
  RRMSE_spectral, peak Accuracy/Precision/Recall/F1, mu_RR, sd_RR, mu_HR, sd_HR, RMSSD.
- **Ours (new):** R-peak timing error in ms (median + IQR), missed-detection rate,
  **Bland-Altman for HR and RMSSD with limits of agreement**, R^2, MAE_HR, MAE_RMSSD,
  **Wilcoxon signed-rank + Holm correction** across folds for every model pair,
  parameters / FLOPs / inference latency per window, and a per-subject box plot of CC_temporal.
- **Fix their bug:** report mu_RR in real milliseconds.

---

## 7. Compute plan (Kaggle 2x T4, 20 GB disk, HF checkpointing)

This project is **compute-light** — the bottleneck is preprocessing I/O, not GPU.

**Stage 0 — one-off preprocessing notebook.**
Download the Kaggle mirror -> parse `.mat` per subject/scenario -> derive the 8 channels -> decimate
to 128 Hz -> window -> save as **sharded `.npz` or a single HDF5** (~350 MB) -> **push the processed
corpus to a Hugging Face dataset repo `Shanmuk4622/cr-rvs-radar-ecg-processed`**. Every later run
downloads that in seconds instead of re-parsing 8 GB of `.mat`. Do this once.

**Stage 1 — training runs.**
~13 k segments of 1024 samples, model < 5 M params. On one T4 an epoch is a few seconds; 300 epochs
is minutes. The full matrix (6 models x 4 scenario settings x 5 folds ≈ 120 runs) plus 30-fold LOSO
is a handful of 12-hour sessions, not weeks. Use `torch.nn.DataParallel` or DDP across the two T4s
only for the LOSO sweep; a single T4 is enough for one run.

**Resumability + HF discipline (your standing requirements, restated as build rules):**
- Every run writes `state.pt` = `{epoch, model, optimizer, scheduler, scaler, rng_states, fold,
  best_metric, history}` to local disk **every epoch**.
- A **background uploader thread** flushes to HF **at most once per 30 min**, and additionally on:
  (a) a new best validation metric, (b) end of a fold, (c) `KeyboardInterrupt` / `SIGINT` /
  `SIGTERM`, (d) `atexit`. Implemented as an `atexit` + `signal` handler around a `threading.Event`.
- HF write budget is ~128 requests/hour: batch every flush into **one** `upload_folder` call with a
  `commit_message` carrying epoch + metrics; keep a token-bucket limiter (`128/h`, refill 1 per
  28 s) with exponential backoff on `429`.
- On startup the notebook **always** tries `snapshot_download` of the run's HF folder first and
  resumes from the exact epoch — restarting the Kaggle session must cost nothing.
- Log to a single append-only `history.jsonl` pushed with the checkpoint so no metric is ever lost.
- `HF_TOKEN` from Kaggle Secrets, never inline.

---

## 8. Known risks and pre-planned troubleshooting

| Risk | Mitigation, decided in advance |
|---|---|
| Kaggle mirror is not the raw `.mat` tree | Step 0 verifies with `kaggle datasets files`; fall back to the figshare DOI or a second mirror |
| `mamba-ssm` needs `nvcc` compilation and often fails on Kaggle | Ship a **pure-PyTorch bidirectional S6/S4D fallback** (~40 lines, no custom CUDA) and a Transformer+RoPE control. Choose at config level, not by editing code |
| Sync offsets in some `.mat` files (the paper mentions ~54 s lags) | Cross-check with `tfm_intervention`; drop or re-align any subject whose radar/ECG cross-correlation peak is implausible; log every exclusion |
| Ellipse fitting on I/Q is fiddly | Use a direct least-squares conic fit (Fitzgibbon), unit-test it on a synthetic ellipse before trusting it; if it fails for a file, fall back to simple DC removal and flag the file |
| Apnea has only 1140 segments (24 subjects) — easy to overfit | Train on RVA-combined and fine-tune per scenario; report both. Use the SSL-pretrained encoder here especially |
| Over-claiming clinical utility | Frame as *waveform/rhythm* reconstruction, never diagnosis. Explicitly state that morphology (ST, T-wave) is not validated for diagnostic use |
| Diffusion stage hallucinating beats | Only report it with the uncertainty band; anchor every sample to the deterministic Stage-1 output |

---

## 9. Timeline (aggressive but realistic)

| Week | Deliverable |
|---|---|
| 1 | ~~Step 0 verification; preprocessing notebook; processed corpus on HF~~ **DONE** — NB01 + NB02 complete, corpus public on HF |
| 2 | Baseline reimplementations (FPN/UNet/LinkNet/MultiResLinkNet) reproducing their Table 2/3 within tolerance. **This is the gate — no new architecture until the baseline reproduces.** |
| 3 | Composite loss (C5) + multi-channel input (C1) on the baseline backbone. Expect the first clear win here |
| 4 | Full CardioMamba-Net (C2+C3+C4); Experiments A and B |
| 5 | Experiments C, D, E, F + full ablation grid |
| 6 | SSL pretraining, optional diffusion stage, statistical tests, all figures |
| 7-8 | Write-up, cover letter, submission |

**Target venues:** *Computers in Biology and Medicine* (same venue as the baseline — reviewers
already know the problem), *Biomedical Signal Processing and Control*, *IEEE JBHI*,
*IEEE Trans. Instrumentation & Measurement*. Companion preprint on arXiv (eess.SP).

---

## 10. Paper skeleton

1. **Introduction** — contactless monitoring; why electrodes fail (skin irritation, motion artefact, ICU/burn/neonate); contributions as 5 bullets.
2. **Related work** — radar vital signs; radar->ECG (Chowdhury 2024, radarODE / radarODE-MTL, LifWavNet, RF2ESG, Air-ECG); SSMs for biosignals; multi-task signal translation.
3. **Dataset and preprocessing** — CR-RVS; the frozen pipeline; the 8-channel physics-informed representation (Fig. 2).
4. **Method** — dual-domain encoder, SSM bottleneck, multi-task decoder with peak-conditioned FiLM, composite loss (Fig. 3 = the architecture figure, Fig. 4 = the loss diagram).
5. **Experiments** — A–F, baselines, folds, hyperparameters, hardware.
6. **Results** — Tables mirroring their 2/3/4/5 plus ablation, LOSO, robustness, Bland-Altman, per-subject box plots, qualitative overlays for all five scenarios.
7. **Discussion** — *why* it works (point at the ablation), the over-smoothing/RMSSD story, the leakage/protocol correction, limitations (healthy young cohort, no arrhythmia, single radar geometry).
8. **Conclusion + future work** — arrhythmic cohorts, mmWave transfer, on-device deployment.

**Figures to budget for:** system overview; 8-channel input example; architecture; loss; qualitative
reconstruction grid (5 scenarios x 4 models); ablation bar chart; Bland-Altman x2; per-subject box
plot; robustness curve vs SNR; parameters-vs-CC scatter.

---

## 11. Immediate next actions (waiting on your go-ahead)

1. ~~Decide the architecture tier~~ — **LOCKED: full CardioMamba-Net (C1–C5)**, C6 parked as a stretch. Baselines: **all four reimplemented by us.** See `DECISIONS.md`.
2. **Step 0**: verify what the Kaggle mirror actually contains.
3. Then I build the notebooks in `03_notebooks/` in this order:
   `01_verify_and_download.ipynb` -> `02_preprocess_to_hf.ipynb` -> `03_baselines.ipynb` ->
   `04_cardiomamba_train.ipynb` -> `05_evaluate_and_figures.ipynb`,
   all Kaggle-ready with the resumable HF checkpointing contract from §7.
