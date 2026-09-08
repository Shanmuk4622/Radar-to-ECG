# Reference verification

Checked on 2026-09-07. The bibliography contains **25 cited references**.

## Verification method

- Seventeen DOI references were resolved against Crossref records for title, authors, year,
  publication and DOI. The local baseline and CR-RVS paper texts were also read for the
  experimental context and acquisition ethics.
- Seven references use the original arXiv records. Their author lists, titles, submission years
  and arXiv DOIs were obtained from citation metadata, and their relevant abstracts were checked.
  They are explicitly cited as preprints; this does not imply that no later journal version exists.
- Holm's 1979 article was checked through the original JSTOR issue/article record and is cited by
  its stable URL. Crossref did not resolve the often-assumed DOI `10.2307/4615733`, so that DOI is
  deliberately not placed in the bibliography.
- The HRV standards corporate author and expanded title were checked against PubMed PMID 8598068.
- Bland and Altman's author-name formatting was checked against PubMed PMID 2868172.
- Machine-readable source records are in `evidence/reference_audit.json` and `evidence/references/`.

## Citation purpose

| Key | Source and role |
|---|---|
| schellenberger2020 | CR-RVS acquisition, cohort, reference synchronisation and original ethics |
| chowdhury2024 | Published MultiResLinkNet comparator and its reported benchmark numbers |
| radarode | ODE-embedded radar reconstruction; conceptual related work only |
| radarodemt | Multi-task radar reconstruction and gradient alignment; conceptual related work only |
| lifwavnet | Prior learned lifting/STFT approach; no numerical comparison claimed |
| s4d | Correct foundation for the implemented diagonal state-space layer |
| mamba | Distinguishes selective dynamics from the implemented static S4D kernel |
| unet, linknet, fpn | Original architectural families adapted by the baseline reimplementations |
| multiresunet | Multiresolution convolutional-block antecedent |
| lifting | Prediction/update construction of lifting transforms |
| film | Feature-wise affine conditioning antecedent |
| eca | Efficient channel-attention antecedent |
| attention | Transformer control architecture |
| adamw | Decoupled weight-decay optimiser |
| huber | Robust error objective |
| wavegan | Multiresolution spectral-loss antecedent |
| focal | Focal classification-loss antecedent |
| welch | Windowed spectral estimation |
| wilcoxon, holm | Paired ranked test and multiple-test correction |
| blandaltman | Descriptive measurement-agreement plots |
| hrvstandards, hrvoverview | HRV definitions and physiological interpretation |

Metadata verification confirms reference identity and bibliographic details. It does not imply
that every referenced paper was independently reproduced. The manuscript uses these sources
only for the claims stated above and clearly identifies methods that were not rerun.

No external similarity database or plagiarism service was available. The prose was newly written
from the verified experiment and attributed sources. No AI-detector score or universal absence
of textual overlap is asserted.
