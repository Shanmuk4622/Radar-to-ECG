# Manuscript package

The complete manuscript is in **`latex/main.tex`**. Its compiled version is **`latex/main.pdf`**.
The portable source package is **`radar_ecg_manuscript.zip`**; upload it to Overleaf and select
`main.tex` as the main document. Author, affiliation, funding, competing-interest and contribution
fields are intentionally blank at the author's request.

## Contents

- `latex/main.tex`: complete original manuscript, including abstract, related work, implemented
  methods, all primary results, discussion, limitations, declarations and five technical appendices.
- `latex/references.bib`: 25 references checked against Crossref, original arXiv records, PubMed
  and the original journal record as appropriate.
- `latex/generated/`: tables, participant effects, bootstrap intervals, sampling-rate audit and
  reproducibility checks generated from the archived evidence.
- `latex/figures/`: five newly plotted scientific figures in vector PDF and high-resolution PNG;
  the architecture diagram is native TikZ in the manuscript.
- `evidence/`: 99 additional pinned Hugging Face files plus reference metadata.
- `tools/`: evidence retrieval, reference verification, analysis and manuscript validation.
- `MANUSCRIPT_AUDIT.md`: newly discovered scientific issues and their publication implications.
- `REFERENCE_CHECK.md`: the scope and results of reference checking.

## Compile

Overleaf: upload the ZIP, select `main.tex`, and compile with **XeLaTeX** (recommended) or pdfLaTeX.
The manuscript uses standard packages and numeric BibTeX citations. No external file paths,
shell escape, network access or credentials are needed after upload.

With a local TeX installation, run from `latex/`:

```text
latexmk -xelatex main.tex
```

Alternatively, run the portable Tectonic compiler provided locally under `.tools/`:

```text
../.tools/tectonic.exe --keep-logs --keep-intermediates main.tex
```

The source is formatted as a clean single-column journal-review manuscript. Journal-specific
class, page layout and declarations can be substituted after choosing the destination. “Q1” is
a journal ranking, not a guarantee of acceptance or a distinct LaTeX format.

## Scientific status

This is a complete manuscript of the experiment that actually ran. It is **not ready for submission
as a validated clinical or SOTA study**. The paper-preparation audit found a confirmed 125-Hz stream
labelled as 128 Hz; the physiological results need reevaluation. It also confirmed that the model
is S4D-based, not selective Mamba, and that the stored FLOP estimates have unverified operator
coverage. These findings are disclosed in the paper and in `MANUSCRIPT_AUDIT.md`.

The source uses original prose with attributed prior work. No external plagiarism-database or
AI-detector certification was performed. A limited local phrase-overlap check, if reported by
the QA tool, is only a comparison against the two supplied paper texts and is not such a certification.

## Regenerate the evidence-based assets

From the project root, using Python with requests, pandas, NumPy, SciPy, Matplotlib and PyArrow:

```text
python 07_paper/tools/prepare_evidence.py
python 07_paper/tools/verify_references.py
python 07_paper/tools/build_paper_assets.py
```

The download script uses public, immutable HF revisions and never sends a token or writes to HF.
The analysis reproduces the nine archived Wilcoxon/Holm comparisons, computes clearly labelled
conditional participant-bootstrap intervals, and checks all 134 recording lengths. The authoring
task does not alter or retrain any model and does not overwrite the original evaluation archive.
