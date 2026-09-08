# Manuscript validation record

Final authoring review: 2026-09-07.

- Complete 22-page LaTeX manuscript, including five technical appendices.
- Six figures: native architecture diagram plus five new scientific vector plots.
- Six tables, including every primary variant and all nine specified paired comparisons.
- All 25 bibliography entries are cited; no missing or duplicate citation keys.
- Cross-references resolve and the compiled TeX log has no overfull boxes, missing characters,
  undefined citations, undefined references or LaTeX errors.
- Every page was rendered for visual review. Diagram labels and waveform legends were adjusted
  after that review. No text extends outside the PDF page bounds.
- Nine Wilcoxon and Holm-adjusted probabilities reproduced from downloaded per-window metrics
  within numerical tolerance 1e-12.
- 20,000 participant bootstrap replicates generated with a fixed seed; intervals explicitly
  labelled conditional, retrospective and unadjusted.
- 134/134 processed recording lengths matched 125-Hz decimation, documenting the timing flaw.
- Matched qualitative examples checked for identical reference arrays and participant IDs
  across displayed models, after conversion to the common target scale.
- Limited local originality diagnostic: no exact 12-word overlap between manuscript body prose
  and either of the two supplied paper texts under lowercase alphabetic-token matching. This is
  not an external plagiarism-database check or an AI-detector certification.
- Author-specific fields remain intentionally blank, as requested.

`qa/validation.json` stores the mechanical checks and final PDF hash. `MANUSCRIPT_PACKAGE.json`
stores the source-package contents and checksums. `MANUSCRIPT_AUDIT.md` distinguishes scientific
limitations from successful typesetting and artifact validation.
