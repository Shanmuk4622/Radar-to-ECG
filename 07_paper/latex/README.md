# Radar-to-ECG manuscript source

Open **main.tex** and compile with **XeLaTeX + BibTeX**, or choose XeLaTeX in Overleaf.
The compiled review copy is **main.pdf**.

This package is self-contained. `generated/` holds data-derived table fragments and analysis
outputs; `figures/` holds vector scientific figures. `references.bib` contains 25 checked entries,
and `main.bbl` is included for convenience. No credentials, network downloads or shell escape
are required to compile the supplied manuscript.

Author and affiliation fields, funding, competing interests and contribution statements are
intentionally blank. The AI-assistance statement must be reviewed along with the author-specific
declarations before submission.

Scientific status: the paper describes a completed experiment and its audit. It discloses a
confirmed 125-Hz stream labelled 128 Hz; the archived physiological outputs require reevaluation.
The full model uses bidirectional S4D rather than selective Mamba. These are substantive findings,
not typographical placeholders. The manuscript does not claim clinical validation or new SOTA.
