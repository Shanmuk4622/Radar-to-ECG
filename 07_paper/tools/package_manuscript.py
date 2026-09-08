"""Package an already-validated, self-contained LaTeX source tree."""
from pathlib import Path
import hashlib
import json
import zipfile

PAPER=Path(__file__).resolve().parents[1]
TEX=PAPER/'latex'
validation=json.loads((PAPER/'qa/validation.json').read_text(encoding='utf-8'))
assert validation['pdf_sha256']==hashlib.sha256((TEX/'main.pdf').read_bytes()).hexdigest()
assert not validation['latex_critical_warnings']
assert all(p['out_of_page_blocks']==0 for p in validation['pages'])
files=[TEX/n for n in ['main.tex','references.bib','main.bbl','main.pdf','README.md']]
files+=sorted((TEX/'generated').glob('*'))
files+=sorted((TEX/'figures').glob('*.pdf'))
archive=PAPER/'radar_ecg_manuscript.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(TEX).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names=set(z.namelist())
    assert {'main.tex','references.bib','main.bbl','main.pdf'}<=names
manifest=dict(archive=archive.name,pdf_pages=validation['pdf_pages'],
              references=validation['cited_references'],
              zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              files=[dict(path=p.relative_to(TEX).as_posix(),bytes=p.stat().st_size,
                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
(PAPER/'MANUSCRIPT_PACKAGE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(f'Packaged {len(files)} files, {archive.stat().st_size:,} bytes; ZIP CRC verified.')
