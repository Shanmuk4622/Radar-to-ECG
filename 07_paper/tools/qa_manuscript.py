"""Read-only source/PDF checks and rendered review sheets; no scientific edits."""
from pathlib import Path
import json
import re
import hashlib
import pdfplumber
import subprocess
from PIL import Image, ImageOps, ImageDraw

PAPER=Path(__file__).resolve().parents[1]
TEX=PAPER/'latex'
QA=PAPER/'qa'

def words(text):
    return re.findall(r'[a-z]+',text.lower())

def main():
    QA.mkdir(parents=True,exist_ok=True)
    source=(TEX/'main.tex').read_text(encoding='utf-8')
    bib=(TEX/'references.bib').read_text(encoding='utf-8')
    cited={k.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}',source) for k in block.split(',')}
    keys=re.findall(r'@\w+\{([^,]+),',bib)
    assert not (cited-set(keys)),f'Missing citations {cited-set(keys)}'
    assert len(keys)==len(set(keys))==25
    assert not (set(keys)-cited),f'Unused references {set(keys)-cited}'
    labels=re.findall(r'\\label\{([^}]+)\}',source)
    refs=re.findall(r'\\ref\{([^}]+)\}',source)
    assert set(refs)<=set(labels) and len(labels)==len(set(labels))
    log=(TEX/'main.log').read_text(encoding='utf-8',errors='replace')
    problems=[x for x in log.splitlines() if any(s in x for s in ('Overfull','undefined','Missing character','LaTeX Error'))]
    assert not problems,problems
    poppler=Path('C:/Users/shanm/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
    subprocess.run([str(poppler),'-r','94','-png',str(TEX/'main.pdf'),str(QA/'page')],check=True)
    doc=pdfplumber.open(TEX/'main.pdf'); page_reports=[]; thumbs=[]
    rendered=sorted(QA.glob('page-*.png'))
    for i,page in enumerate(doc.pages):
        im=Image.open(rendered[i]).convert('RGB')
        im.thumbnail((410,580))
        tile=Image.new('RGB',(430,620),'#e7e9ec');tile.paste(im,((430-im.width)//2,23))
        ImageDraw.Draw(tile).text((12,600),f'Page {i+1}',fill='black');thumbs.append(tile)
        txt=page.extract_text() or ''
        out=[b for b in page.chars if b['x0']<0 or b['top']<0 or b['x1']>page.width+1 or b['bottom']>page.height+1]
        page_reports.append(dict(page=i+1,words=len(txt.split()),out_of_page_blocks=len(out)))
    for batch in range(0,len(thumbs),6):
        sheet=Image.new('RGB',(1290,1240),'white')
        for j,tile in enumerate(thumbs[batch:batch+6]):sheet.paste(tile,((j%3)*430,(j//3)*620))
        sheet.save(QA/f'contact-{batch//6+1}.png')
    text='\n'.join(p.extract_text() or '' for p in doc.pages)
    (QA/'extracted.txt').write_text(text,encoding='utf-8')
    # A deliberately limited overlap diagnostic, not a plagiarism certificate.
    body=text.split('\nReferences\n')[0]
    ww=words(body); n=12; ours={' '.join(ww[i:i+n]) for i in range(len(ww)-n+1)}
    overlaps={}
    for path in (PAPER.parent/'01_literature/extracted_text').glob('*.txt'):
        sw=words(path.read_text(encoding='utf-8',errors='replace'))
        theirs={' '.join(sw[i:i+n]) for i in range(len(sw)-n+1)}
        overlaps[path.name]=sorted(ours&theirs)
    report=dict(pdf_pages=len(doc.pages),pdf_words_including_refs_appendices=len(text.split()),
                cited_references=len(cited),undefined_citations=0,latex_critical_warnings=problems,
                pages=page_reports,local_12_word_overlap=overlaps,
                pdf_sha256=hashlib.sha256((TEX/'main.pdf').read_bytes()).hexdigest())
    (QA/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
