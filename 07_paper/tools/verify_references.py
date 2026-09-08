"""Resolve every reference from Crossref or original arXiv citation metadata."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import html
import json
import re
import requests
import time

def get(url):
    for attempt in range(5):
        r=requests.get(url,timeout=60,headers={'User-Agent':'RadarECG-Manuscript-ReferenceAudit/1.0'})
        if r.status_code in (429,500,502,503,504):
            time.sleep(min(30,3*2**attempt)); continue
        r.raise_for_status()
        return r
    r.raise_for_status()

ROOT = Path(__file__).resolve().parents[1]
DOIS = {
 'schellenberger2020':'10.1038/s41597-020-00629-5',
 'chowdhury2024':'10.1016/j.compbiomed.2024.108555',
 'unet':'10.1007/978-3-319-24574-4_28',
 'fpn':'10.1109/CVPR.2017.106',
 'linknet':'10.1109/VCIP.2017.8305148',
 'multiresunet':'10.1016/j.neunet.2019.08.025',
 'film':'10.1609/aaai.v32i1.11671',
 'eca':'10.1109/CVPR42600.2020.01155',
 'focal':'10.1109/ICCV.2017.324',
 'wavegan':'10.1109/ICASSP40776.2020.9053795',
 'lifting':'10.1137/S0036141095289051',
 'welch':'10.1109/TAU.1967.1161901',
 'wilcoxon':'10.2307/3001968',
 'blandaltman':'10.1016/S0140-6736(86)90837-8',
 'hrvstandards':'10.1161/01.CIR.93.5.1043',
 'hrvoverview':'10.3389/fpubh.2017.00258',
 'huber':'10.1214/aoms/1177703732',
}
ARXIV = {'s4d':'2206.11893', 'mamba':'2312.00752', 'radarode':'2408.01672',
         'radarodemt':'2410.08656', 'lifwavnet':'2510.27692',
         'adamw':'1711.05101', 'attention':'1706.03762'}

def tex(s):
    s = re.sub('<[^>]+>', '', html.unescape(str(s)))
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:
        s=s.replace(a,b)
    return s

def resolve(item):
    key, ident, kind = item
    cache = ROOT/'evidence'/'references'/f'{key}.json'
    if cache.exists():
        data=json.loads(cache.read_text(encoding='utf-8'))
    elif kind=='doi':
        url='https://api.crossref.org/works/'+ident
        r=get(url)
        m=r.json()['message']
        authors=[' '.join([a.get('family',''),',',a.get('given','')]).replace(' ,',',')
                 if a.get('family') else a.get('name','') for a in m.get('author',[])]
        data=dict(key=key,doi=ident,title=m['title'][0],authors=authors,
                  year=m.get('published',m.get('issued'))['date-parts'][0][0],
                  venue=m.get('container-title',[''])[0],volume=m.get('volume',''),
                  pages=m.get('page',m.get('article-number','')),source=url,type=m.get('type',''))
    else:
        url='https://arxiv.org/abs/'+ident
        r=get(url)
        def meta(name):
            return [html.unescape(x) for x in re.findall(r'<meta\s+name="'+name+r'"\s+content="([^"]+)"',r.text)]
        data=dict(key=key,doi='10.48550/arXiv.'+ident,title=meta('citation_title')[0],
                  authors=meta('citation_author'),year=int(meta('citation_date')[0][:4]),
                  venue='arXiv preprint arXiv:'+ident,volume='',pages='',source=url,type='preprint')
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    return data

def main():
    items=[(k,v,'doi') for k,v in DOIS.items()]+[(k,v,'arxiv') for k,v in ARXIV.items()]
    with ThreadPoolExecutor(max_workers=1) as pool:
        refs=list(pool.map(resolve,items))
    refs.append(dict(key='holm',doi='',title='A Simple Sequentially Rejective Multiple Test Procedure',
                     authors=['Holm, Sture'],year=1979,venue='Scandinavian Journal of Statistics',
                     volume='6',pages='65--70',source='https://www.jstor.org/stable/4615733',type='journal-article'))
    for d in refs:
        if d['key']=='hrvstandards':
            d['authors']=['{Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology}']
            d['title']='Heart rate variability: standards of measurement, physiological interpretation and clinical use'
            d['verification_note']='Expanded corporate author and title checked against PubMed PMID 8598068.'
        if d['key']=='blandaltman':
            d['authors']=['Bland, J. Martin','Altman, Douglas G.']
            d['title']='Statistical methods for assessing agreement between two methods of clinical measurement'
            d['verification_note']='Normalised author fields checked against PubMed PMID 2868172.'
    entries=[]
    for d in refs:
        typ='inproceedings' if d['type'] in ('proceedings-article','book-chapter') else 'article'
        fields={'author':' and '.join(d['authors']), 'title':'{'+tex(d['title'])+'}',
                'year':d['year'], 'booktitle' if typ=='inproceedings' else 'journal':tex(d['venue']),
                'doi':d['doi']}
        if not d['doi']:
            fields.pop('doi'); fields['url']=d['source']
        for name in ('volume','pages'):
            if d[name]: fields[name]=d[name]
        if not d['authors']:
            fields['author']='{Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology}' if d['key']=='hrvstandards' else '{Anonymous}'
        entries.append('@'+typ+'{'+d['key']+',\n'+',\n'.join('  '+k+' = {'+str(v)+'}' for k,v in fields.items())+'\n}')
        print(d['key'],d['year'],d['title'],d['authors'][:2],flush=True)
    draft=ROOT/'latex'; draft.mkdir(parents=True,exist_ok=True)
    (draft/'references.bib').write_text('\n\n'.join(entries)+'\n',encoding='utf-8')
    (ROOT/'evidence/reference_audit.json').write_text(json.dumps(refs,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__': main()
