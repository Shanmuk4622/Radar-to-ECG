"""Download pinned, public paper evidence without authentication or remote writes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import requests
import time
import zipfile
import io

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '07_paper' / 'evidence'
REPOS = {
    'cardiomamba': ('model', 'cardiomamba-net-v2', 'dd5764c51031f5d9415ecddbe64ff10ab0d02348'),
    'baselines': ('model', 'cardiomamba-baselines-v2', '84bf3e7992a33aecacfb30ab860872c620ca7699'),
    'inventory': ('dataset', 'cr-rvs-radar-ecg-inventory-v2', 'ef9d72a2dbe5e185644be0c8cd44bb13ab2f3727'),
}

def fetch(job):
    group, name = job
    kind, repo, revision = REPOS[group]
    prefix = 'datasets/' if kind == 'dataset' else ''
    url = f'https://huggingface.co/{prefix}Shanmuk4622/{repo}/resolve/{revision}/{name}'
    dest = OUT / group / name
    if not dest.exists():
        for attempt in range(4):
            r = requests.get(url, timeout=90)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, int(r.headers.get('Retry-After', 5 * 2**attempt))))
                continue
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            break
        else:
            raise RuntimeError(url)
    payload = dest.read_bytes()
    return dict(path=str(dest.relative_to(OUT)).replace('\\', '/'), url=url,
                bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

def main():
    jobs = []
    provenance = ROOT / '06_results/hf_final_snapshot/provenance'
    selected = {'L9_full', 'L8_no_film', 'L7_singletask', 'multireslinknet'}
    for group in ('cardiomamba', 'baselines'):
        for config in sorted((provenance / group / 'runs').glob('B_rva__*/run_config.json')):
            run = config.parent.name
            jobs.append((group, f'runs/{run}/metrics_windows.parquet'))
            if run.split('__')[1] in selected:
                jobs.append((group, f'runs/{run}/metrics_subjects.parquet'))
                if run.endswith('__f0'):
                    jobs.append((group, f'runs/{run}/preds_sample.npz'))
        for module in ('crvs_models.py', 'crvs_engine.py'):
            jobs.append((group, module))
    for name in ('crvs_cmnet.py', 'crvs_data.py', 'crvs_losses.py', 'crvs_metrics.py'):
        jobs.append(('cardiomamba', name))
    jobs.extend(('inventory', name) for name in ('inventory.csv', 'run_manifest.json'))
    with ThreadPoolExecutor(max_workers=3) as executor:
        entries = list(executor.map(fetch, jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'download_manifest.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    print(f'Downloaded/verified {len(entries)} pinned evidence files', flush=True)
    tool = ROOT / '07_paper' / '.tools' / 'tectonic.exe'
    if not tool.exists():
        url = 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip'
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            binary = next(n for n in z.namelist() if n.endswith('tectonic.exe'))
            tool.parent.mkdir(parents=True, exist_ok=True)
            tool.write_bytes(z.read(binary))
    print('Portable LaTeX compiler ready', flush=True)

if __name__ == '__main__':
    main()
