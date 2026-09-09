#!/usr/bin/env python3
import json,tarfile
from pathlib import Path
root=Path('/tmp/cuvarbase-tls-profile/recovery')
rows=[]
with tarfile.open(root/'summaries.tar.gz','w:gz') as tar:
    for p in root.rglob('*'):
        if p.is_file() and (p.suffix in ['.json','.log','.txt'] or p.parent.name=='scripts'):
            tar.add(p,arcname=str(p.relative_to(root)))
    for p in sorted((root/'results').glob('*/summary.json')):
        d=json.loads(p.read_text());cases=d.get('cases',[])
        rows.append(dict(job=p.parent.name,status=d['status'],n=len(cases),
            recall=sum(c['recovered'] is True for c in cases),injections=sum(c['injected'] for c in cases),
            mean_s=sum(c['search_s'] or 0 for c in cases)/len(cases) if cases else None,
            timing=d.get('seconds_per_source'),error=d.get('error','')[-500:]))
(root/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
