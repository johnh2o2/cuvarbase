#!/usr/bin/env python3
import json, os, sys, time, uuid
D = '/workspace/scratch/nvccq'; k = D + '/' + uuid.uuid4().hex
json.dump(dict(argv=sys.argv[1:], cwd=os.getcwd()), open(k + '.tmpreq', 'w')); os.rename(k + '.tmpreq', k + '.req')
while not os.path.exists(k + '.resp'): time.sleep(0.05)
r = json.load(open(k + '.resp')); os.remove(k + '.resp')
sys.stdout.write(r['out']); sys.stderr.write(r['err']); sys.exit(r['rc'])
