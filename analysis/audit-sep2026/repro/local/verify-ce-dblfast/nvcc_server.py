import json, os, subprocess, time, glob, sys
D = '/workspace/scratch/nvccq'; os.makedirs(D, exist_ok=True)
stop = D + '/STOP'
while not os.path.exists(stop):
    for req in sorted(glob.glob(D + '/*.req')):
        j = json.load(open(req)); os.remove(req)
        r = subprocess.run(['/usr/local/cuda-12.4/bin/nvcc'] + j['argv'], cwd=j['cwd'], capture_output=True, text=True)
        json.dump(dict(rc=r.returncode, out=r.stdout, err=r.stderr), open(req[:-4] + '.tmp', 'w')); os.rename(req[:-4] + '.tmp', req[:-4] + '.resp')
    time.sleep(0.05)
