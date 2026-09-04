"""Does eebls_gpu's np.dot (BLAS) still trip CFS throttling on a CPU-quota-limited container? Run with/without OPENBLAS_NUM_THREADS=1."""
import os, time, sys, json
import numpy as np
from prof_common import *
from cuvarbase.bls import eebls_gpu, compile_bls, eebls_gpu_custom
cfg = SURVEYS['TESS']; freqs, qmins, qmaxs = grid_for(cfg); t, y, dy = make_lc(cfg, 3)
fr = compile_bls()
th0 = throttle()
m, allt = med(lambda: eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, functions=fr, max_memory=1.5e9), 5)
th1 = throttle()
# isolate the host prologue that eebls_gpu runs (np.dot on 20K-vectors)
def prologue():
    w = np.power(dy, -2); w /= np.sum(w); ybar = np.dot(w, y); YY = np.dot(w, np.power(np.array(y) - ybar, 2)); return YY
th2 = throttle(); mp, _ = med(prologue, 50); th3 = throttle()
print(json.dumps(dict(env_OPENBLAS=os.environ.get('OPENBLAS_NUM_THREADS'), eebls_gpu_median_ms=m*1e3, all_ms=[x*1e3 for x in allt],
      throttle_delta_call=[a-b for a, b in zip(th1, th0)], prologue_np_dot_ms=mp*1e3, throttle_delta_prologue=[a-b for a, b in zip(th3, th2)]), indent=1))
