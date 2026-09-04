"""eebls_gpu (eebls_transit default for ndata>=500) reduces per-frequency q arrays to batch-wide [min qmin, max qmax]:
count reported solutions outside the caller's per-frequency Keplerian bounds and compare power with eebls_gpu_fast."""
import numpy as np, json
from prof_common import *
from cuvarbase.bls import eebls_gpu, eebls_gpu_fast, compile_bls
cfg = SURVEYS['TESS']; freqs, qmins, qmaxs = grid_for(cfg); t, y, dy = make_lc(cfg, 21)
fr = compile_bls(); out = {}
p_std, sols = eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, functions=fr, max_memory=1.5e9)   # one batch at this size
q_sol = np.array([q for q, phi in sols])
outside = (q_sol < qmins*(1-1e-6)) | (q_sol > qmaxs*(1+1e-6))
p_fast = eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxs, noverlap=3, dlogq=0.2)
out['nfreq'] = len(freqs); out['n_solutions_outside_per_freq_bounds'] = int(outside.sum()); out['frac_outside'] = float(outside.mean())
out['batch_wide_q_range'] = [float(qmins.min()), float(qmaxs.max())]; out['per_freq_q_range_examples'] = [[float(qmins[i]), float(qmaxs[i])] for i in (0, len(freqs)//2, -1)]
out['q_sol_min_max'] = [float(q_sol.min()), float(q_sol.max())]
out['power_std_vs_fast'] = dict(corr=float(np.corrcoef(p_std, p_fast)[0,1]), argmax_same=bool(np.argmax(p_std)==np.argmax(p_fast)),
                                n_std_gt_fast_by_1pct=int(np.sum(p_std > 1.01*p_fast + 1e-4)), max_ratio=float(np.max((p_std+1e-9)/(p_fast+1e-9))))
out['peak_P_std'] = float(1/freqs[np.argmax(p_std)]); out['peak_P_fast'] = float(1/freqs[np.argmax(p_fast)])
print(json.dumps(out, indent=1)); json.dump(out, open('/workspace/scratch/qrange_check.json','w'), indent=1)
