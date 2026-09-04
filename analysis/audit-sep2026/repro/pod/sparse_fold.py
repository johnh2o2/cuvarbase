"""float32 fold precision in sparse BLS at long baselines: power at the true
frequency and the grid maximum, float32-fold implementations vs float64
reference; plus float32 frequency quantization vs Keplerian grid spacing."""
import numpy as np
from cuvarbase.bls import sparse_bls_cpu, sparse_bls_gpu, compile_sparse_bls
from sparse_exp import ref_sparse, make_lc

kern = compile_sparse_bls(block_size=64)
for base, f_inj, q_inj, N in ((3650.0, 4.7, 0.01, 300), (3650.0, 1.0, 0.02, 300), (1000.0, 4.7, 0.01, 300), (365.0, 4.7, 0.01, 300), (365.0, 1.0, 0.03, 100)):
    d_at, d_max, npts_diff, set_diff = [], [], 0, 0
    for seed in range(16):
        t, y, dy = make_lc(N, 'uniform', seed=seed, baseline=base, q=q_inj, f=f_inj, depth_sig=10.0)
        df = q_inj / base / 2
        freqs = f_inj + df * np.arange(-10, 11)
        p64, q64, ph64, n64 = ref_sparse(t, y, dy, freqs, fold='f64')
        p32, q32, ph32, n32 = ref_sparse(t, y, dy, freqs, fold='f32')
        pc, _ = sparse_bls_cpu(t, y, dy, freqs)
        pg, _ = sparse_bls_gpu(t, y, dy, freqs, kernel=kern)
        k = 10
        d_at.append(abs(pc[k] - p64[k]) / p64[k])
        d_max.append(abs(pc.max() - p64.max()) / p64.max())
        npts_diff += int(n32[k] != n64[k])
    ulp = np.spacing(np.float32(base * f_inj))
    print("baseline %5.0fd f=%.1f q=%.3f N=%d ulp(phase)=%.1e (=%.0f%% of q): |cpu-ref64|/ref64 at f_inj median %.3f max %.3f | grid-max median %.3f max %.3f | in-transit set size differs at f_inj in %d/16"
          % (base, f_inj, q_inj, N, ulp, 100 * ulp / q_inj, np.median(d_at), np.max(d_at), np.median(d_max), np.max(d_max), npts_diff))

print("\nfloat32 frequency quantization vs Keplerian grid step (q/(2T)):")
for base, f in ((365.0, 4.7), (3650.0, 4.7), (3650.0, 1.0)):
    q = 0.01
    step = q / (2 * base)
    ferr = abs(float(np.float32(f)) - f)
    print("  T=%.0fd f=%.1f: |f32(f)-f| = %.2e = %.0f%% of grid step %.2e; phase drift over baseline = %.2e cycles"
          % (base, f, ferr, 100 * ferr / step, step, ferr * base))
