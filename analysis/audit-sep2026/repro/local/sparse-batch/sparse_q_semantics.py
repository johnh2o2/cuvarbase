"""Duration (q) semantics: cuvarbase sparse reports phi0 = phi_i (first
in-transit point) and q to the egress midpoint; the paper (Eqs 8-10)
uses ingress AND egress midpoints. Measure the bias of the reported q vs
the injected q and vs the paper definition, and how many candidates the
qmin filter treats differently under the two definitions."""
import numpy as np
from cuvarbase.bls import sparse_bls_cpu, sparse_bls_gpu, compile_sparse_bls
from sparse_exp import ref_sparse, make_lc

kern = compile_sparse_bls(block_size=64)
for N in (50, 100, 200, 400):
    bias_cuv, bias_paper, nflip = [], [], []
    for seed in range(20):
        q_true = 0.04
        t, y, dy = make_lc(N, 'uniform', seed=100 + seed, q=q_true, depth_sig=12.0)
        f0 = 1.3
        freqs = np.array([f0])
        pg, sg = sparse_bls_gpu(t, y, dy, freqs, kernel=kern)
        pc, sc = sparse_bls_cpu(t, y, dy, freqs)
        pr, qr, phr, npr = ref_sparse(t, y, dy, freqs, qdef='paper')
        bias_cuv.append(sg[0][0] - q_true)
        bias_paper.append(qr[0] - q_true)
        # qmin filter disagreement: fraction of candidate sets accepted under
        # one q definition but not the other at qmin = 0.5*q_true
        pa, _, _, _ = ref_sparse(t, y, dy, freqs, qdef='cuv', qmin=0.5 * q_true, qmax=2 * q_true)
        pb, _, _, _ = ref_sparse(t, y, dy, freqs, qdef='paper', qmin=0.5 * q_true, qmax=2 * q_true)
        nflip.append(abs(pa[0] - pb[0]) > 1e-9)
    print("N=%4d  reported q - q_true: cuvarbase mean %+.4f (sd %.4f) | paper-def mean %+.4f (sd %.4f) | mean half-gap 1/(2N)=%.4f | peak power differs under qmin/qmax filter in %d/20 LCs"
          % (N, np.mean(bias_cuv), np.std(bias_cuv), np.mean(bias_paper), np.std(bias_paper), 0.5 / N, sum(nflip)))
