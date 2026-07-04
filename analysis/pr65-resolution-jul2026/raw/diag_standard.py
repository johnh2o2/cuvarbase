"""Diagnostic: replicate test_standard's failing config on GPU and dump
per-frequency numbers to root-cause the mismatch.

Run on the pod from /workspace/cuvarbase.
"""
import json

import numpy as np

from cuvarbase.bls import eebls_gpu, single_bls
from cuvarbase.utils import subtract_epoch


def transit_model(phi0, q, delta):
    def model(t, freq):
        ph = t * freq - phi0
        ph -= np.floor(ph)
        yy = np.zeros(len(t))
        yy[np.absolute(ph) < q] -= delta
        return yy
    return model


def data(seed=100, sigma=0.1, ybar=12., snr=10, ndata=200, freq=10.,
         q=0.01, phi0=None, baseline=1., t0=4.5):
    rand = np.random.RandomState(seed)
    if phi0 is None:
        phi0 = rand.rand()
    delta = snr * sigma / np.sqrt(ndata * q * (1 - q))
    model = transit_model(phi0, q, delta)
    t = baseline * np.sort(rand.rand(ndata)) + t0
    y = model(t, freq) + sigma * rand.randn(len(t))
    y += ybar - np.mean(y)
    err = sigma * np.ones_like(y)
    return t, y, err


def run_config(q_index, phi_index, use_optimized=False):
    q_values = np.logspace(-1.5, np.log10(0.1), num=100)
    phi_values = np.linspace(0, 1, int(np.ceil(2. / min(q_values))))

    q = q_values[q_index]
    phi = phi_values[phi_index]
    freq = 1.0

    t, y, dy = data(snr=10, q=q, phi0=phi, freq=freq, baseline=365.)

    df = min(q_values) / (10 * (max(t) - min(t)))
    delta_f = 5 * df / freq
    freqs = np.linspace(freq * (1 - delta_f), (1 + delta_f) * freq,
                        int(5. * 2 * delta_f * freq / df))

    power, gsols = eebls_gpu(t, y, dy, freqs,
                             qmin=0.1 * q, qmax=2.0 * q,
                             nstreams=1, noverlap=2, dlogq=0.5,
                             freq_batch_size=None,
                             ignore_negative_delta_sols=False,
                             use_optimized=use_optimized)

    t_sub, epoch = subtract_epoch(t)
    bls_c = np.array([single_bls(t, y, dy, x[0], *x[1]) for x in
                      zip(freqs, gsols)])
    diffs = np.abs(power - bls_c)
    iw = int(np.argmax(diffs))

    out = dict(q_index=q_index, phi_index=phi_index,
               use_optimized=use_optimized,
               epoch=float(epoch), q=float(q), phi=float(phi),
               nfreq=len(freqs),
               maxdiff=float(diffs.max()),
               nviol=int(np.sum(diffs > 1e-3 * bls_c + 1e-5)),
               iworst=iw, fworst=float(freqs[iw]),
               p_gpu=float(power[iw]), p_cpu=float(bls_c[iw]),
               sol_q=float(gsols[iw][0]), sol_phi=float(gsols[iw][1]))

    # drill into worst frequency: which points are in the box per
    # single_bls's fold, and their distance to the box edges
    fq = freqs[iw]
    q_sol, phi_sol = gsols[iw]
    phi_sub0 = (phi_sol - epoch * fq) % 1.0
    ph = t_sub.astype(np.float32) * np.float32(fq)
    ph -= np.float32(phi_sub0)
    ph -= np.floor(ph)
    mask = ph < np.float32(q_sol)
    out['n_in_box_cpu'] = int(mask.sum())
    # points close to either edge (within 1e-4 in phase)
    d_lo = np.minimum(ph, 1.0 - ph)      # distance to lower edge (phase 0)
    d_hi = np.abs(ph - np.float32(q_sol))  # distance to upper edge
    close = (d_lo < 1e-4) | (d_hi < 1e-4)
    out['edge_points'] = [
        dict(idx=int(i), phase_rel=float(ph[i]), in_cpu=bool(mask[i]),
             d_lo=float(d_lo[i]), d_hi=float(d_hi[i]))
        for i in np.where(close)[0]]
    return out


if __name__ == '__main__':
    results = []
    for qi, pi in [(0, 0), (0, 10), (0, -1), (5, 0), (-1, 0)]:
        for opt in (False, True):
            r = run_config(qi, pi, use_optimized=opt)
            results.append(r)
            print(f"qi={qi} pi={pi} opt={opt}: maxdiff={r['maxdiff']:.4g} "
                  f"nviol={r['nviol']}/{r['nfreq']} fworst={r['fworst']:.8f} "
                  f"p_gpu={r['p_gpu']:.4f} p_cpu={r['p_cpu']:.4f} "
                  f"sol_q={r['sol_q']:.5f} sol_phi={r['sol_phi']:.6f} "
                  f"n_in_box={r['n_in_box_cpu']} "
                  f"edges={len(r['edge_points'])}")
    with open('/workspace/diag_standard.json', 'w') as f:
        json.dump(results, f, indent=1)
