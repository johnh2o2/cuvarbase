#!/usr/bin/env python
"""
Lighter-gate spot checks for the tls.cu / nufft_lrt.cu hygiene edits
(Jul 2026 kernel-hygiene pass).

tls.cu:      the removed PI macro was dead code, so a TLS search must
             produce IDENTICAL results before/after.
nufft_lrt.cu: PI (unreferenced) moved under the DOUBLE_PRECISION guard
             and the hardcoded float32 helpers (fmaxf/fmodf/fabsf) were
             retyped to the FLT-overloaded forms; float32-mode results
             must be identical, double mode must compile, and the
             kernels must match a numpy port when launched directly
             (the production pipeline runs the matched filter on the
             host, so this launches the .cu kernels explicitly).

Usage:  python tls_nufft_spot.py <label> <out.json>
        python tls_nufft_spot.py --compare before.json after.json
"""
import json
import sys

import numpy as np


def tls_spot():
    from cuvarbase.tls import tls_search_gpu
    rand = np.random.RandomState(7)
    period, q, depth = 3.0, 0.04, 0.01
    ndata, baseline, sigma = 800, 30.0, 0.002
    t = np.sort(baseline * rand.rand(ndata))
    phase = (t / period) % 1.0
    in_transit = np.abs(((phase - 0.25 + 0.5) % 1.0) - 0.5) < q / 2
    y = np.ones(ndata) - depth * in_transit + sigma * rand.randn(ndata)
    dy = sigma * np.ones(ndata)
    periods = np.linspace(2.8, 3.2, 200).astype(np.float32)

    res = tls_search_gpu(t, y, dy, periods=periods)
    out = {k: (float(v) if np.isscalar(v) or getattr(v, 'ndim', 1) == 0
               else np.asarray(v, dtype=np.float64).tolist())
           for k, v in res.items()
           if k in ('period', 'depth', 'SDE', 't0', 'duration')}
    for k in ('chi2', 'power', 'SR'):
        if k in res and res[k] is not None:
            arr = np.asarray(res[k], dtype=np.float64)
            out[k + '_arr'] = np.nan_to_num(arr, nan=-999.0).tolist()
    return out


def nufft_lrt_kernel_spot(use_double):
    """Compile nufft_lrt.cu (f32 or f64 mode) and launch its kernels
    directly, checking against numpy ports."""
    import pycuda.autoprimaryctx  # noqa: F401
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    from cuvarbase.utils import _module_reader, find_kernel

    real_type = np.float64 if use_double else np.float32
    complex_type = np.complex128 if use_double else np.complex64
    defs = {'DOUBLE_PRECISION': None} if use_double else {}
    module_txt = _module_reader(find_kernel('nufft_lrt'), defs)
    mod = SourceModule(module_txt, options=[])

    rand = np.random.RandomState(3)
    nf = 256
    Y = (rand.randn(nf) + 1j * rand.randn(nf)).astype(complex_type)
    T = (rand.randn(nf) + 1j * rand.randn(nf)).astype(complex_type)
    P_s = (0.5 + rand.rand(nf)).astype(real_type)
    weights = np.ones(nf, dtype=real_type)
    eps_floor = real_type(1e-12)

    Y_g, T_g = gpuarray.to_gpu(Y), gpuarray.to_gpu(T)
    P_g, w_g = gpuarray.to_gpu(P_s), gpuarray.to_gpu(weights)
    results_g = gpuarray.zeros(2, dtype=real_type)

    block = (128, 1, 1)
    grid = (int(np.ceil(nf / 128.0)), 1)
    func = mod.get_function('nufft_matched_filter')
    func(Y_g, T_g, P_g, w_g, results_g, np.int32(nf), eps_floor,
         block=block, grid=grid,
         shared=2 * 128 * np.dtype(real_type).itemsize)
    num, den = results_g.get().astype(np.float64)

    # numpy port
    P_inv = 1.0 / np.maximum(P_s.astype(np.float64), float(eps_floor))
    num_ref = np.sum((Y.astype(np.complex128) *
                      np.conj(T.astype(np.complex128))).real * P_inv)
    den_ref = np.sum(np.abs(T.astype(np.complex128)) ** 2 * P_inv)

    # generate_transit_template kernel vs numpy port
    n = 500
    t = np.sort(rand.rand(n) * 27.0).astype(real_type)
    period, epoch, duration, depth = 3.7, 1.2, 0.3, 0.01
    tmpl_g = gpuarray.zeros(n, dtype=real_type)
    t_g = gpuarray.to_gpu(t)
    f2 = mod.get_function('generate_transit_template')
    f2(t_g, tmpl_g, np.int32(n), real_type(period), real_type(epoch),
       real_type(duration), real_type(depth),
       block=block, grid=(int(np.ceil(n / 128.0)), 1))
    tmpl = tmpl_g.get().astype(np.float64)

    ph = np.mod(t.astype(np.float64) - epoch, period) / period
    ph[ph > 0.5] -= 1.0
    tmpl_ref = np.where(np.abs(ph) <= duration / (2 * period), -depth, 0.0)

    return {'num': float(num), 'den': float(den),
            'num_ref': float(num_ref), 'den_ref': float(den_ref),
            'num_rel_err': float(abs(num - num_ref) / abs(num_ref)),
            'den_rel_err': float(abs(den - den_ref) / abs(den_ref)),
            'template': tmpl.tolist(),
            'template_max_abs_diff': float(np.max(np.abs(tmpl - tmpl_ref)))}


def nufft_lrt_pipeline_spot():
    """End-to-end run() sanity: SNR peaks at the injected period."""
    from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
    rand = np.random.RandomState(11)
    n, baseline = 600, 30.0
    t = np.sort(rand.rand(n) * baseline)
    period, duration, depth = 3.0, 0.25, 0.02
    ph = np.mod(t, period) / period
    y = 1.0 - depth * (np.minimum(ph, 1 - ph) < duration / (2 * period))
    y += 0.002 * rand.randn(n)
    proc = NUFFTLRTAsyncProcess()
    periods = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
    snr = proc.run(t, y, periods, durations=np.array([duration]))
    snr = np.asarray(snr, dtype=np.float64).squeeze()
    return {'periods': periods.tolist(), 'snr': snr.tolist(),
            'best_period': float(periods[int(np.argmax(snr))])}


def main(label, outfile):
    out = {'label': label}
    out['tls'] = tls_spot()
    out['nufft_lrt_f32'] = nufft_lrt_kernel_spot(use_double=False)
    out['nufft_lrt_f64'] = nufft_lrt_kernel_spot(use_double=True)
    out['nufft_lrt_pipeline'] = nufft_lrt_pipeline_spot()
    with open(outfile, 'w') as f:
        json.dump(out, f)
    print('TLS: period=%.6f depth=%.6f SDE=%.4f'
          % (out['tls']['period'], out['tls']['depth'], out['tls']['SDE']))
    for m in ('nufft_lrt_f32', 'nufft_lrt_f64'):
        print('%s: num_rel_err=%.3e den_rel_err=%.3e tmpl_max_diff=%.3e'
              % (m, out[m]['num_rel_err'], out[m]['den_rel_err'],
                 out[m]['template_max_abs_diff']))
    print('pipeline best_period=%.3f (expect 3.0), snr=%s'
          % (out['nufft_lrt_pipeline']['best_period'],
             ['%.2f' % s for s in out['nufft_lrt_pipeline']['snr']]))
    print('wrote %s' % outfile)


def compare(before_file, after_file):
    with open(before_file) as f:
        b = json.load(f)
    with open(after_file) as f:
        a = json.load(f)
    # TLS must be identical (dead-macro removal only)
    for k in b['tls']:
        vb, va = b['tls'][k], a['tls'][k]
        if isinstance(vb, list):
            d = float(np.max(np.abs(np.array(vb) - np.array(va))))
        else:
            d = abs(vb - va)
        print('tls[%s] before/after max|d| = %.3e' % (k, d))
    for m in ('nufft_lrt_f32',):
        db = np.max(np.abs(np.array(b[m]['template']) -
                           np.array(a[m]['template'])))
        print('%s template before/after max|d| = %.3e (expect 0)' % (m, db))
        print('%s num: before=%.15g after=%.15g' % (m, b[m]['num'],
                                                    a[m]['num']))
    sb = np.array(b['nufft_lrt_pipeline']['snr'])
    sa = np.array(a['nufft_lrt_pipeline']['snr'])
    print('pipeline snr before/after max|d| = %.3e' % np.max(np.abs(sb - sa)))
    if 'nufft_lrt_f64' in b:
        print('f64 kernel gate ran BEFORE too (num_rel_err=%.3e)'
              % b['nufft_lrt_f64']['num_rel_err'])


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
