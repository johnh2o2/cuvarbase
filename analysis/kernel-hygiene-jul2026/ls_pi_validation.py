#!/usr/bin/env python
"""
Before/after validation for the lomb.cu float32-PI-literal fix
(Jul 2026 kernel-hygiene pass; same defect class as the cunfft.cu A3
fix, commit 2699525).

The only lomb.cu code path that uses PI is the direct-sums pair
(``lomb_dirsum``/``lomb_dirsum_custom_frq``) via ``cossum``/``sinsum``:

    phi = (t + 0.5) * f * 2 * PI        (un-reduced)

With ``#define PI 3.14159...f`` (float32, relative error +2.784e-8) the
double-precision kernel computes cos/sin at ``2*pi_true*(1+eps)*f*(t+0.5)``
with eps = 2.784e-8 -- i.e. it evaluates the exact periodogram on a
frequency axis stretched by (1+eps). Observable error vs a float64
reference at the same nominal frequency is |dP/df| * eps * f, of order
``eps * f * T`` relative to peak structure (T = baseline).

Cases:
  A  low-level API, nonzero epoch (t += 4.5), f ~ 90-110 c/d, T = 30 d
  B  low-level API, BJD-scale epoch (t += 2,450,000), same grid.
     (LombScargleAsyncProcess.run() mean-centers t in float64 before
     upload, so raw BJD epochs reach the kernel only through the
     low-level lomb_scargle_async path.)
  C  run() entry point (t mean-centered internally), same data as A
  D  float32 mode, case-A data: must be bit-identical before/after
     (the float literal's value is unchanged)

References computed per case:
  ref64      float64 CPU port of the kernel (identical op order,
             exact np.pi)   <- acceptance metric
  ref32pi    same port with pi = float64(float32(pi)): the BUGGY kernel
             should match THIS to roundoff (diagnosis confirmation)
  astropy    LombScargle(fit_mean=True, center_data=True), independent

Usage:  python ls_pi_validation.py <label> <out.json>
        python ls_pi_validation.py --compare before.json after.json
"""
import json
import sys

import numpy as np

PI32 = float(np.float32(np.pi))  # 3.1415927410125732


def make_data(seed=42, n=300, T=30.0, epoch=4.5, f0=97.0, sigma=0.1):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.rand(n)) * T + epoch
    y = 0.3 * np.cos(2 * np.pi * f0 * t) + 12.0
    y += sigma * rng.randn(n)
    dy = sigma * (0.8 + 0.4 * rng.rand(n))
    return t, y, dy


def freq_grid(fmin=90.0, fmax=110.0, T=30.0, spp=5):
    df = 1.0 / (spp * T)
    k0 = int(round(fmin / df))
    nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))


def norm_weights(dy):
    w = np.power(np.asarray(dy, dtype=np.float64), -2.0)
    return w / np.sum(w)


def _lspow_flmean64(C, S, C2, S2, YCh, YSh, YY):
    """float64 port of lomb.cu lspow_flmean (reg=NULL)."""
    tan2wt = (S2 - 2.0 * S * C) / (C2 - (C * C - S * S))
    C2w = 1.0 / np.sqrt(1.0 + tan2wt * tan2wt)
    S2w = tan2wt * C2w
    Cw = np.sqrt(0.5 * (1.0 + C2w))
    Sw = np.sqrt(0.5 * (1.0 - C2w))
    if S2w < 0:
        Sw = -Sw
    Cshft = C * Cw + S * Sw
    Sshft = S * Cw - C * Sw
    CC = 0.5 * (1.0 + C2 * C2w + S2 * S2w) - Cshft * Cshft
    SS = 0.5 * (1.0 - C2 * C2w - S2 * S2w) - Sshft * Sshft
    YC = YCh * Cw + YSh * Sw
    YS = YSh * Cw - YCh * Sw
    return (YC * YC / CC + YS * YS / SS) / YY


def kernel_port_cpu(t, y, dy, freqs, pi=np.pi):
    """float64 CPU port of lomb.cu lomb_dirsum, FLOATING_MEAN mode.

    Identical phase convention to the kernel: phi = (t+0.5)*f*2*pi,
    un-reduced, same multiplication order.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = norm_weights(dy)
    ybar = np.dot(w, y)
    yw = w * (y - ybar)
    YY = np.dot(w, (y - ybar) ** 2)

    tp = t + 0.5
    P = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        arg1 = tp * f * 2.0 * pi
        arg2 = tp * (2.0 * f) * 2.0 * pi
        c1, s1 = np.cos(arg1), np.sin(arg1)
        c2, s2 = np.cos(arg2), np.sin(arg2)
        C, S = np.dot(w, c1), np.dot(w, s1)
        C2, S2 = np.dot(w, c2), np.dot(w, s2)
        YCh, YSh = np.dot(yw, c1), np.dot(yw, s1)
        P[i] = _lspow_flmean64(C, S, C2, S2, YCh, YSh, YY)
    return P


def gpu_dirsum_lowlevel(t, y, dy, freqs, use_double=True):
    """GPU dirsum through the low-level API: t reaches the kernel raw
    (no mean-centering), matching LombScargleMemory.setdata()."""
    from cuvarbase.lombscargle import (LombScargleAsyncProcess,
                                       lomb_scargle_async, get_k0)
    proc = LombScargleAsyncProcess(use_double=use_double, sigma=5)
    proc._compile_and_prepare_functions()
    funcs = (proc.function_tuple, proc.nfft_proc.function_tuple)
    mem = proc.allocate_for_single_lc(t, y, dy, nf=len(freqs),
                                      k0=get_k0(freqs))
    p = lomb_scargle_async(mem, funcs, freqs, use_fft=False,
                           block_size=proc.block_size)
    proc.finish()
    return np.asarray(p[:len(freqs)], dtype=np.float64)


def gpu_dirsum_run(t, y, dy, freqs, use_double=True):
    """GPU dirsum through the production run() entry point
    (mean-centers t and y in float64 first)."""
    from cuvarbase.lombscargle import LombScargleAsyncProcess
    proc = LombScargleAsyncProcess(use_double=use_double, sigma=5)
    results = proc.run([(t, y, dy)], freqs=freqs, use_fft=False)
    proc.finish()
    return np.asarray(results[0][1][:len(freqs)], dtype=np.float64)


def astropy_ref(t, y, dy, freqs):
    from astropy.timeseries import LombScargle
    return np.asarray(
        LombScargle(t, y, dy, fit_mean=True, center_data=True)
        .power(freqs), dtype=np.float64)


def summarize(name, p_gpu, refs, freqs):
    out = {'case': name,
           'p_gpu': p_gpu.tolist(),
           'peak_freq_gpu': float(freqs[int(np.argmax(p_gpu))]),
           'p_max_gpu': float(np.max(p_gpu))}
    for rname, pref in refs.items():
        d = p_gpu - pref
        out['max_abs_diff_vs_' + rname] = float(np.max(np.abs(d)))
        out['rms_diff_vs_' + rname] = float(np.sqrt(np.mean(d ** 2)))
        out['peak_freq_' + rname] = float(freqs[int(np.argmax(pref))])
    return out


def main(label, outfile):
    T, f0, spp = 30.0, 97.0, 5
    freqs = freq_grid(90.0, 110.0, T=T, spp=spp)
    results = {'label': label, 'freqs': freqs.tolist(),
               'pi32_rel_err': (PI32 - np.pi) / np.pi}

    # ---- case A: low-level, nonzero epoch --------------------------------
    t, y, dy = make_data(epoch=4.5)
    refs = {'ref64': kernel_port_cpu(t, y, dy, freqs, pi=np.pi),
            'ref32pi': kernel_port_cpu(t, y, dy, freqs, pi=PI32),
            'astropy': astropy_ref(t, y, dy, freqs)}
    p = gpu_dirsum_lowlevel(t, y, dy, freqs, use_double=True)
    results['A_lowlevel_epoch4.5_f64'] = summarize('A', p, refs, freqs)

    # ---- case B: low-level, BJD-scale epoch ------------------------------
    tb = t - 4.5 + 2450000.0  # identical sampling, BJD-scale epoch
    refs_b = {'ref64': kernel_port_cpu(tb, y, dy, freqs, pi=np.pi),
              'ref32pi': kernel_port_cpu(tb, y, dy, freqs, pi=PI32),
              'astropy': astropy_ref(tb, y, dy, freqs)}
    pb = gpu_dirsum_lowlevel(tb, y, dy, freqs, use_double=True)
    results['B_lowlevel_epochBJD_f64'] = summarize('B', pb, refs_b, freqs)

    # ---- case C: run() entry point (t mean-centered internally) ---------
    tc = t - np.nanmean(t)
    yc = y - np.nanmean(y)
    refs_c = {'ref64': kernel_port_cpu(tc, yc, dy, freqs, pi=np.pi),
              'ref32pi': kernel_port_cpu(tc, yc, dy, freqs, pi=PI32),
              'astropy': astropy_ref(t, y, dy, freqs)}
    pc = gpu_dirsum_run(t, y, dy, freqs, use_double=True)
    results['C_run_epoch4.5_f64'] = summarize('C', pc, refs_c, freqs)

    # ---- case D: float32 mode, must be unchanged by the fix -------------
    pd_ = gpu_dirsum_lowlevel(t, y, dy, freqs, use_double=False)
    results['D_lowlevel_epoch4.5_f32'] = {'p_gpu': pd_.tolist(),
                                          'p_max_gpu': float(np.max(pd_))}

    with open(outfile, 'w') as f:
        json.dump(results, f)

    for k in ('A_lowlevel_epoch4.5_f64', 'B_lowlevel_epochBJD_f64',
              'C_run_epoch4.5_f64'):
        r = results[k]
        print('%-28s max|d| vs ref64: %.3e   vs ref32pi: %.3e   '
              'vs astropy: %.3e' % (k, r['max_abs_diff_vs_ref64'],
                                    r['max_abs_diff_vs_ref32pi'],
                                    r['max_abs_diff_vs_astropy']))
    print('float32-pi relative error: %.4e' % results['pi32_rel_err'])
    print('wrote %s' % outfile)


def compare(before_file, after_file):
    with open(before_file) as f:
        b = json.load(f)
    with open(after_file) as f:
        a = json.load(f)
    print('%-28s %14s %14s %10s' % ('case (vs ref64)', 'BEFORE max|d|',
                                    'AFTER max|d|', 'ratio'))
    for k in ('A_lowlevel_epoch4.5_f64', 'B_lowlevel_epochBJD_f64',
              'C_run_epoch4.5_f64'):
        db = b[k]['max_abs_diff_vs_ref64']
        da = a[k]['max_abs_diff_vs_ref64']
        print('%-28s %14.3e %14.3e %10.1f' % (k, db, da,
                                              db / da if da else float('inf')))
        print('%-28s %14.3e %14.3e   (vs ref32pi: before should be ~0)'
              % ('  ... vs ref32pi', b[k]['max_abs_diff_vs_ref32pi'],
                 a[k]['max_abs_diff_vs_ref32pi']))
    pdb_ = np.array(b['D_lowlevel_epoch4.5_f32']['p_gpu'])
    pda = np.array(a['D_lowlevel_epoch4.5_f32']['p_gpu'])
    print('case D (f32) before/after max|d|: %.3e (expect 0.0 -- '
          'bit-identical)' % np.max(np.abs(pdb_ - pda)))


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
