"""Every public entry point wrapped as f(t, y, dy, **opts) -> 1-D array."""
import numpy as np
import warnings
from audit_common import bls_freqs, ls_freqs, P_TR, Q_TR

from cuvarbase import bls as B
from cuvarbase import tls as TL
from cuvarbase.lombscargle import LombScargleAsyncProcess, lomb_scargle_simple
from cuvarbase.cunfft import NFFTAsyncProcess
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.pdm import PDMAsyncProcess
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess

FREQS = bls_freqs()
LSF = ls_freqs()
PERIODS = np.linspace(2.0, 6.0, 400)
QMIN, QMAX = 0.01, 0.2

# ---- BLS family ----
def bls_fast(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    return B.eebls_gpu_fast(t, y, dy, f, qmin=QMIN, qmax=QMAX, **kw)

def bls_fast_opt(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    return B.eebls_gpu_fast_optimized(t, y, dy, f, qmin=QMIN, qmax=QMAX, **kw)

def bls_fast_adaptive(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    return B.eebls_gpu_fast_adaptive(t, y, dy, f, qmin=QMIN, qmax=QMAX, **kw)

def bls_std(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    p, sols = B.eebls_gpu(t, y, dy, f, qmin=QMIN, qmax=QMAX, **kw)
    return p

def bls_std_sols(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    p, sols = B.eebls_gpu(t, y, dy, f, qmin=QMIN, qmax=QMAX, **kw)
    return np.array(sols)

def bls_custom(t, y, dy, freqs=None, **kw):
    f = FREQS[:200] if freqs is None else freqs
    p, sols = B.eebls_gpu_custom(t, y, dy, f, q_values=np.array([0.02, 0.05, 0.1]),
                                 phi_values=np.linspace(0, 1, 400, endpoint=False), **kw)
    return p

def sparse_gpu(t, y, dy, freqs=None, **kw):
    f = FREQS[:300] if freqs is None else freqs
    p, sols = B.sparse_bls_gpu(t, y, dy, f, **kw)
    return p

def sparse_cpu(t, y, dy, freqs=None, **kw):
    f = FREQS[:60] if freqs is None else freqs
    p, sols = B.sparse_bls_cpu(t, y, dy, f, **kw)
    return p

def bls_batch(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    return B.eebls_gpu_batch([(t, y, dy)], f, qmin=QMIN, qmax=QMAX, **kw)[0]

def bls_transit(t, y, dy, **kw):
    f, p, s = B.eebls_transit(t, y, dy, **kw)
    return p

def bls_transit_gpu_fast(t, y, dy, **kw):
    f, p, s = B.eebls_transit_gpu(t, y, dy, use_fast=True, **kw)
    return p

def single(t, y, dy, **kw):
    return np.array([B.single_bls(t, y, dy, 1.0 / P_TR, Q_TR, 0.0)])

# ---- TLS ----
def tls_fast(t, y, dy, periods=None, **kw):
    P = PERIODS if periods is None else periods
    r = TL.tls_search_gpu(t, y, dy, periods=P, **kw)
    return r['chi2']

def tls_fast_full(t, y, dy, periods=None, **kw):
    P = PERIODS if periods is None else periods
    return TL.tls_search_gpu(t, y, dy, periods=P, **kw)

def tls_legacy(t, y, dy, periods=None, **kw):
    P = PERIODS if periods is None else periods
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = TL.tls_search_gpu(t, y, dy, periods=P, use_fast=False, **kw)
    return r['chi2']

def tls_batch(t, y, dy, periods=None, **kw):
    P = PERIODS if periods is None else periods
    r = TL.tls_search_batch([(t, y, dy)], periods=P, return_arrays=True, **kw)[0]
    if 'error' in r:
        raise RuntimeError(r['error'])
    return r['chi2']

def tls_transit(t, y, dy, **kw):
    r = TL.tls_transit(t, y, dy, period_min=2.0, period_max=6.0, **kw)
    return r['chi2']

# ---- Lomb-Scargle ----
_LS = {}
def _ls_proc(key, **kw):
    if key not in _LS:
        _LS[key] = LombScargleAsyncProcess(**kw)
    return _LS[key]

def ls(t, y, dy, freqs=None, **kw):
    f = LSF if freqs is None else freqs
    proc = _ls_proc('plain')
    res = proc.run([(t, y, dy)], freqs=[f], **kw)
    proc.finish()
    return np.copy(res[0][1])

def ls_batched(t, y, dy, freqs=None, **kw):
    f = LSF if freqs is None else freqs
    proc = _ls_proc('plain')
    res = proc.batched_run_const_nfreq([(t, y, dy)], freqs=f, **kw)
    return np.copy(res[0][1])

def ls_dirsum(t, y, dy, freqs=None, **kw):
    f = LSF[:300] if freqs is None else freqs
    proc = _ls_proc('plain')
    res = proc.run([(t, y, dy)], freqs=[f], use_fft=False, **kw)
    proc.finish()
    return np.copy(res[0][1])

def ls_cufinufft(t, y, dy, freqs=None, **kw):
    f = LSF if freqs is None else freqs
    proc = _ls_proc('cufinufft', use_cufinufft=True)
    res = proc.run([(t, y, dy)], freqs=[f], **kw)
    proc.finish()
    return np.copy(res[0][1])

def ls_mh2(t, y, dy, freqs=None, **kw):
    f = LSF if freqs is None else freqs
    proc = _ls_proc('mh2', nharmonics=2)
    res = proc.run([(t, y, dy)], freqs=[f], **kw)
    proc.finish()
    return np.copy(res[0][1])

def ls_double(t, y, dy, freqs=None, **kw):
    f = LSF if freqs is None else freqs
    proc = _ls_proc('double', use_double=True)
    res = proc.run([(t, y, dy)], freqs=[f], **kw)
    proc.finish()
    return np.copy(res[0][1])

def ls_simple(t, y, dy, **kw):
    f, p = lomb_scargle_simple(t, y, dy, **kw)
    return np.copy(p)

_NF = {}
def nfft(t, y, dy, **kw):
    if 'p' not in _NF:
        _NF['p'] = NFFTAsyncProcess()
    proc = _NF['p']
    res = proc.run([(t, y, 512)], **kw)
    proc.finish()
    return np.abs(np.copy(res[0]))

# ---- CE ----
_CE = {}
def _ce_proc(key, **kw):
    if key not in _CE:
        _CE[key] = ConditionalEntropyAsyncProcess(**kw)
    return _CE[key]

def ce(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    proc = _ce_proc('plain')
    res = proc.run([(t, y, dy)], freqs=[f], **kw)
    proc.finish()
    return np.copy(res[0][1])

def ce_fast(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    proc = _ce_proc('fast', use_fast=True)
    res = proc.run([(t, y, dy)], freqs=[f], **kw)
    proc.finish()
    return np.copy(res[0][1])

def ce_weighted(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    proc = _ce_proc('weighted', weighted=True)
    res = proc.run([(t, y, dy)], freqs=[f], **kw)
    proc.finish()
    return np.copy(res[0][1])

def ce_batched(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    proc = _ce_proc('plain')
    res = proc.batched_run_const_nfreq([(t, y, dy)], freqs=f, **kw)
    return np.copy(res[0][1])

def ce_large(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    proc = _ce_proc('plain')
    res = proc.large_run([(t, y, dy)], freqs=[f], **kw)
    return np.copy(res[0][1])

# ---- PDM ----
_PDM = {}
def _pdm(kind, t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    if 'p' not in _PDM:
        _PDM['p'] = PDMAsyncProcess()
    proc = _PDM['p']
    res = proc.run([(t, y, dy)], freqs=f, kind=kind, **kw)
    proc.finish()
    return np.copy(res[0][1])

def pdm_linterp(t, y, dy, **kw): return _pdm('binned_linterp', t, y, dy, **kw)
def pdm_step(t, y, dy, **kw): return _pdm('binned_step', t, y, dy, **kw)
def pdm_linterp_fast(t, y, dy, **kw): return _pdm('binned_linterp_fast', t, y, dy, **kw)
def pdm_step_fast(t, y, dy, **kw): return _pdm('binned_step_fast', t, y, dy, **kw)
def pdm_tophat(t, y, dy, freqs=None, **kw):
    return _pdm('binless_tophat', t, y, dy, freqs=(FREQS[:200] if freqs is None else freqs), **kw)
def pdm_gauss(t, y, dy, freqs=None, **kw):
    return _pdm('binless_gauss', t, y, dy, freqs=(FREQS[:200] if freqs is None else freqs), **kw)
def pdm_tophat_fast(t, y, dy, freqs=None, **kw):
    return _pdm('binless_tophat_fast', t, y, dy, freqs=(FREQS[:200] if freqs is None else freqs), **kw)
def pdm_gauss_fast(t, y, dy, freqs=None, **kw):
    return _pdm('binless_gauss_fast', t, y, dy, freqs=(FREQS[:200] if freqs is None else freqs), **kw)

def pdm_batched(t, y, dy, freqs=None, **kw):
    f = FREQS if freqs is None else freqs
    if 'p' not in _PDM:
        _PDM['p'] = PDMAsyncProcess()
    res = _PDM['p'].batched_run_const_nfreq([(t, y, dy)], freqs=f, **kw)
    return np.copy(res[0][1])

# ---- NUFFT-LRT ----
_LRT = {}
LRT_PERIODS = np.linspace(2.5, 4.5, 21)
LRT_DUR = np.array([0.1, 0.165, 0.3])
def _lrt():
    if 'p' not in _LRT:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _LRT['p'] = NUFFTLRTAsyncProcess()
    return _LRT['p']

def lrt_matched(t, y, dy, **kw):
    return _lrt().run(t, y, LRT_PERIODS, LRT_DUR, **kw).ravel()

def lrt_matched_epochs(t, y, dy, **kw):
    return _lrt().run(t, y, LRT_PERIODS[:5], LRT_DUR[:1], epochs=np.linspace(0, 3.0, 12), **kw).ravel()

def _basis(t):
    tt = (t - t.min()) / max(t.max() - t.min(), 1e-12)
    return np.column_stack([tt - 0.5, np.sin(2 * np.pi * tt)])

def lrt_marginal(t, y, dy, **kw):
    V = _basis(np.asarray(t, dtype=np.float64))
    return _lrt().run(t, y, LRT_PERIODS, LRT_DUR, detector='marginal',
                      systematics_basis=V, coeff_prior_cov=np.eye(2) * 1e-2, **kw).ravel()

def lrt_sequential(t, y, dy, **kw):
    V = _basis(np.asarray(t, dtype=np.float64))
    return _lrt().run(t, y, LRT_PERIODS, LRT_DUR, detector='sequential',
                      systematics_basis=V, **kw).ravel()


ALL = dict(
    bls_fast=bls_fast, bls_fast_opt=bls_fast_opt, bls_fast_adaptive=bls_fast_adaptive,
    bls_std=bls_std, bls_custom=bls_custom, sparse_gpu=sparse_gpu, sparse_cpu=sparse_cpu,
    bls_batch=bls_batch, bls_transit=bls_transit, bls_transit_gpu_fast=bls_transit_gpu_fast,
    single_bls=single,
    tls_fast=tls_fast, tls_legacy=tls_legacy, tls_batch=tls_batch, tls_transit=tls_transit,
    ls=ls, ls_batched=ls_batched, ls_dirsum=ls_dirsum, ls_cufinufft=ls_cufinufft,
    ls_mh2=ls_mh2, ls_double=ls_double, ls_simple=ls_simple, nfft=nfft,
    ce=ce, ce_fast=ce_fast, ce_weighted=ce_weighted, ce_batched=ce_batched, ce_large=ce_large,
    pdm_linterp=pdm_linterp, pdm_step=pdm_step, pdm_linterp_fast=pdm_linterp_fast,
    pdm_step_fast=pdm_step_fast, pdm_tophat=pdm_tophat, pdm_gauss=pdm_gauss,
    pdm_tophat_fast=pdm_tophat_fast, pdm_gauss_fast=pdm_gauss_fast, pdm_batched=pdm_batched,
    lrt_matched=lrt_matched, lrt_matched_epochs=lrt_matched_epochs,
    lrt_marginal=lrt_marginal, lrt_sequential=lrt_sequential,
)
