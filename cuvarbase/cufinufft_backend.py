"""
cuFINUFFT backend for the NFFT in the Lomb-Scargle periodogram.

Optional cross-check backend (``use_cufinufft=True``) replacing the
custom Gaussian-spreading NFFT with cuFINUFFT's type-1 (nonuniform to
uniform) transform.

.. note::

    The custom NFFT kernel remains the default and, in cuvarbase's
    benchmarks (Feb 2026, RTX A5000), was faster end-to-end: the
    cuFINUFFT path ran at 0.63-0.84x the custom kernel's speed because
    plan creation dominated each call. Plans are now cached (LRU,
    keyed on problem shape) to amortize that cost; treat this backend
    as a numerical cross-check unless you benchmark it on your own
    workload.

The key integration point is ``cufinufft_nfft_adjoint()``, which is a
drop-in replacement for ``cunfft.nfft_adjoint_async()`` in the
Lomb-Scargle pipeline.

Requires: pip install cufinufft>=2.2
"""
import threading
from collections import OrderedDict

import numpy as np

try:
    import cufinufft
    HAS_CUFINUFFT = True
except ImportError:
    HAS_CUFINUFFT = False

import pycuda.gpuarray as gpuarray

# LRU cache of cufinufft Plans keyed on (nf_total, eps, n_pts,
# gpu_method). Plan creation (cuFFT plan + GPU workspace allocation)
# dominated the per-call cost of this backend; reuse amortizes it.
# Cached plans hold GPU memory: the cache is small and evicted plans
# free their resources on garbage collection; call free_plan_cache()
# to drop them eagerly (e.g. before tearing down the CUDA context).
_PLAN_CACHE_MAX_SIZE = 8
_plan_cache = OrderedDict()
_plan_cache_lock = threading.Lock()


def check_cufinufft():
    """Raise ImportError if cufinufft is not available."""
    if not HAS_CUFINUFFT:
        raise ImportError(
            "cufinufft is required for the cuFINUFFT LS backend. "
            "Install with: pip install cufinufft>=2.2"
        )


def _get_plan(nf_total, eps, n_pts, gpu_method=1):
    """Return a cached cufinufft Plan for this problem shape."""
    key = (int(nf_total), float(eps), int(n_pts), int(gpu_method))
    with _plan_cache_lock:
        if key in _plan_cache:
            _plan_cache.move_to_end(key)
            return _plan_cache[key]

    plan = cufinufft.Plan(
        nufft_type=1,
        n_modes=(int(nf_total),),
        n_trans=1,
        eps=eps,
        dtype='complex64',
        gpu_method=gpu_method,
    )

    with _plan_cache_lock:
        _plan_cache[key] = plan
        _plan_cache.move_to_end(key)
        while len(_plan_cache) > _PLAN_CACHE_MAX_SIZE:
            _plan_cache.popitem(last=False)

    return plan


def free_plan_cache():
    """Drop all cached cufinufft plans, releasing their GPU resources
    (via the plans' finalizers once unreferenced)."""
    with _plan_cache_lock:
        _plan_cache.clear()


def cufinufft_nfft_adjoint(memory, minimum_frequency=0.0,
                           samples_per_peak=1.0, eps=1e-6,
                           gpu_method=1,
                           transfer_to_device=True,
                           transfer_to_host=True, **kwargs):
    """
    Compute NFFT adjoint (type-1) using cufinufft.

    Drop-in replacement for ``cunfft.nfft_adjoint_async()``. Uses the same
    ``NFFTMemory`` object and produces output in the same ``ghat_g``/``ghat_c``
    arrays with the same indexing convention.

    Output convention
    -----------------
    After this function, ``memory.ghat_g[k]`` contains the Fourier coefficient
    at mode ``k0 + k``, where ``k0 = round(minimum_frequency / df)`` and
    ``df = 1 / (samples_per_peak * baseline)``. This matches the output of
    the custom NFFT pipeline's normalize kernel.

    Time scaling
    ------------
    cufinufft type-1 computes: ``F[m] = sum_j c_j * exp(i * m * x_j)``
    with ``x_j`` in ``[-pi, pi]`` and output modes ``m = -N/2, ..., N/2-1``.

    To match our frequency grid, we scale times:
        ``x = 2*pi * (t - tmin) / (spp * dt) - pi``

    This makes mode m correspond to frequency ``m * df``.

    Parameters
    ----------
    memory : NFFTMemory
        Memory object with t_g, y_g, ghat_g arrays and metadata (tmin, tmax,
        n0, nf). The ghat_g array must be pre-allocated with size >= nf.
    minimum_frequency : float, optional (default: 0)
        First frequency f0 = k0 * df.
    samples_per_peak : float, optional (default: 1)
        Oversampling factor.
    eps : float, optional (default: 1e-6)
        Requested precision for cufinufft.
    gpu_method : int, optional (default: 1)
        cufinufft spreading method (1 = shared-memory subproblem,
        2 = global-memory; see the cufinufft documentation).
    transfer_to_device : bool, optional (default: True)
        Transfer input data to GPU before computation.
    transfer_to_host : bool, optional (default: True)
        Transfer result to CPU after computation.

    Returns
    -------
    ghat_c : ndarray, complex
        The NFFT result on CPU (only if transfer_to_host=True).
    """
    check_cufinufft()

    if transfer_to_device:
        memory.transfer_data_to_gpu()

    nf = memory.nf
    tmin = float(memory.tmin)
    tmax = float(memory.tmax)
    dt = tmax - tmin
    spp = float(samples_per_peak)

    # Frequency spacing and starting mode
    df = 1.0 / (spp * dt)
    k0 = max(0, int(round(float(minimum_frequency) / df)))

    # Maximum mode needed: k0 + nf - 1
    max_mode = k0 + nf - 1

    # cufinufft with default modeord=0 outputs modes -N/2 .. N/2-1
    # For mode M to be available, need N/2 - 1 >= M, so N >= 2*(M+1)
    nf_total = 2 * (max_mode + 1)

    # Scale times to [-pi, pi]
    # x = 2*pi * (t - tmin) / (spp * dt) - pi
    # = scale * t + shift
    scale = np.float32(2.0 * np.pi / (spp * dt))
    shift = np.float32(-scale * tmin - np.pi)

    x_cu = memory.t_g * scale + shift

    # cufinufft needs complex64 strengths
    c = memory.y_g.astype(np.complex64)

    # Output buffer for full transform
    f_out = gpuarray.zeros(nf_total, dtype=np.complex64)

    # Execute with a cached plan (creation dominates the per-call
    # cost); setpts re-bins the points for this call's data
    plan = _get_plan(nf_total, eps, len(x_cu), gpu_method=gpu_method)
    plan.setpts(x_cu)
    plan.execute(c, f_out)

    # Extract modes k0 .. k0+nf-1
    # In default ordering, mode m is at index m + N/2
    offset = nf_total // 2 + k0

    # Write into memory.ghat_g with same indexing as custom NFFT:
    # ghat_g[k] = Fourier coefficient at mode k0 + k
    memory.ghat_g[:nf] = f_out[offset:offset + nf]

    if transfer_to_host:
        memory.transfer_nfft_to_cpu()

    return memory.ghat_c
