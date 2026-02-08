"""
cuFINUFFT backend for GPU-accelerated NFFT in Lomb-Scargle periodogram.

Replaces the custom Gaussian-spreading NFFT with cuFINUFFT's optimized
type-1 (nonuniform to uniform) transform. cuFINUFFT uses exponential-of-
semicircle kernel, bin-sorted shared-memory spreading, and Horner polynomial
evaluation for ~10-100x faster spreading throughput.

The key integration point is ``cufinufft_nfft_adjoint()``, which is a
drop-in replacement for ``cunfft.nfft_adjoint_async()`` in the
Lomb-Scargle pipeline.

Requires: pip install cufinufft>=2.2
"""
import numpy as np

try:
    import cufinufft
    HAS_CUFINUFFT = True
except ImportError:
    HAS_CUFINUFFT = False

import pycuda.gpuarray as gpuarray


def check_cufinufft():
    """Raise ImportError if cufinufft is not available."""
    if not HAS_CUFINUFFT:
        raise ImportError(
            "cufinufft is required for the cuFINUFFT LS backend. "
            "Install with: pip install cufinufft>=2.2"
        )


def cufinufft_nfft_adjoint(memory, minimum_frequency=0.0,
                           samples_per_peak=1.0, eps=1e-6,
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

    # Create and execute cufinufft plan
    plan = cufinufft.Plan(
        nufft_type=1,
        n_modes=(nf_total,),
        n_trans=1,
        eps=eps,
        dtype='complex64',
        gpu_method=1,  # shared-memory subproblem method
    )
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
