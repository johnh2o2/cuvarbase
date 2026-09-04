import numpy as np
import warnings
from typing import Literal

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .core import GPUAsyncProcess
from .memory._host import host_array
from .utils import weights, find_kernel, dphase, normalize_light_curves, autofrequency


def var_tophat(t, y, w, freq, dphi):
    var = 0.
    for i, (T, Y, W) in enumerate(zip(t, y, w)):
        mbar = 0.
        wtot = 0.
        for j, (T2, Y2, W2) in enumerate(zip(t, y, w)):
            dph = dphase(abs(T2 - T), freq)
            if dph < dphi:
                mbar += W2 * Y2
                wtot += W2

        var += W * (Y - mbar / wtot)**2

    return var


def var_gauss(t, y, w, freq, dphi):
    def gaussian(x): return np.exp(-0.5 * x**2)
    var = 0.
    for i, (T, Y, W) in enumerate(zip(t, y, w)):
        mbar = 0.
        wtot = 0.

        for j, (T2, Y2, W2) in enumerate(zip(t, y, w)):
            dph = dphase(abs(T2 - T), freq)
            wgt = W2 * gaussian(dph / dphi)
            mbar += wgt * Y2
            wtot += wgt

        var += W * (Y - mbar / wtot)**2

    return var


def binned_pdm_model(t, y, w, freq, nbins, linterp=True):

    if len(t) == 0:
        return lambda p, **kwargs: np.zeros_like(p)

    bin_means = np.zeros(nbins)
    phase = (t * freq) % 1.0
    bins = [int(p * nbins) % nbins for p in phase]

    for i in range(nbins):
        wtot = max([sum([W for j, W in enumerate(w) if bins[j] == i]), 1E-10])
        bin_means[i] = sum([W * Y for j, (Y, W) in enumerate(zip(y, w))
                            if bins[j] == i]) / wtot

    def pred_y(p, nbins=nbins, linterp=linterp, bin_means=bin_means):
        bs = np.array([int(P * nbins) % nbins for P in p])
        if not linterp:
            return bin_means[bs]
        alphas = p * nbins - np.floor(p * nbins) - 0.5
        di = np.floor(alphas).astype(np.int32)
        bins0 = bs + di
        bins1 = bins0 + 1

        alphas[alphas < 0] += 1
        bins0[bins0 < 0] += nbins
        bins1[bins1 >= nbins] -= nbins

        return (1 - alphas) * bin_means[bins0] + alphas * bin_means[bins1]

    return pred_y


def var_binned(t, y, w, freq, nbins, linterp=True):
    ypred = binned_pdm_model(t, y, w, freq, nbins, linterp=linterp)((t * freq) % 1.0)
    return np.dot(w, np.power(y - ypred, 2))


def binless_pdm_cpu(t, y, w, freqs, dphi=0.05, tophat=True):
    # Prepare data (copies: don't mutate the caller's arrays)
    t = t - np.mean(t)
    y = y - np.mean(y)

    ybar = np.dot(w, y)
    var = np.dot(w, np.power(y - ybar, 2))
    if tophat:
        return [1 - var_tophat(t, y, w, freq, dphi) / var for freq in freqs]
    else:
        return [1 - var_gauss(t, y, w, freq, dphi) / var for freq in freqs]


def pdm2_cpu(t, y, w, freqs, nbins=30, linterp=True):
    # Prepare data (copies: don't mutate the caller's arrays)
    t = t - np.mean(t)
    y = y - np.mean(y)

    ybar = np.dot(w, y)
    var = np.dot(w, np.power(y - ybar, 2))
    return [1 - var_binned(t, y, w, freq,
                           nbins=nbins, linterp=linterp) / var
            for freq in freqs]


def pdm2_single_freq(t, y, w, freq, nbins=30, linterp=True):
    # Prepare data (copies: don't mutate the caller's arrays)
    t = t - np.mean(t)
    y = y - np.mean(y)

    ybar = np.dot(w, y)
    var = np.dot(w, np.power(y - ybar, 2))
    return 1 - var_binned(t, y, w, freq, nbins=nbins, linterp=linterp) / var


def pdm_async(stream, data_cpu, data_gpu, pow_cpu, function,
              dphi=0.05, block_size=256, **kwargs):
    # The *_fast kernels statically allocate shared-memory tiles of
    # MAX_BLOCK_SIZE (= 256) floats; a larger launch would write past them.
    if not (0 < block_size <= 256):
        raise ValueError("block_size must be in (0, 256] "
                         "(the PDM kernels' shared-memory tiles are "
                         "sized for at most 256 threads per block); "
                         "got %r" % (block_size,))

    t, y, w, freqs = data_cpu
    t_g, y_g, w_g, freqs_g, pow_g = data_gpu

    if t_g is None:
        return pow_cpu

    # constants
    nfreqs = np.int32(len(freqs))
    ndata = np.int32(len(t))
    dphi = np.float32(dphi)

    # kernel size
    grid_size = int(np.ceil(float(nfreqs) / block_size))
    grid = (grid_size, 1)
    block = (block_size, 1, 1)

    # weighted mean + weighted variance
    ybar = np.dot(w, y)
    var = np.float32(np.dot(w, np.power(y - ybar, 2)))

    # transfer data
    w_g.set_async(np.asarray(w).astype(np.float32), stream=stream)
    t_g.set_async(np.asarray(t).astype(np.float32), stream=stream)

    # Ensure y is zero-weighted-meaned for fast kernels (one-pass SS_between)
    y_norm = (np.asarray(y) - ybar).astype(np.float32)
    y_g.set_async(y_norm, stream=stream)

    function.prepared_async_call(grid, block, stream,
                                 t_g.ptr, y_g.ptr, w_g.ptr,
                                 freqs_g.ptr, pow_g.ptr,
                                 ndata, nfreqs, dphi, var)

    pow_g.get_async(stream=stream, ary=pow_cpu)

    return pow_cpu


class PDMAsyncProcess(GPUAsyncProcess):
    """
    GPUAsyncProcess for the Phase Dispersion Minimization (PDM) period finder.

    Example
    -------
    >>> proc = PDMAsyncProcess()
    >>> Ndata = 1000
    >>> t = np.sort(365 * np.random.rand(Ndata))
    >>> y = 12 + 0.01 * np.cos(2 * np.pi * t / 5.0)
    >>> y += 0.01 * np.random.randn(len(t))
    >>> dy = 0.01 * np.ones_like(y)
    >>> results = proc.run([(t, y, dy)])
    >>> proc.finish()
    >>> pdm_freqs, pdm_powers = results[0]
    """

    def __init__(self, *args, **kwargs):
        super(PDMAsyncProcess, self).__init__(*args, **kwargs)

    def _compile_and_prepare_functions(self, nbins=10):
        with open(find_kernel('pdm'), 'r') as f:
            pdm2_txt = f.read()
        pdm2_txt = pdm2_txt.replace('//INSERT_NBINS_HERE',
                                    '#define NBINS %d' % nbins)

        self.module = SourceModule(pdm2_txt, options=['--use_fast_math'])

        self.dtypes = [np.intp, np.intp, np.intp, np.intp, np.intp,
                       np.int32, np.int32, np.float32, np.float32]
        for function in ['pdm_binless_tophat', 'pdm_binless_gauss',
                         'pdm_binned_linterp_%dbins' % nbins,
                         'pdm_binned_step_%dbins' % nbins,
                         'pdm_binned_linterp_fast_%dbins' % nbins,
                         'pdm_binned_step_fast_%dbins' % nbins,
                         'pdm_binless_tophat_fast',
                         'pdm_binless_gauss_fast']:
            func = function.replace('_%dbins' % nbins, '')
            func = self.module.get_function(func).prepare(self.dtypes)
            self.prepared_functions[function] = func

    def allocate(self, data, freqs=None, **kwargs):
        """
        Allocate GPU memory for PDM computations.

        Parameters
        ----------
        data: list of tuples
            List of [(t, y, err), ...] or [(t, y, w, freqs), ...] (deprecated)
        freqs: list or np.ndarray, optional
            Frequency grid(s) to search.

        Returns
        -------
        gpu_data: list
            List of GPU arrays.
        pow_cpus: list
            List of CPU arrays for results.
        """
        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        gpu_data, pow_cpus = [], []

        is_deprecated = len(data) > 0 and len(data[0]) == 4

        plot_data = []
        if is_deprecated:
            plot_data = data
        else:
            frqs = freqs
            if frqs is None:
                frqs = [autofrequency(d[0], **kwargs) for d in data]
            elif isinstance(frqs[0], (float, np.floating)):
                frqs = [frqs] * len(data)

            for i, (t, y, err) in enumerate(data):
                # We only need lengths for allocation
                plot_data.append((t, y, None, frqs[i]))

        for t, y, w, freqs in plot_data:

            pow_cpu = host_array((len(freqs),), np.float32)

            t_g, y_g, w_g = None, None, None
            if len(t) > 0:
                t_g, y_g, w_g = tuple([gpuarray.zeros(len(t), dtype=np.float32)
                                       for _ in range(3)])

            pow_g = gpuarray.zeros(len(pow_cpu), dtype=pow_cpu.dtype)
            freqs_g = gpuarray.to_gpu(np.asarray(freqs).astype(np.float32))

            gpu_data.append((t_g, y_g, w_g, freqs_g, pow_g))
            pow_cpus.append(pow_cpu)
        return gpu_data, pow_cpus

    def run(self, data, gpu_data=None, pow_cpus=None, freqs=None,
            kind: Literal['binless_tophat', 'binless_gauss',
                          'binless_tophat_fast', 'binless_gauss_fast',
                          'binned_linterp', 'binned_step',
                          'binned_linterp_fast', 'binned_step_fast'] = 'binned_linterp',
            nbins=10, dphi=0.05, **pdm_kwargs):
        """
        Run PDM on a batch of data.

        Parameters
        ----------
        data: list of tuples
            list of [(t, y, err), ...] containing
            * ``t``: observation times
            * ``y``: observations
            * ``err``: observation uncertainties
            Alternatively, [(t, y, w, freqs), ...] for backward compatibility
            (deprecated). ``w`` are observation weights of any scale (they
            are normalized to sum to one internally).
        gpu_data: list, optional
            list of GPU arrays from ``allocate``
        pow_cpus: list, optional
            list of CPU arrays from ``allocate``
        freqs: list or np.ndarray, optional
            Frequency grid(s) to search.
        kind: str, optional (default: 'binned_linterp')
            PDM variant to use. Available options:
            * 'binless_tophat'
            * 'binless_gauss'
            * 'binless_tophat_fast'
            * 'binless_gauss_fast'
            * 'binned_linterp'
            * 'binned_step'
            * 'binned_linterp_fast'
            * 'binned_step_fast'
        nbins: int, optional (default: 10)
            Number of bins for binned PDM.
        dphi: float, optional (default: 0.05)
            Phase width for binless PDM.
        **pdm_kwargs:
            Extra arguments passed to ``autofrequency`` (when ``freqs``
            is not given) and to ``pdm_async`` (e.g. ``block_size``,
            which must be <= 256).

        Returns
        -------
        results: list
            If depracated format is used: list of power arrays.
            If new format is used: list of (freqs, power) tuples.
            The power arrays are page-locked host buffers filled
            asynchronously: call :meth:`finish` before reading them
            (or use :meth:`batched_run_const_nfreq` / :meth:`large_run`,
            which synchronize for you).

        Notes
        -----
        The returned power is the weighted sum-of-squares ratio
        ``1 - sum(w * (y - model)**2) / sum(w * (y - ybar)**2)`` with
        ``w`` normalized to sum to one and ``model`` the folded-lightcurve
        model of the chosen ``kind`` at each observation's phase. It has
        **no degrees-of-freedom correction**, so it is not Stellingwerf's
        ``1 - Theta``: for pure noise its expectation is
        ``(M - 1) / (N - 1)`` (``M`` occupied bins, ``N`` observations;
        ~0.4 for 20 points in 10 bins) rather than 0, and values are only
        comparable between runs with the same ``nbins`` / ``dphi`` and
        ``N``. See ``docs/source/pdm.rst``.
        """

        if kind in ['binless_tophat', 'binless_gauss',
                    'binless_tophat_fast', 'binless_gauss_fast']:
            function = 'pdm_%s' % kind
        elif kind in ['binned_linterp', 'binned_step',
                      'binned_linterp_fast', 'binned_step_fast']:
            function = 'pdm_%s_%dbins' % (kind, nbins)
        else:
            raise KeyError('Function not available. Please use one of the followings: '
                           'binless_tophat, binless_gauss, '
                           'binless_tophat_fast, binless_gauss_fast, '
                           'binned_linterp, binned_step, '
                           'binned_linterp_fast, binned_step_fast')

        # Backward compatibility check (before kernel compilation, so the
        # warning is emitted even if compilation fails / no GPU is present)
        is_deprecated = len(data) > 0 and len(data[0]) == 4
        if is_deprecated:
            warnings.warn("The (t, y, w, freqs) format is deprecated "
                          "and will be removed in the future. "
                          "Please use the (t, y, err) format "
                          "and pass freqs as a separate argument "
                          "or pass optional keyword arguments "
                          "passed to ``autofrequency``.",
                          DeprecationWarning, stacklevel=2)

        if function not in self.prepared_functions:
            self._compile_and_prepare_functions(nbins=nbins)

        # Prepare data and determine frequencies
        if is_deprecated:
            norm_data = normalize_light_curves(data)
            # The host-side weighted mean/variance and the kernels assume
            # sum(w) == 1; the statistic is invariant to the scale of w,
            # so normalize whatever the caller supplied (raw 1/err^2 or
            # all-ones weights used to give a flat spectrum of 1.0).
            norm_data = [(t, y, np.asarray(w, dtype=np.float64) / np.sum(w), f)
                         for (t, y, w, f) in norm_data]
            frqs = [d[3] for d in data]
        else:
            frqs = freqs
            if frqs is None:
                frqs = [autofrequency(d[0], **pdm_kwargs) for d in data]
            elif isinstance(frqs[0], (float, np.floating)):
                frqs = [frqs] * len(data)

            # Normalize t and y
            norm_data_temp = normalize_light_curves(data)
            norm_data = []
            for i, (t, y, err) in enumerate(norm_data_temp):
                w = weights(err)
                norm_data.append((t, y, w, frqs[i]))

        if pow_cpus is None or gpu_data is None:
            gpu_data, pow_cpus = self.allocate(norm_data, freqs=frqs, **pdm_kwargs)

        streams = [s for i, s in enumerate(self.streams) if i < len(data)]
        func = self.prepared_functions[function]

        results = [pdm_async(stream, cdat, gdat, pcpu, func, dphi=dphi, **pdm_kwargs)
                   for stream, cdat, gdat, pcpu in
                   zip(streams, norm_data, gpu_data, pow_cpus)]

        if is_deprecated:
            return results
        return list(zip(frqs, results))

    @staticmethod
    def _bytes_per_lc(max_ndata, nf):
        """Approximate GPU bytes for one lightcurve's PDM buffers.

        t_g, y_g, w_g (``max_ndata`` float32 each) plus freqs_g and pow_g
        (``nf`` float32 each).
        """
        return (3 * int(max_ndata) + 2 * int(nf)) * 4

    # run() creates one CUDA stream and one page-locked host buffer per
    # lightcurve in the chunk, so device-buffer arithmetic alone would
    # let a huge free-memory pod pick a batch size in the millions --
    # exhausting driver stream/pinned-allocation resources long before
    # GPU memory runs out.
    MAX_BATCH_SIZE = 256

    def _batch_size_from_memory(self, max_ndata, nf, n_lcs, max_memory=None):
        """Largest batch (number of lightcurves held on the GPU at once)
        that fits in ``max_memory`` bytes; capped at ``n_lcs``,
        ``MAX_BATCH_SIZE`` and >= 1.

        ``max_memory`` defaults to 90% of the device's free memory.
        """
        if max_memory is None:
            free, _total = cuda.mem_get_info()
            max_memory = int(0.9 * free)
        per_lc = self._bytes_per_lc(max_ndata, nf)
        batch_size = max(1, int(max_memory // per_lc))
        return min(batch_size, int(n_lcs), self.MAX_BATCH_SIZE)

    def batched_run_const_nfreq(self, data, batch_size=10, freqs=None,
                                **kwargs):
        """Run PDM on many lightcurves that share one frequency grid.

        Processes ``data`` in chunks of ``batch_size`` lightcurves,
        synchronizing and freeing each chunk's GPU memory before the next
        (so peak GPU memory scales with ``batch_size``, not
        ``len(data)``), and resolves the shared frequency grid once.
        Results match per-lightcurve :meth:`run`.

        Parameters
        ----------
        data : list of (t, y, err)
        batch_size : int, optional (default: 10)
            Lightcurves resident on the GPU per chunk.
        freqs : array_like, optional
            Shared frequency grid. If None, it is derived once from the
            longest-baseline lightcurve via ``autofrequency`` and reused.
        **kwargs :
            Passed to :meth:`run` (e.g. ``kind``, ``nbins``, ``dphi``,
            ``block_size``).

        Returns
        -------
        list of (freqs, power)
        """
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1; got %d" % batch_size)
        if any(len(d) != 3 for d in data):
            raise ValueError("batched_run_const_nfreq expects (t, y, err) "
                             "tuples; the deprecated (t, y, w, freqs) "
                             "run() format is not supported here")
        if len(data) == 0:
            return []
        if freqs is None:
            dmax = max(data, key=lambda d: np.max(d[0]) - np.min(d[0]))
            freqs = autofrequency(dmax[0], **kwargs)
        freqs = np.asarray(freqs).astype(np.float32)

        results = []
        for start in range(0, len(data), int(batch_size)):
            chunk = data[start:start + int(batch_size)]
            chunk_res = self.run(chunk, freqs=freqs, **kwargs)
            self.finish()
            for _f, p in chunk_res:
                results.append((freqs, np.copy(p)))
        return results

    def large_run(self, data, freqs=None, max_memory=None, **kwargs):
        """Memory-capped batched PDM for lightcurve collections too large
        to fit on the GPU at once.

        Picks ``batch_size`` so that no more than ``max_memory`` bytes
        (default: 90% of free GPU memory) of lightcurve buffers are
        resident at a time, then defers to :meth:`batched_run_const_nfreq`.
        Results match per-lightcurve :meth:`run`.
        """
        if len(data) == 0:
            return []
        if freqs is None:
            dmax = max(data, key=lambda d: np.max(d[0]) - np.min(d[0]))
            freqs = autofrequency(dmax[0], **kwargs)
        freqs = np.asarray(freqs).astype(np.float32)

        max_ndata = max(len(d[0]) for d in data)
        batch_size = self._batch_size_from_memory(
            max_ndata, len(freqs), len(data), max_memory=max_memory)
        return self.batched_run_const_nfreq(
            data, batch_size=batch_size, freqs=freqs, **kwargs)
