"""
GPU-accelerated Fast Folding BLS (fBLS) periodogram.

Implements the Fast Folding Algorithm (Staelin 1969; Shahaf et al. 2022)
for BLS transit search. Produces identical BLS Signal Residue statistics
to the standard binned BLS, but with O(N + N_p * m * log(N_p)) complexity
instead of O(N_f * N).

References
----------
.. [S2022] Shahaf et al. 2022, MNRAS 513, 2732 (arXiv:2204.02398)
.. [K2002] Kovacs et al. 2002, A&A 391, 369
.. [S1969] Staelin 1969, Proc. IEEE 57, 724
"""
import threading
from collections import OrderedDict

import pycuda.autoprimaryctx
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .utils import find_kernel, _module_reader

import numpy as np

_default_block_size = 256

_all_function_names = [
    'ffa_bin_sections',
    'ffa_combine_pairs',
    'ffa_butterfly',
    'ffa_score',
    'ffa_bin_sections_batch',
    'ffa_butterfly_batch',
    'ffa_score_batch',
]

_function_signatures = {
    'ffa_bin_sections': [
        np.intp, np.intp, np.intp,         # t, yw, w
        np.intp, np.intp,                   # section_starts, section_ends
        np.intp, np.intp,                   # section_yw, section_w
        np.float32,                         # P0
        np.uint32,                          # m
        np.uint32,                          # N_p
    ],
    'ffa_combine_pairs': [
        np.intp, np.intp,                   # section_yw, section_w
        np.intp, np.intp,                   # folds_yw, folds_w
        np.uint32,                          # m
        np.uint32,                          # N_p
    ],
    'ffa_butterfly': [
        np.intp, np.intp,                   # in_yw, in_w
        np.intp, np.intp,                   # out_yw, out_w
        np.intp,                            # shift_vector
        np.uint32,                          # m
        np.uint32,                          # N_p
        np.uint32,                          # folds_per_group
        np.uint32,                          # half_folds
    ],
    'ffa_score': [
        np.intp, np.intp,                   # folds_yw, folds_w
        np.intp,                            # sr_out
        np.uint32,                          # m
        np.uint32,                          # N_p
        np.uint32,                          # nbins0
        np.uint32,                          # nbinsf
        np.float32,                         # dlogq
        np.uint32,                          # ignore_negative_delta_sols
    ],
    'ffa_bin_sections_batch': [
        np.intp, np.intp, np.intp,         # t, yw, w
        np.intp,                            # P0_arr
        np.intp,                            # m_oct_arr
        np.intp,                            # fold_offsets
        np.intp, np.intp,                   # starts_all, ends_all
        np.intp, np.intp,                   # section_yw, section_w
        np.uint32,                          # N_p
        np.uint32,                          # max_m
    ],
    'ffa_butterfly_batch': [
        np.intp, np.intp,                   # in_yw, in_w
        np.intp, np.intp,                   # out_yw, out_w
        np.intp,                            # m_oct_arr
        np.intp,                            # fold_offsets
        np.uint32,                          # N_p
        np.uint32,                          # level
    ],
    'ffa_score_batch': [
        np.intp, np.intp,                   # folds_yw, folds_w
        np.intp,                            # sr_out
        np.intp,                            # m_oct_arr
        np.intp,                            # fold_offsets
        np.intp,                            # nbins0_arr
        np.intp,                            # nbinsf_arr
        np.intp,                            # sr_offsets
        np.uint32,                          # N_p
        np.uint32,                          # max_m
        np.float32,                         # dlogq
        np.uint32,                          # ignore_negative_delta_sols
    ],
}

# Kernel cache
_KERNEL_CACHE_MAX_SIZE = 10
_kernel_cache = OrderedDict()
_kernel_cache_lock = threading.Lock()


def compile_ffa_bls(block_size=_default_block_size,
                    function_names=None, prepare=True):
    """
    Compile FFA-BLS CUDA kernels.

    Parameters
    ----------
    block_size : int
        CUDA threads per block.
    function_names : list, optional
        Kernel function names to compile. Default: all.
    prepare : bool
        Whether to prepare functions for faster launching.

    Returns
    -------
    functions : dict
        Compiled kernel functions.
    """
    if function_names is None:
        function_names = _all_function_names

    cppd = dict(BLOCK_SIZE=block_size)
    kernel_txt = _module_reader(find_kernel('ffa_bls'), cpp_defs=cppd)
    module = SourceModule(kernel_txt, options=['--use_fast_math'])

    functions = {name: module.get_function(name) for name in function_names}

    if prepare:
        for name in functions:
            functions[name] = functions[name].prepare(_function_signatures[name])

    return functions


def _get_cached_ffa_kernels(block_size, function_names=None):
    """Get compiled FFA kernels from cache, compiling if needed."""
    if function_names is None:
        function_names = _all_function_names

    key = (block_size, tuple(sorted(function_names)))

    with _kernel_cache_lock:
        if key in _kernel_cache:
            _kernel_cache.move_to_end(key)
            return _kernel_cache[key]

        compiled = compile_ffa_bls(block_size=block_size,
                                   function_names=function_names)
        _kernel_cache[key] = compiled
        _kernel_cache.move_to_end(key)

        if len(_kernel_cache) > _KERNEL_CACHE_MAX_SIZE:
            _kernel_cache.popitem(last=False)

        return compiled


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_shift_vectors(n_levels):
    """
    Build FFA shift vectors for each butterfly level.

    The shift vector follows the recurrence from Shahaf et al. (2022):
        S_1 = (0, 1)
        S_{l+1} = concat(S_l, S_l + 2^(l-1))

    Parameters
    ----------
    n_levels : int
        Total number of butterfly levels (log2(N_p)).

    Returns
    -------
    shift_vectors : list of ndarray
        shift_vectors[l] has 2^(l+1) elements for level l.
        Level 0 is the initial pair-combine (shifts 0, 1).
        Levels 1..n_levels-1 are butterfly levels.
    """
    # S_1 = (0, 1) — used by combine_pairs (level 0)
    S = np.array([0, 1], dtype=np.int32)
    vectors = [S.copy()]

    for l in range(1, n_levels):
        # S_{l+1} = concat(S_l, S_l + 2^(l-1))
        S = np.concatenate([S, S + (1 << (l - 1))])
        vectors.append(S.copy())

    return vectors


def _compute_section_boundaries(t, P0, N_p):
    """
    Compute section boundaries for irregularly sampled data.

    Section s spans time [t[0] + s*P0, t[0] + (s+1)*P0).

    Parameters
    ----------
    t : ndarray
        Sorted observation times.
    P0 : float
        Base period for this octave.
    N_p : int
        Number of sections.

    Returns
    -------
    starts, ends : ndarray (uint32)
        Start and end indices into t for each section.
    """
    boundaries = np.arange(N_p + 1, dtype=np.float64) * P0 + t[0]
    starts = np.searchsorted(t, boundaries[:-1]).astype(np.uint32)
    ends = np.searchsorted(t, boundaries[1:]).astype(np.uint32)
    return starts, ends


def _compute_butterfly_shifts(m_oct, N_p, level):
    """
    Compute classic FFA shift vector for one butterfly level.

    Uses the Staelin (1969) binary decomposition: at level l,
    fold d gets shift -(2^l) if bit l of d is set, else 0.
    The negative sign corresponds to P(d) > P0 (longer period
    = backward phase drift).

    Total accumulated drift for fold d = -d bins (mod m).

    Parameters
    ----------
    m_oct : int
        Number of phase bins.
    N_p : int
        Total number of folds (trial periods).
    level : int
        Butterfly level (0-indexed).

    Returns
    -------
    shifts : ndarray (int32), shape (N_p,)
        Shift for each output fold.
    """
    shift_mag = (1 << level) % m_oct  # 2^level mod m
    shifts = np.zeros(N_p, dtype=np.int32)
    for d in range(N_p):
        if (d >> level) & 1:
            shifts[d] = (-shift_mag) % m_oct  # negative shift
        # else: shifts[d] = 0 (already initialized)
    return shifts


def _ffa_single_octave_cpu(t, yw, w_f32, P0, m_oct, N_p,
                            qmin, qmax, dlogq,
                            ignore_negative_delta_sols):
    """
    Run FFA for a single octave on CPU.

    Uses the classic FFA butterfly (Staelin 1969): at level l,
    the shift is 0 or -(2^l) bins depending on bit l of the
    drift index d. Total drift = -d bins for fold d, matching
    the period grid P(d) = (m + d/(N_p-1)) * dt.

    Parameters
    ----------
    t : ndarray (float64, sorted)
    yw : ndarray (float32, weighted residuals)
    w_f32 : ndarray (float32, weights)
    P0 : float, base period
    m_oct : int, number of phase bins (= section length in dt units)
    N_p : int, number of sections (power of 2)
    qmin, qmax, dlogq : BLS scoring parameters
    ignore_negative_delta_sols : bool

    Returns
    -------
    sr : ndarray (float32), shape (N_p,)
    """
    n_levels = int(np.log2(N_p))
    starts, ends = _compute_section_boundaries(t, P0, N_p)
    dt = P0 / m_oct

    # Scoring parameters scale with m_oct
    nbins0 = max(1, int(np.ceil(m_oct * qmin)))
    nbinsf = min(m_oct, max(nbins0, int(np.floor(m_oct * qmax))))

    # --- Bin observations into per-section profiles ---
    fold_yw = np.zeros((N_p, m_oct), dtype=np.float32)
    fold_w = np.zeros((N_p, m_oct), dtype=np.float32)

    # Assign each observation to its section
    phases_all = (t.astype(np.float64) / P0)
    phases_frac = (phases_all - np.floor(phases_all)).astype(np.float32)
    bins_all = np.clip(np.floor(m_oct * phases_frac).astype(np.int32),
                       0, m_oct - 1)
    # Flat index: section * m_oct + bin
    for sec in range(N_p):
        s, e = int(starts[sec]), int(ends[sec])
        if s >= e:
            continue
        np.add.at(fold_yw[sec], bins_all[s:e], yw[s:e])
        np.add.at(fold_w[sec], bins_all[s:e], w_f32[s:e])

    # --- Butterfly: classic FFA shifts (0 or -2^l at level l) ---
    # Vectorized: at each level, only two shifts exist (0 and -(2^l)),
    # and we compute all folds at once using fancy indexing.
    d_global = np.arange(N_p)
    for level in range(n_levels):
        half = 1 << level
        group_size = 1 << (level + 1)

        # Compute left/right source indices for all folds
        group = d_global // group_size
        d_local = d_global % group_size
        d_sub = d_local % half
        left_idx = (2 * group) * half + d_sub
        right_idx = (2 * group + 1) * half + d_sub

        # Mask: which folds get shift vs no-shift
        needs_shift = ((d_global >> level) & 1).astype(bool)

        new_yw = fold_yw[left_idx].copy()
        new_w = fold_w[left_idx].copy()

        # Add unshifted right for no-shift folds
        no_shift_mask = ~needs_shift
        new_yw[no_shift_mask] += fold_yw[right_idx[no_shift_mask]]
        new_w[no_shift_mask] += fold_w[right_idx[no_shift_mask]]

        # Add shifted right for shift folds (shift = -(2^l) mod m)
        if np.any(needs_shift):
            shift_amount = (-(1 << level)) % m_oct
            # Circular shift via column reindexing
            shifted_cols = np.arange(m_oct)
            shifted_cols = (shifted_cols - shift_amount) % m_oct
            new_yw[needs_shift] += fold_yw[right_idx[needs_shift]][:, shifted_cols]
            new_w[needs_shift] += fold_w[right_idx[needs_shift]][:, shifted_cols]

        fold_yw = new_yw
        fold_w = new_w

    # --- Score: vectorized BLS box scan ---
    # Build list of trial widths (log-spaced)
    trial_widths = []
    w_val = nbins0
    while w_val <= nbinsf:
        trial_widths.append(w_val)
        if dlogq > 0:
            step = max(1, int(np.floor(dlogq * w_val)))
            w_val += step
        else:
            w_val += 1

    # Circular cumulative sum: duplicate bins for wrap-around
    yw_ext = np.concatenate([fold_yw, fold_yw[:, :nbinsf]], axis=1)
    w_ext = np.concatenate([fold_w, fold_w[:, :nbinsf]], axis=1)
    yw_cumsum = np.cumsum(yw_ext, axis=1)
    w_cumsum = np.cumsum(w_ext, axis=1)

    sr = np.zeros(N_p, dtype=np.float32)
    for width in trial_widths:
        # Sum over [start, start+width) for all start bins
        # cumsum[start+width] - cumsum[start] for start in 0..m_oct-1
        starts_arr = np.arange(m_oct)
        box_yw = yw_cumsum[:, starts_arr + width] - yw_cumsum[:, starts_arr]
        box_w = w_cumsum[:, starts_arr + width] - w_cumsum[:, starts_arr]

        # BLS value: yw^2 / (w * (1-w))
        valid = (box_w > 1e-10) & (box_w < 1.0 - 1e-10)
        if ignore_negative_delta_sols:
            valid &= (box_yw < 0)
        denom = np.where(valid, box_w * (1.0 - box_w), 1.0)
        bls = np.where(valid,
                       box_yw ** 2 / denom,
                       0.0).astype(np.float32)

        # Max over all start bins for each fold
        max_bls = np.max(bls, axis=1)
        sr = np.maximum(sr, max_bls)

    return sr


def _preprocess(t, y, dy):
    """Sort, compute weights, weighted residuals, and normalization."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)

    order = np.argsort(t)
    t = t[order]
    y = y[order]
    dy = dy[order]

    w = np.power(dy, -2)
    w /= w.sum()
    ybar = np.dot(w, y)
    yy = np.dot(w, (y - ybar) ** 2)
    yw = ((y - ybar) * w).astype(np.float32)
    w_f32 = w.astype(np.float32)

    return t, yw, w_f32, yy


def _compute_octave_params(period_min, period_max, m_bins, T_total):
    """
    Compute FFA octave parameters.

    The reference cadence dt = period_min / m_bins ensures that at
    period_min, we have exactly m_bins phase bins. Each octave at
    section length m_oct has m_oct phase bins.

    Returns
    -------
    dt : float
    m_min : int (= m_bins)
    m_max : int
    """
    dt = period_min / m_bins
    m_min = m_bins
    m_max = int(np.ceil(period_max / dt))
    return dt, m_min, m_max


def _group_octaves_by_np(m_min, m_max, dt, T_total, qmin, qmax,
                         shmem_lim, block_size):
    """
    Pre-compute octave parameters and group by N_p value.

    Returns
    -------
    groups : dict
        N_p -> list of octave metadata dicts, each containing:
        P0, m_oct, N_p, nbins0, nbinsf, n_levels
    """
    float_size = 4  # sizeof(float32)
    groups = {}

    for m_oct in range(m_min, m_max + 1):
        P0 = m_oct * dt

        n_sections = max(1, int(np.round(T_total / P0)))
        N_p = _next_power_of_2(n_sections)

        if N_p < 4:
            continue

        # Check shared memory for both bin and score kernels
        shmem_bin = 2 * m_oct * float_size
        shmem_score = (2 * m_oct + block_size) * float_size
        if shmem_bin > shmem_lim or shmem_score > shmem_lim:
            continue

        nbins0 = max(1, int(np.ceil(m_oct * qmin)))
        nbinsf = min(m_oct, max(nbins0, int(np.floor(m_oct * qmax))))
        n_levels = int(np.log2(N_p))

        info = {
            'P0': P0,
            'm_oct': m_oct,
            'N_p': N_p,
            'nbins0': nbins0,
            'nbinsf': nbinsf,
            'n_levels': n_levels,
        }

        if N_p not in groups:
            groups[N_p] = []
        groups[N_p].append(info)

    return groups


def _process_np_group_gpu(N_p, octaves, t, t_g, yw_g, w_g, functions,
                          block_size, stream, dlogq,
                          ignore_negative_delta_sols, yy, dt,
                          shmem_lim, max_gpu_bytes=None):
    """
    Process all octaves sharing the same N_p using batch kernels.

    Parameters
    ----------
    N_p : int
        Number of sections/folds (power of 2).
    octaves : list of dict
        Octave metadata from _group_octaves_by_np.
    t : ndarray (float64, sorted)
        Observation times.
    t_g, yw_g, w_g : gpuarray
        GPU arrays of observation data.
    functions : dict
        Compiled batch kernel functions.
    max_gpu_bytes : int or None
        Max GPU memory per sub-batch. None = no limit.

    Returns
    -------
    all_periods : list of ndarray
    all_sr : list of ndarray
    """
    float_size = 4
    n_levels = int(np.log2(N_p))

    bin_func   = functions['ffa_bin_sections_batch']
    bfly_func  = functions['ffa_butterfly_batch']
    score_func = functions['ffa_score_batch']
    block = (block_size, 1, 1)

    # Sub-batch by memory budget
    if max_gpu_bytes is None:
        dev = pycuda.autoprimaryctx.device
        max_gpu_bytes = int(dev.total_memory() * 0.5)

    all_periods = []
    all_sr = []

    # Split octaves into sub-batches that fit in memory
    sub_batches = []
    current_batch = []
    current_bytes = 0

    for oct_info in octaves:
        m_oct = oct_info['m_oct']
        # 4 fold buffers + 1 sr buffer per octave
        oct_bytes = (4 * N_p * m_oct + N_p) * float_size
        if current_batch and current_bytes + oct_bytes > max_gpu_bytes:
            sub_batches.append(current_batch)
            current_batch = []
            current_bytes = 0
        current_batch.append(oct_info)
        current_bytes += oct_bytes

    if current_batch:
        sub_batches.append(current_batch)

    for batch in sub_batches:
        n_oct = len(batch)

        # Build parameter arrays
        P0_arr = np.array([o['P0'] for o in batch], dtype=np.float32)
        m_oct_arr = np.array([o['m_oct'] for o in batch], dtype=np.uint32)
        nbins0_arr = np.array([o['nbins0'] for o in batch], dtype=np.uint32)
        nbinsf_arr = np.array([o['nbinsf'] for o in batch], dtype=np.uint32)
        max_m = int(np.max(m_oct_arr))

        # Compute fold offsets (cumulative)
        fold_sizes = np.array([N_p * o['m_oct'] for o in batch],
                              dtype=np.uint32)
        fold_offsets = np.zeros(n_oct, dtype=np.uint32)
        fold_offsets[1:] = np.cumsum(fold_sizes[:-1])
        total_fold_size = int(np.sum(fold_sizes))

        # SR offsets (N_p per octave)
        sr_offsets = np.arange(n_oct, dtype=np.uint32) * N_p
        total_sr_size = n_oct * N_p

        # Section boundaries: [n_oct * N_p] flat arrays
        starts_all = np.zeros(n_oct * N_p, dtype=np.uint32)
        ends_all = np.zeros(n_oct * N_p, dtype=np.uint32)
        for i, o in enumerate(batch):
            s, e = _compute_section_boundaries(t, o['P0'], N_p)
            starts_all[i * N_p:(i + 1) * N_p] = s
            ends_all[i * N_p:(i + 1) * N_p] = e

        # Upload parameter arrays
        P0_g = gpuarray.to_gpu(P0_arr)
        m_oct_g = gpuarray.to_gpu(m_oct_arr)
        fold_offsets_g = gpuarray.to_gpu(fold_offsets)
        nbins0_g = gpuarray.to_gpu(nbins0_arr)
        nbinsf_g = gpuarray.to_gpu(nbinsf_arr)
        sr_offsets_g = gpuarray.to_gpu(sr_offsets)
        starts_g = gpuarray.to_gpu(starts_all)
        ends_g = gpuarray.to_gpu(ends_all)

        # Allocate fold buffers (A/B for ping-pong) and SR
        fold_yw_A = gpuarray.zeros(total_fold_size, dtype=np.float32)
        fold_w_A = gpuarray.zeros(total_fold_size, dtype=np.float32)
        fold_yw_B = gpuarray.zeros(total_fold_size, dtype=np.float32)
        fold_w_B = gpuarray.zeros(total_fold_size, dtype=np.float32)
        sr_g = gpuarray.zeros(total_sr_size, dtype=np.float32)

        # Shared memory for bin/score kernels
        shmem_bin = 2 * max_m * float_size
        shmem_score = (2 * max_m + block_size) * float_size

        # --- Kernel 1: Bin sections ---
        bin_grid = (int(N_p), int(n_oct))
        args = (bin_grid, block)
        if stream is not None:
            args += (stream,)
        args += (t_g.ptr, yw_g.ptr, w_g.ptr)
        args += (P0_g.ptr, m_oct_g.ptr, fold_offsets_g.ptr)
        args += (starts_g.ptr, ends_g.ptr)
        args += (fold_yw_A.ptr, fold_w_A.ptr)
        args += (np.uint32(N_p), np.uint32(max_m))

        if stream is not None:
            bin_func.prepared_async_call(*args, shared_size=int(shmem_bin))
        else:
            bin_func.prepared_call(*args, shared_size=int(shmem_bin))

        # --- Kernel 2: Butterfly levels ---
        read_yw, read_w = fold_yw_A, fold_w_A
        write_yw, write_w = fold_yw_B, fold_w_B

        for level in range(n_levels):
            bfly_grid = (int(N_p), int(n_oct))
            args = (bfly_grid, block)
            if stream is not None:
                args += (stream,)
            args += (read_yw.ptr, read_w.ptr)
            args += (write_yw.ptr, write_w.ptr)
            args += (m_oct_g.ptr, fold_offsets_g.ptr)
            args += (np.uint32(N_p), np.uint32(level))

            if stream is not None:
                bfly_func.prepared_async_call(*args)
            else:
                bfly_func.prepared_call(*args)

            read_yw, write_yw = write_yw, read_yw
            read_w, write_w = write_w, read_w

        # --- Kernel 3: Score ---
        score_grid = (int(N_p), int(n_oct))
        args = (score_grid, block)
        if stream is not None:
            args += (stream,)
        args += (read_yw.ptr, read_w.ptr)
        args += (sr_g.ptr,)
        args += (m_oct_g.ptr, fold_offsets_g.ptr)
        args += (nbins0_g.ptr, nbinsf_g.ptr, sr_offsets_g.ptr)
        args += (np.uint32(N_p), np.uint32(max_m))
        args += (np.float32(dlogq),)
        args += (np.uint32(ignore_negative_delta_sols),)

        if stream is not None:
            score_func.prepared_async_call(*args, shared_size=int(shmem_score))
        else:
            score_func.prepared_call(*args, shared_size=int(shmem_score))

        # Download and slice results
        sr_all = sr_g.get()

        for i, o in enumerate(batch):
            m_oct = o['m_oct']
            sr = sr_all[i * N_p:(i + 1) * N_p].copy()
            if yy > 0:
                sr /= yy

            if N_p > 1:
                periods = (m_oct + np.arange(N_p, dtype=np.float64)
                           / (N_p - 1)) * dt
            else:
                periods = np.array([m_oct * dt])

            all_periods.append(periods)
            all_sr.append(sr)

    return all_periods, all_sr


def eebls_ffa_gpu(t, y, dy, period_min, period_max, m_bins=None,
                  qmin=0.01, qmax=0.15, dlogq=0.2,
                  ignore_negative_delta_sols=True,
                  block_size=None, stream=None,
                  functions=None, use_batch=True):
    """
    BLS periodogram using Fast Folding Algorithm on GPU.

    Parameters
    ----------
    t : array_like
        Observation times (days).
    y : array_like
        Flux observations.
    dy : array_like
        Flux uncertainties.
    period_min : float
        Minimum search period (same units as t).
    period_max : float
        Maximum search period (same units as t).
    m_bins : int, optional
        Number of phase bins at the shortest period. If None,
        auto-select as ceil(1/qmin). Longer periods will have
        proportionally more bins.
    qmin : float
        Minimum transit duty cycle (fractional phase).
    qmax : float
        Maximum transit duty cycle (fractional phase).
    dlogq : float
        Logarithmic spacing of trial transit widths.
    ignore_negative_delta_sols : bool
        Ignore solutions where the in-transit mean is brighter.
    block_size : int, optional
        CUDA block size. Auto-selected if None.
    stream : pycuda.driver.Stream, optional
        CUDA stream for async execution.
    functions : dict, optional
        Pre-compiled kernel functions.
    use_batch : bool
        Use Phase 2 batch kernels (default True). Set False for
        sequential fallback (Phase 1).

    Returns
    -------
    periods : ndarray
        Trial periods (FFA native grid), sorted.
    power : ndarray
        BLS Signal Residue at each trial period.
    """
    t, yw, w_f32, yy = _preprocess(t, y, dy)
    ndata = len(t)
    T_total = t[-1] - t[0]
    t_f32 = t.astype(np.float32)

    if m_bins is None:
        m_bins = int(np.ceil(1.0 / qmin))

    dt, m_min, m_max = _compute_octave_params(period_min, period_max,
                                               m_bins, T_total)

    if block_size is None:
        block_size = _default_block_size

    if functions is None:
        functions = _get_cached_ffa_kernels(block_size)

    dev = pycuda.autoprimaryctx.device
    att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
    shmem_lim = dev.get_attribute(att)

    t_g  = gpuarray.to_gpu(t_f32)
    yw_g = gpuarray.to_gpu(yw)
    w_g  = gpuarray.to_gpu(w_f32)

    if use_batch:
        return _eebls_ffa_gpu_batched(
            t, t_g, yw_g, w_g, yy, dt, m_min, m_max, T_total,
            qmin, qmax, dlogq, ignore_negative_delta_sols,
            block_size, stream, functions, shmem_lim)
    else:
        return _eebls_ffa_gpu_sequential(
            t, t_g, yw_g, w_g, yy, dt, m_min, m_max, T_total,
            qmin, qmax, dlogq, ignore_negative_delta_sols,
            block_size, stream, functions, shmem_lim)


def _eebls_ffa_gpu_batched(t, t_g, yw_g, w_g, yy, dt, m_min, m_max,
                            T_total, qmin, qmax, dlogq,
                            ignore_negative_delta_sols,
                            block_size, stream, functions, shmem_lim):
    """Phase 2: Batch octaves by N_p for minimal kernel launches."""
    groups = _group_octaves_by_np(m_min, m_max, dt, T_total,
                                  qmin, qmax, shmem_lim, block_size)

    all_periods = []
    all_sr = []

    for N_p in sorted(groups.keys()):
        octaves = groups[N_p]
        periods_list, sr_list = _process_np_group_gpu(
            N_p, octaves, t, t_g, yw_g, w_g, functions,
            block_size, stream, dlogq,
            ignore_negative_delta_sols, yy, dt, shmem_lim)
        all_periods.extend(periods_list)
        all_sr.extend(sr_list)

    if len(all_periods) == 0:
        return np.array([]), np.array([])

    periods = np.concatenate(all_periods)
    power = np.concatenate(all_sr)

    order = np.argsort(periods)
    return periods[order], power[order]


def _eebls_ffa_gpu_sequential(t, t_g, yw_g, w_g, yy, dt, m_min, m_max,
                               T_total, qmin, qmax, dlogq,
                               ignore_negative_delta_sols,
                               block_size, stream, functions, shmem_lim):
    """Phase 1: Sequential per-octave kernel launches (fallback)."""
    float_size = np.float32(1).nbytes
    block = (block_size, 1, 1)

    bin_func   = functions['ffa_bin_sections']
    bfly_func  = functions['ffa_butterfly']
    score_func = functions['ffa_score']

    all_periods = []
    all_sr = []

    for m_oct in range(m_min, m_max + 1):
        P0 = m_oct * dt

        n_sections = max(1, int(np.round(T_total / P0)))
        N_p = _next_power_of_2(n_sections)

        if N_p < 4:
            continue

        n_levels = int(np.log2(N_p))

        # Scoring parameters scale with m_oct
        nbins0 = max(1, int(np.ceil(m_oct * qmin)))
        nbinsf = min(m_oct, max(nbins0, int(np.floor(m_oct * qmax))))

        starts, ends = _compute_section_boundaries(t, P0, N_p)

        # GPU memory
        section_yw_g = gpuarray.zeros(N_p * m_oct, dtype=np.float32)
        section_w_g  = gpuarray.zeros(N_p * m_oct, dtype=np.float32)
        folds_yw_A = gpuarray.zeros(N_p * m_oct, dtype=np.float32)
        folds_w_A  = gpuarray.zeros(N_p * m_oct, dtype=np.float32)
        folds_yw_B = gpuarray.zeros(N_p * m_oct, dtype=np.float32)
        folds_w_B  = gpuarray.zeros(N_p * m_oct, dtype=np.float32)
        sr_g = gpuarray.zeros(N_p, dtype=np.float32)
        starts_g = gpuarray.to_gpu(starts)
        ends_g   = gpuarray.to_gpu(ends)

        # Kernel 1: Bin sections
        shmem_bin = 2 * m_oct * float_size
        if shmem_bin > shmem_lim:
            continue

        bin_grid = (int(N_p), 1)
        args = (bin_grid, block)
        if stream is not None:
            args += (stream,)
        args += (t_g.ptr, yw_g.ptr, w_g.ptr)
        args += (starts_g.ptr, ends_g.ptr)
        args += (section_yw_g.ptr, section_w_g.ptr)
        args += (np.float32(P0),)
        args += (np.uint32(m_oct), np.uint32(N_p))

        if stream is not None:
            bin_func.prepared_async_call(*args, shared_size=int(shmem_bin))
        else:
            bin_func.prepared_call(*args, shared_size=int(shmem_bin))

        # Kernel 2: Butterfly (all levels, with analytical shifts)
        read_yw, read_w = section_yw_g, section_w_g
        write_yw, write_w = folds_yw_A, folds_w_A

        for level in range(n_levels):
            folds_per_group = 1 << (level + 1)
            half_folds = 1 << level

            # Classic FFA shifts: 0 or -(2^l) at level l
            shifts = _compute_butterfly_shifts(m_oct, N_p, level)
            shifts_g = gpuarray.to_gpu(shifts)

            bfly_grid = (int(N_p), 1)
            args = (bfly_grid, block)
            if stream is not None:
                args += (stream,)
            args += (read_yw.ptr, read_w.ptr)
            args += (write_yw.ptr, write_w.ptr)
            args += (shifts_g.ptr,)
            args += (np.uint32(m_oct), np.uint32(N_p))
            args += (np.uint32(folds_per_group), np.uint32(half_folds))

            if stream is not None:
                bfly_func.prepared_async_call(*args)
            else:
                bfly_func.prepared_call(*args)

            read_yw, write_yw = write_yw, read_yw
            read_w, write_w = write_w, read_w

        # Kernel 3: Score
        shmem_score = (2 * m_oct + block_size) * float_size
        if shmem_score > shmem_lim:
            continue

        score_grid = (int(N_p), 1)
        args = (score_grid, block)
        if stream is not None:
            args += (stream,)
        args += (read_yw.ptr, read_w.ptr)
        args += (sr_g.ptr,)
        args += (np.uint32(m_oct), np.uint32(N_p))
        args += (np.uint32(nbins0), np.uint32(nbinsf))
        args += (np.float32(dlogq),)
        args += (np.uint32(ignore_negative_delta_sols),)

        if stream is not None:
            score_func.prepared_async_call(*args, shared_size=int(shmem_score))
        else:
            score_func.prepared_call(*args, shared_size=int(shmem_score))

        sr = sr_g.get()
        if yy > 0:
            sr /= yy

        # Period grid: P(i) = (m_oct + i/(N_p-1)) * dt
        if N_p > 1:
            periods = (m_oct + np.arange(N_p, dtype=np.float64) / (N_p - 1)) * dt
        else:
            periods = np.array([m_oct * dt])

        all_periods.append(periods)
        all_sr.append(sr)

    if len(all_periods) == 0:
        return np.array([]), np.array([])

    periods = np.concatenate(all_periods)
    power = np.concatenate(all_sr)

    order = np.argsort(periods)
    return periods[order], power[order]


def eebls_ffa_cpu(t, y, dy, period_min, period_max, m_bins=None,
                  qmin=0.01, qmax=0.15, dlogq=0.2,
                  ignore_negative_delta_sols=True):
    """
    CPU reference implementation of FFA-BLS for testing.

    Same algorithm as the GPU version but runs entirely on CPU.
    Useful for correctness testing without GPU access.

    Parameters are identical to eebls_ffa_gpu.
    """
    t, yw, w_f32, yy = _preprocess(t, y, dy)
    ndata = len(t)
    T_total = t[-1] - t[0]

    if m_bins is None:
        m_bins = int(np.ceil(1.0 / qmin))

    dt, m_min, m_max = _compute_octave_params(period_min, period_max,
                                               m_bins, T_total)

    all_periods = []
    all_sr = []

    for m_oct in range(m_min, m_max + 1):
        P0 = m_oct * dt

        n_sections = max(1, int(np.round(T_total / P0)))
        N_p = _next_power_of_2(n_sections)

        if N_p < 4:
            continue

        sr = _ffa_single_octave_cpu(t, yw, w_f32, P0, m_oct, N_p,
                                     qmin, qmax, dlogq,
                                     ignore_negative_delta_sols)
        if yy > 0:
            sr /= yy

        if N_p > 1:
            periods = (m_oct + np.arange(N_p, dtype=np.float64) / (N_p - 1)) * dt
        else:
            periods = np.array([m_oct * dt])

        all_periods.append(periods)
        all_sr.append(sr)

    if len(all_periods) == 0:
        return np.array([]), np.array([])

    periods = np.concatenate(all_periods)
    power = np.concatenate(all_sr)

    order = np.argsort(periods)
    return periods[order], power[order]
