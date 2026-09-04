"""
Memory management for batch BLS GPU operations.

Handles padded multi-lightcurve data layout with page-aligned CPU
arrays (NOT page-locked/pinned: async transfers fall back to
synchronous staged copies) and GPU arrays for batch processing.
"""
import numpy as np

import pycuda.driver as cuda  # noqa: F401  (kept for transfer methods / API)
import pycuda.gpuarray as gpuarray

from ..base import ensure_context
from ._host import host_array
from ..utils import (subtract_epoch, conflict_scatter_perm,
                     check_lightcurve, check_freqs)


class BLSBatchMemory:
    """
    Memory manager for multi-lightcurve batch BLS.

    Data layout: all LC arrays padded to max_ndata and concatenated.
        t_all[lc_idx * max_ndata + i]    for i < ndata_per_lc[lc_idx]
        yw_all[lc_idx * max_ndata + i]
        w_all[lc_idx * max_ndata + i]

    Output layout:
        bls_all[lc_idx * nfreqs + freq_idx]

    Parameters
    ----------
    max_ndata : int
        Maximum observations per lightcurve (arrays padded to this).
    n_lcs : int
        Number of lightcurves in this batch.
    nfreqs : int
        Number of trial frequencies.
    stream : pycuda.driver.Stream, optional
        CUDA stream for async transfers.
    """

    def __init__(self, max_ndata, n_lcs, nfreqs, stream=None, pinned=True):
        # Constructing GPU memory is a "first GPU use" -- retain the CUDA
        # primary context now (no longer created eagerly at import).
        ensure_context()
        self.max_ndata = int(max_ndata)
        self.n_lcs = int(n_lcs)
        self.nfreqs = int(nfreqs)
        self.stream = stream
        self.rtype = np.float32
        # Pinned (page-locked) host buffers by default for async overlap;
        # graceful fallback to page-aligned if pinning fails.
        self.pinned = pinned

        # Per-LC normalization factors
        self.yy = np.zeros(n_lcs, dtype=np.float64)

        # Per-LC epochs: floor(min(t)) subtracted from each lightcurve's times
        # before the float32 cast (phases are relative to it)
        self.epochs = np.zeros(n_lcs, dtype=np.float64)

        # Pinned (or page-aligned fallback) host arrays
        p = self.pinned
        total_data = self.max_ndata * self.n_lcs
        total_bls = self.nfreqs * self.n_lcs

        self.t = host_array((total_data,), self.rtype, pinned=p)
        self.yw = host_array((total_data,), self.rtype, pinned=p)
        self.w = host_array((total_data,), self.rtype, pinned=p)
        self.ndata_per_lc = host_array((self.n_lcs,), np.uint32, pinned=p)

        self.freqs = host_array((self.nfreqs,), self.rtype, pinned=p)
        self.nbins0 = host_array((self.nfreqs,), np.uint32, pinned=p)
        self.nbinsf = host_array((self.nfreqs,), np.uint32, pinned=p)

        self.bls = host_array((total_bls,), self.rtype, pinned=p)

        # GPU arrays (allocated on first transfer)
        self.t_g = None
        self.yw_g = None
        self.w_g = None
        self.ndata_per_lc_g = None
        self.freqs_g = None
        self.nbins0_g = None
        self.nbinsf_g = None
        self.bls_g = None

    def set_freqs(self, freqs, qmin=1e-2, qmax=0.5):
        """
        Set frequency grid and compute bin counts.

        Parameters
        ----------
        freqs : array_like
            Frequency array (1/days).
        qmin : float or array_like
            Minimum fractional transit duration.
        qmax : float or array_like
            Maximum fractional transit duration.

        Returns
        -------
        max_nbins : int
            Maximum number of fine bins (for shared memory sizing).
        """
        check_freqs(freqs, name='BLSBatchMemory.set_freqs')
        freqs = np.asarray(freqs, dtype=self.rtype)
        nf = len(freqs)
        if nf > self.nfreqs:
            raise ValueError(
                f"Got {nf} freqs but allocated for {self.nfreqs}")

        self.freqs[:nf] = freqs

        # Validate before the uint32 cast below: a NaN, a zero qmin or
        # a qmax >= 1 becomes a bin count of 0, which divides by zero
        # in the kernel and atomicAdds outside the shared-memory
        # histogram -- an illegal memory access that kills the CUDA
        # context (Sep 2026 audit, defect 23). Imported lazily to
        # avoid a circular import with cuvarbase.bls.
        from ..bls import _validate_fast_q_bounds
        _validate_fast_q_bounds(nf, qmin, qmax)

        qmin_arr = np.broadcast_to(np.asarray(qmin, dtype=self.rtype), (nf,))
        qmax_arr = np.broadcast_to(np.asarray(qmax, dtype=self.rtype), (nf,))

        self.nbinsf[:nf] = (1.0 / qmin_arr).astype(np.uint32)
        self.nbins0[:nf] = (1.0 / qmax_arr).astype(np.uint32)

        max_nbins = int(self.nbinsf[:nf].max())
        return max_nbins

    def set_lightcurve(self, idx, t, y, dy):
        """
        Set data for one lightcurve in the batch.

        Computes weights, weighted-mean-subtracted observations, and
        stores the yy normalization factor.

        Parameters
        ----------
        idx : int
            Index of this lightcurve within the batch (0-based).
        t : array_like
            Observation times.
        y : array_like
            Observations.
        dy : array_like
            Observation uncertainties.
        """
        check_lightcurve(t, y, dy, min_n=2,
                         name='BLSBatchMemory.set_lightcurve %d' % idx)
        # Epoch-subtract in float64 before the float32 cast: absolute
        # timestamps (e.g. BJD) would otherwise destroy the phase fold.
        t, epoch = subtract_epoch(t)
        y = np.asarray(y, dtype=np.float64)
        dy = np.asarray(dy, dtype=np.float64)
        ndata = len(t)

        if idx >= self.n_lcs:
            raise ValueError(f"idx={idx} >= n_lcs={self.n_lcs}")
        if ndata > self.max_ndata:
            raise ValueError(
                f"ndata={ndata} > max_ndata={self.max_ndata}")

        self.ndata_per_lc[idx] = np.uint32(ndata)
        self.epochs[idx] = epoch

        offset = idx * self.max_ndata

        # Compute weights
        w = np.power(dy, -2)
        w /= w.sum()

        # Weighted mean and normalization. einsum, not np.dot: BLAS
        # ddot spawns a threadpool for large vectors and trips CFS
        # throttling on CPU-quota-limited hosts (see BLSMemory.setdata).
        ybar = float(np.einsum('i,i->', y, w))
        self.yy[idx] = float(np.einsum('i,i->', w, (y - ybar) ** 2))

        # Store (use float64 for computation, cast to float32 for GPU)
        # in conflict-scattered order: time-sorted input serializes the
        # batch kernel's shared-memory atomics (warp-adjacent samples
        # fold into the same phase bin; 3.1x measured on TESS-like
        # cadence). Binning is a sum, so order is semantically free.
        perm = conflict_scatter_perm(ndata)
        if perm is None:
            self.t[offset:offset + ndata] = t.astype(self.rtype)
            self.yw[offset:offset + ndata] = \
                ((y - ybar) * w).astype(self.rtype)
            self.w[offset:offset + ndata] = w.astype(self.rtype)
        else:
            self.t[offset:offset + ndata] = t.astype(self.rtype)[perm]
            self.yw[offset:offset + ndata] = \
                ((y - ybar) * w).astype(self.rtype)[perm]
            self.w[offset:offset + ndata] = w.astype(self.rtype)[perm]

        # Zero-pad remainder (should already be zero from aligned_zeros,
        # but be explicit in case of reuse)
        self.t[offset + ndata:offset + self.max_ndata] = 0.0
        self.yw[offset + ndata:offset + self.max_ndata] = 0.0
        self.w[offset + ndata:offset + self.max_ndata] = 0.0

    def transfer_to_gpu(self, n_lcs_active=None, transfer_freqs=True):
        """Transfer host arrays to GPU asynchronously.

        Parameters
        ----------
        n_lcs_active : int, optional
            Transfer only the first ``n_lcs_active`` lightcurve slots
            (chunked reuse: a batch call processing fewer LCs than the
            allocation avoids re-uploading the padded tail). Default:
            all slots.
        transfer_freqs : bool, optional (default: True)
            Upload the frequency grid + bin-count arrays. Chunk loops
            reusing the same grid only need this once.
        """
        n_act = self.n_lcs if n_lcs_active is None else int(n_lcs_active)
        n_act = min(n_act, self.n_lcs)
        total_data = self.max_ndata * self.n_lcs
        total_bls = self.nfreqs * self.n_lcs

        if self.t_g is None:
            self.t_g = gpuarray.zeros(total_data, dtype=self.rtype)
            self.yw_g = gpuarray.zeros(total_data, dtype=self.rtype)
            self.w_g = gpuarray.zeros(total_data, dtype=self.rtype)
            self.ndata_per_lc_g = gpuarray.zeros(
                self.n_lcs, dtype=np.uint32)
            self.freqs_g = gpuarray.zeros(self.nfreqs, dtype=self.rtype)
            self.nbins0_g = gpuarray.zeros(self.nfreqs, dtype=np.uint32)
            self.nbinsf_g = gpuarray.zeros(self.nfreqs, dtype=np.uint32)
            self.bls_g = gpuarray.zeros(total_bls, dtype=self.rtype)

        nd = self.max_ndata * n_act
        # driver-level prefix copies (contiguous views of the pinned
        # buffers stay page-locked, so these are genuinely async)
        if self.stream is not None:
            cuda.memcpy_htod_async(self.t_g.gpudata, self.t[:nd],
                                   self.stream)
            cuda.memcpy_htod_async(self.yw_g.gpudata, self.yw[:nd],
                                   self.stream)
            cuda.memcpy_htod_async(self.w_g.gpudata, self.w[:nd],
                                   self.stream)
            cuda.memcpy_htod_async(self.ndata_per_lc_g.gpudata,
                                   self.ndata_per_lc[:n_act], self.stream)
        else:
            cuda.memcpy_htod(self.t_g.gpudata, self.t[:nd])
            cuda.memcpy_htod(self.yw_g.gpudata, self.yw[:nd])
            cuda.memcpy_htod(self.w_g.gpudata, self.w[:nd])
            cuda.memcpy_htod(self.ndata_per_lc_g.gpudata,
                             self.ndata_per_lc[:n_act])

        if transfer_freqs:
            self.freqs_g.set_async(self.freqs, stream=self.stream)
            self.nbins0_g.set_async(self.nbins0, stream=self.stream)
            self.nbinsf_g.set_async(self.nbinsf, stream=self.stream)

    def transfer_to_cpu(self, n_lcs_active=None):
        """Transfer BLS results from GPU to host.

        Parameters
        ----------
        n_lcs_active : int, optional
            Read back only the first ``n_lcs_active`` result rows.
        """
        n_act = self.n_lcs if n_lcs_active is None else int(n_lcs_active)
        n_act = min(n_act, self.n_lcs)
        nb = self.nfreqs * n_act
        if self.stream is not None:
            cuda.memcpy_dtoh_async(self.bls[:nb], self.bls_g.gpudata,
                                   self.stream)
            self.stream.synchronize()
        else:
            cuda.memcpy_dtoh(self.bls[:nb], self.bls_g.gpudata)

    def get_results(self, n_lcs_active=None, nfreq_active=None):
        """
        Return normalized BLS results per lightcurve.

        Parameters
        ----------
        n_lcs_active : int, optional
            Number of populated lightcurve slots to return (chunked
            reuse). Default: all slots.
        nfreq_active : int, optional
            Number of valid frequencies per row (a memory allocated
            for more frequencies than the current call uses -- the
            ``memory=`` reuse path -- keeps its allocation pitch, and
            the row tails are stale). Default: the full allocation.

        Returns
        -------
        results : list of ndarray
            BLS power for each lightcurve, normalized by yy.
        """
        n_act = self.n_lcs if n_lcs_active is None else int(n_lcs_active)
        n_act = min(n_act, self.n_lcs)
        nf = self.nfreqs if nfreq_active is None else int(nfreq_active)
        nf = min(nf, self.nfreqs)
        results = []
        for i in range(n_act):
            offset = i * self.nfreqs
            raw = self.bls[offset:offset + nf].copy()
            if self.yy[i] > 0:
                raw /= self.yy[i]
            results.append(raw)
        return results
