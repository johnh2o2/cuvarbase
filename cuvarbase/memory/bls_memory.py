"""
Memory management for batch BLS GPU operations.

Handles padded multi-lightcurve data layout with pinned CPU arrays
and GPU arrays for efficient batch processing.
"""
import resource
import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from ..utils import subtract_epoch


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

    def __init__(self, max_ndata, n_lcs, nfreqs, stream=None):
        self.max_ndata = int(max_ndata)
        self.n_lcs = int(n_lcs)
        self.nfreqs = int(nfreqs)
        self.stream = stream
        self.rtype = np.float32

        # Per-LC normalization factors
        self.yy = np.zeros(n_lcs, dtype=np.float64)

        # Per-LC epochs: min(t) subtracted from each lightcurve's times
        # before the float32 cast (phases are relative to it)
        self.epochs = np.zeros(n_lcs, dtype=np.float64)

        # Allocate pinned host arrays
        align = resource.getpagesize()
        total_data = self.max_ndata * self.n_lcs
        total_bls = self.nfreqs * self.n_lcs

        self.t = cuda.aligned_zeros(
            shape=(total_data,), dtype=self.rtype, alignment=align)
        self.yw = cuda.aligned_zeros(
            shape=(total_data,), dtype=self.rtype, alignment=align)
        self.w = cuda.aligned_zeros(
            shape=(total_data,), dtype=self.rtype, alignment=align)
        self.ndata_per_lc = cuda.aligned_zeros(
            shape=(self.n_lcs,), dtype=np.uint32, alignment=align)

        self.freqs = cuda.aligned_zeros(
            shape=(self.nfreqs,), dtype=self.rtype, alignment=align)
        self.nbins0 = cuda.aligned_zeros(
            shape=(self.nfreqs,), dtype=np.uint32, alignment=align)
        self.nbinsf = cuda.aligned_zeros(
            shape=(self.nfreqs,), dtype=np.uint32, alignment=align)

        self.bls = cuda.aligned_zeros(
            shape=(total_bls,), dtype=self.rtype, alignment=align)

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
        freqs = np.asarray(freqs, dtype=self.rtype)
        nf = len(freqs)
        assert nf <= self.nfreqs, (
            f"Got {nf} freqs but allocated for {self.nfreqs}")

        self.freqs[:nf] = freqs

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
        # Epoch-subtract in float64 before the float32 cast: absolute
        # timestamps (e.g. BJD) would otherwise destroy the phase fold.
        t, epoch = subtract_epoch(t)
        y = np.asarray(y, dtype=np.float64)
        dy = np.asarray(dy, dtype=np.float64)
        ndata = len(t)

        assert idx < self.n_lcs, f"idx={idx} >= n_lcs={self.n_lcs}"
        assert ndata <= self.max_ndata, (
            f"ndata={ndata} > max_ndata={self.max_ndata}")

        self.ndata_per_lc[idx] = np.uint32(ndata)
        self.epochs[idx] = epoch

        offset = idx * self.max_ndata

        # Compute weights
        w = np.power(dy, -2)
        w /= w.sum()

        # Weighted mean and normalization
        ybar = np.dot(y, w)
        self.yy[idx] = np.dot(w, (y - ybar) ** 2)

        # Store (use float64 for computation, cast to float32 for GPU)
        self.t[offset:offset + ndata] = t.astype(self.rtype)
        self.yw[offset:offset + ndata] = ((y - ybar) * w).astype(self.rtype)
        self.w[offset:offset + ndata] = w.astype(self.rtype)

        # Zero-pad remainder (should already be zero from aligned_zeros,
        # but be explicit in case of reuse)
        self.t[offset + ndata:offset + self.max_ndata] = 0.0
        self.yw[offset + ndata:offset + self.max_ndata] = 0.0
        self.w[offset + ndata:offset + self.max_ndata] = 0.0

    def transfer_to_gpu(self):
        """Transfer all host arrays to GPU asynchronously."""
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

        self.t_g.set_async(self.t, stream=self.stream)
        self.yw_g.set_async(self.yw, stream=self.stream)
        self.w_g.set_async(self.w, stream=self.stream)
        self.ndata_per_lc_g.set_async(
            self.ndata_per_lc, stream=self.stream)
        self.freqs_g.set_async(self.freqs, stream=self.stream)
        self.nbins0_g.set_async(self.nbins0, stream=self.stream)
        self.nbinsf_g.set_async(self.nbinsf, stream=self.stream)

    def transfer_to_cpu(self):
        """Transfer BLS results from GPU to host."""
        if self.stream is not None:
            self.bls_g.get_async(ary=self.bls, stream=self.stream)
            self.stream.synchronize()
        else:
            self.bls[:] = self.bls_g.get()

    def get_results(self):
        """
        Return normalized BLS results per lightcurve.

        Returns
        -------
        results : list of ndarray
            BLS power for each lightcurve, normalized by yy.
        """
        results = []
        for i in range(self.n_lcs):
            offset = i * self.nfreqs
            raw = self.bls[offset:offset + self.nfreqs].copy()
            if self.yy[i] > 0:
                raw /= self.yy[i]
            results.append(raw)
        return results
