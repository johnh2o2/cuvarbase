"""Runtime monkeypatches (scratch only; the installed tree is untouched) for the three candidate fixes:
  F: floorf -> floor in fast_gaussian_grid (cunfft.cu)
  P: give the w-spectrum NFFTMemory its own precompute_psi (its grid size differs from the yw grid's)
  B: size the NFFT grids to cover modes up to k0+nf (yw) and 2(k0+nf) (w) instead of shaving k0 off
"""
import os
import cuvarbase.cunfft as cunfft_mod
import cuvarbase.memory.lombscargle_memory as lsm
from cuvarbase.utils import find_kernel as _orig_find_kernel

_orig_alloc = lsm.LombScargleMemory.allocate_grids

def apply(F=True, P=True, B=True):
    if F:
        src = open(_orig_find_kernel('cunfft')).read()
        assert 'floorf(' in src
        src = src.replace('floorf(', 'floor(')
        path = '/workspace/scratch/cunfft_patched.cu'
        open(path, 'w').write(src)
        cunfft_mod.find_kernel = lambda name: path if name == 'cunfft' else _orig_find_kernel(name)
    else:
        cunfft_mod.find_kernel = _orig_find_kernel

    def allocate_grids(self, **kwargs):
        k0 = kwargs.get('k0', self.k0)
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)
        self.nf = kwargs.get('nf', self.nf)
        if self.use_fft:
            if self.nfft_mem_yw.precomp_psi:
                self.nfft_mem_yw.allocate_precomp_psi(n0=n0)
            if P:
                self.nfft_mem_w.precomp_psi = True
                self.nfft_mem_w.allocate_precomp_psi(n0=n0)
            else:
                self.nfft_mem_w.precomp_psi = False
                self.nfft_mem_w.q1 = self.nfft_mem_yw.q1
                self.nfft_mem_w.q2 = self.nfft_mem_yw.q2
                self.nfft_mem_w.q3 = self.nfft_mem_yw.q3
            fft_size = self.nharmonics * (self.nf + k0)
            if B:
                self.nfft_mem_yw.allocate_grid(nf=fft_size)
                self.nfft_mem_w.allocate_grid(nf=2 * fft_size)
            else:
                self.nfft_mem_yw.allocate_grid(nf=fft_size - k0)
                self.nfft_mem_w.allocate_grid(nf=2 * fft_size - k0)
        import pycuda.gpuarray as gpuarray
        self.lsp_g = gpuarray.zeros(self.nf, dtype=self.real_type)
        return self
    lsm.LombScargleMemory.allocate_grids = allocate_grids if (P or B) else _orig_alloc
