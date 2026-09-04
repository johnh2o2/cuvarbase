"""Run the shipped LS test-suite with the B fix monkeypatched in (no tree edits)."""
import sys, pytest
import cuvarbase.memory.lombscargle_memory as lsm
import pycuda.gpuarray as gpuarray
def alloc_fixed(self, **kwargs):
    k0 = kwargs.get('k0', self.k0); n0 = kwargs.get('n0', self.n0)
    if self.buffered_transfer: n0 = kwargs.get('n0_buffer', self.n0_buffer)
    self.nf = kwargs.get('nf', self.nf)
    if self.use_fft:
        if self.nfft_mem_yw.precomp_psi: self.nfft_mem_yw.allocate_precomp_psi(n0=n0)
        self.nfft_mem_w.precomp_psi = False
        self.nfft_mem_w.q1 = self.nfft_mem_yw.q1; self.nfft_mem_w.q2 = self.nfft_mem_yw.q2; self.nfft_mem_w.q3 = self.nfft_mem_yw.q3
        fft_size = self.nharmonics * (self.nf + k0)
        self.nfft_mem_yw.allocate_grid(nf=fft_size); self.nfft_mem_w.allocate_grid(nf=2 * fft_size)
    self.lsp_g = gpuarray.zeros(self.nf, dtype=self.real_type)
    return self
lsm.LombScargleMemory.allocate_grids = alloc_fixed
sys.exit(pytest.main(['-q', '-x', '-p', 'no:cacheprovider', '/workspace/cuvarbase/cuvarbase/tests/test_lombscargle.py']))
