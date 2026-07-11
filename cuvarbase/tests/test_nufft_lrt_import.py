"""
Test NUFFT LRT module import and basic structure.

These tests verify that the NUFFT LRT module is properly structured
and can be imported when CUDA is available.
"""
import pytest
import os
import ast


class TestNUFFTLRTImport:
    """Test NUFFT LRT module structure and imports"""

    def test_module_syntax_valid(self):
        """Test that nufft_lrt.py has valid Python syntax"""
        module_path = os.path.join(os.path.dirname(__file__), '..', 'nufft_lrt.py')
        with open(module_path) as f:
            content = f.read()

        # Should parse without errors
        ast.parse(content)

    def test_cuda_kernel_exists(self):
        """Test that CUDA kernel file exists"""
        kernel_path = os.path.join(os.path.dirname(__file__), '..', 'kernels', 'nufft_lrt.cu')
        assert os.path.exists(kernel_path), f"CUDA kernel not found: {kernel_path}"

    def test_cuda_kernel_has_required_functions(self):
        """Test that CUDA kernel contains required __global__ functions"""
        kernel_path = os.path.join(os.path.dirname(__file__), '..', 'kernels', 'nufft_lrt.cu')

        with open(kernel_path) as f:
            content = f.read()

        # Should have at least one __global__ function
        assert '__global__' in content, "No CUDA kernels found"

        # Check for key kernel functions
        required_kernels = [
            'nufft_matched_filter',
            'estimate_power_spectrum',
            'compute_frequency_weights'
        ]

        for kernel in required_kernels:
            assert kernel in content, f"Required kernel '{kernel}' not found"

    def test_module_imports(self):
        """Test that NUFFT LRT module can be imported (requires CUDA)"""
        pytest.importorskip("pycuda")

        # Try to import the module
        from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, NUFFTLRTMemory

        # Check that classes are defined
        assert NUFFTLRTAsyncProcess is not None
        assert NUFFTLRTMemory is not None

    def test_documentation_exists(self):
        """Test that NUFFT LRT documentation exists"""
        # Check for README in docs/
        readme_path = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'NUFFT_LRT_README.md')
        assert os.path.exists(readme_path), "NUFFT_LRT_README.md not found in docs/"

    def test_example_exists(self):
        """Test that example code exists"""
        example_path = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'nufft_lrt_example.py')
        assert os.path.exists(example_path), "nufft_lrt_example.py not found in examples/"

    def test_example_syntax_valid(self):
        """Test that example has valid syntax"""
        example_path = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'nufft_lrt_example.py')

        with open(example_path) as f:
            content = f.read()

        # Should parse without errors
        ast.parse(content)


class TestDetectorAlgebra:
    """CPU tests for the Detector-A (marginalized joint detector)
    algebra. The Woodbury frequency-domain path is verified against a
    dense inverse of the realified combined covariance -- an
    independent computation of the same statistic."""

    @staticmethod
    def _realify(a):
        import numpy as np
        return np.concatenate([np.real(a), np.imag(a)])

    def _dense_statistic(self, Y, T, V_ks, psd, weights, prior_cov):
        # Cov_s^{-1} is diagonal (w/P) in the realified space; the
        # combined covariance is Cov_z = Cov_s + R Cov_c R^T with R the
        # realified basis. Invert it densely (small nf) and evaluate
        # the matched filter directly.
        import numpy as np
        d = np.concatenate([weights / psd, weights / psd])
        Cov_s = np.diag(1.0 / d)
        R = np.stack([self._realify(v) for v in V_ks], axis=1)
        Cov_z = Cov_s + R @ np.atleast_2d(prior_cov) @ R.T
        Wz = np.linalg.inv(Cov_z)
        ry, rt = self._realify(Y), self._realify(T)
        return float(ry @ Wz @ rt / np.sqrt(rt @ Wz @ rt))

    def test_marginal_matches_dense_inverse(self):
        import numpy as np
        from cuvarbase.nufft_lrt import _marginal_statistic

        rng = np.random.RandomState(7)
        nf, K = 24, 3
        Y = rng.randn(nf) + 1j * rng.randn(nf)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        V_ks = [rng.randn(nf) + 1j * rng.randn(nf) for _ in range(K)]
        psd = 0.5 + rng.rand(nf)
        weights = np.ones(nf)
        A = rng.randn(K, K)
        prior_cov = A @ A.T + 0.5 * np.eye(K)   # positive definite

        got = _marginal_statistic(Y, T, V_ks, psd, weights, prior_cov)
        want = self._dense_statistic(Y, T, V_ks, psd, weights, prior_cov)
        np.testing.assert_allclose(got, want, rtol=1e-9)

    def test_no_basis_reduces_to_matched_filter(self):
        import numpy as np
        from cuvarbase.nufft_lrt import (_marginal_statistic,
                                         _whitened_inner)

        rng = np.random.RandomState(1)
        nf = 32
        Y = rng.randn(nf) + 1j * rng.randn(nf)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        psd = 1.0 + rng.rand(nf)
        w = np.ones(nf)
        got = _marginal_statistic(Y, T, [], psd, w, np.zeros((0, 0)))
        want = (_whitened_inner(Y, T, psd, w)
                / np.sqrt(_whitened_inner(T, T, psd, w)))
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_wide_prior_suppresses_basis_component(self):
        # With a very wide prior, any data component along the basis is
        # marginalized away: adding a huge basis-aligned contaminant to
        # Y must not change the statistic (while it wrecks the plain
        # matched filter).
        import numpy as np
        from cuvarbase.nufft_lrt import (_marginal_statistic,
                                         _whitened_inner)

        rng = np.random.RandomState(3)
        nf = 24
        Y = rng.randn(nf) + 1j * rng.randn(nf)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        v = rng.randn(nf) + 1j * rng.randn(nf)
        psd = np.ones(nf)
        w = np.ones(nf)
        prior = np.array([[1e8]])

        clean = _marginal_statistic(Y, T, [v], psd, w, prior)
        contaminated = _marginal_statistic(Y + 50.0 * v, T, [v], psd, w,
                                           prior)
        np.testing.assert_allclose(contaminated, clean, rtol=1e-4)

        plain = _whitened_inner(Y, T, psd, w) \
            / np.sqrt(_whitened_inner(T, T, psd, w))
        plain_cont = _whitened_inner(Y + 50.0 * v, T, psd, w) \
            / np.sqrt(_whitened_inner(T, T, psd, w))
        assert abs(plain_cont - plain) > 10 * abs(contaminated - clean)

    def test_sequential_detrend_removes_basis(self):
        import numpy as np
        from cuvarbase.nufft_lrt import _sequential_detrend

        rng = np.random.RandomState(5)
        n = 200
        t = np.sort(rng.rand(n)) * 30
        V = np.stack([t - t.mean(), (t - t.mean()) ** 2], axis=1)
        y = 1.0 + 0.01 * rng.randn(n) + V @ np.array([0.3, -0.02])
        r = _sequential_detrend(t, y, V)
        # residual orthogonal to the basis
        np.testing.assert_allclose(V.T @ r, 0.0, atol=1e-8 * n)


class TestPsdSmoothing:
    """CPU tests for the edge-corrected periodogram smoothing (audit
    finding: plain np.convolve 'same' depressed the PSD at the spectrum
    edges, overweighting those bins by up to ~2x after 1/P whitening)."""

    def test_flat_periodogram_stays_flat_at_edges(self):
        import numpy as np
        from cuvarbase.nufft_lrt import _smoothed_periodogram

        power = np.ones(64, dtype=np.float32)
        smoothed = _smoothed_periodogram(power, 5)
        # Un-corrected smoothing gives 3/5 and 4/5 at the edges; the
        # count-normalized version is exactly flat everywhere.
        np.testing.assert_allclose(smoothed, 1.0, rtol=1e-6)

    def test_interior_matches_plain_boxcar(self):
        import numpy as np
        from cuvarbase.nufft_lrt import _smoothed_periodogram

        rng = np.random.RandomState(0)
        power = rng.rand(128).astype(np.float64)
        k = 7
        smoothed = _smoothed_periodogram(power, k)
        plain = np.convolve(power, np.ones(k) / k, mode='same')
        # away from the edges the two agree
        np.testing.assert_allclose(smoothed[k:-k], plain[k:-k], rtol=1e-12)

    def test_window_one_is_identity(self):
        import numpy as np
        from cuvarbase.nufft_lrt import _smoothed_periodogram

        power = np.arange(16, dtype=np.float32)
        out = _smoothed_periodogram(power, 1)
        np.testing.assert_array_equal(out, power)
