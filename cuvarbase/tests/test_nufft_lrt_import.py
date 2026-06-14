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
