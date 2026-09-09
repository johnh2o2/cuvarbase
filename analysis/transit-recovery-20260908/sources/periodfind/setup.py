import os
from os.path import join as pjoin

import numpy
from setuptools import setup

try:
    from Cython.Distutils import build_ext

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME or CUDA_HOME env variable is in use
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    elif "CUDA_HOME" in os.environ:
        home = os.environ["CUDA_HOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise OSError(
                "The nvcc binary could not be "
                "located in your $PATH. Either add it to your path, "
                "or set $CUDAHOME / $CUDA_HOME"
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise OSError(f"The CUDA {k} path could not be located in {v}")

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


try:
    CUDA = locate_cuda()
except OSError:
    CUDA = None

if CUDA is not None and HAS_CYTHON:
    from distutils.extension import Extension

    # Run the customize_compiler
    class custom_build_ext(build_ext):
        def build_extensions(self):
            customize_compiler_for_nvcc(self.compiler)
            build_ext.build_extensions(self)

    # Obtain the numpy include directory. This logic works across numpy versions.
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()

    # Arguments for both NVCC and GCC
    compiler_flags = ["-std=c++14", "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]
    gcc_only_flags = []
    nvcc_only_flags = ["-c", "--compiler-options", "'-fPIC'", "-use_fast_math"]

    # Generate the gencode arguments
    compute_capabilities = [50, 52, 60, 61, 70, 75, 80, 86, 89, 90]
    cuda_arch_flags = [f"-gencode=arch=compute_{cap},code=sm_{cap}" for cap in compute_capabilities]
    cuda_arch_flags.append(
        f"-gencode=arch=compute_{compute_capabilities[-1]},code=compute_{compute_capabilities[-1]}"
    )

    # Final Args
    gcc_flags = gcc_only_flags + compiler_flags
    nvcc_flags = nvcc_only_flags + compiler_flags + cuda_arch_flags

    extensions = [
        Extension(
            "periodfind.ce",
            sources=["periodfind/cuda/ce.cu", "periodfind/ce.pyx"],
            language="c++",
            libraries=["cudart"],
            library_dirs=[CUDA["lib64"]],
            runtime_library_dirs=[CUDA["lib64"]],
            include_dirs=[numpy_include, CUDA["include"]],
            extra_compile_args={
                "gcc": gcc_flags,
                "nvcc": nvcc_flags,
            },
        ),
        Extension(
            "periodfind.aov",
            sources=["periodfind/cuda/aov.cu", "periodfind/aov.pyx"],
            language="c++",
            libraries=["cudart"],
            library_dirs=[CUDA["lib64"]],
            runtime_library_dirs=[CUDA["lib64"]],
            include_dirs=[numpy_include, CUDA["include"]],
            extra_compile_args={
                "gcc": gcc_flags,
                "nvcc": nvcc_flags,
            },
        ),
        Extension(
            "periodfind.ls",
            sources=["periodfind/cuda/ls.cu", "periodfind/ls.pyx"],
            language="c++",
            libraries=["cudart"],
            library_dirs=[CUDA["lib64"]],
            runtime_library_dirs=[CUDA["lib64"]],
            include_dirs=[numpy_include, CUDA["include"]],
            extra_compile_args={
                "gcc": gcc_flags,
                "nvcc": nvcc_flags,
            },
        ),
        Extension(
            "periodfind.fpw",
            sources=["periodfind/cuda/fpw.cu", "periodfind/fpw.pyx"],
            language="c++",
            libraries=["cudart"],
            library_dirs=[CUDA["lib64"]],
            runtime_library_dirs=[CUDA["lib64"]],
            include_dirs=[numpy_include, CUDA["include"]],
            extra_compile_args={
                "gcc": gcc_flags,
                "nvcc": nvcc_flags,
            },
        ),
        Extension(
            "periodfind.bls",
            sources=["periodfind/cuda/bls.cu", "periodfind/bls.pyx"],
            language="c++",
            libraries=["cudart"],
            library_dirs=[CUDA["lib64"]],
            runtime_library_dirs=[CUDA["lib64"]],
            include_dirs=[numpy_include, CUDA["include"]],
            extra_compile_args={
                "gcc": gcc_flags,
                "nvcc": nvcc_flags,
            },
        ),
        Extension(
            "periodfind.mf",
            sources=["periodfind/cuda/mf.cu", "periodfind/mf.pyx"],
            language="c++",
            libraries=["cudart"],
            library_dirs=[CUDA["lib64"]],
            runtime_library_dirs=[CUDA["lib64"]],
            include_dirs=[numpy_include, CUDA["include"]],
            extra_compile_args={
                "gcc": gcc_flags,
                "nvcc": nvcc_flags,
            },
        ),
        Extension(
            "periodfind.vn",
            sources=["periodfind/cuda/vn.cu", "periodfind/vn.pyx"],
            language="c++",
            libraries=["cudart"],
            library_dirs=[CUDA["lib64"]],
            runtime_library_dirs=[CUDA["lib64"]],
            include_dirs=[numpy_include, CUDA["include"]],
            extra_compile_args={
                "gcc": gcc_flags,
                "nvcc": nvcc_flags,
            },
        ),
        Extension(
            "periodfind.mhf",
            sources=["periodfind/cuda/mhf.cu", "periodfind/mhf.pyx"],
            language="c++",
            libraries=["cudart"],
            library_dirs=[CUDA["lib64"]],
            runtime_library_dirs=[CUDA["lib64"]],
            include_dirs=[numpy_include, CUDA["include"]],
            extra_compile_args={
                "gcc": gcc_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ]
    cmdclass = {"build_ext": custom_build_ext}
else:
    extensions = []
    cmdclass = {}

# All metadata (name, version, description, dependencies, etc.) lives in
# pyproject.toml.  setup.py only contributes CUDA extensions and cmdclass.
setup(
    ext_modules=extensions,
    cmdclass=cmdclass,
)
