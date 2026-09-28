Install instructions
********************

These instructions describe the v1 candidate on ``v1.0-fixes``. As of
10 September 2026, `PyPI <https://pypi.org/project/cuvarbase/>`_ still provides
0.2.5, which has no TLS implementation. The commands below install the candidate
from its Git branch.

Requirements
------------

* **Python 3.9 – 3.14**
* An **NVIDIA GPU** with a working CUDA driver, and the **CUDA toolkit** (``nvcc`` must be on your ``PATH``). cuvarbase 1.0 is validated against **CUDA 12.4** (every archived release-gate record was produced with it); other 11.x/12.x toolkits may well work but are untested.
* `PyCUDA <https://documen.tician.de/pycuda/>`_ >= 2017.1.1 (except 2024.1.2), installed automatically as a dependency.

GPU execution requires Linux or Windows via WSL2. NVIDIA dropped CUDA support on macOS in 2019, so modern Macs cannot run the GPU code. ``import cuvarbase`` itself needs neither a GPU nor pycuda, and the pure-numpy helpers in ``cuvarbase.utils`` (``check_lightcurve``, ``autofrequency``, ...), ``cuvarbase.bls_frequencies``, ``cuvarbase.tls_grids``, ``cuvarbase.tls_models`` and ``cuvarbase.tls_stats`` work on any machine. The method modules — ``cuvarbase.bls`` (including its CPU routines ``sparse_bls_cpu`` and ``single_bls``), ``cuvarbase.lombscargle`` (including ``fap_baluev``), ``ce``, ``pdm``, ``tls`` — import ``pycuda.driver`` at module top, so they need the pycuda *package* installed; a device is only touched at the first GPU call. See *GPU-less installs* below.

Installing the CUDA toolkit
---------------------------

Get the toolkit for your distribution from the `NVIDIA download page <https://developer.nvidia.com/cuda-downloads>`_ (or your package manager). Then make sure the CUDA binaries and libraries are visible:

.. code:: bash

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

Verify with ``which nvcc`` — if nothing prints, PyCUDA will not be able to compile kernels. Adjust ``/usr/local/cuda`` to your install location (versioned paths like ``/usr/local/cuda-12.4`` also work).

Installing cuvarbase
--------------------

In a fresh virtual environment (venv or conda, Python 3.9+):

.. code:: bash

    pip install 'cuvarbase @ git+https://github.com/johnh2o2/cuvarbase@v1.0-fixes'

numpy, scipy and pycuda are installed automatically (astropy is only needed by the test suite). PyCUDA builds against your CUDA toolkit during installation, so set the environment variables above first. See *GPU-less installs* for the path without CUDA dependencies.

Optional extras:

.. code:: bash

    pip install 'cuvarbase[tls] @ git+https://github.com/johnh2o2/cuvarbase@v1.0-fixes'
    pip install 'cuvarbase[cufinufft] @ git+https://github.com/johnh2o2/cuvarbase@v1.0-fixes'
    pip install 'cuvarbase[test] @ git+https://github.com/johnh2o2/cuvarbase@v1.0-fixes'

``tls`` installs the standard TLS dependencies; ``cufinufft`` enables the optional
cuFINUFFT Lomb-Scargle backend; ``test`` supplies pytest, nfft, astropy, batman and
transitleastsquares. In a checkout, ``pip install -r docs/requirements.txt``
installs Sphinx and matplotlib for building the documentation.

The standard TLS engine requires both CuPy and ``batman-package`` and does not
substitute an approximate template when either is absent. The ``tls`` extra
uses CuPy 13 and supports Python 3.9–3.13; the current GPU validation uses
Python 3.11, CuPy 13.6 and CUDA 12.4. For a different CUDA runtime, install its
matching CuPy 13 wheel and ``batman-package`` separately (only one CuPy
distribution per environment). See the `CuPy installation guide
<https://docs.cupy.dev/en/v13.6.0/install.html>`_.

``method='binned'`` keeps the earlier TLS engine and its optional analytic
template fallback. For GPU testing of all engines, install ``.[test,tls]``.

Installing from source
----------------------

.. code:: bash

    git clone --branch v1.0-fixes https://github.com/johnh2o2/cuvarbase
    cd cuvarbase
    pip install -e .

GPU-less installs
-----------------

To use the pure helpers (frequency grids, TLS duration grids and statistics,
``check_lightcurve``, ...) without installing CUDA dependencies, skip dependency
resolution:

.. code:: bash

    pip install numpy scipy
    pip install --no-deps 'cuvarbase @ git+https://github.com/johnh2o2/cuvarbase@v1.0-fixes'

``import cuvarbase`` and the pure modules listed under *Requirements* then work; importing a method module (``cuvarbase.bls``, ``cuvarbase.lombscargle``, ...) raises ``ImportError`` because pycuda is absent. The test suite ships its own pycuda stub (``cuvarbase/tests/conftest.py``), so ``pytest --pyargs cuvarbase`` also runs on such a machine: the CPU tests pass and the GPU tests skip.

Verifying the installation
--------------------------

.. code:: bash

    python -c "import cuvarbase; print(cuvarbase.__version__)"          # works even without a GPU or pycuda
    python -c "from cuvarbase.bls import eebls_gpu_fast; print('GPU BLS ready')"   # needs pycuda

For a real end-to-end check on a GPU machine, install the test extra and run the test suite (on a GPU-less machine the same command runs the CPU tests and skips the rest):

.. code:: bash

    pip install 'cuvarbase[test] @ git+https://github.com/johnh2o2/cuvarbase@v1.0-fixes'
    pytest --pyargs cuvarbase

Troubleshooting
---------------

* **``nvcc`` not found / ``CompileError`` at first GPU call** — the CUDA toolkit is missing from ``PATH``. Kernels are compiled at first use (then cached), so a working ``nvcc`` is required at runtime, not just install time.
* **PyCUDA fails to build** — check that the toolkit version matches your driver (``nvidia-smi`` shows the maximum supported CUDA version) and that you are not hitting the excluded ``pycuda==2024.1.2``.
* **``import cuvarbase`` succeeds but GPU calls fail** — importing no longer initializes CUDA (new in 1.0.0); the context is created at first GPU use, which is where driver problems will surface. ``python -c "import pycuda.autoprimaryctx"`` isolates driver/toolkit issues from cuvarbase itself.
* **Selecting a GPU** — set the ``CUDA_DEVICE`` environment variable before the first GPU call.

If you hit something not covered here, please open an `issue <https://github.com/johnh2o2/cuvarbase/issues>`_.
