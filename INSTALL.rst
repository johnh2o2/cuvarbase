Install instructions
********************

Requirements
------------

* **Python 3.9 – 3.12**
* An **NVIDIA GPU** with a working CUDA driver, and the **CUDA toolkit** (11.x or 12.x; ``nvcc`` must be on your ``PATH``). cuvarbase is developed and validated against CUDA 11.8 and 12.4.
* `PyCUDA <https://documen.tician.de/pycuda/>`_ >= 2017.1.1 (except 2024.1.2), installed automatically as a dependency.

GPU execution requires Linux or Windows via WSL2. NVIDIA dropped CUDA support on macOS in 2019, so modern Macs cannot run the GPU code — although ``import cuvarbase`` and the CPU-only helpers (``sparse_bls_cpu``, ``single_bls``, ``fap_baluev``, the frequency-grid builders) work on any machine, GPU or not.

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

    pip install cuvarbase

That's it. numpy, scipy, astropy, and pycuda are installed automatically. PyCUDA builds against your CUDA toolkit during installation, so the environment variables above must be set first.

Optional extras:

.. code:: bash

    pip install cuvarbase[cufinufft]   # optional cuFINUFFT backend for Lomb-Scargle
    pip install batman-package         # limb-darkened templates for the experimental TLS module
    pip install cuvarbase[test]        # test-suite dependencies

Installing from source
----------------------

.. code:: bash

    git clone https://github.com/johnh2o2/cuvarbase
    cd cuvarbase
    pip install -e .

Docker
------

A ``Dockerfile`` (CUDA 11.8 base image) ships with the repository for containerized use:

.. code:: bash

    docker build -t cuvarbase .
    docker run --gpus all -it cuvarbase python -c "import cuvarbase; print(cuvarbase.__version__)"

Verifying the installation
--------------------------

.. code:: bash

    python -c "import cuvarbase; print(cuvarbase.__version__)"          # works even without a GPU
    python -c "from cuvarbase.bls import eebls_gpu_fast; print('GPU BLS ready')"

For a real end-to-end check on a GPU machine, install the test extra and run the test suite:

.. code:: bash

    pip install cuvarbase[test]
    pytest --pyargs cuvarbase

Troubleshooting
---------------

* **``nvcc`` not found / ``CompileError`` at first GPU call** — the CUDA toolkit is missing from ``PATH``. Kernels are compiled at first use (then cached), so a working ``nvcc`` is required at runtime, not just install time.
* **PyCUDA fails to build** — check that the toolkit version matches your driver (``nvidia-smi`` shows the maximum supported CUDA version) and that you are not hitting the excluded ``pycuda==2024.1.2``.
* **``import cuvarbase`` succeeds but GPU calls fail** — importing no longer initializes CUDA (new in 1.0.0); the context is created at first GPU use, which is where driver problems will surface. ``python -c "import pycuda.autoprimaryctx"`` isolates driver/toolkit issues from cuvarbase itself.
* **Selecting a GPU** — set the ``CUDA_DEVICE`` environment variable before the first GPU call.

If you hit something not covered here, please open an `issue <https://github.com/johnh2o2/cuvarbase/issues>`_.
