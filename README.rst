cuvarbase 0.1.9
===============

John Hoffman
(c) 2017

``cuvarbase`` is a Python library that uses `PyCUDA <https://mathema.tician.de/software/pycuda/>`_ to implement several time series tools used in astronomy to GPUs.

This project is under active development, and currently includes implementations of

- Generalized `Lomb Scargle <https://arxiv.org/abs/0901.2573>`_ periodogram
- Box-least squares (`BLS <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_ )
- Non-equispaced fast Fourier transform (adjoint operation) (`NFFT paper <http://epubs.siam.org/doi/abs/10.1137/0914081>`_)
- Conditional entropy period finder (`CE <http://adsabs.harvard.edu/abs/2013MNRAS.434.2629G>`_)
- Phase dispersion minimization (`PDM2 <http://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=29>`_)
	- Currently operational but minimal unit testing or documentation (yet)

Hopefully future developments will have

- (Weighted) wavelet transforms
- Spectrograms (for PDM and GLS)
- Multiharmonic extensions for GLS


Dependencies
------------

- `PyCUDA <https://mathema.tician.de/software/pycuda/>`_ **<-essential**
- `scikit cuda <https://scikit-cuda.readthedocs.io/en/latest/>`_ **<-also essential**
	- used for access to the CUDA FFT runtime library
- `matplotlib <https://matplotlib.org/>`_ (for plotting utilities)
- `nfft <https://github.com/jakevdp/nfft>`_ (for unit testing)
- `astropy <http://www.astropy.org/>`_ (for unit testing)


Using multiple GPUs
-------------------

If you have more than one GPU, you can choose which one to
use in a given script by setting the ``CUDA_DEVICE`` environment
variable:

.. code:: sh

    CUDA_DEVICE=1 python script.py

If anyone is interested in implementing multi-device load-balancing
solution, they are encouraged to do so! At some point this may
become important, but for the time being manually splitting up the
jobs to different GPU's will have to suffice.
