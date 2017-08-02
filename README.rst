cuvarbase
=========

John Hoffman -- 2017

``cuvarbase`` is a Python (2.7) library that uses `PyCUDA <https://mathema.tician.de/software/pycuda/>`_ to implement several time series tools used in astronomy.


This project is under active development, and currently includes implementations of

- Generalized `Lomb Scargle <https://arxiv.org/abs/0901.2573>`_ periodogram
- Box-least squares (`BLS <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_ )
	- Currently operational but no unit tests or documentation yet.
- Non-equispaced fast Fourier transform (adjoint operation) (`NFFT <http://epubs.siam.org/doi/abs/10.1137/0914081>`_)
- Phase dispersion minimization (`PDM2 <http://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=29>`_)
	- Currently operational but minimal unit testing or documentation (yet)

Hopefully future developments will have

- (Weighted) wavelet transforms
- Spectrograms (for PDM and GLS)

Dependencies
------------

- `PyCUDA <https://mathema.tician.de/software/pycuda/>`_ **<-essential**
- `scikit cuda <https://scikit-cuda.readthedocs.io/en/latest/>`_ **<-also essential**
	- used for access to the CUDA FFT runtime library
- `matplotlib <https://matplotlib.org/>`_ (for plotting utilities)
- `nfft <https://github.com/jakevdp/nfft>`_ (for unit testing)
- `astropy <http://www.astropy.org/>`_ (for unit testing)

Install instructions
--------------------

`Conda <https://www.continuum.io/downloads>`_ is a great way to do this in a safe, isolated environment.

First create a new conda environment (named ``pycu`` here) that uses Python 2.7 (future versions should be compatible with 3.x), and install some required dependencies (``numpy``, ``astropy`` and ``matplotlib``).

.. code:: bash

	conda create -n pycu python=2.7 numpy astropy matplotlib

Then activate the virtual environment

.. code:: bash

	source activate pycu

and use ``pip`` to install the other dependencies.

.. code:: bash

	pip install nfft scikit-cuda pycuda

You should test if ``pycuda`` is working correctly.

.. code:: bash

	python -c "import pycuda.autoinit"

If everything works up until now, we should be ready to install ``cuvarbase``

.. code:: bash

	python setup.py install

and run the unit tests

.. code:: bash

	py.test cuvarbase

**If you don't want to use conda** the following should work with just pip (assuming you're using Python 2.7):

.. code:: bash

	pip install numpy scikit-cuda pycuda astropy nfft matplotlib
	python setup.py install
	py.test cuvarbase



Example Usage
-------------


For a Lomb-Scargle periodogram

.. code:: python


	from cuvarbase.lombscargle import LombScargleAsyncProcess
	import numpy as np

	# random observation times (1 year baseline)
	t = 365 * np.random.rand(100)

	# some signal (10 day period, 0.1 amplitude)
	y = 12 + 0.1 * np.cos(2 * np.pi * t / 10.)

	# data uncertainties (0.01)
	dy = 0.01 * np.ones_like(y)

	# add noise to observations
	y += dy * np.random.randn(len(t))

	# start an asynchronous process
	ls_proc = LombScargleAsyncProcess()

	# run on our data (only one lightcurve)
	result = ls_proc.run([(t, y, dy)])

	freqs, pows = zip(*(result[0]))

	# print peak frequency
	print(freqs[np.argmax(pows)])