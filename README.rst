cuvarbase 0.1.5
===============

John Hoffman -- 2017

``cuvarbase`` is a Python (2.7) library that uses `PyCUDA <https://mathema.tician.de/software/pycuda/>`_ to implement several time series tools used in astronomy to GPUs.

This project is under active development, and currently includes implementations of

- Generalized `Lomb Scargle <https://arxiv.org/abs/0901.2573>`_ periodogram
- Box-least squares (`BLS <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_ )
- Non-equispaced fast Fourier transform (adjoint operation) (`NFFT <http://epubs.siam.org/doi/abs/10.1137/0914081>`_)
- Conditional entropy period finder (`CE <http://adsabs.harvard.edu/abs/2013MNRAS.434.2629G>`_)
- Phase dispersion minimization (`PDM2 <http://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=29>`_)
	- Currently operational but minimal unit testing or documentation (yet)

Hopefully future developments will have

- (Weighted) wavelet transforms
- Spectrograms (for PDM and GLS)
- Multiharmonic extensions for GLS

What's new
----------

* **0.1.5**
	* Conditional Entropy period finder now unit tested
		* Weighted variant also implemented -- accounts for heteroskedasticity if
		  that's important
	* BLS
		* New unit tests
		* A new transiting exoplanet BLS function: ``eebls_transit_gpu``
			* Only searches plausible parameter space for Keplerian orbit
	* GLS
		* False alarm probability: ``fap_baluev``
			* Implements `Baluev 2008 <http://adsabs.harvard.edu/abs/2008MNRAS.385.1279B>`_ false alarm probability measure based on extreme value theory


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


Installing on a Mac
-------------------

Nvidia offers `CUDA for Mac OSX <https://developer.nvidia.com/cuda-downloads>`_. After installing the
package via downloading and running the ``.dmg`` file, you'll have to make a couple of edits to your
``~/.bash_profile``:

.. code:: sh
    
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:/usr/local/cuda/lib"
    export PATH="/usr/local/cuda/bin:${PATH}"

and then source these changes in your current shell by running ``. ~/.bash_profile``. 

Another important note: **nvcc (8.0.61) does not appear to support the latest clang compiler**. If this is
the case, running ``python example.py`` should produce the following error:

.. code::

    nvcc fatal   : The version ('80100') of the host compiler ('Apple clang') is not supported

You can fix this problem by temporarily downgrading your clang compiler. To do this:

- `Download Xcode command line tools 7.3.1 <http://adcdownload.apple.com/Developer_Tools/Command_Line_Tools_OS_X_10.11_for_Xcode_7.3.1/Command_Line_Tools_OS_X_10.11_for_Xcode_7.3.1.dmg>`_
- Install.
- Run ``sudo xcode-select --switch /Library/Developer/CommandLineTools`` until ``clang --version`` says ``7.3``.


Example Usage
-------------

For a Lomb-Scargle periodogram

.. code:: python


	from cuvarbase.lombscargle import LombScargleAsyncProcess
	import numpy as np


	def generate_lightcurve(nobs=300, baseline=10.,
						    frequency=3.,
		                    mean_mag=12., amplitude=0.1,
		                    uncertainty=0.01):
		# random observation times (baseline in yrs)
		t = baseline * 365 * np.sort(np.random.rand(nobs))

		# some sinusoidal signal
		y = mean_mag + amplitude * np.cos(2 * np.pi * t * frequency)

		# add noise to data
		dy = uncertainty * np.ones_like(y)
		y += dy * np.random.randn(len(t))

		return t, y, dy

	# generate a fake lightcurve
	f0 = 3.
	t, y, dy = generate_lightcurve(frequency=f0)

	# start an asynchronous process
	ls_proc = LombScargleAsyncProcess()

	# run on our data (only one lightcurve)
	result = ls_proc.run([(t, y, dy)],
		                 minimum_frequency=0.5,
		                 maximum_frequency=10.)

	freqs, pows = result[0]

	# print peak frequency
	print(f0, freqs[np.argmax(pows)])


	# For a large number of lightcurves, you'll want
	# to do things in batches on the GPU.

	# lets try a thousand lightcurves
	nlc = 1000

	# with 3000 observations each
	nobs = 3000

	# and do 30 lightcurves at a time
	batch_size = 30

	# generate the lightcurves
	lightcurves = [generate_lightcurve(nobs=nobs)
	               for i in range(nlc)]

	from time import time

	t0 = time()
	r = ls_proc.batched_run_const_nfreq(lightcurves,
		                                batch_size=batch_size)
	dt = time() - t0

	print("batching:\n"
		  " %e sec. / lc [%e sec. total]"%( dt / nlc, dt))

	# How long would that have taken if we hadn't reused
	# the memory for each batch?

	# save the frequencies (same for all lightcurves)
	freqs = r[0][0]

	# generate batches
	batches = []
	while len(batches) * batch_size < len(lightcurves):
		start = len(batches) * batch_size
		end = start + min([batch_size, len(lightcurves) - start])
		batches.append([lightcurves[i] for i in range(start, end)])

	# and run!
	t0 = time()
	results = []
	for batch in batches:
		result = ls_proc.run(batch, freqs=freqs)
		ls_proc.finish()
		results.extend(result)

	dt = time() - t0

	print("batching but not reusing memory:\n"
		  " %e sec. / lc [%e sec. total]"%( dt / nlc, dt))

	# ... what about if we didn't do any batching at all?

	# and run!
	t0 = time()
	results = []
	for lightcurve in lightcurves:
		result = ls_proc.run([lightcurve], freqs=freqs)
		ls_proc.finish()
		results.extend(result)

	dt = time() - t0

	print("no batching:\n"
		  " %e sec. / lc [%e sec. total]"%( dt / nlc, dt))


For me, running this script (``example.py``) gives the following
output to stdout:

.. code:: sh

	(3.0, 2.9999814655808299)
	batching:
	 3.164886e-03 sec. / lc [3.164886e+00 sec. total]
	batching but not reusing memory:
	 5.288674e-02 sec. / lc [5.288674e+01 sec. total]
	no batching:
	 5.464483e-02 sec. / lc [5.464483e+01 sec. total]

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
