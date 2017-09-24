Install instructions
********************

Using conda
-----------

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


Using pip
---------

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
