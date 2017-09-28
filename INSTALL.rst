Install instructions
********************

These installation instructions are for Linux/BSD-based systems (OS X/macOS, Ubuntu, etc.). Windows users, your suggestions and feedback is welcome if we can make your life easier!

Installing the Nvidia Toolkit
-----------------------------

``cuvarbase`` requires PyCUDA and scikit-cuda, which both require the Nvidia toolkit for access to the Nvidia compiler, drivers, and runtime libraries.

Go to the `NVIDIA Download page <https://developer.nvidia.com/cuda-downloads>`_ and select the distribution for your operating system. Everything has been developed and tested using **version 8.0**, so it may be best to stick with that version for now until we verify that later versions are OK.

.. warning::

	Make sure that your ``$PATH`` environment variable contains the location of the ``CUDA`` binaries. You can test this by trying
	``which nvcc`` from your terminal. If nothing is printed, you'll have to amend your ``~/.bashrc`` file: 

	``echo "export PATH=/usr/local/cuda/bin:${PATH}" >> ~/.bashrc && . ~/.bashrc``

	The ``>>`` is not a typo -- using one ``>`` will *overwrite* the ``~/.bashrc`` file. Make sure you change ``/usr/local/cuda`` to the appropriate location of your Nvidia install.

	**Also important**

	Make sure your ``$LD_LIBRARY_PATH`` and ``$DYLD_LIBRARY_PATH`` are also similarly modified to include the ``/lib`` directory of the CUDA install:

	``echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib:${LD_LIBRARY_PATH}" >> ~/.bashrc && . ~/.bashrc``
	``echo "export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:${DYLD_LIBRARY_PATH}" >> ~/.bashrc && . ~/.bashrc``


Using conda
-----------

`Conda <https://www.continuum.io/downloads>`_ is a great way to do this in a safe, isolated environment.

First create a new conda environment (named ``pycu`` here) that will use Python 2.7 (python 2.7, 3.4, 3.5, and 3.6
have been tested), with the numpy library installed. 

.. code:: bash

	conda create -n pycu python=2.7 numpy

.. note::

	The numpy library *has* to be installed *before* PyCUDA is installed with pip. 
	The PyCUDA setup needs to be able to access the numpy library for building against it. You can do this with
	the above command, or alternatively just do ``pip install numpy && pip install cuvarbase``

Then activate the virtual environment

.. code:: bash

	source activate pycu

and then use ``pip`` to install ``cuvarbase``

.. code:: bash

	pip install cuvarbase


Installing with just ``pip``
----------------------------

**If you don't want to use conda** the following should work with just pip

.. code:: bash

	pip install numpy 
	pip install cuvarbase


Troubleshooting PyCUDA installation problems
--------------------------------------------

The ``PyCUDA`` installation step may be a hiccup in this otherwise orderly process. If you run into problems installing ``PyCUDA`` with pip, you may have to install PyCUDA from source yourself. It's not too bad, but if you experience any problems, please submit an `Issue <https://github.com/johnh2o2/cuvarbase/issues>`_ at the ``cuvarbase`` Github page and I'll amend this documentation.

Below is a small bash script that (hopefully) automates the process of installing PyCUDA in the event of any problems you've encountered at this point.

.. code-block:: bash
	
	PYCUDA="pycuda-2017.1.1"
	PYCUDA_URL="https://pypi.python.org/packages/b3/30/9e1c0a4c10e90b4c59ca7aa3c518e96f37aabcac73ffe6b5d9658f6ef843/pycuda-2017.1.1.tar.gz#md5=9e509f53a23e062b31049eb8220b2e3d"
	CUDA_ROOT=/usr/local/cuda

	# Download
	wget $PYCUDA_URL

	# Unpack
	tar xvf ${PYCUDA}.tar.gz
	cd $PYCUDA

	# Configure with current python exe
	./configure.py --python-exe=`which python` --cuda-root=$CUDA_ROOT
	python setup.py build
	python setup.py install

If everything goes smoothly, you should now test if ``pycuda`` is working correctly.

.. code:: bash

	python -c "import pycuda.autoinit; print 'Hurray!'"

If everything works up until now, we should be ready to install ``cuvarbase``

.. code:: bash

	pip install cuvarbase

Installing from source
----------------------

You can also install directly from the repository. Clone the ``git`` repository on your machine:

.. code:: bash
	
	git clone https://github.com/johnh2o2/cuvarbase

Then install!

.. code:: bash

	cd cuvarbase
	python setup.py install

The last command can also be done with pip:

.. code:: bash

	pip install -e .



Troubleshooting on a Mac
------------------------

Nvidia offers `CUDA for Mac OSX <https://developer.nvidia.com/cuda-downloads>`_. After installing the
package via downloading and running the ``.dmg`` file, you'll have to make a couple of edits to your
``~/.bash_profile``:

.. code:: sh
    
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:/usr/local/cuda/lib"
    export PATH="/usr/local/cuda/bin:${PATH}"

and then source these changes in your current shell by running ``. ~/.bash_profile``. 

Another important note: **nvcc (8.0.61) does not appear to support the latest clang compiler**. If this is
the case, running ``python example.py`` should produce the following error:

.. code:: bash

    nvcc fatal   : The version ('80100') of the host compiler ('Apple clang') is not supported

You can fix this problem by temporarily downgrading your clang compiler. To do this:

- `Download Xcode command line tools 7.3.1 <http://adcdownload.apple.com/Developer_Tools/Command_Line_Tools_OS_X_10.11_for_Xcode_7.3.1/Command_Line_Tools_OS_X_10.11_for_Xcode_7.3.1.dmg>`_
- Install.
- Run ``sudo xcode-select --switch /Library/Developer/CommandLineTools`` until ``clang --version`` says ``7.3``.
