#!/usr/bin/env python

import io
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def read(path, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py

    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = version('cuvarbase/__init__.py')

setup(name='cuvarbase',
      version=VERSION,
      description="Period-finding and variability on the GPU",
      author='John Hoffman',
      author_email='johnh2o2@gmail.com',
      packages=['cuvarbase',
                'cuvarbase.tests'],
      package_data={'cuvarbase': ['kernels/*cu']},
      url='https://github.com/johnh2o2/cuvarbase',
      setup_requires=['pytest-runner'],
      install_requires=['numpy>=1.17',
                        'scipy>=1.3',
                        'pycuda>=2017.1.1,!=2024.1.2',
                        'scikit-cuda'],
      tests_require=['pytest',
                     'nfft',
                     'matplotlib',
                     'astropy'],
      python_requires='>=3.7',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C',
        'Programming Language :: C++'])
