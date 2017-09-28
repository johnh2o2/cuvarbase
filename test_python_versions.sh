#!/bin/bash
#
# Very rough script for testing cuvarbase compatibility across python 
# versions
#
# (c) John Hoffman
#
# Run this from the top-level cuvarbase directory


# Print everything you do.
set -x

# Decide which python version to test
PYTHON_VERSION=3.6

# Put your cuda installation directory here
export CUDA_ROOT=/usr/local/cuda

########################################################################
CONDA_ENVIRONMENT_NAME=cuvar
CUVARBASE_DIR=$PWD

# Export the library paths
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib:${LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="${CUDA_ROOT}/lib:${DYLD_LIBRARY_PATH}"
export PATH="${CUDA_ROOT}/bin:${PATH}"

# Erase the testing conda environment if it already exists
test_str=`conda info --envs | grep ${CONDA_ENVIRONMENT_NAME}`
if [ "$test_str" != "" ]; then
        echo "removing conda environment ${CONDA_ENVIRONMENT_NAME}"
        conda remove -y --name ${CONDA_ENVIRONMENT_NAME} --all
fi

# Create the conda environment for testing with the right Python version
conda create -y -n $CONDA_ENVIRONMENT_NAME python=$PYTHON_VERSION numpy

# Activate the conda environment
source activate $CONDA_ENVIRONMENT_NAME

cd $CUVARBASE_DIR

# Install from the present directory, ignoring caches
pip install --no-cache-dir -e .

# test
python setup.py test

# (optionally) clean up conda environment
#source deactivate
#conda remove -y --name $CONDA_ENVIRONMENT_NAME --all
