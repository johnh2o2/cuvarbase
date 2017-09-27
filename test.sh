set -x
PYTHON_VERSION=2.7
CONDA_ENVIRONMENT_NAME=CUVARBASE_TESTING
PIP_DEPS="numpy astropy matplotlib nfft scikit-cuda pycuda"
CUDA_DIR=/usr/local/cuda/lib


test_str=`conda info --envs | grep ${CONDA_ENVIRONMENT_NAME}`
if [ "$test_str" != "" ]; then
        echo "removing conda environment ${CONDA_ENVIRONMENT_NAME}"
        conda remove -y --name ${CONDA_ENVIRONMENT_NAME} --all
fi

conda create -y -n $CONDA_ENVIRONMENT_NAME python=$PYTHON_VERSION
source activate $CONDA_ENVIRONMENT_NAME
pip install $PIP_DEPS

python -c "import pycuda.autoinit"

export LD_LIBRARY_PATH="${CUDA_DIR}:${LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="${CUDA_DIR}:${DYLD_LIBRARY_PATH}"

python setup.py install
py.test -x cuvarbase

source deactivate
conda remove -y --name $CONDA_ENVIRONMENT_NAME --all
