#!/bin/bash 
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

# nvcc --ptx ${SCRIPTPATH}//GPUFun.cu -rdc=true -DCUDA_FORCE_CDP1_IF_SUPPORTED -o ${SCRIPTPATH}/GPUFun.ptx
Fun=$(cat ${SCRIPTPATH}//GPUFun.cu)
# Fun=$(cat ${SCRIPTPATH}/GPUFun.ptx)
rm ${SCRIPTPATH}/GPUFun.py
echo "def getGPUCode():
    GPUCode = \"\"\"
$Fun
\"\"\"
    return GPUCode" >> ${SCRIPTPATH}/GPUFun.py

# rm ${SCRIPTPATH}/GPUFun.ptx