FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install cuvarbase dependencies
RUN pip3 install numpy>=1.17 scipy>=1.3

# Install PyCUDA (may need to be compiled from source)
RUN pip3 install pycuda

# Install scikit-cuda
RUN pip3 install scikit-cuda

# Create working directory
WORKDIR /workspace

# Install cuvarbase (when ready)
# COPY . /workspace
# RUN pip3 install -e .

# Default command
CMD ["/bin/bash"]
