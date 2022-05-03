# Setting environment arguments
ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"
# ARG PYTORCH = "1.6.0"
# ARG CUDA="10.1"
# ARG CUDNN="7"

# Define the base image to start the building process.
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Setting environment variables.
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Update and install various libraries.
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
#TODO: The CUDA version (i.e. 11.3) and PYTORCH version (i.e. 1.10.0) needs to be changed accordingly.

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git -b v2.21.0 /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Install Masterclass-Biodiversity
RUN git clone https://github.com/sfoucher/Masterclass-Biodiversity --branch code-reorg /CRIM_notebooks