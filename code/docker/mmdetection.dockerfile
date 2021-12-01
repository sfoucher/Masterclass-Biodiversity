# docker run --gpus all --shm-size=8g -it --rm mmdetection:latest bash
# docker run --gpus all --shm-size=8g -v $pwd/jupyter_notebooks:/mmdetection/jupyter_notebooks -it --rm mmdetection:latest bash
ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection -b v2.18.0
WORKDIR /mmdetection
# COPY jupyter_notebooks jupyter_notebooks/
RUN pip install --src /mmddetection/jupyter_notebooks -e "git+https://github.com/sfoucher/Masterclass-Biodiversity.git@code-reorg#egg=MyPackageName&subdirectory=jupyter_notebooks" 
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN conda install -y -c conda-forge jupyterlab
