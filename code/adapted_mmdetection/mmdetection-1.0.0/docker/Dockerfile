# docker run --gpus all --shm-size=8g -it --rm masterclass:latest bash
ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install mmdetection
RUN conda clean --all
RUN git clone  -b code-reorg https://github.com/sfoucher/Masterclass-Biodiversity.git /Masterclass-Biodiversity
WORKDIR /Masterclass-Biodiversity/code/adapted_mmdetection/mmdetection-1.0.0
ENV FORCE_CUDA="1"
RUN pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
RUN pip install mmcv==0.3.1
RUN pip install --no-cache-dir -e .
