FROM ubuntu:18.04

# Set timezone
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

##Set CUDA
FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pu
ENV CUDA_MAJOR_VERSION=11
ENV CUDA_MINOR_VERSION=2
ENV CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
ENV LD_LIBRARY_PATH_NO_STUBS="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/bin/lib"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/local/bin/lib"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION"


# SET LINUX COMMEND
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    g++ \
    gcc \
    openjdk-8-jdk \
    python3-dev \
    python3-pip \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libssl-dev \
    libzmq3-dev \
    vim \
    git

# SET PYTHON
RUN apt-get update \
  && apt-get install -y python3.8 python3-pip python3.8-dev \
  && apt-get install -y --reinstall systemd \
  && apt-get install libaio1 \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN ln -sf /usr/bin/python3.8 /usr/bin/python &&  ln -sf /usr/bin/python3.8 /usr/bin/python3 \
    && rm /usr/local/bin/python && ln -sf /usr/bin/python3.8 /usr/local/bin/python &&  ln -sf /usr/bin/python3.8 /usr/local/bin/python3 \
    && apt-get install -y python3-pip python-dev python3.8-dev && python3 -m pip install pip --upgrade

# SET mirror server 
RUN sed -i 's@archive.ubuntu.com@mirror.kakao.com@g' /etc/apt/sources.list

# Install miniconda3
ARG CONDA_DIR=/opt/conda

# add to path
ENV PATH $CONDA_DIR/bin:$PATH

# Install miniconda
RUN echo "export PATH=$CONDA_DIR/bin:"'$PATH' > /etc/profile.d/conda.sh && \
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh -o ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Conda 가상환경 생성
RUN conda config --set always_yes yes --set changeps1 no && \
    conda create -y -q -n py37 python=3.7

ENV PATH /opt/conda/envs/py37/bin:$PATH
ENV CONDA_DEFAULT_ENV py37
ENV CONDA_PREFIX /opt/conda/envs/py37

## set DL Framework
RUN pip install tensorflow==2.6.0
RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
RUN pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

RUN pip install setuptools && \
    pip install mkl && \
    pip install pymysql && \
    pip install numpy && \
    pip install scipy && \
    pip install pandas==1.2.5 && \
    pip install jupyter notebook && \
    pip install matplotlib && \
    pip install seaborn && \
    pip install hyperopt && \
    pip install optuna && \
    pip install missingno && \
    pip install mlxtend && \
    pip install catboost && \
    pip install kaggle && \
    pip install folium && \
    pip install librosa && \
    pip install nbconvert && \
    pip install Pillow && \
    pip install tqdm && \
    pip install gensim && \
    pip install cupy-cuda112





COPY setting/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

RUN mkdir /root/.jupyter/custom/

COPY setting/custom.js /root/.jupyter/custom/custom.js
COPY setting /home/setting/
# 기본
EXPOSE 9000
