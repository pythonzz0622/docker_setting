FROM ubuntu:18.04
FROM nvidia/cuda:11.0-base-ubuntu18.04
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
# CUDA
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

RUN apt-get update && apt-get install -y vim supervisor nginx net-tools curl cron \
  && apt-get install -y zip htop tree && apt-get -y install libgl1-mesa-glx

RUN mkdir user
RUN mkdir requirements
ADD requirements.txt /requirements
RUN pip install -r /requirements/requirements.txt