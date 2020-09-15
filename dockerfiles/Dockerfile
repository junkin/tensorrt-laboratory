#FROM gitlab-master.nvidia.com/compute/mlperf-inference:ampere-base
#FROM bert-mig-image
FROM nvcr.io/nvidian/tensorrt:20.06-py3

RUN apt update
RUN apt install -y clang-format libssl-dev openssl libz-dev software-properties-common

# remove base cmake
RUN apt remove --purge -y cmake
RUN apt autoremove -y
RUN apt autoclean -y

# install cmake ppa from kitware - https://apt.kitware.com/
RUN apt install -y apt-transport-https ca-certificates gnupg software-properties-common wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt update && apt install -y cmake

# then remove FindGTest.cmake installed by cmake
RUN find / -name "FindGTest.cmake" -exec rm -f {} \;

RUN apt install -y libcufft-dev-11-0 cuda-nvml-dev-11-0

ENV LD_LIBRARY_PATH=/externals/myelin/x86_64/cuda-11.0/lib:/externals/cudnn/x86_64/8.0/cuda-11.0/lib64:/usr/local/cuda-11.0/targets/x86_64-linux/lib
ENV CCACHE_DIR=/tmp/.ccache
#RUN cd /usr/lib/x86_64-linux-gnu && ln -s libnvidia-ml.so.1 libnvidia-ml.so
