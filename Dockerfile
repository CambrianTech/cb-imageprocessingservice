ARG UBUNTU_VERSION=18.04

ARG ARCH
ARG CUDA=10.0
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base

# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=7.4.1.5-1

# Needed for string substitution 
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        cuda-cublas-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        libsm6 \
        libxext6 \
        libxrender-dev

RUN [ "${ARCH}" = ppc64le ] || (apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda${CUDA} \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*)

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Needed to find libcuda.so.1
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

ARG USE_PYTHON_3_NOT_2=true
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow-gpu
ARG TF_PACKAGE_VERSION=1.13.1
RUN ${PIP} install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

#OpenCV based upon https://github.com/jacobs-robotics/docker-opencv3-py3-cuda/blob/master/Dockerfile
ARG WRITE_PATH=~/
ARG CUDA_ROOT=/usr/local/cuda-${CUDA}
ARG CUDA_LIB_PATH=${CUDA_ROOT}/targets/x86_64-linux/lib
ARG OPENCV_VERSION=3.3.0

#update system repos and libraries
RUN apt-get update -y && apt-get upgrade -y
#essential extra packages (I think some of these are silly - Joel)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    wget vim git cmake

#opencv dependencies 1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev \
    libavformat-dev libswscale-dev libv4l-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran
#opencv dependencies 2
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    libopencv-dev checkinstall yasm libdc1394-22-dev libxine2-dev \
    libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libtbb-dev libqt4-dev libfaac-dev \
    libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev \
    v4l-utils ffmpeg qt5-default libhdf5-dev

#opencv python dependencies
RUN ${PIP} --no-cache-dir install --upgrade \
    numpy scipy matplotlib scikit-image scikit-learn jupyter notebook pandas

#download opencv from repos
RUN cd ${WRITE_PATH} && wget -O opencv.zip https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.zip && unzip opencv.zip
RUN cd ${WRITE_PATH} && wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/${OPENCV_VERSION}.zip && unzip opencv_contrib.zip

RUN cd ${WRITE_PATH}/opencv-${OPENCV_VERSION}/ && mkdir opencv_build && cd opencv_build

RUN cmake \
   -D CMAKE_BUILD_TYPE=RELEASE \
   -D CMAKE_INSTALL_PREFIX=/usr/local \
   -D BUILD_opencv_java=OFF \
   -D INSTALL_C_EXAMPLES=OFF \
   -D OPENCV_EXTRA_MODULES_PATH=${WRITE_PATH}/opencv_contrib-${OPENCV_VERSION}/modules \
   -D PYTHON_EXECUTABLE=/usr/local/bin/python \
   -D WITH_CUDA=ON \
   -D WITH_CUBLAS=ON \
   -D WITH_TBB=ON \
   -D WITH_V4L=ON \
   -D WITH_QT=ON \
   -D WITH_OPENGL=ON \
   -D BUILD_PERF_TESTS=OFF \
   -D BUILD_TESTS=OFF \
   -D CUDA_CUDA_LIBRARY=${CUDA_LIB_PATH}/stubs/libcuda.so \
   -D CUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT} \
   -D CUDA_CUDART_LIBRARY=${CUDA_LIB_PATH}/libcudart.so \
   -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" \
   -D CUDA_GENERATION=Auto \
   -D ENABLE_FAST_MATH=1 \
   -D CUDA_FAST_MATH=1 \
   -D WITH_NVCUVID=OFF \
   -D WITH_CUFFT=ON \
   -D WITH_EIGEN=ON \
   -D WITH_IPP=ON

#bash setup
COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

#FROM base

COPY .  /

RUN chmod 755 start.sh

# libglib needed for OpenCV
RUN apt-get update && apt-get install --no-install-recommends -y libglib2.0-0 && apt-get clean

RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install /cb-core/ --no-cache-dir

EXPOSE 8080

ENTRYPOINT [ "/start.sh" ]