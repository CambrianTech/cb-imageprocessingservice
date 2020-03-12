FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

# Install dependencies needed for OpenCV and CUDA headers
RUN apt-get update && apt-get install --no-install-recommends -y \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        && apt-get clean

COPY . .

RUN pip install --upgrade pip
RUN cd planercnn && pip install -r requirements.txt

# Compile custom ops of planercnn (https://github.com/NVlabs/planercnn#getting-started)
RUN cd planercnn/nms/src/cuda/ && nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61 && cd ../../ && python build.py
RUN cd planercnn/roialign/roi_align/src/cuda/ && nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61 && cd ../../ && python build.py

EXPOSE 8081
CMD cd planercnn && python serve.py --suffix=warping_refine --dataset=inference --methods=f