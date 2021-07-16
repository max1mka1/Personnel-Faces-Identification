#FROM nvidia/cuda:10.2-cudnn7-devel
FROM jhonatans01/python-dlib-opencv

WORKDIR /app

COPY requirements.txt /app
COPY ./api /app/

RUN apt update -y; apt-get upgrade -y;
RUN apt install -y \
        python3 \
        python3-pip \
        python3-dev

RUN python3 --version
RUN pip3 install --upgrade pip

RUN apt-get update && apt-get install -y --no-install-recommends \
            git \
            cmake \
            libsm6 \
            libxext6 \
            libxrender-dev \
            build-essential cmake pkg-config \
            libx11-dev libatlas-base-dev \
            libgtk-3-dev libboost-python-dev

RUN pip3 install -r requirements.txt
RUN pip install -v --install-option="--no" --install-option="DLIB_USE_CUDA" dlib

CMD ["python3", "app.py"]
