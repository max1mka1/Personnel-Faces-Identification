# FROM python:3.7
FROM nvidia/cuda:10.1-cudnn7-devel

WORKDIR /app

COPY requirements.txt /app
COPY ./api /app/

RUN apt update -y; apt-get upgrade -y;
RUN apt install -y \
        python3 \
        python3-pip \
        python3-dev

RUN pip3 install --upgrade pip

# Install face recognition dependencies
RUN apt install -y \
            git \
            cmake \
            libsm6 \
            libxext6 \
            libxrender-dev \
            build-essential cmake pkg-config \
            libx11-dev libatlas-base-dev \
            libgtk-3-dev libboost-python-dev

RUN pip3 install -r requirements.txt

CMD ["python3", "app.py"]
