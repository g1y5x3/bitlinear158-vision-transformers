FROM --platform=linux/amd64 pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN apt-get update && \
    apt-get install -y ssh && \ 
    apt-get install -y vim  && \
    apt-get install -y git  && \
    apt-get install -y tmux && \
    apt-get install -y unzip && \
    apt-get -y install python3-pip && \
    pip install wandb && \
    pip install transformers && \
    pip install ipywidgets && \
    pip install pycocotools && \
    pip install scipy && \
    pip install tensorboard && \
    pip install deepspeed && \
    pip install ipykernel && \
    pip install notebook
