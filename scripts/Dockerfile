FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgtk2.0-dev \
	vim
RUN pip install sklearn opencv-python
RUN pip install wandb
RUN pip install ipdb
