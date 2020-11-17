FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN pip install sklearn opencv-python
RUN pip install matplotlib
RUN pip install wandb
RUN pip install ipdb
ENTRYPOINT wandb login $WANDB_KEY && /bin/bash
