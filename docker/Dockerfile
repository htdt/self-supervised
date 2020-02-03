FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN pip install sklearn opencv-python
RUN pip install wandb
RUN pip install ipdb
RUN cd ~ && git clone https://github.com/NVIDIA/apex
RUN cd ~/apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
