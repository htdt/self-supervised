#!/usr/bin/env bash
GPU=${1:-0}
docker run -v ~/white_ss:/workspace --rm --gpus device=$GPU --shm-size 16G -it white_ss /bin/bash
