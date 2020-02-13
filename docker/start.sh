#!/usr/bin/env bash
if [ -z "$1" ]
	then
		GPU="all"
	else
		GPU="device=${1}"
fi
docker run -v ~/white_ss:/workspace -v ~/ILSVRC2012:/imagenet --rm --gpus $GPU --shm-size 16G -it white_ss /bin/bash
