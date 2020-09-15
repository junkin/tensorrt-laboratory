#!/bin/bash
docker run --gpus=1 --rm -ti -v $PWD:/tmp nvcr.io/nvidian/ac-jarvis/nemo:20.02-py3

