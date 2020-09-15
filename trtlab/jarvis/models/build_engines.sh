#!/bin/bash

python3 export_jasper_onnx_to_trt.py nn_encoder.onnx jasper_encoder.engine --max-seq-len 251 --seq-len 251 --batch-size 1 --max-batch-size 64
python3 export_jasper_onnx_to_trt.py nn_decoder.onnx jasper_decoder.engine --max-seq-len 126 --seq-len 126 --batch-size 1 --max-batch-size 64
