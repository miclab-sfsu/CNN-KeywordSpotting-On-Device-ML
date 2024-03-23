#!/bin/bash
# python training.py && \
python pytorch_to_onnx.py && \
onnx2tf -i Weight/Onnx/Conv2D_Voice_Model_ACC87%.onnx -osd && \
python TF_to_C_header.py && \
cp Weight/TFLite_C_model/model.h keywordspotting_sonyspresense/