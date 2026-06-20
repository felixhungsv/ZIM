#!/bin/bash

if [ ! -d "venv" ]; then
    python3.12 -m venv venv
fi

source venv/bin/activate

pip install onnx onnxruntime matplotlib pycocotools opencv-python 
pip install torch torchvision easydict numpy gradio==4.38.1 "huggingface-hub<1.0" "pydantic==2.10.6" gradio-image-prompter fastapi==0.112.2 git+https://github.com/facebookresearch/segment-anything.git
pip install -e . --no-deps