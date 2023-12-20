#!/usr/bin/env bash
conda create -y --name any_grasp python=3.8 && \
	 conda activate any_grasp && \
	 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
	 pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine && \
	 pip install -r requirements.txt && \
	 cd pointnet2 && python setup.py install && cd .. && \
	 pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
