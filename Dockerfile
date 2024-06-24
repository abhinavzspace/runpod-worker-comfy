# Use Nvidia CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

ARG SKIP_DEFAULT_MODELS
# Download checkpoints/vae/LoRA to include in image.
RUN if [ -z "$SKIP_DEFAULT_MODELS" ]; then wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors; fi
RUN if [ -z "$SKIP_DEFAULT_MODELS" ]; then wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors; fi
RUN if [ -z "$SKIP_DEFAULT_MODELS" ]; then wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; fi
RUN wget -O models/checkpoints/zavychromaxl_v80.safetensors https://huggingface.co/tiddu/checkpoints/resolve/main/zavychromaxl_v80.safetensors
RUN wget -O models/controlnet/diffusion_pytorch_model.safetensors https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors
RUN bash -c 'mkdir -p /comfyui/models/instantid'
RUN wget -O models/instantid/ip-adapter.bin https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin
RUN bash -c 'mkdir -p /comfyui/models/insightface/models/antelopev2'
RUN wget -O models/insightface/models/antelopev2/1k3d68.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx
RUN wget -O models/insightface/models/antelopev2/2d106det.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx
RUN wget -O models/insightface/models/antelopev2/genderage.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx
RUN wget -O models/insightface/models/antelopev2/glintr100.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx
RUN wget -O models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx
RUN wget -O models/loras/ExpressionsXL.safetensors https://huggingface.co/tiddu/loras/resolve/main/ExpressionsXL.safetensors
RUN wget -O models/loras/SDXLHighDetail_v5.safetensors https://huggingface.co/tiddu/loras/resolve/main/SDXLHighDetail_v5.safetensors
RUN wget -O models/loras/epiCPhotoXL.safetensors https://huggingface.co/tiddu/loras/resolve/main/epiCPhotoXL.safetensors
RUN wget -O models/loras/sdxl_lightning_4step_lora.safetensors https://huggingface.co/tiddu/loras/resolve/main/sdxl_lightning_4step_lora.safetensors

# Add custom nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/ComfyUI-Impact-Pack
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git custom_nodes/was-node-suite-comfyui
RUN git clone https://github.com/rgthree/rgthree-comfy.git custom_nodes/rgthree-comfy
RUN git clone https://github.com/cubiq/ComfyUI_InstantID.git custom_nodes/ComfyUI_InstantID
RUN git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git custom_nodes/ComfyUI-Inspire-Pack
RUN git clone https://github.com/shadowcz007/comfyui-mixlab-nodes.git custom_nodes/comfyui-mixlab-nodes
RUN git clone https://github.com/sipherxyz/comfyui-art-venture.git custom_nodes/comfyui-art-venture
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes
RUN git clone https://github.com/bash-j/mikey_nodes.git custom_nodes/mikey_nodes
RUN git clone https://github.com/chflame163/ComfyUI_LayerStyle.git custom_nodes/ComfyUI_LayerStyle

# Install ComfyUI dependencies
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir xformers==0.0.21 \
    && pip3 install -r requirements.txt

# Install runpod
RUN pip3 install runpod requests

# For instantid
RUN pip3 install insightface onnxruntime onnxruntime-gpu

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
