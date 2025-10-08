# Diffusion_pipe_in_ComfyUI_Win Custom Node

*click to see [中文文档](./READMEChinese.md)*

## Project Overview

Diffusion-Pipe In ComfyUI Custom Node is a powerful extension plugin that provides complete Diffusion model training and fine-tuning capabilities for ComfyUI. This project allows users to configure and launch training for various advanced AI models within ComfyUI's graphical interface, supporting both LoRA and full fine-tuning, covering the most popular image generation and video generation models available today.You can train Qwen lora with 16g Vram

***Video Demo: https://www.bilibili.com/video/BV1DAnKzTEup/?share_source=copy_web&vd_source=5a2c3d8b60d05e98a2e7f4f58f77eba5***

***[📋 View Supported Models](./docs/supported_models.md)***


# Quick Start
## You can use my pre configured portable environment pack

***You still need to download Microsoft MPI to prepare the deepspeed environment for Windows: https://www.microsoft.com/en-us/download/details.aspx?id=105289 ***

*Download and restart the computer*

```bash
git clone --recurse-submodules https://github.com/TianDongL/Diffusion_pipe_in_ComfyUI_Win.git
```
* If you haven't installed the submodules, follow these steps
* If you don't complete this step, training will not work

```bash
git submodule init
```
```bash
git submodule update
```

## Conda Environment Installation Guide

```bash
conda create -n comfyui_DP python=3.11
```
```bash
conda activate comfyui_DP
```
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

* You need to install pre-compiled wheels for Windows. You can find the compiled wheels in my Releases. This project requires deepspeed==0.17.0 https://github.com/TianDongL/Diffusion_pipe_in_ComfyUI_Win/releases

```bash
pip install E:/ComfyUI/deepspeed-0.17.0+720787e7-cp311-cp311-win_amd64.whl
```

* And flash-attn==2.8.1

```bash
pip install E:/ComfyUI/flash_attn-2.8.1-cp311-cp311-win_amd64.whl
```

* Also bitsandbytes compiled for Windows

```bash
pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-wheels/windows/index.html
```
```bash
cd /ComfyUI/custom_nodes/Diffusion_pipe_in_ComfyUI_Win.git
```
```bash
pip install -r requirements.txt
```

## Portable Environment Installation Guide

* You are responsible for backing up your portable environment
* My wheels are all compiled under Torch 2.7.1+cu128-cp311

*Skip this step if you already meet the requirements*

```bash
E:/ComfyUI_windows_portable/python_embeded/python.exe -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

*Install necessary dependencies directly*

*You need to install pre-compiled wheels for Windows. You can find the compiled wheels in my Releases. This project requires deepspeed==0.17.0 https://github.com/TianDongL/Diffusion_pipe_in_ComfyUI_Win/releases*

```bash
E:/ComfyUI_windows_portable/python_embeded/python.exe -m pip install E:/ComfyUI_windows_portable/python_embeded_DP/deepspeed-0.17.0+720787e7-cp311-cp311-win_amd64.whl
```

*And flash-attn==2.8.1*

```bash
E:/ComfyUI_windows_portable/python_embeded/python.exe -m pip install E:/ComfyUI_windows_portable/python_embeded_DP/flash_attn-2.8.1-cp311-cp311-win_amd64.whl
```

*And bitsandbytes compiled for Windows*

```bash
E:/ComfyUI_windows_portable/python_embeded/python.exe -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-wheels/windows/index.html
```

```bash
cd /ComfyUI/custom_nodes/Diffusion_pipe_in_ComfyUI_Win.git
```
```bash
E:/ComfyUI_windows_portable/python_embeded/python.exe -m pip install -r requirements.txt
```

## 🚀 One-Click Workflow Import

To get you started quickly, I've provided a pre-configured ComfyUI workflow file:

***[📋 Click to Import Complete Workflow](./examworkflow_DP.json)***

Simply drag this file into the ComfyUI interface to import the complete training workflow with all necessary node configurations.

## Please read the prompts in the workflow carefully, as they can help you build your dataset


# 📷 Workflow Interface Preview

<div align="center">

![Model Loading Node](./img/11.png)
Models can be stored in the ComfyUI model directory

![Launch Training and Monitoring](./img/22.png)
*Disable the Train node when debugging*

![Model Configuration](./img/33.png)


![Dataset Configuration](./img/44.png)

![Workflow Overview](./img/55.png)

![Monitoring Options](./img/66.png)
*kill port will stop all monitoring processes on the current port*

</div>


### Core Features

- 🎯 **Visual Training Configuration**: Graphically configure training parameters through ComfyUI nodes
- 🚀 **Multi-Model Support**: Support for 20+ latest Diffusion models
- 💾 **Flexible Training Methods**: Support for both LoRA training and full fine-tuning
- ⚡ **High-Performance Training**: Distributed training support based on DeepSpeed
- 📊 **Real-Time Monitoring**: Integrated TensorBoard for monitoring training progress
- 🎥 **Video Training**: Support for training video generation models
- 🖼️ **Image Editing**: Support for training image editing models

## System Requirements

### Hardware Requirements
- On Windows, it seems 16GB VRAM can train Qwen, which is quite Confusing

### Software Requirements
- **Operating System**: Windows 10/11 
- **ComfyUI**: Latest version


## Supported Models

This plugin supports over 20 of the latest Diffusion models, including:

| Model          | LoRA | Full Fine Tune | fp8/quantization |
|----------------|------|----------------|------------------|
|SDXL            |✅    |✅              |❌                |
|Flux            |✅    |✅              |✅                |
|LTX-Video       |✅    |❌              |❌                |
|HunyuanVideo    |✅    |❌              |✅                |
|Cosmos          |✅    |❌              |❌                |
|Lumina Image 2.0|✅    |✅              |❌                |
|Wan2.1          |✅    |✅              |✅                |
|Chroma          |✅    |✅              |✅                |
|HiDream         |✅    |❌              |✅                |
|SD3             |✅    |❌              |✅                |
|Cosmos-Predict2 |✅    |✅              |✅                |
|OmniGen2        |✅    |❌              |❌                |
|Flux Kontext    |✅    |✅              |✅                |
|Wan2.2          |✅    |✅              |✅                |
|Qwen-Image      |✅    |✅              |✅                |
|Qwen-Image-Edit |✅    |✅              |✅                |
|HunyuanImage-2.1|✅    |✅              |✅                |


## Node System Detailed Explanation

### 🗂️ Dataset Configuration Nodes

#### GeneralDatasetConfig (General Dataset Configuration)
Configure core parameters for training datasets:
- **Input Path**: Dataset directory path
- **Resolution Settings**: Training resolution configuration `[512]` or `[1280, 720]`
- **Aspect Ratio Bucketing**: Automatically handle images with different aspect ratios
- **Dataset Repetition**: Control data usage frequency
- **Cache Settings**: Optimize data loading performance

#### GeneralDatasetPathNode (General Dataset Node)
Handle standard image-text pair datasets:
```
dataset/
├── image1.jpg
├── image1.txt
├── image2.png
└── image2.txt
```

#### EditModelDatasetPathNode (Edit Model Dataset)
Handle image editing datasets:
```
dataset/
├── source_images/
└── target_images/
```
source_images and target_images must have the same file names

#### FrameBucketsNode (Frame Bucketing Configuration)
Frame count configuration for video training:
- Support for multiple frame length training
- Automatic batch organization

#### ArBucketsNode (Aspect Ratio Bucketing Configuration)
Custom aspect ratio bucketing strategy:
- Precise control over bucket count
- Optimize VRAM usage

### 🤖 Model Configuration Nodes

- **SDXLModelNode**: SDXL model configuration
- **FluxModelNode**: Flux model configuration
- **SD3ModelNode**: SD3 model configuration
- **QwenImageModelNode**: Qwen image model
- **HiDreamModelNode**: HiDream model configuration
- **ChromaModelNode**: Chroma model configuration
- **Lumina2ModelNode**: Lumina2 model configuration
- **LTXVideoModelNode**: LTX-Video configuration
- **HunyuanVideoModelNode**: Hunyuan video configuration
- **Wan21ModelNode**: Wan2.1 configuration
- **Wan22ModelNode**: Wan2.2 configuration
- **FluxKontextModelNode**: Flux Kontext configuration
- **QwenImageEditModelNode**: Qwen image edit configuration
- **HunyuanImage-2.1Node**: Hunyuan image model configuration

### ⚙️ Training Configuration Nodes

#### GeneralConfig (General Training Settings)
Core training parameter configuration:
- **Training Epochs**: Control training duration
- **Batch Size**: GPU memory optimization
- **Learning Rate Schedule**: Warmup and decay strategies
- **Gradient Configuration**: Accumulation and clipping settings
- **Optimizer Settings**: AdamW, AdamW8bit, etc.
- **Memory Optimization**: Block swap, activation checkpointing

#### ModelConfig (Model Configuration)
Model-specific configuration:
- **Data Types**: bfloat16, float16, float8
- **LoRA Settings**: rank, alpha, dropout
- **Quantization Options**: FP8, 4bit quantization

#### AdapterConfigNode (Adapter Configuration)
Detailed LoRA adapter configuration:
- **Target Modules**: Select which model parts to train
- **LoRA Parameters**: rank, alpha, target dimensions
- **Training Strategy**: Partial freezing, layered learning rates

#### OptimizerConfigNode (Optimizer Configuration)
Detailed optimizer settings:
- **Optimizer Type**: AdamW, Lion, Adafactor
- **Learning Rate**: Base learning rate and scheduling
- **Regularization**: Weight decay, gradient clipping

### 🚀 Training Control Nodes

#### Train (Training Launcher)
Start and control the training process:
- **Configuration Merging**: Automatically merge dataset and training configs
- **Process Management**: Start and monitor training
- **Error Handling**: Exception capture and recovery
- **Log Output**: Real-time training status

#### TensorBoardMonitor (TensorBoard Monitor)
Real-time training monitoring:
- **Loss Curves**: Training and validation loss
- **Learning Rate Tracking**: Learning rate change curves
- **GPU Utilization**: Hardware usage statistics
- **Sample Preview**: Generated sample quality monitoring

#### OutputDirPassthrough (Output Directory Passthrough)
Utility node to simplify path passing.


## License

This project is open-sourced under the Apache License 2.0.

## Contributing

Issues and Pull Requests are welcome!

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Submit a Pull Request

## Acknowledgments

Thanks to the following projects and teams:
- ComfyUI team
- @tdrussell, the original author of Diffusion_Piped
- Hugging Face Diffusers
- DeepSpeed team
- Original authors of all models



