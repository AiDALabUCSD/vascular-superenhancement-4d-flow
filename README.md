# Vascular Superenhancement 4D Flow

## Table of Contents
- [Introduction](#introduction)
- [Project Goals](#project-goals)
- [Data](#data)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [About the Author](#about-the-author)

## Introduction
This project aims to develop a PyTorch model that performs **vascular superenhancement**, 
enhancing the visibility of vascular structures in MRI data.

## Project Goals
The model processes **4D flow magnitude** and **velocity** data to produce a 4D flow magnitude 
volume with enhanced vasculature—simulating the effect of contrast injection.

## Data
- **Input**: 4D flow magnitude and velocity data  
- **Ground Truth**: 3D cine MRI data for vascular enhancement

## Model
The model is built using PyTorch, leveraging advanced neural network architectures 
for accurate vascular enhancement.

## Installation
1. Ensure **Python 3.10** and **CUDA 12.1** (or whichever version your GPU requires) are installed.
2. Create and activate a conda environment:
   ```bash
   conda create -n vascular_superenhancement python=3.10 \
       pytorch torchvision torchaudio pytorch-cuda=11.8 \
       -c pytorch -c nvidia
   conda activate vascular_superenhancement
   ```
3. Install `uv` (optional) and other dependencies from your `pyproject.toml`:
   ```bash
   pip install uv
   # or uv pip install -e .
   ```

## Usage
1. Prepare your 4D flow MRI data in a designated folder.
2. Run the model training or inference script (example):
   ```bash
   python src/vascular_superenhancement/modeling/train_superenhancement.py
   ```
   (Adjust based on your actual scripts and file structure.)
3. The script outputs enhanced vascular images simulating contrast injection.
4. Check logs, metrics, or any configured experiment tracking for results.

## About the Author
This repository is maintained by **Akhilesh Yeluru**, a graduate student. It serves as a 
platform to experiment with the creation of a vascular superenhancement model, facilitating 
both development and application in 4D Flow MRI research.

