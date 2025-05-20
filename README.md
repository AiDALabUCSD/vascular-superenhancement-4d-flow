# Vascular Superenhancement 4D Flow

## Table of Contents
- [Introduction](#introduction)
- [Project Goals](#project-goals)
- [Data](#data)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Project Status](#project-status)
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

### Data Management CLI Tools

1. **Extract Archives**
   ```bash
   python -m vascular_superenhancement.commands.extract_archives [--config CONFIG] [--overwrite]
   ```
   - Extracts all archives from the zipped directory to the unzipped directory
   - Options:
     - `--config`: Name of the config file to use (default: "default")
     - `--overwrite`: Overwrite existing extracted files if they exist

2. **Catalog DICOM Files**
   ```bash
   python -m vascular_superenhancement.commands.catalog_dicoms [--config CONFIG] [--overwrite]
   ```
   - Catalogs DICOM files for all patients in the unzipped directory
   - Options:
     - `--config`: Name of the config file to use (default: "default")
     - `--overwrite`: Overwrite existing catalog files if they exist

3. **Data Synchronization**
   ```bash
   python scripts/sync_to_nas.py
   ```
   - Synchronizes data with NAS storage system
   - Can be automated using the included cron job script:
     ```bash
     # Add to crontab (runs every 6 hours)
     0 */6 * * * /path/to/scripts/cron_sync.sh
     ```
   - Manual sync can be triggered using the `backup` alias:
     ```bash
     backup
     ```
     This runs `~/vascular-superenhancement-4d-flow/scripts/trigger_backup.sh`
     
     To set up the alias, add this line to your `~/.bashrc`:
     ```bash
     alias backup="~/vascular-superenhancement-4d-flow/scripts/trigger_backup.sh"
     ```
     Then reload your shell configuration:
     ```bash
     source ~/.bashrc
     ```

### Model Training and Inference
The model training and inference pipeline is currently under development. Documentation will be updated as these features become available.

## Project Status
- **Data Pipeline**: ✅ Complete
  - ✅ Archive extraction implemented
  - ✅ DICOM file cataloging implemented
  - ✅ Patient class with DICOM catalog management
  - ✅ 3D Cine MRI organization
    - ✅ Identified key DICOM tags for slice organization:
      - `CardiacNumberOfImages`
      - `LocationsInAcquisition`
      - `InstanceNumber`
    - ✅ Implementing 3D cine catalog creation
    - ✅ Added slice direction handling and flipping state tracking
  - ✅ 4D Flow MRI organization
    - ✅ Extending catalog logic for flow data
    - ✅ Incorporating velocity data from Tempus
    - ✅ Adding flow-specific DICOM tag filtering
    - ✅ Added slice direction handling and flipping state tracking
  - ✅ NIfTI conversion pipeline
    - ✅ Using organized catalogs for proper 4D conversion
    - ✅ Preserving temporal and spatial information
    - ✅ Automatic slice direction correction
    - ✅ Tracking of flipped state for both 3D cine and 4D flow data
  - ✅ Data preprocessing pipeline
    - ✅ Integration with torchio for medical image transformations
    - ✅ Support for standard preprocessing operations:
      - Resampling
      - Normalization
      - Spatial transformations
      - Intensity transformations
- **Model Development**: 🚧 In Progress
  - 🚧 Building training architecture and pipeline
  - 🚧 Testing datasets.py and dataloading.py
    - Need to implement train/val/test splits CSV
  - Neural network architecture design
- **Inference Pipeline**: 🚧 In Progress
  - Development of inference tools
  - Performance optimization
- **Documentation**: 🚧 In Progress
  - API documentation
  - Usage examples
  - Performance benchmarks

## About the Author
This repository is maintained by **Akhilesh Yeluru**, a graduate student. It serves as a 
platform to experiment with the creation of a vascular superenhancement model, facilitating 
both development and application in 4D Flow MRI research.

