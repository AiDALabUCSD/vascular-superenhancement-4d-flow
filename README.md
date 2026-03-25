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
- [Developer Notes](#developer-notes)

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

3. **Build Patient Images**
   ```bash
   python -m vascular_superenhancement.commands.build_patients [--config CONFIG] [--overwrite-images] [--overwrite-catalogs] [--max-processors N] [--debug]
   ```
   - Builds all patient images (3D cine and 4D flow) from DICOM catalogs
   - Options:
     - `--config`: Name of the config file to use (default: "default")
     - `--overwrite-images`: Overwrite existing image files if they exist
     - `--overwrite-catalogs`: Overwrite existing catalog files if they exist
     - `--max-processors`: Maximum number of processors to use (default: CPU count - 1)
     - `--debug`: Enable debug logging

4. **Data Synchronization**
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

5. **Paired Video Crop GUI (PowerPoint Overlay Reveal)**
   ```bash
   # Opens file picker dialogs for both videos:
   python scripts/crop_pair_square_gui.py --out-dir cropped/
   ```
   - Opens an interactive preview to set one value: `TOP_CUT` (pixels removed from top).
   - Lets you select the two input videos through file dialogs each run.
   - Exports both videos with identical **1:1 square crop**, anchored to the **bottom-right**.
   - Designed for side-by-side/slide-by-slide reveal workflows where both videos must stay perfectly aligned.
   - Optional: pass files directly via `--without-overlay ... --with-overlay ...` if preferred.
   - Dependencies: `ffmpeg` + `opencv-python` (`pip install opencv-python`).

### Model Training

The training pipeline is now fully implemented and ready for use:

```bash
python -m vascular_superenhancement.training.train
```

**Training Features:**
- **GAN Architecture**: Uses a UNet generator with PatchGAN discriminator
- **Loss Functions**: Combined GAN loss and L1 loss for realistic image generation
- **Data Loading**: TorchIO-based patch sampling with queue system
- **Transforms**: Comprehensive preprocessing pipeline with resampling and normalization
- **Monitoring**: Weights & Biases integration for experiment tracking
- **Checkpointing**: Automatic model saving and early stopping
- **Visualization**: Sample predictions saved during training

**Configuration:**
- Training parameters are configurable via Hydra configs in `hydra_configs/`
- Model architecture, data transforms, and training hyperparameters can be customized
- Supports multiple experiment configurations and hyperparameter sweeps

### Model Architecture

**Generator (UNet):**
- 3D UNet architecture using MONAI
- Input: 2 channels (magnitude + speed)
- Output: 1 channel (synthetic contrast-enhanced image)
- Configurable channels, strides, and activation functions

**Discriminator (PatchGAN):**
- 3D PatchGAN-style discriminator
- Receptive field: 46x46x46 voxels
- Input: 3 channels (magnitude + speed + target/prediction)
- Output: Patch-wise real/fake classification

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
  - ✅ **NEW**: Patient image building CLI tool with multiprocessing support
  - ✅ **NEW**: Data alignment strategy using 3D cine as spatial reference
- **Model Development**: ✅ Complete
  - ✅ **NEW**: Full training pipeline implemented
  - ✅ **NEW**: GAN architecture with UNet generator and PatchGAN discriminator
  - ✅ **NEW**: Comprehensive loss functions (GAN + L1)
  - ✅ **NEW**: TorchIO-based data loading with patch sampling
  - ✅ **NEW**: Hydra configuration system for experiment management
  - ✅ **NEW**: Weights & Biases integration for experiment tracking
  - ✅ **NEW**: Automatic checkpointing and early stopping
  - ✅ **NEW**: Training visualization and sample prediction saving
  - ✅ **NEW**: Train/val/test splits implemented with CSV management
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

## Developer Notes

### Immediate Tasks & Improvements

**Training Pipeline Optimizations:**
- ✅ **Reduce epoch size**: Implemented timepoints_as_augmentation system to use one timepoint per patient per epoch efficiently
- ✅ **Patch overlapping**: Implemented patch overlapping for smoother full patient inference during evaluation
- ✅ **W&B visualizations**: Added comprehensive visualizations to Weights & Biases for better experiment monitoring
- ✅ **Increase batch size**: Optimized memory usage to allow larger batch sizes (increased to 32) for better training efficiency
- ✅ **Fix visualization loop**: Fixed the visualization loop that selects patients for visualization

**Future Enhancements:**
- [ ] Add inference pipeline
- [ ] Implement model evaluation metrics
- [ ] Add data augmentation strategies
- [ ] Optimize memory usage for larger datasets
- [ ] Add model export functionality
- [ ] Implement ensemble methods

**Bug Fixes:**
- [ ] Address any training convergence issues
- [ ] Fix potential memory leaks in data loading
- [ ] Resolve any CUDA out-of-memory errors

**Documentation:**
- [ ] Add API documentation
- [ ] Create usage examples
- [ ] Add performance benchmarks
- [ ] Document model architecture decisions

