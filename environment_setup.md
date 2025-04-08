# Environment Setup Information

This document provides the setup information for the `vascular-superenhancement-4d-flow` project. The project involves building a PyTorch model to enhance vascular structures in 4D flow MRI data using 3D cine MRI as the ground truth.

## Path Structure
- `/home/ayeluru` is a symbolic link to `/mnt/yeluru/`
- This means paths that start with either prefix refer to the same physical location

## Conda/Mamba Installation
- Physical location: `/mnt/yeluru/miniconda3`
- Accessible through: `/home/ayeluru/miniconda3` (symlink)
- Both `conda` and `mamba` are installed in this location

## Environment Directories
Default environment locations (same physical location, different paths):
1. Primary: `/mnt/yeluru/miniconda3/envs` 
   - Also accessible as: `/home/ayeluru/miniconda3/envs`
2. Secondary: `/mnt/yeluru/.conda/envs`
   - Also accessible as: `/home/ayeluru/.conda/envs`

## Project Environment
- Name: `vascular-superenhancement-4d-flow`
- Python version: 3.10
- CUDA version: 12.1 (system has CUDA 12.4, but PyTorch uses 12.1)

## Notes for AI Assistants
- Always use the physical paths (`/mnt/yeluru/...`) rather than symlinked paths to avoid confusion
- When managing environments, check for duplicate paths that might point to the same location
- The system has CUDA 12.4 installed, but PyTorch packages should use CUDA 12.1 for compatibility
- Environment operations (create/remove) should target the primary environment directory at `/mnt/yeluru/miniconda3/envs` 