# Vascular Superenhancement Inference Module

This module provides inference capabilities for the Vascular Superenhancement 4D Flow model, allowing you to generate superenhanced images from 4D flow MRI data.

## Overview

The inference module consists of:
- `inference.py`: Main inference script that loads a trained model and generates predictions
- `default.yaml`: Configuration file with default inference parameters
- `README.md`: This documentation file

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support (recommended)
- TorchIO
- Hydra configuration framework
- Trained model checkpoint file

## Configuration

### Default Configuration (`default.yaml`)

The default configuration file contains the following parameters:

```yaml
inference_name: senate                    # Name for the inference run
checkpoint_path: /path/to/checkpoint.pt  # Path to trained model checkpoint
output_dir: ${path_config.base_working_dir}/${path_config.project_name}/working_dir/inference/${inference.inference_name}
patch_size: [48, 48, 48]                # Size of patches for inference
patch_overlap: 24                        # Overlap between patches
patch_aggregation_overlap_mode: "hann"   # Aggregation method for overlapping regions
num_workers: 1                           # Number of worker processes
batch_size: 200                          # Batch size for inference
patient_id: null                         # Patient ID (must be provided via command line)
time_point: 3                            # Default time point for 4D flow data
```

### Required Parameters

- `patient_id`: The phonetic ID of the patient to process
- `time_point`: The time point in the 4D flow sequence (default: 3)

### Optional Parameters

- `patch_overlap`: Controls the overlap between patches (higher values = smoother results but slower inference)
- `patch_size`: Size of 3D patches for processing large volumes
- `batch_size`: Number of patches processed simultaneously

## Usage

### Basic Usage

Run inference for a specific patient:

```bash
python -m vascular_superenhancement.inference.inference \
    path_config=all_patients \
    inference.patient_id=Fejoba
```

This will use the default time point (3) and all other default parameters.

### Custom Time Point

Specify a different time point:

```bash
python -m vascular_superenhancement.inference.inference \
    path_config=all_patients \
    inference.patient_id=Fejoba \
    inference.time_point=5
```

### Custom Patch Overlap

Adjust patch overlap for different quality/speed trade-offs:

```bash
python -m vascular_superenhancement.inference.inference \
    path_config=all_patients \
    inference.patient_id=Fejoba \
    inference.patch_overlap=16
```

### Custom Batch Size

Modify batch size based on available GPU memory:

```bash
python -m vascular_superenhancement.inference.inference \
    path_config=all_patients \
    inference.patient_id=Fejoba \
    inference.batch_size=100
```

## Output

The inference script generates a NIfTI file with the following naming convention:

```
pred_{patient_id}_t{time_point:02d}_overlap_{patch_overlap}_overlap-mode_{aggregation_mode}.nii.gz
```

Example output filename:
```
pred_Fejoba_t03_overlap_24_overlap-mode_hann.nii.gz
```

The output file contains the superenhanced image and is saved in the configured output directory.

## Performance Considerations

### Patch Overlap Trade-offs

- **Higher overlap (e.g., 24)**: Smoother results, fewer artifacts, slower inference
- **Lower overlap (e.g., 8)**: Faster inference, potential artifacts at patch boundaries
- **No overlap (0)**: Fastest inference, visible patch boundaries

### GPU Memory

- Increase `batch_size` if you have sufficient GPU memory
- Decrease `patch_size` if you encounter out-of-memory errors
- Monitor GPU usage during inference

### Processing Time

Typical processing times (varies by hardware and data size):
- Small volumes (128³): 1-5 minutes
- Medium volumes (256³): 5-15 minutes  
- Large volumes (512³): 15-60 minutes

## Troubleshooting

### Common Issues

1. **Checkpoint not found**: Verify the `checkpoint_path` in your configuration
2. **Patient not found**: Ensure the `patient_id` exists in your dataset
3. **Out of memory**: Reduce `batch_size` or `patch_size`
4. **CUDA errors**: Check GPU availability and PyTorch CUDA installation

### Debug Mode

For debugging, you can modify the configuration to use CPU instead of GPU:

```yaml
# In your config or via command line
device: cpu
```

## Advanced Configuration

### Custom Configuration Files

Create custom configuration files by extending the default:

```yaml
# custom_inference.yaml
defaults:
  - inference/default
  - _self_

inference:
  patch_overlap: 32
  batch_size: 150
  patch_aggregation_overlap_mode: "gaussian"
```

### Environment Variables

Override configuration values using environment variables:

```bash
export INFERENCE_PATCH_OVERLAP=16
export INFERENCE_BATCH_SIZE=100
python -m vascular_superenhancement.inference.inference path_config=all_patients inference.patient_id=Fejoba
```

## Examples

### Batch Processing Multiple Patients

```bash
# Process multiple patients with different time points
for patient in "Fejoba" "Smith" "Johnson"; do
    for time in 1 2 3 4; do
        python -m vascular_superenhancement.inference.inference \
            path_config=all_patients \
            inference.patient_id=$patient \
            inference.time_point=$time
    done
done
```

### Quality vs Speed Comparison

```bash
# High quality (slow)
python -m vascular_superenhancement.inference.inference \
    path_config=all_patients \
    inference.patient_id=Fejoba \
    inference.patch_overlap=32

# Medium quality (balanced)
python -m vascular_superenhancement.inference.inference \
    path_config=all_patients \
    inference.patient_id=Fejoba \
    inference.patch_overlap=16

# Fast (lower quality)
python -m vascular_superenhancement.inference.inference \
    path_config=all_patients \
    inference.patient_id=Fejoba \
    inference.patch_overlap=8
```

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify configuration parameters
3. Ensure all dependencies are properly installed
4. Check GPU memory and CUDA compatibility

## License

This module is part of the Vascular Superenhancement 4D Flow project. Please refer to the main project license for usage terms.
