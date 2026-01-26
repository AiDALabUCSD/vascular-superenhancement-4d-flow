"""Dataset utilities for inference.

This module provides functions for creating TorchIO subjects from patient data
specifically for inference purposes. It uses full FOV (field of view) data
that has not been resampled, ensuring predictions cover the entire original volume.

Supports both single-timepoint and multi-timepoint modes.
"""

from pathlib import Path
from typing import Optional, List
import torchio as tio
from torchio import ScalarImage, Subject

from vascular_superenhancement.data_management.patients import Patient


def make_multi_timepoint_subject_full_fov(
    patient: Patient,
    center_time_index: int,
    window_size: int = 5,
    transforms=None
) -> Subject:
    """
    Create a TorchIO Subject from a temporal window using full FOV data.

    This function loads data from multiple consecutive timepoints for temporal
    inference. Each timepoint's data is stored with a suffix (_t0, _t1, etc.).

    Args:
        patient: Patient object containing data paths
        center_time_index: The center timepoint index
        window_size: Number of timepoints in the window (must be odd)
        transforms: Optional TorchIO transforms to apply

    Returns:
        TorchIO Subject containing full FOV images from all timepoints in the window.
        Images are named with suffixes _t0, _t1, etc. where _t{window_size//2}
        is the center timepoint.

    Note:
        Timepoints wrap around using modulo arithmetic at boundaries.
    """
    assert window_size % 2 == 1, "window_size must be odd"
    half_window = window_size // 2
    num_timepoints = patient.num_timepoints

    # Calculate timepoint indices with wrapping
    time_indices = [
        (center_time_index + offset) % num_timepoints
        for offset in range(-half_window, half_window + 1)
    ]

    subject_dict = {}

    for i, t_idx in enumerate(time_indices):
        suffix = f'_t{i}'

        # Load from full FOV directories
        mag_path = patient.flow_mag_per_timepoint_full_fov_dir / f'4d_flow_mag_{patient.identifier}_frame_{t_idx:02d}.nii.gz'
        fvx_path = patient.flow_vx_per_timepoint_full_fov_dir / f'4d_flow_vx_{patient.identifier}_frame_{t_idx:02d}.nii.gz'
        fvy_path = patient.flow_vy_per_timepoint_full_fov_dir / f'4d_flow_vy_{patient.identifier}_frame_{t_idx:02d}.nii.gz'
        fvz_path = patient.flow_vz_per_timepoint_full_fov_dir / f'4d_flow_vz_{patient.identifier}_frame_{t_idx:02d}.nii.gz'

        subject_dict[f'mag{suffix}'] = ScalarImage(mag_path)
        subject_dict[f'flow_vx{suffix}'] = ScalarImage(fvx_path)
        subject_dict[f'flow_vy{suffix}'] = ScalarImage(fvy_path)
        subject_dict[f'flow_vz{suffix}'] = ScalarImage(fvz_path)

        # Store paths for reference
        subject_dict[f'mag{suffix}_path'] = str(mag_path)
        subject_dict[f'flow_vx{suffix}_path'] = str(fvx_path)
        subject_dict[f'flow_vy{suffix}_path'] = str(fvy_path)
        subject_dict[f'flow_vz{suffix}_path'] = str(fvz_path)

    # Store metadata
    subject_dict['patient_id'] = patient.identifier
    subject_dict['time_index'] = center_time_index
    subject_dict['time_indices'] = time_indices
    subject_dict['window_size'] = window_size

    subject = tio.Subject(**subject_dict)
    subject.name = f"{patient.identifier}_{center_time_index:02d}_w{window_size}_full_fov"

    if transforms:
        subject = transforms(subject)
    return subject


def make_subject_full_fov(patient: Patient, time_index: int, transforms=None) -> Subject:
    """
    Create a TorchIO Subject from one timepoint of 4D Flow data using full FOV (non-resampled) files.
    
    This function is specifically designed for inference where we need predictions for the
    entire original field of view. It loads data from the full FOV per-timepoint directories
    that preserve the original 4D flow volume dimensions and spacing.
    
    Args:
        patient: Patient object containing the data
        time_index: Timepoint index (0-based)
        transforms: Optional transforms to apply to the subject
        
    Returns:
        TorchIO Subject with mag, flow_vx, flow_vy, flow_vz images from full FOV directories
    """
    # Load all flow components for this timepoint from full FOV directories
    mag_path = patient.flow_mag_per_timepoint_full_fov_dir / f'4d_flow_mag_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    fvx_path = patient.flow_vx_per_timepoint_full_fov_dir / f'4d_flow_vx_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    fvy_path = patient.flow_vy_per_timepoint_full_fov_dir / f'4d_flow_vy_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    fvz_path = patient.flow_vz_per_timepoint_full_fov_dir / f'4d_flow_vz_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    
    subject = tio.Subject(
        mag=ScalarImage(mag_path),
        flow_vx=ScalarImage(fvx_path),
        flow_vy=ScalarImage(fvy_path),
        flow_vz=ScalarImage(fvz_path),
        mag_path=str(mag_path),
        flow_vx_path=str(fvx_path),
        flow_vy_path=str(fvy_path),
        flow_vz_path=str(fvz_path),
        patient_id=patient.identifier,
        time_index=time_index
    )
    
    subject.name = f"{patient.identifier}_{time_index:02d}_full_fov"
    
    if transforms:
        subject = transforms(subject)
    return subject

