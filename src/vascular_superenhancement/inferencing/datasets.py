"""Dataset utilities for inference.

This module provides functions for creating TorchIO subjects from patient data
specifically for inference purposes. It uses full FOV (field of view) data
that has not been resampled, ensuring predictions cover the entire original volume.
"""

from pathlib import Path
from typing import Optional
import torchio as tio
from torchio import ScalarImage, Subject

from vascular_superenhancement.data_management.patients import Patient


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

