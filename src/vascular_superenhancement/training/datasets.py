from pathlib import Path
from typing import List
import pandas as pd
import torchio as tio
import nibabel as nib
from torchio import ScalarImage, Subject, SubjectsDataset

from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.training.transforms import build_transforms


def make_subject(patient: Patient, time_index: int, transforms=None) -> Subject:
    """
    Create a TorchIO Subject from one timepoint of 4D Flow data and the target cine volume.
    """
    # Load all flow components for this timepoint
    mag_path = patient.flow_mag_per_timepoint_dir / f'4d_flow_mag_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    fvx_path = patient.flow_vx_per_timepoint_dir / f'4d_flow_vx_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    fvy_path = patient.flow_vy_per_timepoint_dir / f'4d_flow_vy_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    fvz_path = patient.flow_vz_per_timepoint_dir / f'4d_flow_vz_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    
    # Load cine target for this timepoint
    cine_path = patient.cine_per_timepoint_dir / f'3d_cine_{patient.identifier}_frame_{time_index:02d}.nii.gz'

    subject = tio.Subject(
        mag=tio.ScalarImage(mag_path),
        fvx=tio.ScalarImage(fvx_path),
        fvy=tio.ScalarImage(fvy_path),
        fvz=tio.ScalarImage(fvz_path),
        cine=tio.ScalarImage(cine_path),
        mag_path=str(mag_path),
        fvx_path=str(fvx_path),
        fvy_path=str(fvy_path),
        fvz_path=str(fvz_path),
        cine_path=str(cine_path),
        patient_id=patient.identifier,
        time_index=time_index
    )
    
    subject.name = f"{patient.identifier}_{time_index:02d}"

    if transforms:
        subject = transforms(subject)
    return subject

def build_subjects_dataset(
    split: str,
    split_csv_path: Path,
    path_config: dict,
    transforms=None
) -> SubjectsDataset:
    """
    Build a TorchIO SubjectsDataset for a given split (train/val/test).
    """
    df = pd.read_csv(split_csv_path)
    patient_ids = df[df.split == split].patient_id.tolist()
    
    subjects: List[Subject] = []
    for pid in patient_ids:
        patient = Patient(pid, path_config)
        for t in range(patient.num_timepoints):
            subjects.append(make_subject(patient, t, transforms))

    return SubjectsDataset(subjects)


# Example usage from training script:
# transforms = build_transforms(cfg)
# dataset = build_subjects_dataset('train', Path(cfg.splits_path), cfg.path_config, transforms=transforms)
