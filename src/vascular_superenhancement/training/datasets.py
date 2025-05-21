from pathlib import Path
from typing import List
import pandas as pd
import torchio as tio
import nibabel as nib
from torchio import ScalarImage, Subject, SubjectsDataset

from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.training.transforms import build_transforms
from vascular_superenhancement.utils.path_config import load_path_config


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
        flow_vx=tio.ScalarImage(fvx_path),
        flow_vy=tio.ScalarImage(fvy_path),
        flow_vz=tio.ScalarImage(fvz_path),
        cine=tio.ScalarImage(cine_path),
        mag_path=str(mag_path),
        flow_vx_path=str(fvx_path),
        flow_vy_path=str(fvy_path),
        flow_vz_path=str(fvz_path),
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
    path_config: str,
    transforms=None
) -> SubjectsDataset:
    """
    Build a TorchIO SubjectsDataset for a given split (train/val/test).
    """
    path_config = load_path_config(path_config)
    
    df = pd.read_csv(split_csv_path)
    patient_ids = df[df.split == split].patient_id.tolist()
    
    subjects: List[Subject] = []
    for pid in patient_ids:
        try:
            patient = Patient(
                path_config=path_config,
                phonetic_id=pid,
                debug=False # optional, but preferred if available
            )
            for t in range(patient.num_timepoints):
                try:
                    subjects.append(make_subject(patient, t))
                except Exception as e:
                    patient._logger.error(f"Error creating subject for patient {pid} at timepoint {t}: {e}")
                    continue
            patient._logger.info(f"Added {patient.num_timepoints} subjects for patient {pid}")
        except ValueError as e:
            patient._logger.warning(f"Warning: Not adding patient {pid} as a subject to dataset due to error: {e}")
            continue
        except Exception as e:
            patient._logger.error(f"Error creating subject in dataset for patient {pid}: {e}")
            continue
    
    if not subjects:
        raise ValueError("No valid subjects found")

    return SubjectsDataset(subjects, transform=transforms)

# Example usage from training script:
# transforms = build_transforms(cfg)
# dataset = build_subjects_dataset('train', Path(cfg.splits_path), cfg.path_config, transforms=transforms)
