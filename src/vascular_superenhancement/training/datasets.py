from pathlib import Path
from typing import List, Optional
import time
import logging
import pandas as pd
import torchio as tio
# import nibabel as nib
from torchio import ScalarImage, Subject, SubjectsDataset
from torch.utils.data.sampler import Sampler
import random

from vascular_superenhancement.data_management.patients import Patient
# from vascular_superenhancement.training.transforms import build_transforms
from vascular_superenhancement.utils.path_config import load_path_config

hydra_logger = logging.getLogger(__name__)


def make_subject(patient: Patient, time_index: int, transforms=None, peak_systolic_only: bool = False) -> Subject:
    """
    Create a TorchIO Subject from one timepoint of 4D Flow data and the target cine volume.
    """
    # Load all flow components for this timepoint
    if peak_systolic_only:
        # pick a random number between 3 and 5 inclusive
        random_frame = random.randint(3, 5)
        mag_path = patient.flow_mag_per_timepoint_dir / f'4d_flow_mag_{patient.identifier}_frame_{time_index:02d}.nii.gz'
        fvx_path = patient.flow_vx_per_timepoint_dir / f'4d_flow_vx_{patient.identifier}_frame_{random_frame:02d}.nii.gz'
        fvy_path = patient.flow_vy_per_timepoint_dir / f'4d_flow_vy_{patient.identifier}_frame_{random_frame:02d}.nii.gz'
        fvz_path = patient.flow_vz_per_timepoint_dir / f'4d_flow_vz_{patient.identifier}_frame_{random_frame:02d}.nii.gz'
    else:
        mag_path = patient.flow_mag_per_timepoint_dir / f'4d_flow_mag_{patient.identifier}_frame_{time_index:02d}.nii.gz'
        fvx_path = patient.flow_vx_per_timepoint_dir / f'4d_flow_vx_{patient.identifier}_frame_{time_index:02d}.nii.gz'
        fvy_path = patient.flow_vy_per_timepoint_dir / f'4d_flow_vy_{patient.identifier}_frame_{time_index:02d}.nii.gz'
        fvz_path = patient.flow_vz_per_timepoint_dir / f'4d_flow_vz_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    
    # Load cine target for this timepoint
    cine_path = patient.cine_per_timepoint_dir / f'3d_cine_{patient.identifier}_frame_{time_index:02d}.nii.gz'

    subject = tio.Subject(
        mag=ScalarImage(mag_path),
        flow_vx=ScalarImage(fvx_path),
        flow_vy=ScalarImage(fvy_path),
        flow_vz=ScalarImage(fvz_path),
        cine=ScalarImage(cine_path),
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

class TimepointCyclingSampler(Sampler):
    """
    Custom sampler that cycles through timepoints epoch by epoch.
    Each epoch uses only subjects from one specific timepoint.
    """
    def __init__(self, dataset, num_timepoints=20, shuffle_within_timepoint=True):
        self.dataset = dataset
        self.num_timepoints = num_timepoints
        self.shuffle_within_timepoint = shuffle_within_timepoint
        
        # Group dataset indices by timepoint
        self.timepoint_indices = {}
        for timepoint in range(num_timepoints):
            self.timepoint_indices[timepoint] = []
        
        #time each loop
        start_time = time.time()
        hydra_logger.debug(f"Beginning TimepointCyclingSampler initialization with {len(dataset)} subjects")
        for idx, subject in enumerate(dataset.dry_iter()):
            if idx % 100 == 0:
                hydra_logger.debug(f"TimepointCyclingSampler initialization still in progress {time.time() - start_time:.2f} seconds: {idx}/{len(dataset)} subjects processed")
            timepoint = subject.time_index
            if timepoint in self.timepoint_indices:
                self.timepoint_indices[timepoint].append(idx)
        end_time = time.time()
        hydra_logger.debug(f"TimepointCyclingSampler initialization completed in {end_time - start_time} seconds")

        self.current_epoch = 0
        hydra_logger.info(f"TimepointCyclingSampler initialized with {len(self.timepoint_indices)} timepoints")
        for tp, indices in self.timepoint_indices.items():
            hydra_logger.info(f"  Timepoint {tp}: {len(indices)} subjects")
        
    def __iter__(self):
        # Get current timepoint for this epoch
        current_timepoint = self.current_epoch % self.num_timepoints
        indices = self.timepoint_indices[current_timepoint].copy()
        
        if self.shuffle_within_timepoint:
            random.shuffle(indices)
            
        hydra_logger.debug(f"Epoch {self.current_epoch}: Using timepoint {current_timepoint} with {len(indices)} subjects")
        return iter(indices)
    
    def __len__(self):
        # Return length of current timepoint's data
        current_timepoint = self.current_epoch % self.num_timepoints
        return len(self.timepoint_indices[current_timepoint])
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch

def build_subjects_dataset(
    split: str,
    split_csv_path: Path,
    path_config: str,
    transforms=None,
    debug: bool = False,
    time_index: Optional[int] = None,
    include_all_timepoints: bool = False,
    peak_systolic_only: bool = False
) -> SubjectsDataset:
    """
    Build a TorchIO SubjectsDataset for a given split (train/val/test).
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        split_csv_path: Path to the CSV file containing split information
        path_config: Name of the path configuration to use
        transforms: Optional transforms to apply to subjects
        debug: Whether to enable debug logging for patient objects
        time_index: Optional timepoint index to use, if None, all timepoints are used
        include_all_timepoints: Whether to include all timepoints for each patient, if True, time_index is ignored
    """
    path_config = load_path_config(path_config)
    
    df = pd.read_csv(split_csv_path)
    patient_ids = df[df.split == split].patient_id.tolist()
    
    subjects: List[Subject] = []
    hydra_logger.debug(f"Starting with {len(subjects)} subjects")
    for pid in patient_ids:
        try:
            patient = Patient(
                path_config=path_config,
                phonetic_id=pid,
                debug=debug  # Use the debug parameter
            )
            if time_index is not None:
                try:
                    subjects.append(make_subject(patient, time_index, peak_systolic_only=peak_systolic_only))
                except Exception as e:
                    patient._logger.error(f"Error creating subject for patient {pid} at timepoint {time_index}: {e}")
                    continue
                patient._logger.debug(f"Added timepoint {time_index} for patient {pid}. Total subjects: {len(subjects)}")
                hydra_logger.debug(f"Added timepoint {time_index} for patient {pid}. Total subjects: {len(subjects)}")
            elif include_all_timepoints:
                for t in range(patient.num_timepoints):
                    try:
                        subjects.append(make_subject(patient, t, peak_systolic_only=peak_systolic_only))
                    except Exception as e:
                        patient._logger.error(f"Error creating subject for patient {pid} at timepoint {t}: {e}")
                        continue
                patient._logger.debug(f"Added {patient.num_timepoints} subjects for patient {pid}")
                hydra_logger.debug(f"Added {patient.num_timepoints} subjects for patient {pid}. Total subjects: {len(subjects)}")
            else:
                # Legacy mode - include all timepoints for each patient (same as include_all_timepoints=True)
                for t in range(patient.num_timepoints):
                    try:
                        subjects.append(make_subject(patient, t, peak_systolic_only=peak_systolic_only))
                    except Exception as e:
                        patient._logger.error(f"Error creating subject for patient {pid} at timepoint {t}: {e}")
                        continue
                patient._logger.debug(f"Added {patient.num_timepoints} subjects for patient {pid}")
                hydra_logger.debug(f"Added {patient.num_timepoints} subjects for patient {pid}. Total subjects: {len(subjects)}")
        except ValueError as e:
            patient._logger.warning(f"Warning: Not adding patient {pid} as a subject to dataset due to error: {e}")
            hydra_logger.warning(f"Warning: Not adding patient {pid} as a subject to dataset due to error: {e}")
            continue
        except Exception as e:
            patient._logger.error(f"Error creating subject in dataset for patient {pid}: {e}")
            hydra_logger.error(f"Error creating subject in dataset for patient {pid}: {e}")
            continue
    patient._logger.debug(f"Finished with {len(subjects)} subjects")
    hydra_logger.debug(f"Finished with {len(subjects)} subjects")
    
    if not subjects:
        raise ValueError("No valid subjects found")

    return SubjectsDataset(subjects, transform=transforms)

# Example usage from training script:
# transforms = build_transforms(cfg)
# dataset = build_subjects_dataset('train', Path(cfg.splits_path), cfg.path_config, transforms=transforms)
