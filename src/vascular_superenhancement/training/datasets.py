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


def make_multi_timepoint_subject(
    patient: Patient,
    center_time_index: int,
    window_size: int = 5,
    transforms=None,
) -> Subject:
    """
    Create a TorchIO Subject from a temporal window of timepoints for training.

    This function loads data from multiple consecutive timepoints to enable
    temporal context in the model. The center timepoint is the primary target,
    with surrounding timepoints providing temporal context.
    
    Uses precomputed speed volumes (sqrt(vx^2 + vy^2 + vz^2)) for efficiency.

    Args:
        patient: Patient object containing data paths
        center_time_index: The center timepoint index (t=0 in the window)
        window_size: Number of timepoints to include (must be odd, default 5)
        transforms: Optional TorchIO transforms to apply

    Returns:
        TorchIO Subject containing images from all timepoints in the window.
        Images are named with suffixes _t0, _t1, _t2, etc. where _t{window_size//2}
        is the center timepoint.

    Example:
        For window_size=5 and center_time_index=10:
        - _t0 contains data from timepoint 8  (center - 2)
        - _t1 contains data from timepoint 9  (center - 1)
        - _t2 contains data from timepoint 10 (center)
        - _t3 contains data from timepoint 11 (center + 1)
        - _t4 contains data from timepoint 12 (center + 2)

    Note:
        Timepoints wrap around using modulo arithmetic, so for a patient with
        20 timepoints and center_time_index=1, the window would be [19, 0, 1, 2, 3].
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
        suffix = f'_t{i}'  # _t0, _t1, _t2, _t3, _t4

        # Load magnitude and precomputed speed for this timepoint
        mag_path = patient.flow_mag_per_timepoint_dir / f'4d_flow_mag_{patient.identifier}_frame_{t_idx:02d}.nii.gz'
        speed_path = patient.flow_speed_per_timepoint_dir / f'4d_flow_speed_{patient.identifier}_frame_{t_idx:02d}.nii.gz'
        cine_path = patient.cine_per_timepoint_dir / f'3d_cine_{patient.identifier}_frame_{t_idx:02d}.nii.gz'

        subject_dict[f'mag{suffix}'] = ScalarImage(mag_path)
        subject_dict[f'speed{suffix}'] = ScalarImage(speed_path)
        subject_dict[f'cine{suffix}'] = ScalarImage(cine_path)

        # Store paths for reference
        subject_dict[f'mag{suffix}_path'] = str(mag_path)
        subject_dict[f'speed{suffix}_path'] = str(speed_path)
        subject_dict[f'cine{suffix}_path'] = str(cine_path)

    # Store metadata
    subject_dict['patient_id'] = patient.identifier
    subject_dict['time_index'] = center_time_index
    subject_dict['time_indices'] = time_indices
    subject_dict['window_size'] = window_size

    subject = tio.Subject(**subject_dict)
    subject.name = f"{patient.identifier}_{center_time_index:02d}_w{window_size}"

    if transforms:
        subject = transforms(subject)
    return subject


def make_subject(patient: Patient, time_index: int, transforms=None) -> Subject:
    """
    Create a TorchIO Subject from one timepoint of 4D Flow data and the target cine volume.
    
    Uses precomputed speed volumes (sqrt(vx^2 + vy^2 + vz^2)) for efficiency.
    """
    # Load magnitude, precomputed speed, and cine for this timepoint
    mag_path = patient.flow_mag_per_timepoint_dir / f'4d_flow_mag_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    speed_path = patient.flow_speed_per_timepoint_dir / f'4d_flow_speed_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    cine_path = patient.cine_per_timepoint_dir / f'3d_cine_{patient.identifier}_frame_{time_index:02d}.nii.gz'
    
    subject = tio.Subject(
        mag=ScalarImage(mag_path),
        speed=ScalarImage(speed_path),
        cine=ScalarImage(cine_path),
        mag_path=str(mag_path),
        speed_path=str(speed_path),
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
    cfg,
    split: str,
    transforms=None,
    time_index: Optional[int] = None,
    include_all_timepoints: bool = False,
    patient_ids: Optional[List[str]] = None,
) -> SubjectsDataset:
    """
    Build a TorchIO SubjectsDataset for a given split (train/val/test) or explicit patient list.
    
    Uses precomputed speed volumes for efficient training.
    
    Args:
        cfg: Hydra configuration object
        split: Dataset split ('train', 'validation', 'test')
        transforms: Optional transforms to apply to subjects
        time_index: Optional timepoint index to use, if None, all timepoints are used
        include_all_timepoints: Whether to include all timepoints for each patient
        patient_ids: Optional explicit list of patient IDs, bypassing the CSV split
    """
    path_config = load_path_config(cfg.path_config.path_config_name)
    splits_path = Path(cfg.data.splits_path)
    debug = cfg.train.debug
    
    if patient_ids is None:
        df = pd.read_csv(splits_path)
        patient_ids = df[df.split == split].patient_id.tolist()
        hydra_logger.info(f"Split CSV path: {splits_path}")
        hydra_logger.info(f"Building subjects dataset for split {split} with {patient_ids} patients")
    else:
        hydra_logger.info(f"Building subjects dataset from explicit patient list: {patient_ids}")
    
    subjects: List[Subject] = []
    hydra_logger.debug(f"Starting with {len(subjects)} subjects")
    for pid in patient_ids:
        try:
            patient = Patient(
                path_config=path_config,
                phonetic_id=pid,
                debug=debug
            )
            if time_index is not None:
                try:
                    subjects.append(make_subject(patient, time_index))
                except Exception as e:
                    patient._logger.error(f"Error creating subject for patient {pid} at timepoint {time_index}: {e}")
                    continue
                patient._logger.debug(f"Added timepoint {time_index} for patient {pid}. Total subjects: {len(subjects)}")
                hydra_logger.debug(f"Added timepoint {time_index} for patient {pid}. Total subjects: {len(subjects)}")
            elif include_all_timepoints:
                for t in range(patient.num_timepoints):
                    try:
                        subjects.append(make_subject(patient, t))
                    except Exception as e:
                        patient._logger.error(f"Error creating subject for patient {pid} at timepoint {t}: {e}")
                        continue
                patient._logger.debug(f"Added {patient.num_timepoints} subjects for patient {pid}")
                hydra_logger.debug(f"Added {patient.num_timepoints} subjects for patient {pid}. Total subjects: {len(subjects)}")
            else:
                # Default: include all timepoints for each patient
                for t in range(patient.num_timepoints):
                    try:
                        subjects.append(make_subject(patient, t))
                    except Exception as e:
                        patient._logger.error(f"Error creating subject for patient {pid} at timepoint {t}: {e}")
                        continue
                patient._logger.debug(f"Added {patient.num_timepoints} subjects for patient {pid}")
                hydra_logger.debug(f"Added {patient.num_timepoints} subjects for patient {pid}. Total subjects: {len(subjects)}")
        except ValueError as e:
            hydra_logger.warning(f"Warning: Not adding patient {pid} as a subject to dataset due to error: {e}")
            continue
        except Exception as e:
            hydra_logger.error(f"Error creating subject in dataset for patient {pid}: {e}")
            continue
    hydra_logger.debug(f"Finished with {len(subjects)} subjects")
    
    if not subjects:
        raise ValueError("No valid subjects found")

    return SubjectsDataset(subjects, transform=transforms)

def build_multi_timepoint_subjects_dataset(
    cfg,
    split: str,
    transforms=None,
    time_index: Optional[int] = None,
    include_all_timepoints: bool = False,
    patient_ids: Optional[List[str]] = None,
) -> SubjectsDataset:
    """
    Build a TorchIO SubjectsDataset with multi-timepoint subjects.

    Each subject contains data from a window of consecutive timepoints
    (e.g., 5 timepoints for temporal context).
    
    Uses precomputed speed volumes for efficient training.

    Args:
        cfg: Hydra configuration object
        split: Dataset split ('train', 'validation', 'test')
        transforms: Optional transforms to apply to subjects
        time_index: Optional center timepoint index to use, if None, all center timepoints are used
        include_all_timepoints: Whether to include all possible center timepoints for each patient
        patient_ids: Optional explicit list of patient IDs, bypassing CSV split

    Returns:
        SubjectsDataset containing multi-timepoint subjects
    """
    path_config = load_path_config(cfg.path_config.path_config_name)
    splits_path = Path(cfg.data.splits_path)
    debug = cfg.train.debug
    window_size = cfg.train.temporal_window_size

    if patient_ids is None:
        df = pd.read_csv(splits_path)
        patient_ids = df[df.split == split].patient_id.tolist()
        hydra_logger.info(f"Split CSV path: {splits_path}")
        hydra_logger.info(f"Building multi-timepoint subjects dataset for split {split} with {len(patient_ids)} patients")
    else:
        hydra_logger.info(f"Building multi-timepoint subjects dataset from explicit patient list: {patient_ids}")

    hydra_logger.info(f"Using temporal window size: {window_size}")

    subjects: List[Subject] = []
    for pid in patient_ids:
        try:
            patient = Patient(
                path_config=path_config,
                phonetic_id=pid,
                debug=debug
            )

            if time_index is not None:
                # Single center timepoint
                try:
                    subjects.append(
                        make_multi_timepoint_subject(
                            patient,
                            center_time_index=time_index,
                            window_size=window_size,
                        )
                    )
                except Exception as e:
                    hydra_logger.error(f"Error creating multi-timepoint subject for patient {pid} at center timepoint {time_index}: {e}")
                    continue
                hydra_logger.debug(f"Added center timepoint {time_index} for patient {pid}. Total subjects: {len(subjects)}")

            elif include_all_timepoints:
                # All possible center timepoints
                for t in range(patient.num_timepoints):
                    try:
                        subjects.append(
                            make_multi_timepoint_subject(
                                patient,
                                center_time_index=t,
                                window_size=window_size,
                            )
                        )
                    except Exception as e:
                        hydra_logger.error(f"Error creating multi-timepoint subject for patient {pid} at center timepoint {t}: {e}")
                        continue
                hydra_logger.debug(f"Added {patient.num_timepoints} multi-timepoint subjects for patient {pid}. Total subjects: {len(subjects)}")

            else:
                # Default: include all timepoints
                for t in range(patient.num_timepoints):
                    try:
                        subjects.append(
                            make_multi_timepoint_subject(
                                patient,
                                center_time_index=t,
                                window_size=window_size,
                            )
                        )
                    except Exception as e:
                        hydra_logger.error(f"Error creating multi-timepoint subject for patient {pid} at center timepoint {t}: {e}")
                        continue
                hydra_logger.debug(f"Added {patient.num_timepoints} multi-timepoint subjects for patient {pid}. Total subjects: {len(subjects)}")

        except ValueError as e:
            hydra_logger.warning(f"Warning: Not adding patient {pid} to dataset due to error: {e}")
            continue
        except Exception as e:
            hydra_logger.error(f"Error creating patient object for {pid}: {e}")
            continue

    hydra_logger.info(f"Built multi-timepoint dataset with {len(subjects)} subjects")

    if not subjects:
        raise ValueError("No valid subjects found")

    return SubjectsDataset(subjects, transform=transforms)
