from pathlib import Path
from typing import List, Optional, Tuple
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
    peak_systolic_only: bool = False,
    inference_mode: bool = False
) -> Subject:
    """
    Create a TorchIO Subject from a temporal window of timepoints.

    This function loads data from multiple consecutive timepoints to enable
    temporal context in the model. The center timepoint is the primary target,
    with surrounding timepoints providing temporal context.

    Args:
        patient: Patient object containing data paths
        center_time_index: The center timepoint index (t=0 in the window)
        window_size: Number of timepoints to include (must be odd, default 5)
        transforms: Optional TorchIO transforms to apply
        peak_systolic_only: Whether to use peak systolic velocity frames for flow
        inference_mode: Whether to skip loading cine targets (for inference only)

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

    # For peak systolic, pick a random frame between 3-5 for velocity
    if peak_systolic_only:
        random_flow_frame = random.randint(3, 5)

    subject_dict = {}

    for i, t_idx in enumerate(time_indices):
        suffix = f'_t{i}'  # _t0, _t1, _t2, _t3, _t4

        # Magnitude always uses the actual timepoint
        mag_path = patient.flow_mag_per_timepoint_dir / f'4d_flow_mag_{patient.identifier}_frame_{t_idx:02d}.nii.gz'

        # Velocity: use peak systolic frame if requested, otherwise use actual timepoint
        if peak_systolic_only:
            fvx_path = patient.flow_vx_per_timepoint_dir / f'4d_flow_vx_{patient.identifier}_frame_{random_flow_frame:02d}.nii.gz'
            fvy_path = patient.flow_vy_per_timepoint_dir / f'4d_flow_vy_{patient.identifier}_frame_{random_flow_frame:02d}.nii.gz'
            fvz_path = patient.flow_vz_per_timepoint_dir / f'4d_flow_vz_{patient.identifier}_frame_{random_flow_frame:02d}.nii.gz'
        else:
            fvx_path = patient.flow_vx_per_timepoint_dir / f'4d_flow_vx_{patient.identifier}_frame_{t_idx:02d}.nii.gz'
            fvy_path = patient.flow_vy_per_timepoint_dir / f'4d_flow_vy_{patient.identifier}_frame_{t_idx:02d}.nii.gz'
            fvz_path = patient.flow_vz_per_timepoint_dir / f'4d_flow_vz_{patient.identifier}_frame_{t_idx:02d}.nii.gz'

        subject_dict[f'mag{suffix}'] = ScalarImage(mag_path)
        subject_dict[f'flow_vx{suffix}'] = ScalarImage(fvx_path)
        subject_dict[f'flow_vy{suffix}'] = ScalarImage(fvy_path)
        subject_dict[f'flow_vz{suffix}'] = ScalarImage(fvz_path)

        # Store paths for reference
        subject_dict[f'mag{suffix}_path'] = str(mag_path)
        subject_dict[f'flow_vx{suffix}_path'] = str(fvx_path)
        subject_dict[f'flow_vy{suffix}_path'] = str(fvy_path)
        subject_dict[f'flow_vz{suffix}_path'] = str(fvz_path)

        if not inference_mode:
            cine_path = patient.cine_per_timepoint_dir / f'3d_cine_{patient.identifier}_frame_{t_idx:02d}.nii.gz'
            subject_dict[f'cine{suffix}'] = ScalarImage(cine_path)
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


def get_multi_timepoint_image_keys(window_size: int = 5) -> Tuple[List[str], List[str], List[str]]:
    """
    Get the image key names for multi-timepoint subjects.

    Args:
        window_size: Number of timepoints in the window

    Returns:
        Tuple of (mag_keys, cine_keys, flow_keys) where each is a list of strings
    """
    mag_keys = [f'mag_t{i}' for i in range(window_size)]
    cine_keys = [f'cine_t{i}' for i in range(window_size)]
    flow_keys = []
    for i in range(window_size):
        flow_keys.extend([f'flow_vx_t{i}', f'flow_vy_t{i}', f'flow_vz_t{i}'])
    return mag_keys, cine_keys, flow_keys


def make_subject(patient: Patient, time_index: int, transforms=None, peak_systolic_only: bool = False, inference_mode: bool = False) -> Subject:
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
    
    if inference_mode:
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
    else:
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
    split: Optional[str],
    split_csv_path: Optional[Path],
    path_config: str,
    transforms=None,
    debug: bool = False,
    time_index: Optional[int] = None,
    include_all_timepoints: bool = False,
    peak_systolic_only: bool = False,
    patient_ids: Optional[List[str]] = None,
    inference_mode: bool = False
) -> SubjectsDataset:
    """
    Build a TorchIO SubjectsDataset for a given split (train/val/test) or explicit patient list.
    
    Args:
        split: Dataset split ('train', 'validation', 'test'). Optional when patient_ids is provided
        split_csv_path: Path to the CSV file containing split information. Optional when patient_ids is provided
        path_config: Name of the path configuration to use
        transforms: Optional transforms to apply to subjects
        debug: Whether to enable debug logging for patient objects
        time_index: Optional timepoint index to use, if None, all timepoints are used
        include_all_timepoints: Whether to include all timepoints for each patient, if True, time_index is ignored
        peak_systolic_only: Whether to use peak systolic only
        patient_ids: Optional explicit list of patient IDs to build the dataset from, bypassing the CSV split
        inference_mode: Whether to skip cine targets when creating subjects (inference-only)
    """
    path_config = load_path_config(path_config)
    
    if patient_ids is None:
        if split is None or split_csv_path is None:
            raise ValueError("Either provide split and split_csv_path or an explicit list of patient_ids")
        df = pd.read_csv(split_csv_path)
        patient_ids = df[df.split == split].patient_id.tolist()
        hydra_logger.info(f"Split CSV path: {split_csv_path}")
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
                debug=debug  # Use the debug parameter
            )
            if time_index is not None:
                try:
                    subjects.append(
                        make_subject(
                            patient,
                            time_index,
                            peak_systolic_only=peak_systolic_only,
                            inference_mode=inference_mode
                        )
                    )
                except Exception as e:
                    patient._logger.error(f"Error creating subject for patient {pid} at timepoint {time_index}: {e}")
                    continue
                patient._logger.debug(f"Added timepoint {time_index} for patient {pid}. Total subjects: {len(subjects)}")
                hydra_logger.debug(f"Added timepoint {time_index} for patient {pid}. Total subjects: {len(subjects)}")
            elif include_all_timepoints:
                for t in range(patient.num_timepoints):
                    try:
                        subjects.append(
                            make_subject(
                                patient,
                                t,
                                peak_systolic_only=peak_systolic_only,
                                inference_mode=inference_mode
                            )
                        )
                    except Exception as e:
                        patient._logger.error(f"Error creating subject for patient {pid} at timepoint {t}: {e}")
                        continue
                patient._logger.debug(f"Added {patient.num_timepoints} subjects for patient {pid}")
                hydra_logger.debug(f"Added {patient.num_timepoints} subjects for patient {pid}. Total subjects: {len(subjects)}")
            else:
                # Legacy mode - include all timepoints for each patient (same as include_all_timepoints=True)
                for t in range(patient.num_timepoints):
                    try:
                        subjects.append(
                            make_subject(
                                patient,
                                t,
                                peak_systolic_only=peak_systolic_only,
                                inference_mode=inference_mode
                            )
                        )
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
    hydra_logger.debug(f"Finished with {len(subjects)} subjects")
    
    if not subjects:
        raise ValueError("No valid subjects found")

    return SubjectsDataset(subjects, transform=transforms)

def build_multi_timepoint_subjects_dataset(
    split: Optional[str],
    split_csv_path: Optional[Path],
    path_config: str,
    window_size: int = 5,
    transforms=None,
    debug: bool = False,
    time_index: Optional[int] = None,
    include_all_timepoints: bool = False,
    peak_systolic_only: bool = False,
    patient_ids: Optional[List[str]] = None,
    inference_mode: bool = False
) -> SubjectsDataset:
    """
    Build a TorchIO SubjectsDataset with multi-timepoint subjects.

    Each subject contains data from a window of consecutive timepoints
    (e.g., 5 timepoints for temporal context).

    Args:
        split: Dataset split ('train', 'validation', 'test'). Optional when patient_ids is provided
        split_csv_path: Path to the CSV file containing split information
        path_config: Name of the path configuration to use
        window_size: Number of timepoints per subject (must be odd, default 5)
        transforms: Optional transforms to apply to subjects
        debug: Whether to enable debug logging for patient objects
        time_index: Optional center timepoint index to use, if None, all center timepoints are used
        include_all_timepoints: Whether to include all possible center timepoints for each patient
        peak_systolic_only: Whether to use peak systolic velocity frames
        patient_ids: Optional explicit list of patient IDs, bypassing CSV split
        inference_mode: Whether to skip cine targets (for inference only)

    Returns:
        SubjectsDataset containing multi-timepoint subjects
    """
    path_config_obj = load_path_config(path_config)

    if patient_ids is None:
        if split is None or split_csv_path is None:
            raise ValueError("Either provide split and split_csv_path or an explicit list of patient_ids")
        df = pd.read_csv(split_csv_path)
        patient_ids = df[df.split == split].patient_id.tolist()
        hydra_logger.info(f"Split CSV path: {split_csv_path}")
        hydra_logger.info(f"Building multi-timepoint subjects dataset for split {split} with {len(patient_ids)} patients")
    else:
        hydra_logger.info(f"Building multi-timepoint subjects dataset from explicit patient list: {patient_ids}")

    hydra_logger.info(f"Using temporal window size: {window_size}")

    subjects: List[Subject] = []
    for pid in patient_ids:
        try:
            patient = Patient(
                path_config=path_config_obj,
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
                            peak_systolic_only=peak_systolic_only,
                            inference_mode=inference_mode
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
                                peak_systolic_only=peak_systolic_only,
                                inference_mode=inference_mode
                            )
                        )
                    except Exception as e:
                        hydra_logger.error(f"Error creating multi-timepoint subject for patient {pid} at center timepoint {t}: {e}")
                        continue
                hydra_logger.debug(f"Added {patient.num_timepoints} multi-timepoint subjects for patient {pid}. Total subjects: {len(subjects)}")

            else:
                # Default: include all timepoints (same as include_all_timepoints=True)
                for t in range(patient.num_timepoints):
                    try:
                        subjects.append(
                            make_multi_timepoint_subject(
                                patient,
                                center_time_index=t,
                                window_size=window_size,
                                peak_systolic_only=peak_systolic_only,
                                inference_mode=inference_mode
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


# Example usage from training script:
# transforms = build_transforms(cfg)
# dataset = build_subjects_dataset('train', Path(cfg.splits_path), cfg.path_config, transforms=transforms)
