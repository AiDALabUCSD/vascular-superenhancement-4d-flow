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


# =============================================================================
# Downsampled dual-task subject/dataset builders
# =============================================================================


def _offset_key(offset: int) -> str:
    """Convert a temporal offset integer to a subject key suffix.

    Examples:
        -2 -> 'mag_offset_n2'
        -1 -> 'mag_offset_n1'
         1 -> 'mag_offset_p1'
         2 -> 'mag_offset_p2'
    """
    if offset < 0:
        return f"mag_offset_n{abs(offset)}"
    else:
        return f"mag_offset_p{offset}"


def get_downsampled_mag_keys(temporal_mag_offsets: List[int]) -> List[str]:
    """Return the ordered list of magnitude subject keys for the given offsets.

    The keys are sorted by offset value with the center ('mag_center') inserted
    in order.  E.g. for offsets ``[-2, -1, 1, 2]`` the result is:
    ``['mag_offset_n2', 'mag_offset_n1', 'mag_center', 'mag_offset_p1', 'mag_offset_p2']``
    """
    sorted_offsets = sorted(temporal_mag_offsets)
    keys: List[str] = []
    center_inserted = False
    for off in sorted_offsets:
        if not center_inserted and off > 0:
            keys.append("mag_center")
            center_inserted = True
        keys.append(_offset_key(off))
    if not center_inserted:
        keys.append("mag_center")
    return keys


def make_downsampled_subject(
    patient: Patient,
    center_time_index: int,
    temporal_mag_offsets: List[int],
    downsampled_folder: str,
) -> Subject:
    """Create a TorchIO Subject from one centre timepoint of downsampled data.

    Loads:
      - Magnitude images for the centre timepoint *and* each temporal offset
      - Uncorrected velocity (vx, vy, vz) for the centre timepoint
      - 3D cine target for the centre timepoint
      - Cine mask (time-independent)
      - Ground-truth correction fields vx, vy, vz (per-timepoint raw diff)

    Args:
        patient: Patient object with data paths.
        center_time_index: The centre timepoint index.
        temporal_mag_offsets: List of offsets (e.g. ``[-2, -1, 1, 2]``).
        downsampled_folder: Subfolder name under ``patient.nifti_dir``
            (e.g. ``"downsampled_full_fov_128x128x64"``).

    Returns:
        A TorchIO Subject ready for transforms / DataLoader.
    """
    root = patient.nifti_dir / downsampled_folder
    num_tp = patient.num_timepoints
    pid = patient.identifier

    subject_dict = {}

    # --- Centre magnitude -------------------------------------------------
    mag_center_path = root / "4d_flow_mag" / f"4d_flow_mag_{pid}_frame_{center_time_index:02d}.nii.gz"
    subject_dict["mag_center"] = ScalarImage(mag_center_path)

    # --- Offset magnitudes ------------------------------------------------
    for offset in temporal_mag_offsets:
        t_idx = (center_time_index + offset) % num_tp
        mag_path = root / "4d_flow_mag" / f"4d_flow_mag_{pid}_frame_{t_idx:02d}.nii.gz"
        subject_dict[_offset_key(offset)] = ScalarImage(mag_path)

    # --- Uncorrected velocity (centre timepoint) --------------------------
    for comp in ("vx", "vy", "vz"):
        vel_path = root / f"4d_flow_{comp}" / f"4d_flow_{comp}_{pid}_frame_{center_time_index:02d}.nii.gz"
        subject_dict[f"uncorrected_{comp}"] = ScalarImage(vel_path)

    # --- Cine target (centre timepoint) -----------------------------------
    cine_path = root / "3d_cine" / f"3d_cine_{pid}_frame_{center_time_index:02d}.nii.gz"
    subject_dict["cine"] = ScalarImage(cine_path)

    # --- Cine mask (time-independent) -------------------------------------
    cine_mask_path = root / f"3d_cine_mask_{pid}.nii.gz"
    subject_dict["cine_mask"] = ScalarImage(cine_mask_path)

    # --- Ground-truth correction fields (per-timepoint raw diff) ----------
    for comp in ("vx", "vy", "vz"):
        gt_path = root / f"4d_flow_diff_{comp}" / f"4d_flow_diff_{comp}_{pid}_frame_{center_time_index:02d}.nii.gz"
        subject_dict[f"gt_correction_{comp}"] = ScalarImage(gt_path)

    # --- Correction air mask (time-independent, precomputed) --------------
    correction_mask_path = root / f"correction_air_mask_{pid}.nii.gz"
    if correction_mask_path.exists():
        subject_dict["correction_mask"] = ScalarImage(correction_mask_path)

    # --- Metadata ---------------------------------------------------------
    subject_dict["patient_id"] = pid
    subject_dict["time_index"] = center_time_index
    subject_dict["venc"] = float(patient.venc)

    subject = tio.Subject(**subject_dict)
    subject.name = f"{pid}_{center_time_index:02d}_ds"
    return subject


def make_downsampled_subject_inference(
    patient: Patient,
    center_time_index: int,
    temporal_mag_offsets: List[int],
    downsampled_folder: str,
) -> Subject:
    """Create a TorchIO Subject with inputs only (no ground-truth targets).

    Use this for visualization-only patients that lack cine / correction data.
    Loads magnitude images and uncorrected velocity -- skips cine, cine_mask,
    and ground-truth correction fields.

    Args:
        patient: Patient object with data paths.
        center_time_index: The centre timepoint index.
        temporal_mag_offsets: List of offsets (e.g. ``[-2, -1, 1, 2]``).
        downsampled_folder: Subfolder name under ``patient.nifti_dir``.

    Returns:
        A TorchIO Subject with model inputs and metadata only.
    """
    root = patient.nifti_dir / downsampled_folder
    num_tp = patient.num_timepoints
    pid = patient.identifier

    subject_dict = {}

    # --- Centre magnitude -------------------------------------------------
    mag_center_path = root / "4d_flow_mag" / f"4d_flow_mag_{pid}_frame_{center_time_index:02d}.nii.gz"
    subject_dict["mag_center"] = ScalarImage(mag_center_path)

    # --- Offset magnitudes ------------------------------------------------
    for offset in temporal_mag_offsets:
        t_idx = (center_time_index + offset) % num_tp
        mag_path = root / "4d_flow_mag" / f"4d_flow_mag_{pid}_frame_{t_idx:02d}.nii.gz"
        subject_dict[_offset_key(offset)] = ScalarImage(mag_path)

    # --- Uncorrected velocity (centre timepoint) --------------------------
    for comp in ("vx", "vy", "vz"):
        vel_path = root / f"4d_flow_{comp}" / f"4d_flow_{comp}_{pid}_frame_{center_time_index:02d}.nii.gz"
        subject_dict[f"uncorrected_{comp}"] = ScalarImage(vel_path)

    # --- Metadata ---------------------------------------------------------
    subject_dict["patient_id"] = pid
    subject_dict["time_index"] = center_time_index
    subject_dict["venc"] = float(patient.venc)

    subject = tio.Subject(**subject_dict)
    subject.name = f"{pid}_{center_time_index:02d}_ds_inf"
    return subject


def build_downsampled_dataset(
    cfg,
    split: str,
    transforms=None,
    patient_ids: Optional[List[str]] = None,
    time_index: Optional[int] = None,
    exclude_patient_ids: Optional[List[str]] = None,
) -> SubjectsDataset:
    """Build a TorchIO SubjectsDataset of downsampled subjects for dual-task training.

    Args:
        cfg: Hydra configuration object.
        split: Dataset split ('train', 'validation', 'test').
        transforms: Optional TorchIO transforms to apply.
        patient_ids: Optional explicit patient IDs (overrides CSV split).
        time_index: If provided, only create subjects for this single timepoint
            per patient.  If ``None``, creates subjects for all timepoints.
        exclude_patient_ids: Optional list of patient IDs to exclude from the
            dataset.  Useful for removing visualization-only patients that lack
            ground-truth targets from the validation set.

    Returns:
        ``tio.SubjectsDataset``
    """
    path_config = load_path_config(cfg.path_config.path_config_name)
    splits_path = Path(cfg.data.splits_path)
    debug = cfg.train.debug
    downsampled_folder = cfg.data.downsampled_folder
    temporal_mag_offsets = list(cfg.train.temporal_mag_offsets)

    if patient_ids is None:
        df = pd.read_csv(splits_path)
        patient_ids = df[df.split == split].patient_id.tolist()
        hydra_logger.info(f"Building downsampled dataset for split '{split}' with {len(patient_ids)} patients")
    else:
        hydra_logger.info(f"Building downsampled dataset from explicit patient list: {patient_ids}")

    if exclude_patient_ids:
        before = len(patient_ids)
        patient_ids = [pid for pid in patient_ids if pid not in set(exclude_patient_ids)]
        hydra_logger.info(f"Excluded {before - len(patient_ids)} patients (visualization-only): {exclude_patient_ids}")

    if time_index is not None:
        hydra_logger.info(f"Using single timepoint: {time_index}")

    subjects: List[Subject] = []
    for pid in patient_ids:
        try:
            patient = Patient(
                path_config=path_config,
                phonetic_id=pid,
                debug=debug,
            )
            timepoints = [time_index] if time_index is not None else range(patient.num_timepoints)
            for t in timepoints:
                try:
                    subj = make_downsampled_subject(
                        patient,
                        center_time_index=t,
                        temporal_mag_offsets=temporal_mag_offsets,
                        downsampled_folder=downsampled_folder,
                    )
                    subjects.append(subj)
                except Exception as e:
                    hydra_logger.error(
                        f"Error creating downsampled subject for {pid} at timepoint {t}: {e}"
                    )
                    continue
            hydra_logger.debug(f"Added {len(timepoints)} downsampled subjects for {pid}. Total: {len(subjects)}")
        except ValueError as e:
            hydra_logger.warning(f"Skipping patient {pid}: {e}")
            continue
        except Exception as e:
            hydra_logger.error(f"Error creating patient object for {pid}: {e}")
            continue

    hydra_logger.info(f"Built downsampled dataset with {len(subjects)} subjects")
    if not subjects:
        raise ValueError("No valid downsampled subjects found")

    return SubjectsDataset(subjects, transform=transforms)


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
