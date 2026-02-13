"""End-to-end inference pipeline for the dual-task (cine enhancement + phase-error
correction) model.

Pipeline per patient
--------------------
1. Run the dual-head generator on every timepoint at the downsampled (128×128×64)
   resolution.  The model outputs 4 channels:
     - channel 0:   predicted enhanced magnitude (Sigmoid, [0, 1])
     - channels 1–3: predicted VENC-normalised velocity corrections (Tanh, [-1, 1])

2. **Magnitude post-processing** – upsample the predicted enhanced magnitude from
   the downsampled grid back to the original corrected-velocity FOV.  Several
   combination modes are supported so that different strategies can be compared
   quickly:
     - ``naive``    : trilinear upsample only
     - ``blend``    : α · prediction + (1-α) · original_mag
     - ``multiply`` : original_mag × prediction

3. **Velocity-correction post-processing** – mirror the ground-truth correction
   pipeline:
     a. Stack per-timepoint predicted corrections → (T, 3, Z, Y, X)
     b. Build a magnitude mask at the downsampled resolution
     c. Fit a 3rd-order polynomial per velocity component, taking the median
        coefficients across timepoints (same as ``Patient.build_velocity_correction_data``)
     d. Rebuild the polynomial basis at the **original** corrected-velocity-FOV
        resolution and reconstruct the correction field
     e. De-normalise:  correction_physical = correction_venc_normalised × VENC
     f. Add to uncorrected velocity:  v_corrected = v_uncorrected + correction_physical

4. Write DICOM series that contain the predicted enhanced magnitude and the
   predicted corrected velocities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from omegaconf import DictConfig

from vascular_superenhancement.data_management.dicom_to_nifti import (
    DicomToNiftiConverter,
)
from vascular_superenhancement.data_management.nifti_to_dicom import (
    NiftiToDicomConverter,
)
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.training.model_factory import build_generator
from vascular_superenhancement.utils.logger import setup_patient_logger
from vascular_superenhancement.utils.path_config import load_path_config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (static / module-level)
# ──────────────────────────────────────────────────────────────────────────────

def _save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Save a numpy array as a NIfTI-1 file."""
    nii = nib.Nifti1Image(data.astype(np.float32), affine)
    nii.set_qform(affine, code=1)
    nii.set_sform(affine, code=1)
    hdr = nii.header
    hdr["dim"][0] = data.ndim
    hdr["xyzt_units"] = 2 | 8  # mm + seconds
    nib.save(nii, path)


def _resample_nifti_to_reference(
    source_path: Path,
    reference_img: sitk.Image,
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """Resample a NIfTI to a SimpleITK reference grid."""
    source_img = sitk.ReadImage(str(source_path))
    return DicomToNiftiConverter.resample_to_target_grid(
        source_img, reference_img, interpolator=interpolator
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class DualTaskInferencer:
    """Complete inference pipeline for the dual-task model.

    Parameters are read from the merged Hydra ``DictConfig`` – see
    ``hydra_configs/inference/dual_task.yaml`` for the available knobs.
    """

    # ── construction ─────────────────────────────────────────────────────
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Checkpoint / output paths
        self.checkpoint_path = Path(cfg.inference.checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.output_dir = Path(cfg.inference.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inference_name = cfg.inference.inference_name

        # Data layout
        self.downsampled_folder: str = cfg.data.downsampled_folder
        self.temporal_mag_offsets: list[int] = list(
            cfg.train.get("temporal_mag_offsets", [])
        )

        # Magnitude post-processing
        self.mag_mode: str = cfg.inference.get("mag_mode", "naive")
        self.mag_blend_alpha: float = cfg.inference.get("mag_blend_alpha", 0.5)

        # Polynomial-fit settings
        self.n_poly_coeffs: int = cfg.inference.get("n_poly_coeffs", 20)
        self.mag_mask_threshold: float = cfg.inference.get("mag_mask_threshold", 0.05)
        self.mag_mask_shrink_margin: int = cfg.inference.get(
            "mag_mask_shrink_margin", 4
        )
        self.mag_mask_norm_percentile: float = cfg.inference.get(
            "mag_mask_norm_percentile", 99.0
        )

        # Misc
        self.overwrite: bool = cfg.inference.get("overwrite", False)

        # Build & load model
        self.generator = build_generator(cfg).to(self.device)
        self._load_checkpoint()
        self.generator.eval()

    def _load_checkpoint(self) -> None:
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator_state_dict"])
        logger.info(f"Loaded checkpoint from {self.checkpoint_path}")

    # ── patient loading ──────────────────────────────────────────────────
    def _load_patient(self, patient_id: str) -> Patient:
        config_name = self.cfg.path_config.path_config_name
        path_config_obj = load_path_config(config_name)
        return Patient(
            path_config=path_config_obj,
            phonetic_id=patient_id,
            debug=False,
            config=config_name,
        )

    # ── prerequisite checks ─────────────────────────────────────────────
    def _ensure_prerequisites(
        self, patient: Patient, plog: logging.Logger
    ) -> None:
        """Verify (and build if missing) data required by the inference pipeline.

        Required directories / files:
          1. Downsampled per-timepoint data  (``downsampled_full_fov_128x128x64/``)
          2. Uncorrected velocity at corrected-velocity FOV
             (``4d_flow_vx_{id}_per_timepoint_corr_fov/`` etc.)
          3. Corrected-velocity reference (``4d_flow_vx_corr_{id}_frame_00.nii.gz``)
             — needed for the target geometry.
          4. Magnitude at corrected-velocity FOV (only when ``mag_mode`` != 'naive')
        """
        pid = patient.identifier
        ds_root = patient.nifti_dir / self.downsampled_folder

        # (1) Downsampled data
        ds_mag_dir = ds_root / "4d_flow_mag"
        if not ds_mag_dir.exists() or not list(ds_mag_dir.glob("*.nii.gz")):
            raise FileNotFoundError(
                f"Downsampled data not found at {ds_root}. "
                f"Run patient.build_downsampled_full_fov_per_timepoint() first."
            )
        plog.info(f"  ✓ downsampled data found at {ds_root}")

        # (2) Uncorrected velocity at corrected FOV
        for comp, comp_dir in [
            ("vx", patient.flow_vx_per_timepoint_corr_fov_dir),
            ("vy", patient.flow_vy_per_timepoint_corr_fov_dir),
            ("vz", patient.flow_vz_per_timepoint_corr_fov_dir),
        ]:
            if not list(comp_dir.glob("*.nii.gz")):
                plog.info(
                    f"  uncorrected {comp} at corrected FOV not found – "
                    "building via build_uncorrected_per_timepoint_corr_fov() …"
                )
                patient.build_uncorrected_per_timepoint_corr_fov()
                break
        plog.info("  ✓ uncorrected velocity at corrected FOV ready")

        # (3) Corrected-velocity reference
        ref_path = (
            patient.flow_vx_corr_per_timepoint_dir
            / f"4d_flow_vx_corr_{pid}_frame_00.nii.gz"
        )
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Corrected-velocity reference not found: {ref_path}. "
                "Ensure build_corrected_velocities_per_timepoint() has been run."
            )
        plog.info("  ✓ corrected-velocity reference found")

        # (4) Magnitude at corrected FOV (for blend / multiply modes)
        if self.mag_mode in ("blend", "multiply"):
            mag_corr_dir = patient.flow_mag_per_timepoint_corr_fov_dir
            if not list(mag_corr_dir.glob("*.nii.gz")):
                plog.info(
                    "  magnitude at corrected FOV not found – "
                    "building via build_uncorrected_per_timepoint_corr_fov() …"
                )
                patient.build_uncorrected_per_timepoint_corr_fov()
            plog.info("  ✓ magnitude at corrected FOV ready")

    # ── public entry point ───────────────────────────────────────────────
    def predict_patient(self, patient_id: str) -> Path:
        """Run the full dual-task inference pipeline for one patient.

        Returns the patient-level output directory.
        """
        patient_logger = setup_patient_logger(
            patient_id, config=self.cfg.path_config.path_config_name
        )

        patient = self._load_patient(patient_id)
        venc = float(patient.venc)
        num_tp = patient.num_timepoints
        patient_logger.info(
            f"Starting dual-task inference for {patient_id} "
            f"({num_tp} timepoints, VENC={venc})"
        )

        # Verify / build prerequisite data
        patient_logger.info("Checking prerequisites …")
        self._ensure_prerequisites(patient, patient_logger)

        patient_dir = self.output_dir / patient_id
        raw_dir = patient_dir / "raw_predictions"
        mag_dir = patient_dir / "predicted_mag"
        vel_dir = patient_dir / "predicted_corrected_velocity"
        dicom_dir = patient_dir / "dicoms"
        for d in (raw_dir, mag_dir, vel_dir, dicom_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ---- Step 1: run model per timepoint --------------------------------
        patient_logger.info("Step 1 / 4: running model on each timepoint …")
        self._run_model_all_timepoints(patient, raw_dir, patient_logger)

        # ---- Step 2: post-process magnitude ---------------------------------
        patient_logger.info("Step 2 / 4: post-processing magnitude …")
        self._postprocess_magnitude(patient, raw_dir, mag_dir, patient_logger)

        # ---- Step 3: post-process velocity corrections ----------------------
        patient_logger.info("Step 3 / 4: post-processing velocity corrections …")
        self._postprocess_velocity_corrections(
            patient, raw_dir, vel_dir, patient_logger
        )

        # ---- Step 4: DICOM generation ---------------------------------------
        patient_logger.info("Step 4 / 4: generating DICOMs …")
        self._generate_dicoms(patient, mag_dir, vel_dir, dicom_dir, patient_logger)

        patient_logger.info(f"Pipeline complete for {patient_id}")
        return patient_dir

    # =====================================================================
    # Step 1 – run model on each timepoint
    # =====================================================================

    def _run_model_all_timepoints(
        self,
        patient: Patient,
        raw_dir: Path,
        plog: logging.Logger,
    ) -> None:
        """Run the dual-head model on each timepoint at downsampled resolution.

        Saves two files per timepoint under *raw_dir*:
          - ``pred_cine_t{tt}.nii.gz``      (1 channel, [0, 1])
          - ``pred_correction_t{tt}.nii.gz`` (3 channels, [-1, 1])
        """
        pid = patient.identifier
        venc = float(patient.venc)
        num_tp = patient.num_timepoints
        ds_root = patient.nifti_dir / self.downsampled_folder

        # Load the downsampled reference NIfTI for affine / shape
        ref_nib = nib.load(ds_root / "reference.nii.gz")
        ds_affine = ref_nib.affine

        with torch.no_grad():
            for t in range(num_tp):
                cine_path = raw_dir / f"pred_cine_t{t:02d}.nii.gz"
                corr_path = raw_dir / f"pred_correction_t{t:02d}.nii.gz"

                if not self.overwrite and cine_path.exists() and corr_path.exists():
                    plog.info(f"  t={t:02d} – raw predictions exist, skipping")
                    continue

                # -- load downsampled inputs ----------------------------------
                mag_path = (
                    ds_root / "4d_flow_mag"
                    / f"4d_flow_mag_{pid}_frame_{t:02d}.nii.gz"
                )
                vx_path = (
                    ds_root / "4d_flow_vx"
                    / f"4d_flow_vx_{pid}_frame_{t:02d}.nii.gz"
                )
                vy_path = (
                    ds_root / "4d_flow_vy"
                    / f"4d_flow_vy_{pid}_frame_{t:02d}.nii.gz"
                )
                vz_path = (
                    ds_root / "4d_flow_vz"
                    / f"4d_flow_vz_{pid}_frame_{t:02d}.nii.gz"
                )

                mag = self._load_tensor(mag_path)  # [1, 1, X, Y, Z]
                vx = self._load_tensor(vx_path)
                vy = self._load_tensor(vy_path)
                vz = self._load_tensor(vz_path)

                # -- normalise exactly as in DualTaskTrainer.prepare_batch -----
                # Mag → [0, 1]
                mag_min, mag_max = mag.min(), mag.max()
                if mag_max > mag_min:
                    mag = (mag - mag_min) / (mag_max - mag_min)
                else:
                    mag = torch.zeros_like(mag)

                # Velocity → v / VENC, clamp [-1, 1]
                vx = (vx / venc).clamp(-1.0, 1.0)
                vy = (vy / venc).clamp(-1.0, 1.0)
                vz = (vz / venc).clamp(-1.0, 1.0)

                # -- assemble input and run model ------------------------------
                input_tensor = torch.cat([mag, vx, vy, vz], dim=1)  # [1, 4, X, Y, Z]
                pred = self.generator(input_tensor)  # [1, 4, X, Y, Z]

                pred_cine = pred[0, 0:1].cpu().numpy()  # (1, X, Y, Z)
                pred_corr = pred[0, 1:4].cpu().numpy()  # (3, X, Y, Z)

                # -- save with nibabel (channel-first → spatial) ---------------
                _save_nifti(pred_cine[0], ds_affine, cine_path)
                # Save corrections as 4-D NIfTI (X, Y, Z, 3)
                corr_vol = np.transpose(pred_corr, (1, 2, 3, 0))  # (X, Y, Z, 3)
                _save_nifti(corr_vol, ds_affine, corr_path)

                plog.info(
                    f"  t={t:02d} – pred_cine [{pred_cine.min():.3f}, {pred_cine.max():.3f}], "
                    f"pred_corr [{pred_corr.min():.3f}, {pred_corr.max():.3f}]"
                )

    # =====================================================================
    # Step 2 – magnitude post-processing
    # =====================================================================

    def _postprocess_magnitude(
        self,
        patient: Patient,
        raw_dir: Path,
        mag_dir: Path,
        plog: logging.Logger,
    ) -> None:
        """Upsample predicted cine to the original corrected-velocity FOV.

        The ``mag_mode`` configuration controls how the upsampled prediction is
        combined with the original magnitude image:
          - ``naive``    – trilinear upsample only
          - ``blend``    – alpha-blend with the original mag
          - ``multiply`` – element-wise multiply with the original mag
        """
        pid = patient.identifier
        num_tp = patient.num_timepoints

        # Reference at original corrected-velocity FOV resolution
        ref_path = (
            patient.flow_vx_corr_per_timepoint_dir
            / f"4d_flow_vx_corr_{pid}_frame_00.nii.gz"
        )
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Corrected-velocity reference not found: {ref_path}. "
                "Ensure build_corrected_velocities_per_timepoint() has been run."
            )
        ref_img = sitk.ReadImage(str(ref_path))

        for t in range(num_tp):
            out_path = mag_dir / f"pred_mag_{pid}_frame_{t:02d}.nii.gz"
            if not self.overwrite and out_path.exists():
                plog.info(f"  t={t:02d} – upsampled mag exists, skipping")
                continue

            # Upsample prediction
            pred_path = raw_dir / f"pred_cine_t{t:02d}.nii.gz"
            upsampled_img = _resample_nifti_to_reference(pred_path, ref_img)
            upsampled_arr = sitk.GetArrayFromImage(upsampled_img).astype(np.float32)

            if self.mag_mode in ("blend", "multiply"):
                # Load original magnitude at corrected-velocity FOV
                orig_mag_path = (
                    patient.flow_mag_per_timepoint_corr_fov_dir
                    / f"4d_flow_mag_{pid}_frame_{t:02d}.nii.gz"
                )
                if not orig_mag_path.exists():
                    plog.warning(
                        f"  t={t:02d} – original mag not found at {orig_mag_path}, "
                        "falling back to naive mode"
                    )
                else:
                    orig_arr = sitk.GetArrayFromImage(
                        sitk.ReadImage(str(orig_mag_path))
                    ).astype(np.float32)

                    if self.mag_mode == "blend":
                        alpha = self.mag_blend_alpha
                        # Normalise original mag to [0,1] for blending
                        omin, omax = orig_arr.min(), orig_arr.max()
                        if omax > omin:
                            orig_norm = (orig_arr - omin) / (omax - omin)
                        else:
                            orig_norm = np.zeros_like(orig_arr)
                        upsampled_arr = (
                            alpha * upsampled_arr + (1 - alpha) * orig_norm
                        )
                    elif self.mag_mode == "multiply":
                        upsampled_arr = orig_arr * upsampled_arr

            # Save
            result_img = sitk.GetImageFromArray(upsampled_arr)
            result_img.CopyInformation(ref_img)
            sitk.WriteImage(result_img, str(out_path))
            plog.info(f"  t={t:02d} – saved upsampled mag ({self.mag_mode})")

    # =====================================================================
    # Step 3 – velocity-correction post-processing
    # =====================================================================

    def _postprocess_velocity_corrections(
        self,
        patient: Patient,
        raw_dir: Path,
        vel_dir: Path,
        plog: logging.Logger,
    ) -> None:
        """Polyfit the predicted corrections and produce corrected velocity NIfTIs.

        Sub-steps:
          a. Collect per-timepoint predicted corrections at downsampled resolution.
          b. Build a magnitude mask.
          c. Fit 3rd-order polynomial, median across timepoints.
          d. Reconstruct correction at original corrected-velocity-FOV resolution.
          e. De-normalise (× VENC) and add to uncorrected velocity.
        """
        pid = patient.identifier
        venc = float(patient.venc)
        num_tp = patient.num_timepoints
        ds_root = patient.nifti_dir / self.downsampled_folder

        # ---- (a) load all predicted corrections (T, 3, Z, Y, X) -------------
        plog.info("  3a: loading predicted corrections …")
        corrections_list = []
        for t in range(num_tp):
            corr_path = raw_dir / f"pred_correction_t{t:02d}.nii.gz"
            corr_nib = nib.load(corr_path)
            corr_data = corr_nib.get_fdata(dtype=np.float32)  # (X, Y, Z, 3)
            # Transpose to (3, Z, Y, X) to match _fit_polynomial_coefficients
            corr_zyx = np.transpose(corr_data, (3, 2, 1, 0))  # (3, Z, Y, X)
            corrections_list.append(corr_zyx)

        corrections = np.stack(corrections_list, axis=0)  # (T, 3, Z, Y, X)
        ds_shape_zyx = corrections.shape[2:]  # (Z, Y, X)
        ds_shape_xyz = (ds_shape_zyx[2], ds_shape_zyx[1], ds_shape_zyx[0])  # (X, Y, Z)
        plog.info(
            f"  corrections shape (T, 3, Z, Y, X): {corrections.shape}, "
            f"value range [{corrections.min():.4f}, {corrections.max():.4f}]"
        )

        # ---- (b) magnitude mask at downsampled resolution --------------------
        plog.info("  3b: building magnitude mask at downsampled resolution …")
        mag_arrays = []
        for t in range(num_tp):
            mag_path = (
                ds_root
                / "4d_flow_mag"
                / f"4d_flow_mag_{pid}_frame_{t:02d}.nii.gz"
            )
            mag_img = sitk.ReadImage(str(mag_path))
            mag_arrays.append(
                sitk.GetArrayFromImage(mag_img).astype(np.float32)
            )  # (Z, Y, X) each

        mag_4d = np.stack(mag_arrays, axis=0)  # (T, Z, Y, X)
        mag_xyz_t = np.transpose(mag_4d, (3, 2, 1, 0))  # (X, Y, Z, T)

        mask = Patient._create_magnitude_mask(
            mag_xyz_t,
            threshold_fraction=self.mag_mask_threshold,
            shrink_margin=self.mag_mask_shrink_margin,
            normalization_percentile=self.mag_mask_norm_percentile,
        )  # (X, Y, Z)
        n_valid = int(np.sum(mask > 0))
        plog.info(f"  mask valid voxels: {n_valid} / {int(np.prod(ds_shape_xyz))}")

        # ---- (c) polynomial fit – median coefficients ------------------------
        plog.info("  3c: fitting polynomial coefficients …")
        basis_ds = Patient._build_polynomial_basis(ds_shape_xyz, self.n_poly_coeffs)
        coefficients = Patient._fit_polynomial_coefficients(
            corrections, basis_ds, mask
        )  # (n_coeffs, 3)
        plog.info(f"  median coefficients shape: {coefficients.shape}")

        # Save coefficients for reproducibility
        coeff_path = vel_dir / f"pred_poly_coefficients_{pid}.npz"
        np.savez(
            coeff_path,
            coefficients=coefficients,
            venc=venc,
            downsampled_shape=np.array(ds_shape_xyz),
        )

        # ---- (d) reconstruct at original corrected-velocity-FOV res ----------
        plog.info("  3d: reconstructing correction at original resolution …")

        ref_path = (
            patient.flow_vx_corr_per_timepoint_dir
            / f"4d_flow_vx_corr_{pid}_frame_00.nii.gz"
        )
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Corrected-velocity reference not found: {ref_path}"
            )
        ref_sitk = sitk.ReadImage(str(ref_path))
        orig_size = ref_sitk.GetSize()  # (X, Y, Z) in SimpleITK order
        orig_shape_xyz = (orig_size[0], orig_size[1], orig_size[2])
        plog.info(f"  original corrected-velocity FOV shape (X,Y,Z): {orig_shape_xyz}")

        basis_orig = Patient._build_polynomial_basis(orig_shape_xyz, self.n_poly_coeffs)
        gt_dict = Patient._reconstruct_from_coefficients(
            coefficients, basis_orig, orig_shape_xyz
        )  # {'vx': (X,Y,Z), 'vy': …, 'vz': …}  – VENC-normalised

        # Get affine from original-res reference
        ref_nib = nib.load(ref_path)
        orig_affine = ref_nib.affine

        # Save reconstructed corrections (VENC-normalised)
        for comp in ("vx", "vy", "vz"):
            corr_out = vel_dir / f"pred_correction_{comp}_{pid}.nii.gz"
            _save_nifti(gt_dict[comp], orig_affine, corr_out)

        # ---- (e-f) de-normalise and apply to uncorrected velocity ------------
        plog.info("  3e-f: applying corrections to uncorrected velocity …")

        comp_dirs = {
            "vx": patient.flow_vx_per_timepoint_corr_fov_dir,
            "vy": patient.flow_vy_per_timepoint_corr_fov_dir,
            "vz": patient.flow_vz_per_timepoint_corr_fov_dir,
        }

        for comp in ("vx", "vy", "vz"):
            # Correction in physical units
            correction_physical = gt_dict[comp] * venc  # (X, Y, Z)
            # Transpose to (Z, Y, X) for SimpleITK array ordering
            correction_zyx = np.transpose(correction_physical, (2, 1, 0))

            comp_out_dir = vel_dir / f"4d_flow_{comp}_corr"
            comp_out_dir.mkdir(parents=True, exist_ok=True)

            for t in range(num_tp):
                out_path = (
                    comp_out_dir
                    / f"4d_flow_{comp}_corr_{pid}_frame_{t:02d}.nii.gz"
                )
                if not self.overwrite and out_path.exists():
                    continue

                # Load uncorrected velocity at corrected-velocity FOV
                uncorr_path = (
                    comp_dirs[comp]
                    / f"4d_flow_{comp}_{pid}_frame_{t:02d}.nii.gz"
                )
                if not uncorr_path.exists():
                    plog.warning(
                        f"  uncorrected {comp} t={t:02d} not found at "
                        f"{uncorr_path}, skipping"
                    )
                    continue

                uncorr_img = sitk.ReadImage(str(uncorr_path))
                uncorr_arr = sitk.GetArrayFromImage(uncorr_img).astype(
                    np.float32
                )  # (Z, Y, X)

                corrected_arr = uncorr_arr + correction_zyx

                corrected_img = sitk.GetImageFromArray(corrected_arr)
                corrected_img.CopyInformation(uncorr_img)
                sitk.WriteImage(corrected_img, str(out_path))

            plog.info(f"  saved corrected {comp} for all timepoints")

        plog.info("  velocity correction post-processing complete")

    # =====================================================================
    # Step 4 – DICOM generation
    # =====================================================================

    def _generate_dicoms(
        self,
        patient: Patient,
        mag_dir: Path,
        vel_dir: Path,
        dicom_dir: Path,
        plog: logging.Logger,
    ) -> Optional[Path]:
        """Write predicted mag + corrected velocity to DICOM series.

        Uses the extended ``NiftiToDicomConverter.write_timepoint_with_velocities_to_dicoms``.
        """
        pid = patient.identifier
        num_tp = patient.num_timepoints

        converter = NiftiToDicomConverter.from_patient(patient)

        from pydicom.uid import generate_uid

        study_uid = generate_uid()
        series_uids = {
            2: generate_uid(),  # Magnitude
            3: generate_uid(),  # Vx
            4: generate_uid(),  # Vy
            5: generate_uid(),  # Vz
        }

        for t in range(num_tp):
            mag_path = mag_dir / f"pred_mag_{pid}_frame_{t:02d}.nii.gz"
            vx_path = (
                vel_dir / "4d_flow_vx_corr"
                / f"4d_flow_vx_corr_{pid}_frame_{t:02d}.nii.gz"
            )
            vy_path = (
                vel_dir / "4d_flow_vy_corr"
                / f"4d_flow_vy_corr_{pid}_frame_{t:02d}.nii.gz"
            )
            vz_path = (
                vel_dir / "4d_flow_vz_corr"
                / f"4d_flow_vz_corr_{pid}_frame_{t:02d}.nii.gz"
            )

            if not mag_path.exists():
                plog.warning(f"  t={t:02d} – predicted mag not found, skipping")
                continue

            vel_paths = {}
            if vx_path.exists() and vy_path.exists() and vz_path.exists():
                vel_paths = {"vx": vx_path, "vy": vy_path, "vz": vz_path}

            converter.write_timepoint_with_velocities_to_dicoms(
                mag_prediction_path=mag_path,
                velocity_paths=vel_paths,
                output_dir=dicom_dir,
                timepoint=t,
                study_uid=study_uid,
                series_uids=series_uids,
                overwrite=self.overwrite,
            )

        plog.info(f"  wrote DICOMs to {dicom_dir}")

        # Optional zip
        zip_path = dicom_dir.parent / f"{pid}_dual_task_dicoms.zip"
        import zipfile

        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in dicom_dir.rglob("*"):
                if fp.is_file():
                    zf.write(fp, arcname=fp.relative_to(dicom_dir))
        plog.info(f"  created zip archive: {zip_path}")
        return zip_path

    # =====================================================================
    # Utility helpers
    # =====================================================================

    def _load_tensor(self, path: Path) -> torch.Tensor:
        """Load a 3-D NIfTI and return a ``[1, 1, X, Y, Z]`` float tensor on device."""
        nii = nib.load(path)
        arr = nii.get_fdata(dtype=np.float32)  # (X, Y, Z)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, X, Y, Z)
        return tensor.to(self.device)


# ──────────────────────────────────────────────────────────────────────────────
# Hydra entry point
# ──────────────────────────────────────────────────────────────────────────────

import hydra
import pandas as pd


@hydra.main(
    version_base="1.1",
    config_path=str(
        (Path(__file__).resolve().parents[3] / "hydra_configs").as_posix()
    ),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Run dual-task inference from the command line.

    Examples::

        # Single patient
        python -m vascular_superenhancement.inferencing.dual_task_inference \\
            inference.patient_id=Foxtrot

        # All test patients
        python -m vascular_superenhancement.inferencing.dual_task_inference \\
            inference.all_test_patients=true
    """
    if cfg.inference.get("all_test_patients"):
        df = pd.read_csv(cfg.data.splits_path)
        patient_ids = df[df.split == "test"].patient_id.tolist()
        logger.info(f"Running dual-task inference for {len(patient_ids)} test patients")
    elif cfg.inference.get("patient_id"):
        patient_ids = [cfg.inference.patient_id]
    else:
        raise ValueError("Provide inference.patient_id or inference.all_test_patients=true")

    inferencer = DualTaskInferencer(cfg)

    for pid in patient_ids:
        logger.info(f"{'='*60}")
        logger.info(f"Processing patient: {pid}")
        logger.info(f"{'='*60}")
        try:
            output_dir = inferencer.predict_patient(pid)
            logger.info(f"Done → {output_dir}")
        except Exception as e:
            logger.error(f"Failed for {pid}: {e}", exc_info=True)
            continue


if __name__ == "__main__":
    main()
