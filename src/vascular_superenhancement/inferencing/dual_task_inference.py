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
     - ``naive``        : trilinear upsample only
     - ``blend_{pct}``  : α · prediction + (1-α) · original_mag, one per configured alpha

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

Configuration flags
-------------------
- ``write_dicoms`` (default ``true``): set ``false`` to skip step 4 for faster
  iteration when only the NIfTI outputs are needed.
- ``dicoms_only`` (default ``false``): set ``true`` to run *only* step 4 from
  existing NIfTI predictions — the model is never loaded, so no GPU is required.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

        # Checkpoint / output paths
        self.checkpoint_path = Path(cfg.inference.checkpoint_path)
        self.output_dir = Path(cfg.inference.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inference_name = cfg.inference.inference_name

        # Data layout
        self.downsampled_folder: str = cfg.data.downsampled_folder
        self.temporal_mag_offsets: list[int] = list(
            cfg.train.get("temporal_mag_offsets", [])
        )

        # Magnitude post-processing
        raw_alphas = cfg.inference.get("mag_blend_alphas", None)
        if raw_alphas is not None:
            self.mag_blend_alphas: list[float] = list(raw_alphas)
        else:
            self.mag_blend_alphas = [cfg.inference.get("mag_blend_alpha", 0.5)]
        self.include_naive: bool = cfg.inference.get("include_naive", True)

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
        _mw = cfg.inference.get("max_workers", 0)
        self.max_workers: int | None = None if _mw <= 0 else _mw
        self.write_dicoms: bool = cfg.inference.get("write_dicoms", True)
        self.dicoms_only: bool = cfg.inference.get("dicoms_only", False)

        if self.dicoms_only and not self.write_dicoms:
            raise ValueError(
                "dicoms_only=true requires write_dicoms=true "
                "(nothing to do when both DICOM writing and inference are disabled)"
            )

        # Only load the model when we actually need to run inference
        if not self.dicoms_only:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(f"Using device: {self.device}")
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.checkpoint_path}"
                )
            self.generator = build_generator(cfg).to(self.device)
            self._load_checkpoint()
            self.generator.eval()
        else:
            logger.info(
                "dicoms_only mode – skipping model loading, "
                "will generate DICOMs from existing NIfTI predictions"
            )

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

        # (4) Magnitude at corrected FOV (needed for blend modes)
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
        """Run the dual-task inference pipeline for one patient.

        Which steps are executed depends on the configuration flags:
          - Default (``write_dicoms=true, dicoms_only=false``): full pipeline
            (steps 1–4).
          - ``write_dicoms=false``: steps 1–3 only (fast NIfTI-only run).
          - ``dicoms_only=true``: step 4 only, from existing NIfTI predictions.

        Returns the patient-level output directory.
        """
        patient_logger = setup_patient_logger(
            patient_id, config=self.cfg.path_config.path_config_name
        )

        patient = self._load_patient(patient_id)
        venc = float(patient.venc)
        num_tp = patient.num_timepoints

        run_inference = not self.dicoms_only
        run_dicoms = self.write_dicoms

        total_steps = (3 if run_inference else 0) + (1 if run_dicoms else 0)
        mode_label = (
            "dicoms-only" if self.dicoms_only
            else ("no-dicoms" if not self.write_dicoms else "full")
        )
        patient_logger.info(
            f"Starting dual-task inference for {patient_id} "
            f"({num_tp} timepoints, VENC={venc}, mode={mode_label})"
        )

        patient_dir = self.output_dir / patient_id
        raw_dir = patient_dir / "raw_predictions"
        mag_dir = patient_dir / "predicted_mag"
        vel_dir = patient_dir / "predicted_corrected_velocity"
        dicom_dir = patient_dir / "dicoms"

        step = 0

        if run_inference:
            # Verify / build prerequisite data
            patient_logger.info("Checking prerequisites …")
            self._ensure_prerequisites(patient, patient_logger)

            for d in (raw_dir, mag_dir, vel_dir):
                d.mkdir(parents=True, exist_ok=True)

            # ---- Step 1: run model per timepoint (GPU, sequential) -----------
            step += 1
            patient_logger.info(
                f"Step {step} / {total_steps}: running model on each timepoint …"
            )
            self._run_model_all_timepoints(patient, raw_dir, patient_logger)

            # ---- Steps 2 & 3: post-process (concurrent) ---------------------
            step += 1
            next_step = step + 1
            patient_logger.info(
                f"Steps {step}–{next_step} / {total_steps}: "
                "post-processing magnitude & velocity (concurrent) …"
            )
            with ThreadPoolExecutor(max_workers=2) as orchestrator:
                mag_future = orchestrator.submit(
                    self._postprocess_magnitude,
                    patient, raw_dir, mag_dir, patient_logger,
                )
                vel_future = orchestrator.submit(
                    self._postprocess_velocity_corrections,
                    patient, raw_dir, vel_dir, patient_logger,
                )
                mag_future.result()
                vel_future.result()
            step = next_step

        if run_dicoms:
            dicom_dir.mkdir(parents=True, exist_ok=True)

            # Validate that NIfTI predictions exist when running dicoms-only
            if self.dicoms_only:
                for d, label in [
                    (mag_dir, "predicted_mag"),
                    (vel_dir, "predicted_corrected_velocity"),
                ]:
                    if not d.exists() or not list(d.rglob("*.nii.gz")):
                        raise FileNotFoundError(
                            f"Cannot generate DICOMs: {label} directory "
                            f"is missing or empty ({d}). "
                            "Run inference first with dicoms_only=false."
                        )

            step += 1
            patient_logger.info(
                f"Step {step} / {total_steps}: "
                "generating DICOMs (one set per mag mode) …"
            )
            self._generate_dicoms(
                patient, mag_dir, vel_dir, dicom_dir, patient_dir, patient_logger
            )

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

        Saves four files per timepoint under *raw_dir*:
          - ``pred_mag_t{tt}.nii.gz``               (3-D, [0, 1])
          - ``pred_correction_vx_t{tt}.nii.gz``     (3-D, [-1, 1])
          - ``pred_correction_vy_t{tt}.nii.gz``     (3-D, [-1, 1])
          - ``pred_correction_vz_t{tt}.nii.gz``     (3-D, [-1, 1])
        """
        pid = patient.identifier
        venc = float(patient.venc)
        num_tp = patient.num_timepoints
        ds_root = patient.nifti_dir / self.downsampled_folder

        # Load the downsampled reference NIfTI for affine / shape
        ref_nib = nib.load(ds_root / "reference.nii.gz")
        ds_affine = ref_nib.affine

        # Build the ordered list of temporal mag offsets matching training:
        # e.g. offsets [-2, -1, 1, 2] → keys [n2, n1, center, p1, p2] → 5 mags
        sorted_offsets = sorted(self.temporal_mag_offsets)

        with torch.no_grad():
            for t in range(num_tp):
                mag_out = raw_dir / f"pred_mag_t{t:02d}.nii.gz"
                corr_vx_out = raw_dir / f"pred_correction_vx_t{t:02d}.nii.gz"
                corr_vy_out = raw_dir / f"pred_correction_vy_t{t:02d}.nii.gz"
                corr_vz_out = raw_dir / f"pred_correction_vz_t{t:02d}.nii.gz"

                all_exist = (
                    mag_out.exists()
                    and corr_vx_out.exists()
                    and corr_vy_out.exists()
                    and corr_vz_out.exists()
                )
                if not self.overwrite and all_exist:
                    plog.info(f"  t={t:02d} – raw predictions exist, skipping")
                    continue

                # -- load temporal magnitude window (matching training order) ---
                mag_tensors: list[torch.Tensor] = []
                center_inserted = False
                for off in sorted_offsets:
                    if not center_inserted and off > 0:
                        mag_tensors.append(
                            self._load_and_normalise_mag(ds_root, pid, t, num_tp)
                        )
                        center_inserted = True
                    t_off = (t + off) % num_tp
                    mag_tensors.append(
                        self._load_and_normalise_mag(ds_root, pid, t_off, num_tp)
                    )
                if not center_inserted:
                    mag_tensors.append(
                        self._load_and_normalise_mag(ds_root, pid, t, num_tp)
                    )

                # -- load downsampled velocity --------------------------------
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

                vx = self._load_tensor(vx_path)
                vy = self._load_tensor(vy_path)
                vz = self._load_tensor(vz_path)

                # Velocity → v / VENC, clamp [-1, 1]
                vx = (vx / venc).clamp(-1.0, 1.0)
                vy = (vy / venc).clamp(-1.0, 1.0)
                vz = (vz / venc).clamp(-1.0, 1.0)

                # -- assemble input and run model ------------------------------
                # [1, num_mag + 3, X, Y, Z]  (e.g. 5 mag + 3 vel = 8 channels)
                input_tensor = torch.cat(mag_tensors + [vx, vy, vz], dim=1)
                pred = self.generator(input_tensor)  # [1, 4, X, Y, Z]

                pred_mag_arr = pred[0, 0].cpu().numpy()   # (X, Y, Z)
                pred_corr = pred[0, 1:4].cpu().numpy()    # (3, X, Y, Z)

                # -- save as separate 3-D NIfTIs ------------------------------
                _save_nifti(pred_mag_arr, ds_affine, mag_out)
                _save_nifti(pred_corr[0], ds_affine, corr_vx_out)  # vx
                _save_nifti(pred_corr[1], ds_affine, corr_vy_out)  # vy
                _save_nifti(pred_corr[2], ds_affine, corr_vz_out)  # vz

                plog.info(
                    f"  t={t:02d} – pred_mag [{pred_mag_arr.min():.3f}, {pred_mag_arr.max():.3f}], "
                    f"pred_corr [{pred_corr.min():.3f}, {pred_corr.max():.3f}]"
                )

    # =====================================================================
    # Step 2 – magnitude post-processing
    # =====================================================================

    def _mag_modes(self) -> list[str]:
        """Return the list of magnitude post-processing mode directory names."""
        modes = ["naive"] if self.include_naive else []
        for alpha in self.mag_blend_alphas:
            modes.append(f"blend_{int(round(alpha * 100))}")
        return modes

    def _postprocess_magnitude(
        self,
        patient: Patient,
        raw_dir: Path,
        mag_dir: Path,
        plog: logging.Logger,
    ) -> None:
        """Upsample predicted magnitude and produce combination modes.

        Outputs under *mag_dir*:
          - ``naive/``        – trilinear upsample only
          - ``blend_{pct}/``  – one directory per blend alpha, where *pct* is
            ``int(alpha * 100)`` (prediction weight percentage)

        Timepoints are processed in parallel via :pyclass:`ThreadPoolExecutor`.
        """
        pid = patient.identifier
        num_tp = patient.num_timepoints

        modes = self._mag_modes()
        mode_dirs: dict[str, Path] = {}
        for mode in modes:
            d = mag_dir / mode
            d.mkdir(parents=True, exist_ok=True)
            mode_dirs[mode] = d

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
        orig_mag_dir = patient.flow_mag_per_timepoint_corr_fov_dir

        # -- per-timepoint worker (executed in thread pool) --------------------
        def _process_timepoint(t: int) -> tuple[int, bool]:
            out_paths = {
                mode: mode_dirs[mode] / f"pred_mag_{pid}_frame_{t:02d}.nii.gz"
                for mode in modes
            }
            if not self.overwrite and all(p.exists() for p in out_paths.values()):
                return t, True  # skipped

            # Upsample prediction (shared across modes)
            pred_path = raw_dir / f"pred_mag_t{t:02d}.nii.gz"
            upsampled_img = _resample_nifti_to_reference(pred_path, ref_img)
            naive_arr = sitk.GetArrayFromImage(upsampled_img).astype(np.float32)

            # Load original magnitude at corrected-velocity FOV (for blending)
            orig_mag_path = (
                orig_mag_dir / f"4d_flow_mag_{pid}_frame_{t:02d}.nii.gz"
            )
            have_orig = orig_mag_path.exists()
            orig_norm = None
            if have_orig:
                orig_arr = sitk.GetArrayFromImage(
                    sitk.ReadImage(str(orig_mag_path))
                ).astype(np.float32)
                omin, omax = orig_arr.min(), orig_arr.max()
                if omax > omin:
                    orig_norm = (orig_arr - omin) / (omax - omin)
                else:
                    orig_norm = np.zeros_like(orig_arr)

            for mode in modes:
                if not self.overwrite and out_paths[mode].exists():
                    continue

                if mode == "naive":
                    result_arr = naive_arr
                elif mode.startswith("blend_") and have_orig:
                    alpha = int(mode.split("_", 1)[1]) / 100.0
                    result_arr = alpha * naive_arr + (1 - alpha) * orig_norm
                else:
                    result_arr = naive_arr

                result_img = sitk.GetImageFromArray(result_arr)
                result_img.CopyInformation(ref_img)
                sitk.WriteImage(result_img, str(out_paths[mode]))

            return t, False  # processed

        # -- dispatch ----------------------------------------------------------
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(_process_timepoint, t): t for t in range(num_tp)
            }
            for future in as_completed(futures):
                t, skipped = future.result()
                if skipped:
                    plog.info(f"  t={t:02d} – all mag modes exist, skipping")
                else:
                    plog.info(f"  t={t:02d} – saved all mag modes")

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
            comp_arrays = []
            for comp in ("vx", "vy", "vz"):
                corr_path = raw_dir / f"pred_correction_{comp}_t{t:02d}.nii.gz"
                corr_nib = nib.load(corr_path)
                corr_data = corr_nib.get_fdata(dtype=np.float32)  # (X, Y, Z)
                # Transpose to (Z, Y, X) to match _fit_polynomial_coefficients
                comp_arrays.append(np.transpose(corr_data, (2, 1, 0)))  # (Z, Y, X)
            corrections_list.append(np.stack(comp_arrays, axis=0))  # (3, Z, Y, X)

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

        # ---- (e-f) de-normalise and apply to uncorrected velocity (parallel) --
        plog.info("  3e-f: applying corrections to uncorrected velocity …")

        comp_dirs = {
            "vx": patient.flow_vx_per_timepoint_corr_fov_dir,
            "vy": patient.flow_vy_per_timepoint_corr_fov_dir,
            "vz": patient.flow_vz_per_timepoint_corr_fov_dir,
        }

        # Pre-compute corrections in physical units for all components
        corrections_zyx: dict[str, np.ndarray] = {}
        for comp in ("vx", "vy", "vz"):
            correction_physical = gt_dict[comp] * venc  # (X, Y, Z)
            corrections_zyx[comp] = np.transpose(correction_physical, (2, 1, 0))
            (vel_dir / f"4d_flow_{comp}_corr").mkdir(parents=True, exist_ok=True)

        def _apply_correction(comp: str, t: int) -> tuple[str, int, str]:
            out_path = (
                vel_dir / f"4d_flow_{comp}_corr"
                / f"4d_flow_{comp}_corr_{pid}_frame_{t:02d}.nii.gz"
            )
            if not self.overwrite and out_path.exists():
                return comp, t, "skip"

            uncorr_path = (
                comp_dirs[comp]
                / f"4d_flow_{comp}_{pid}_frame_{t:02d}.nii.gz"
            )
            if not uncorr_path.exists():
                return comp, t, "missing"

            uncorr_img = sitk.ReadImage(str(uncorr_path))
            uncorr_arr = sitk.GetArrayFromImage(uncorr_img).astype(
                np.float32
            )  # (Z, Y, X)

            corrected_arr = uncorr_arr + corrections_zyx[comp]

            corrected_img = sitk.GetImageFromArray(corrected_arr)
            corrected_img.CopyInformation(uncorr_img)
            sitk.WriteImage(corrected_img, str(out_path))
            return comp, t, "done"

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(_apply_correction, comp, t)
                for comp in ("vx", "vy", "vz")
                for t in range(num_tp)
            ]
            for future in as_completed(futures):
                comp, t, status = future.result()
                if status == "missing":
                    plog.warning(
                        f"  uncorrected {comp} t={t:02d} not found, skipping"
                    )

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
        patient_dir: Path,
        plog: logging.Logger,
    ) -> None:
        """Write predicted mag + corrected velocity to DICOM series.

        Produces one DICOM set per magnitude mode (naive + one per blend alpha),
        each under ``dicoms/{mode}/``.  All (mode, timepoint) pairs are
        dispatched to a flat thread pool for maximum I/O parallelism.
        """
        import zipfile

        from pydicom.uid import generate_uid

        pid = patient.identifier
        num_tp = patient.num_timepoints
        modes = self._mag_modes()

        mag_descriptions: dict[str, str] = {"naive": "Raw Mag Prediction"}
        for alpha in self.mag_blend_alphas:
            pct = int(round(alpha * 100))
            mag_descriptions[f"blend_{pct}"] = (
                f"Blended Mag ({pct}/{100 - pct})"
            )

        vel_descriptions: dict[int, str] = {
            3: "Corrected Vx",
            4: "Corrected Vy",
            5: "Corrected Vz",
        }

        # Pre-build per-mode resources (UIDs, dirs, converter)
        mode_ctx: dict[str, dict] = {}
        for mode in modes:
            mode_dicom_dir = dicom_dir / mode
            mode_dicom_dir.mkdir(parents=True, exist_ok=True)
            mode_ctx[mode] = {
                "dicom_dir": mode_dicom_dir,
                "mag_dir": mag_dir / mode,
                "converter": NiftiToDicomConverter.from_patient(patient),
                "study_uid": generate_uid(),
                "series_uids": {
                    2: generate_uid(),
                    3: generate_uid(),
                    4: generate_uid(),
                    5: generate_uid(),
                },
                "series_descs": {
                    2: mag_descriptions[mode],
                    **vel_descriptions,
                },
            }

        def _write_timepoint(mode: str, t: int) -> tuple[str, int]:
            ctx = mode_ctx[mode]
            mag_path = ctx["mag_dir"] / f"pred_mag_{pid}_frame_{t:02d}.nii.gz"
            if not mag_path.exists():
                return mode, t

            vx_path = vel_dir / "4d_flow_vx_corr" / f"4d_flow_vx_corr_{pid}_frame_{t:02d}.nii.gz"
            vy_path = vel_dir / "4d_flow_vy_corr" / f"4d_flow_vy_corr_{pid}_frame_{t:02d}.nii.gz"
            vz_path = vel_dir / "4d_flow_vz_corr" / f"4d_flow_vz_corr_{pid}_frame_{t:02d}.nii.gz"

            vel_paths = {}
            if vx_path.exists() and vy_path.exists() and vz_path.exists():
                vel_paths = {"vx": vx_path, "vy": vy_path, "vz": vz_path}

            ctx["converter"].write_timepoint_with_velocities_to_dicoms(
                mag_prediction_path=mag_path,
                velocity_paths=vel_paths,
                output_dir=ctx["dicom_dir"],
                timepoint=t,
                study_uid=ctx["study_uid"],
                series_uids=ctx["series_uids"],
                overwrite=self.overwrite,
                series_descriptions=ctx["series_descs"],
            )
            return mode, t

        # Dispatch all (mode, timepoint) pairs; slice-level parallelism is
        # handled inside write_timepoint_with_velocities_to_dicoms, so cap
        # the outer pool to avoid excessive thread nesting.
        outer_workers = min(self.max_workers or 6, 6)
        with ThreadPoolExecutor(max_workers=outer_workers) as pool:
            futures = [
                pool.submit(_write_timepoint, mode, t)
                for mode in modes
                for t in range(num_tp)
            ]
            for future in as_completed(futures):
                future.result()  # re-raises exceptions

        # Zip archives (fast, sequential)
        for mode in modes:
            ctx = mode_ctx[mode]
            zip_path = patient_dir / f"{pid}_dual_task_dicoms_{mode}.zip"
            if zip_path.exists():
                zip_path.unlink()
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in ctx["dicom_dir"].rglob("*"):
                    if fp.is_file():
                        zf.write(fp, arcname=fp.relative_to(ctx["dicom_dir"]))
            plog.info(f"  wrote DICOMs + zip ({mode})")

    # =====================================================================
    # Utility helpers
    # =====================================================================

    def _load_tensor(self, path: Path) -> torch.Tensor:
        """Load a 3-D NIfTI and return a ``[1, 1, X, Y, Z]`` float tensor on device."""
        nii = nib.load(path)
        arr = nii.get_fdata(dtype=np.float32)  # (X, Y, Z)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, X, Y, Z)
        return tensor.to(self.device)

    def _load_and_normalise_mag(
        self, ds_root: Path, pid: str, t_idx: int, num_tp: int
    ) -> torch.Tensor:
        """Load a single downsampled magnitude frame and min-max normalise to [0, 1]."""
        mag_path = (
            ds_root / "4d_flow_mag"
            / f"4d_flow_mag_{pid}_frame_{t_idx:02d}.nii.gz"
        )
        mag = self._load_tensor(mag_path)
        mag_min, mag_max = mag.min(), mag.max()
        if mag_max > mag_min:
            mag = (mag - mag_min) / (mag_max - mag_min)
        else:
            mag = torch.zeros_like(mag)
        return mag


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

        # Single patient (full pipeline)
        python -m vascular_superenhancement.inferencing.dual_task_inference \\
            inference.patient_id=Foxtrot

        # All test patients
        python -m vascular_superenhancement.inferencing.dual_task_inference \\
            inference.all_test_patients=true

        # NIfTI predictions only (skip DICOM generation)
        python -m vascular_superenhancement.inferencing.dual_task_inference \\
            inference.patient_id=Foxtrot inference.write_dicoms=false

        # Generate DICOMs from existing predictions (no model/GPU needed)
        python -m vascular_superenhancement.inferencing.dual_task_inference \\
            inference.patient_id=Foxtrot inference.dicoms_only=true
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
