"""Utility module for converting NIfTI predictions back to DICOM format.

This module handles resampling predictions from NIfTI space back to original
DICOM space, preserving metadata and creating properly formatted DICOM series.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import logging
import zipfile
import numpy as np
import pandas as pd
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from datetime import datetime
import SimpleITK as sitk

if TYPE_CHECKING:
    from .patients import Patient


class NiftiToDicomConverter:
    """
    Converter for writing NIfTI predictions back to DICOM format.
    
    Handles:
    - Resampling from NIfTI space to DICOM space
    - Percentile-based intensity mapping
    - Processing all 4 DICOM components (magnitude + 3 velocities)
    - Generating new UIDs and updating metadata
    - Creating zip archives
    """
    
    def __init__(
        self,
        catalog: pd.DataFrame,
        logger: logging.Logger,
        patient_id: str = "unknown",
        dataset_logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the converter.
        
        Args:
            catalog: 4D Flow DICOM catalog DataFrame
            logger: Logger instance
            patient_id: Patient identifier
            dataset_logger: Optional dataset-level logger
        """
        self.catalog = catalog
        self.logger = logger
        self.patient_id = patient_id
        self.dataset_logger = dataset_logger
        
    @classmethod
    def from_patient(cls, patient: "Patient") -> "NiftiToDicomConverter":
        """Build a converter using patient's DICOM catalog and logger."""
        from .patients import Patient
        if not isinstance(patient, Patient):
            raise TypeError(f"Expected Patient instance, got {type(patient)}")
            
        return cls(
            catalog=patient.dicom_catalog_4d_flow,
            logger=patient._logger,
            patient_id=patient.identifier,
            dataset_logger=patient._dataset_logger if hasattr(patient, '_dataset_logger') else None,
        )
    
    def _resample_prediction_to_dicom_space(
        self,
        prediction_path: Path,
        dicom_filepaths: list[Path],
    ) -> np.ndarray:
        """
        Resample 3D prediction from NIfTI space to DICOM space.
        
        Uses SimpleITK ImageSeriesReader to load DICOMs as reference volume,
        which automatically handles coordinate system transformations.
        
        Args:
            prediction_path: Path to prediction NIfTI file
            dicom_filepaths: List of DICOM file paths for this timepoint (sorted by slice)
            
        Returns:
            3D numpy array in DICOM space [Z, Y, X]
        """
        # Load prediction NIfTI
        prediction_img = sitk.ReadImage(str(prediction_path))
        
        # Load DICOMs as 3D reference volume
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(fp) for fp in dicom_filepaths])
        dicom_3d = reader.Execute()
        
        # Resample prediction to match DICOM space
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(dicom_3d)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0.0)
        
        resampled_img = resampler.Execute(prediction_img)
        
        # Convert to numpy array [Z, Y, X]
        return sitk.GetArrayFromImage(resampled_img)
    
    def _map_intensity_to_dicom_range(
        self,
        prediction_array: np.ndarray,
        dicom_array: np.ndarray,
    ) -> np.ndarray:
        """
        Map prediction intensities to match DICOM intensity range using percentiles.
        
        Args:
            prediction_array: Prediction array to map
            dicom_array: Reference DICOM array for intensity range
            
        Returns:
            Mapped array as int16
        """
        # Get intensity ranges using percentiles
        dicom_low, dicom_high = np.percentile(dicom_array, [1, 99.9])
        self.logger.info(f"DICOM intensity range: {dicom_low}, {dicom_high}")
        # dicom_low, dicom_high = 0, 5000
        self.logger.info(f"Using fixed DICOM intensity range: {dicom_low}, {dicom_high}")
        pred_low, pred_high = np.percentile(prediction_array, [1, 99.9])
        self.logger.info(f"Prediction intensity range: {pred_low}, {pred_high}")
        
        # Clip and normalize prediction
        pred_clipped = np.clip(prediction_array, pred_low, pred_high)
        normalized = (pred_clipped - pred_low) / (pred_high - pred_low)
        
        # Map to DICOM range
        mapped = normalized * (dicom_high - dicom_low) + dicom_low
        
        return np.rint(mapped).astype(np.int16)
    
    def _update_dicom_metadata(
        self,
        ds: pydicom.Dataset,
        study_uid: str,
        series_uid: str,
        content_date: str,
        content_time: str,
        series_number_offset: int = 0,
        series_description: Optional[str] = None,
    ) -> None:
        """
        Update DICOM metadata in-place.
        
        Args:
            ds: DICOM dataset to update
            study_uid: New StudyInstanceUID
            series_uid: New SeriesInstanceUID
            content_date: Content date string (YYYYMMDD)
            content_time: Content time string (HHMMSS.fff)
            series_number_offset: Offset to add to series number
            series_description: If provided, overwrite SeriesDescription with
                this value.  If ``None``, the original description is kept.
        """
        # Update ImageType to indicate derived data
        ds.ImageType = r"DERIVED\SECONDARY"
        
        # Update UIDs
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = generate_uid()
        
        # Update SeriesNumber
        original_series = getattr(ds, 'SeriesNumber', 1)
        ds.SeriesNumber = 9000 + (int(original_series) % 100) + series_number_offset
        
        # Update SeriesDescription
        if series_description is not None:
            ds.SeriesDescription = series_description
        
        # Update PatientID
        ds.PatientID = self.patient_id
        
        # Update date/time
        ds.ContentDate = content_date
        ds.ContentTime = content_time
    
    def write_timepoint_to_dicoms(
        self,
        prediction_path: Path,
        output_dir: Path,
        timepoint: int,
        study_uid: str,
        series_uids: dict[int, str],
        overwrite: bool = False,
    ) -> None:
        """
        Write prediction for one timepoint to DICOM files.
        
        Args:
            prediction_path: Path to prediction NIfTI file
            output_dir: Directory to save DICOM files
            timepoint: Timepoint index
            study_uid: StudyInstanceUID for all files
            series_uids: Dict mapping component codes (2,3,4,5) to SeriesInstanceUIDs
            overwrite: Whether to overwrite existing files
        """
        # Get catalog entries for this timepoint (all components)
        catalog_tp = self.catalog[
            (self.catalog['time_index'] == timepoint) &
            (self.catalog['tag_0x0043_0x1030'].isin([2, 3, 4, 5]))
        ].copy()
        
        if len(catalog_tp) == 0:
            self.logger.warning(f"No DICOM entries found for timepoint {timepoint}")
            return
        
        # CRITICAL FIX: Sort by Z coordinate (same as dicom_to_nifti.py does)
        # This ensures slice ordering matches the NIfTI creation process
        catalog_tp['ipp'] = catalog_tp['imagepositionpatient'].apply(lambda x: np.array(eval(x)))
        catalog_tp['z'] = catalog_tp['ipp'].apply(lambda x: x[2])
        catalog_tp = catalog_tp.sort_values('z', ascending=True).reset_index(drop=True)  # Inferior → Superior

        
        # Separate by component
        catalog_mag = catalog_tp[catalog_tp['tag_0x0043_0x1030'] == 2].copy()
        catalog_vx = catalog_tp[catalog_tp['tag_0x0043_0x1030'] == 3].copy()
        catalog_vy = catalog_tp[catalog_tp['tag_0x0043_0x1030'] == 4].copy()
        catalog_vz = catalog_tp[catalog_tp['tag_0x0043_0x1030'] == 5].copy()
        
        # Get filepaths
        mag_paths = [Path(row['filepath']) for _, row in catalog_mag.iterrows()]
        vx_paths = [Path(row['filepath']) for _, row in catalog_vx.iterrows()]
        vy_paths = [Path(row['filepath']) for _, row in catalog_vy.iterrows()]
        vz_paths = [Path(row['filepath']) for _, row in catalog_vz.iterrows()]
        
        # Resample prediction to DICOM space
        resampled_pred = self._resample_prediction_to_dicom_space(prediction_path, mag_paths)
        
        # Load original DICOM data for intensity mapping
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(fp) for fp in mag_paths])
        dicom_3d = reader.Execute()
        dicom_array = sitk.GetArrayFromImage(dicom_3d)
        
        # Map prediction to DICOM intensity range
        mapped_pred = self._map_intensity_to_dicom_range(resampled_pred, dicom_array)
        
        # Get current date/time
        now = datetime.now()
        content_date = now.strftime('%Y%m%d')
        content_time = now.strftime('%H%M%S.%f')[:-3]
        
        # Process each slice
        for z, (mag_path, vx_path, vy_path, vz_path) in enumerate(zip(mag_paths, vx_paths, vy_paths, vz_paths)):
            # Read all 4 DICOMs for this slice
            dcm_mag = pydicom.dcmread(mag_path)
            dcm_vx = pydicom.dcmread(vx_path)
            dcm_vy = pydicom.dcmread(vy_path)
            dcm_vz = pydicom.dcmread(vz_path)
            
            # Decompress magnitude if needed
            if dcm_mag.file_meta.TransferSyntaxUID.is_compressed:
                dcm_mag.decompress()
                dcm_mag.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                # dcm_mag.is_little_endian = True
            
            # Replace magnitude pixel data
            slice_data = mapped_pred[z, :, :]
            assert slice_data.shape == (dcm_mag.Rows, dcm_mag.Columns), \
                f"Shape mismatch: {slice_data.shape} != ({dcm_mag.Rows}, {dcm_mag.Columns})"
            dcm_mag.PixelData = slice_data.tobytes()
            
            # Update metadata for all 4 components
            original_mag_desc = getattr(dcm_mag, 'SeriesDescription', '')
            self._update_dicom_metadata(dcm_mag, study_uid, series_uids[2], content_date, content_time, 0, f"{original_mag_desc} VSE")
            self._update_dicom_metadata(dcm_vx, study_uid, series_uids[3], content_date, content_time, 1)
            self._update_dicom_metadata(dcm_vy, study_uid, series_uids[4], content_date, content_time, 2)
            self._update_dicom_metadata(dcm_vz, study_uid, series_uids[5], content_date, content_time, 3)
            
            # Save all 4 DICOMs
            output_dir.mkdir(parents=True, exist_ok=True)
            dcm_mag.save_as(output_dir / f"{mag_path.stem}_vse.dcm", enforce_file_format=False)
            dcm_vx.save_as(output_dir / f"{vx_path.stem}_v3.dcm", enforce_file_format=False)
            dcm_vy.save_as(output_dir / f"{vy_path.stem}_v4.dcm", enforce_file_format=False)
            dcm_vz.save_as(output_dir / f"{vz_path.stem}_v5.dcm", enforce_file_format=False)
        
        self.logger.info(f"Processed timepoint {timepoint}: {len(mag_paths)} slices")
    
    def _map_velocity_to_dicom_range(
        self,
        prediction_array: np.ndarray,
        dicom_array: np.ndarray,
    ) -> np.ndarray:
        """Map predicted velocity intensities to match DICOM velocity pixel range.

        Uses a linear mapping that preserves the zero point, which is important
        for signed velocity data.  The scale is derived from the 1st / 99th
        percentiles of the original DICOM and the prediction.

        Args:
            prediction_array: Predicted velocity array (physical units).
            dicom_array: Reference DICOM pixel array for this velocity component.

        Returns:
            Mapped array as int16.
        """
        dicom_low, dicom_high = np.percentile(dicom_array, [1, 99])
        pred_low, pred_high = np.percentile(prediction_array, [1, 99])

        dicom_range = dicom_high - dicom_low
        pred_range = pred_high - pred_low

        if pred_range < 1e-9:
            self.logger.warning("Predicted velocity has near-zero range; returning zeros")
            return np.zeros_like(prediction_array, dtype=np.int16)

        # Scale so that [pred_low, pred_high] → [dicom_low, dicom_high]
        scale = dicom_range / pred_range
        mapped = (prediction_array - pred_low) * scale + dicom_low

        return np.rint(mapped).astype(np.int16)

    def write_timepoint_with_velocities_to_dicoms(
        self,
        mag_prediction_path: Path,
        velocity_paths: dict[str, Path],
        output_dir: Path,
        timepoint: int,
        study_uid: str,
        series_uids: dict[int, str],
        overwrite: bool = False,
        series_descriptions: Optional[dict[int, str]] = None,
    ) -> None:
        """Write predicted magnitude AND corrected-velocity DICOMs for one timepoint.

        This is the dual-task variant of :meth:`write_timepoint_to_dicoms`.
        In addition to replacing the magnitude pixel data, it also replaces the
        velocity (vx, vy, vz) pixel data with the model-predicted corrected
        velocities.

        Args:
            mag_prediction_path: Path to predicted-magnitude NIfTI.
            velocity_paths: Dict with keys ``'vx'``, ``'vy'``, ``'vz'`` mapping
                to the predicted corrected-velocity NIfTI paths.  If empty, the
                velocity DICOMs are written with metadata changes only (no pixel
                data modification).
            output_dir: Directory to save DICOM files.
            timepoint: Timepoint index.
            study_uid: StudyInstanceUID for all files.
            series_uids: Dict mapping component codes (2,3,4,5) to SeriesInstanceUIDs.
            overwrite: Whether to overwrite existing files.
            series_descriptions: Optional dict mapping component codes
                (2, 3, 4, 5) to SeriesDescription strings.  If ``None``,
                original descriptions are preserved.
        """
        # Get catalog entries for this timepoint (all 4 components)
        catalog_tp = self.catalog[
            (self.catalog['time_index'] == timepoint) &
            (self.catalog['tag_0x0043_0x1030'].isin([2, 3, 4, 5]))
        ].copy()

        if len(catalog_tp) == 0:
            self.logger.warning(f"No DICOM entries found for timepoint {timepoint}")
            return

        # Sort by Z coordinate (same as dicom_to_nifti.py)
        catalog_tp['ipp'] = catalog_tp['imagepositionpatient'].apply(lambda x: np.array(eval(x)))
        catalog_tp['z'] = catalog_tp['ipp'].apply(lambda x: x[2])
        catalog_tp = catalog_tp.sort_values('z', ascending=True).reset_index(drop=True)

        # Separate by component
        catalog_mag = catalog_tp[catalog_tp['tag_0x0043_0x1030'] == 2].copy()
        catalog_vx  = catalog_tp[catalog_tp['tag_0x0043_0x1030'] == 3].copy()
        catalog_vy  = catalog_tp[catalog_tp['tag_0x0043_0x1030'] == 4].copy()
        catalog_vz  = catalog_tp[catalog_tp['tag_0x0043_0x1030'] == 5].copy()

        mag_paths = [Path(row['filepath']) for _, row in catalog_mag.iterrows()]
        vx_paths  = [Path(row['filepath']) for _, row in catalog_vx.iterrows()]
        vy_paths  = [Path(row['filepath']) for _, row in catalog_vy.iterrows()]
        vz_paths  = [Path(row['filepath']) for _, row in catalog_vz.iterrows()]

        # ---- Magnitude -------------------------------------------------------
        resampled_mag = self._resample_prediction_to_dicom_space(mag_prediction_path, mag_paths)

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(fp) for fp in mag_paths])
        dicom_mag_3d = reader.Execute()
        dicom_mag_array = sitk.GetArrayFromImage(dicom_mag_3d)

        mapped_mag = self._map_intensity_to_dicom_range(resampled_mag, dicom_mag_array)

        # ---- Velocities (optional) ------------------------------------------
        vel_comp_map = {3: 'vx', 4: 'vy', 5: 'vz'}
        vel_dicom_paths = {3: vx_paths, 4: vy_paths, 5: vz_paths}
        mapped_vels: dict[int, np.ndarray | None] = {3: None, 4: None, 5: None}

        if velocity_paths:
            for code, comp in vel_comp_map.items():
                pred_path = velocity_paths.get(comp)
                if pred_path is None or not pred_path.exists():
                    continue

                dicom_paths_for_comp = vel_dicom_paths[code]
                resampled_vel = self._resample_prediction_to_dicom_space(
                    pred_path, dicom_paths_for_comp
                )

                # Load original DICOM velocity data for intensity mapping
                vel_reader = sitk.ImageSeriesReader()
                vel_reader.SetFileNames([str(fp) for fp in dicom_paths_for_comp])
                dicom_vel_3d = vel_reader.Execute()
                dicom_vel_array = sitk.GetArrayFromImage(dicom_vel_3d)

                mapped_vels[code] = self._map_velocity_to_dicom_range(
                    resampled_vel, dicom_vel_array
                )

        # ---- Write slices (parallel) -----------------------------------------
        now = datetime.now()
        content_date = now.strftime('%Y%m%d')
        content_time = now.strftime('%H%M%S.%f')[:-3]
        descs = series_descriptions or {}

        output_dir.mkdir(parents=True, exist_ok=True)

        def _write_slice(z: int, mag_p: Path, vx_p: Path, vy_p: Path, vz_p: Path) -> None:
            dcm_mag = pydicom.dcmread(mag_p)
            dcm_vx  = pydicom.dcmread(vx_p)
            dcm_vy  = pydicom.dcmread(vy_p)
            dcm_vz  = pydicom.dcmread(vz_p)

            if dcm_mag.file_meta.TransferSyntaxUID.is_compressed:
                dcm_mag.decompress()
                dcm_mag.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            mag_slice = mapped_mag[z, :, :]
            assert mag_slice.shape == (dcm_mag.Rows, dcm_mag.Columns), \
                f"Mag shape mismatch: {mag_slice.shape} != ({dcm_mag.Rows}, {dcm_mag.Columns})"
            dcm_mag.PixelData = mag_slice.tobytes()

            vel_dcms = {3: dcm_vx, 4: dcm_vy, 5: dcm_vz}
            for code, dcm_vel in vel_dcms.items():
                if mapped_vels[code] is not None:
                    if dcm_vel.file_meta.TransferSyntaxUID.is_compressed:
                        dcm_vel.decompress()
                        dcm_vel.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    vel_slice = mapped_vels[code][z, :, :]
                    assert vel_slice.shape == (dcm_vel.Rows, dcm_vel.Columns), \
                        f"Vel shape mismatch for code {code}"
                    dcm_vel.PixelData = vel_slice.tobytes()

            self._update_dicom_metadata(dcm_mag, study_uid, series_uids[2], content_date, content_time, 0, descs.get(2))
            self._update_dicom_metadata(dcm_vx,  study_uid, series_uids[3], content_date, content_time, 1, descs.get(3))
            self._update_dicom_metadata(dcm_vy,  study_uid, series_uids[4], content_date, content_time, 2, descs.get(4))
            self._update_dicom_metadata(dcm_vz,  study_uid, series_uids[5], content_date, content_time, 3, descs.get(5))

            dcm_mag.save_as(output_dir / f"{mag_p.stem}_vse.dcm", enforce_file_format=False)
            dcm_vx.save_as(output_dir / f"{vx_p.stem}_pec_v3.dcm", enforce_file_format=False)
            dcm_vy.save_as(output_dir / f"{vy_p.stem}_pec_v4.dcm", enforce_file_format=False)
            dcm_vz.save_as(output_dir / f"{vz_p.stem}_pec_v5.dcm", enforce_file_format=False)

        with ThreadPoolExecutor() as pool:
            futures = [
                pool.submit(_write_slice, z, mag_p, vx_p, vy_p, vz_p)
                for z, (mag_p, vx_p, vy_p, vz_p)
                in enumerate(zip(mag_paths, vx_paths, vy_paths, vz_paths))
            ]
            for future in as_completed(futures):
                future.result()

        self.logger.info(f"Processed timepoint {timepoint}: {len(mag_paths)} slices (mag + velocity)")

    def write_all_timepoints_to_dicoms(
        self,
        prediction_dir: Path,
        output_dir: Path,
        num_timepoints: int = 20,
        create_zip: bool = True,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """
        Write all timepoint predictions to DICOM files.
        
        Args:
            prediction_dir: Directory containing prediction NIfTI files
            output_dir: Directory to save DICOM files
            num_timepoints: Number of timepoints to process
            create_zip: Whether to create a zip archive
            overwrite: Whether to overwrite existing files
            
        Returns:
            Path to zip file if created, None otherwise
        """
        # Generate UIDs once for all timepoints
        study_uid = generate_uid()
        series_uids = {
            2: generate_uid(),  # Magnitude
            3: generate_uid(),  # Vx
            4: generate_uid(),  # Vy
            5: generate_uid(),  # Vz
        }
        
        self.logger.info(f"Processing {num_timepoints} timepoints")
        
        # Process each timepoint
        for t in range(num_timepoints):
            # Find prediction file for this timepoint
            pred_pattern = f"*_t{t:02d}_*.nii.gz"
            pred_files = list(prediction_dir.glob(pred_pattern))
            
            if not pred_files:
                self.logger.warning(f"No prediction file found for timepoint {t}")
                continue
            
            if len(pred_files) > 1:
                self.logger.warning(f"Multiple files found for timepoint {t}, using first")
            
            prediction_path = pred_files[0]
            self.logger.info(f"Processing timepoint {t}/{num_timepoints-1}: {prediction_path.name}")
            
            self.write_timepoint_to_dicoms(
                prediction_path=prediction_path,
                output_dir=output_dir,
                timepoint=t,
                study_uid=study_uid,
                series_uids=series_uids,
                overwrite=overwrite,
            )
        
        # Create zip archive if requested
        zip_path = None
        if create_zip:
            zip_path = output_dir.parent / f"{self.patient_id}_dicom_predictions.zip"
            
            # Remove existing zip if present
            if zip_path.exists():
                zip_path.unlink()
            
            self.logger.info(f"Creating zip archive: {zip_path}")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file_path, arcname=arcname)
            
            self.logger.info(f"Created zip archive: {zip_path}")
        
        self.logger.info("Completed writing all predictions to DICOMs")
        return zip_path