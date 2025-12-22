"""Utility module for converting NIfTI predictions back to DICOM format.

This module handles resampling predictions from NIfTI space back to original
DICOM space, preserving metadata, data types, and compression.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import logging
import numpy as np
import pandas as pd
import pydicom
from pydicom.encaps import encapsulate
from pydicom import uid
from datetime import datetime
import SimpleITK as sitk

if TYPE_CHECKING:
    from .patients import Patient  # Only imported during type checking


class NiftiToDicomConverter:
    """
    Converter for writing NIfTI predictions back to DICOM format.
    
    The predictions are assumed to be 3D volumes in the original magnitude FOV space
    (same space as the original 4D flow magnitude NIfTI).
    
    Handles:
    - Resampling from NIfTI space (magnitude FOV) to DICOM space (LPS)
    - Mapping predictions to correct DICOM files by timepoint and slice
    - Replacing magnitude pixel data while preserving velocity data
    - Preserving metadata, data types, and compression
    """
    
    def __init__(
        self,
        catalog: pd.DataFrame,
        logger: logging.Logger,
        patient_id: str = "unknown",
        dataset_logger: Optional[logging.Logger] = None,
        mag_nifti_path: Optional[Path] = None,
    ) -> None:
        """Initialize the converter.
        
        Args:
            catalog: 4D Flow DICOM catalog DataFrame
            logger: Logger instance
            patient_id: Patient identifier
            dataset_logger: Optional dataset-level logger
            mag_nifti_path: Path to original 4D flow magnitude NIfTI (for coordinate space reference)
        """
        self.catalog = catalog
        self.logger = logger
        self.patient_id = patient_id
        self.dataset_logger = dataset_logger
        self.mag_nifti_path = mag_nifti_path
        
    @classmethod
    def from_patient(cls, patient: "Patient") -> "NiftiToDicomConverter":
        """
        Build a converter using patient.dicom_catalog_4d_flow and patient._logger.
        Automatically sets the magnitude NIfTI path for coordinate space reference.
        """
        from .patients import Patient  # Imported at runtime when method is called
        if not isinstance(patient, Patient):
            raise TypeError(f"Expected Patient instance, got {type(patient)}")
        
        # Get path to original 4D flow magnitude NIfTI for coordinate space reference
        mag_nifti_path = patient.nifti_dir / f"4d_flow_mag_{patient.identifier}.nii.gz"
        if not mag_nifti_path.exists():
            mag_nifti_path = None
            patient._logger.warning(
                f"Original 4D flow magnitude NIfTI not found at {mag_nifti_path}. "
                f"Coordinate space mapping may be less accurate."
            )
            
        return cls(
            catalog=patient.dicom_catalog_4d_flow,
            logger=patient._logger,
            patient_id=patient.identifier,
            dataset_logger=patient._dataset_logger if hasattr(patient, '_dataset_logger') else None,
            mag_nifti_path=mag_nifti_path
        )
    
    def _resample_3d_prediction_to_dicom_space(
        self,
        prediction_nifti: Path,
        dicom_filepaths: list[Path],
    ) -> np.ndarray:
        """
        Resample entire 3D prediction from NIfTI space (RAS) to DICOM space (LPS).
        
        Uses SimpleITK's ImageSeriesReader (same approach as dicom_to_nifti._load_series) to load
        all DICOMs into a 3D volume, which automatically computes proper spacing/origin/direction.
        
        Args:
            prediction_nifti: Path to prediction NIfTI file (3D volume in RAS)
            dicom_filepaths: List of DICOM file paths for this timepoint (sorted by slice)
            
        Returns:
            3D numpy array in DICOM space [Z, Y, X] (SimpleITK array format)
        """
        # Load prediction (3D volume in RAS space)
        pred_img = sitk.ReadImage(str(prediction_nifti))
        
        # Load all DICOMs for this timepoint into a 3D volume (LPS space)
        # This matches the approach in dicom_to_nifti._load_series
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(fp) for fp in dicom_filepaths])
        dicom_3d = reader.Execute()  # 3D volume in LPS space with proper spacing/origin/direction
        
        # Resample prediction to match DICOM 3D space
        # SimpleITK automatically handles RAS (prediction) -> LPS (DICOM) conversion
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(dicom_3d)  # Reference is in LPS
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0.0)
        
        resampled_3d = resampler.Execute(pred_img)  # Resampled from RAS to LPS
        
        # Convert to numpy array [Z, Y, X] format (same as dicom_to_nifti._load_series)
        resampled_array = sitk.GetArrayFromImage(resampled_3d)
        
        return resampled_array
    
    def write_predictions_to_dicoms(
        self,
        prediction_dir: Path,
        output_dir: Path,
        timepoint: int,
        overwrite: bool = False,
    ) -> None:
        """
        Write predictions for a specific timepoint to DICOM files.
        
        Args:
            prediction_dir: Directory containing prediction NIfTI files
            output_dir: Directory to save modified DICOM files
            timepoint: Timepoint index (0-based)
            overwrite: Whether to overwrite existing files
        """
        # Find prediction file for this timepoint
        # Assuming naming convention: pred_<patient_id>_t<timepoint:02d>_*.nii.gz
        pred_pattern = f"*_t{timepoint:02d}_*.nii.gz"
        pred_files = list(prediction_dir.glob(pred_pattern))
        
        if not pred_files:
            self.logger.warning(
                f"No prediction file found for timepoint {timepoint} in {prediction_dir}"
            )
            return
        
        if len(pred_files) > 1:
            self.logger.warning(
                f"Multiple prediction files found for timepoint {timepoint}, using first: {pred_files[0]}"
            )
        
        prediction_file = pred_files[0]
        self.logger.info(f"Using prediction file: {prediction_file}")
        
        # Filter catalog for this timepoint - get both magnitude and velocity files
        catalog_tp_all = self.catalog[
            (self.catalog['time_index'] == timepoint) &
            (self.catalog['tag_0x0043_0x1030'].isin([2, 3, 4, 5]))  # Magnitude (2), Vx (3), Vy (4), Vz (5)
        ].copy()
        
        if len(catalog_tp_all) == 0:
            self.logger.warning(
                f"No 4D flow DICOMs found for timepoint {timepoint}"
            )
            return
        
        # Separate magnitude and velocity files
        catalog_tp_mag = catalog_tp_all[catalog_tp_all['tag_0x0043_0x1030'] == 2].copy()
        catalog_tp_vel = catalog_tp_all[catalog_tp_all['tag_0x0043_0x1030'].isin([3, 4, 5])].copy()
        
        if len(catalog_tp_mag) == 0:
            self.logger.warning(
                f"No magnitude DICOMs found for timepoint {timepoint}"
            )
            return
        
        # Sort by slice index (same as dicom_to_nifti._load_series)
        catalog_tp_mag = catalog_tp_mag.sort_values('slice_index').reset_index(drop=True)
        dicom_filepaths = [Path(row['filepath']) for _, row in catalog_tp_mag.iterrows()]
        
        self.logger.info(
            f"Processing {len(catalog_tp_mag)} magnitude DICOMs and {len(catalog_tp_vel)} velocity DICOMs for timepoint {timepoint}"
        )
        
        # Resample entire 3D prediction to DICOM space using ImageSeriesReader
        # (same approach as dicom_to_nifti._load_series)
        resampled_3d = self._resample_3d_prediction_to_dicom_space(
            prediction_file,
            dicom_filepaths
        )  # Shape: [Z, Y, X]
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a new SeriesInstanceUID for all files in this series (magnitude + velocity)
        # This ensures the new series doesn't conflict with the original
        new_series_instance_uid = uid.generate_uid()
        
        # Get current date/time for ContentDate/Time and SeriesDate/Time
        current_datetime = datetime.now()
        content_date = current_datetime.strftime('%Y%m%d')
        content_time = current_datetime.strftime('%H%M%S.%f')[:-3]  # Include milliseconds
        
        # Helper function to update DICOM metadata
        def update_dicom_metadata(ds, is_velocity=False):
            """Update DICOM metadata for both magnitude and velocity files."""
            # Generate new SeriesInstanceUID (same for all files in series)
            ds.SeriesInstanceUID = new_series_instance_uid
            
            # Generate new SOPInstanceUID (unique for each file)
            ds.SOPInstanceUID = uid.generate_uid()
            
            # Update SeriesNumber (use a high number to avoid conflicts)
            original_series_number = getattr(ds, 'SeriesNumber', 1)
            if isinstance(original_series_number, (int, str)):
                try:
                    original_num = int(original_series_number)
                    # Use 9000+ range to clearly distinguish from original series
                    ds.SeriesNumber = 9000 + (original_num % 100)
                except (ValueError, TypeError):
                    ds.SeriesNumber = 9000
            else:
                ds.SeriesNumber = 9000
            
            # Update SeriesDescription to identify as VSE prediction (only for magnitude)
            if not is_velocity:
                original_description = getattr(ds, 'SeriesDescription', '')
                if original_description:
                    ds.SeriesDescription = f"{original_description} VSE"
                else:
                    ds.SeriesDescription = "Magnitude VSE Prediction"
            # For velocity files, keep original SeriesDescription unchanged
            
            # Update date/time tags
            ds.ContentDate = content_date
            ds.ContentTime = content_time
            if hasattr(ds, 'SeriesDate'):
                ds.SeriesDate = content_date
            if hasattr(ds, 'SeriesTime'):
                ds.SeriesTime = content_time
        
        # Process magnitude files (with prediction data)
        for idx, row in catalog_tp_mag.iterrows():
            dicom_path = Path(row['filepath'])
            slice_idx = row['slice_index']
            
            if not dicom_path.exists():
                self.logger.warning(f"DICOM file not found: {dicom_path}")
                continue
            
            # Create output path with "_vse" suffix
            dicom_name = dicom_path.stem
            dicom_ext = dicom_path.suffix
            output_path = output_dir / f"{dicom_name}_vse{dicom_ext}"
            
            if output_path.exists() and not overwrite:
                self.logger.debug(f"Skipping existing file: {output_path}")
                continue
            
            try:
                # Load DICOM
                ds = pydicom.dcmread(dicom_path)
                
                # Decompress if necessary
                if hasattr(ds.file_meta, 'TransferSyntaxUID'):
                    if ds.file_meta.TransferSyntaxUID.is_compressed:
                        ds.decompress()
                
                # Extract the corresponding slice from resampled 3D volume
                if slice_idx >= resampled_3d.shape[0]:
                    self.logger.warning(
                        f"Slice index {slice_idx} >= volume depth {resampled_3d.shape[0]}, "
                        f"using last slice"
                    )
                    slice_idx = resampled_3d.shape[0] - 1
                
                resampled_pred = resampled_3d[slice_idx, :, :]  # Extract [Y, X] slice
                
                # Scale from normalized [0, 1] range back to uint16 [0, 65535]
                resampled_pred = resampled_pred * 65535.0
                resampled_pred = np.clip(resampled_pred, 0, 65535)
                resampled_pred_uint16 = resampled_pred.astype(np.uint16)
                
                # Update metadata
                update_dicom_metadata(ds, is_velocity=False)
                
                # Store original transfer syntax to preserve compression format
                original_transfer_syntax = None
                if hasattr(ds.file_meta, 'TransferSyntaxUID'):
                    original_transfer_syntax = ds.file_meta.TransferSyntaxUID
                
                # Replace pixel data
                if original_transfer_syntax and original_transfer_syntax.is_compressed:
                    if 'RLE' in str(original_transfer_syntax):
                        ds.PixelData = encapsulate(resampled_pred_uint16.tobytes())
                    else:
                        self.logger.warning(
                            f"Original DICOM uses {original_transfer_syntax} compression. "
                            f"Saving as uncompressed."
                        )
                        ds.PixelData = resampled_pred_uint16.tobytes()
                        ds.file_meta.TransferSyntaxUID = uid.ExplicitVRLittleEndian
                else:
                    ds.PixelData = resampled_pred_uint16.tobytes()
                
                # Update pixel data related tags for unsigned 16-bit integer
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0  # Unsigned integer (0-65535)
                
                # Save modified DICOM
                ds.save_as(output_path, write_like_original=False)
                
                self.logger.debug(
                    f"Saved modified DICOM: {output_path} (slice {slice_idx}, timepoint {timepoint})"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Error processing DICOM {dicom_path}: {str(e)}",
                    exc_info=True
                )
                if self.dataset_logger:
                    self.dataset_logger.error(
                        f"Error processing DICOM for patient {self.patient_id}: {str(e)}"
                    )
                continue
        
        # Process velocity files (copy unchanged, just update metadata)
        for idx, row in catalog_tp_vel.iterrows():
            dicom_path = Path(row['filepath'])
            
            if not dicom_path.exists():
                self.logger.warning(f"DICOM file not found: {dicom_path}")
                continue
            
            # Create output path with original filename (no "_vse" suffix for velocity)
            output_path = output_dir / dicom_path.name
            
            if output_path.exists() and not overwrite:
                self.logger.debug(f"Skipping existing file: {output_path}")
                continue
            
            try:
                # Load DICOM
                ds = pydicom.dcmread(dicom_path)
                
                # Decompress if necessary
                if hasattr(ds.file_meta, 'TransferSyntaxUID'):
                    if ds.file_meta.TransferSyntaxUID.is_compressed:
                        ds.decompress()
                
                # Update metadata (same SeriesInstanceUID as magnitude files, but keep original description)
                update_dicom_metadata(ds, is_velocity=True)
                
                # Pixel data stays unchanged for velocity files
                # Just save with updated metadata
                ds.save_as(output_path, write_like_original=False)
                
                self.logger.debug(
                    f"Saved velocity DICOM: {output_path} (timepoint {timepoint})"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Error processing velocity DICOM {dicom_path}: {str(e)}",
                    exc_info=True
                )
                if self.dataset_logger:
                    self.dataset_logger.error(
                        f"Error processing velocity DICOM for patient {self.patient_id}: {str(e)}"
                    )
                continue
        
        self.logger.info(
            f"Completed writing predictions to DICOMs for timepoint {timepoint}"
        )
    
    def write_all_timepoints_to_dicoms(
        self,
        prediction_dir: Path,
        output_dir: Path,
        num_timepoints: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Write predictions for all timepoints to DICOM files.
        
        Args:
            prediction_dir: Directory containing prediction NIfTI files
            output_dir: Base directory to save modified DICOM files
            num_timepoints: Number of timepoints (if None, inferred from catalog)
            overwrite: Whether to overwrite existing files
        """
        if num_timepoints is None:
            num_timepoints = int(self.catalog['time_index'].max() + 1)
        
        self.logger.info(
            f"Writing predictions for {num_timepoints} timepoints to DICOMs"
        )
        
        for timepoint in range(num_timepoints):
            self.logger.info(f"Processing timepoint {timepoint}/{num_timepoints-1}")
            self.write_predictions_to_dicoms(
                prediction_dir=prediction_dir,
                output_dir=output_dir,
                timepoint=timepoint,
                overwrite=overwrite,
            )
        
        self.logger.info("Completed writing all predictions to DICOMs")


