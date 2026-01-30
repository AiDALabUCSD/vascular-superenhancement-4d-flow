"""Utility module for converting DICOM catalogs to NIfTI volumes.

Stateless: it does **not** depend on `Patient` at import‑time.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Union, Optional, TYPE_CHECKING
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
import SimpleITK as sitk
import warnings

if TYPE_CHECKING:
    from .patients import Patient  # Only imported during type checking
class DicomToNiftiConverter:
    """
    Core converter.  Accepts a DICOM *catalog* (DataFrame),
    an output directory, and a logger.

    Use `DicomToNiftiConverter.from_patient()` for a convenience factory
    that extracts those pieces from a `Patient` object.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        nifti_dir: Path,
        logger: logging.Logger,
        patient_id: str = "unknown",
        dataset_logger: Optional[logging.Logger] = None,
    ) -> None:
        """Store minimal data required for conversion."""
        self.catalog = catalog
        self.nifti_dir = nifti_dir
        self.logger = logger
        self.patient_id = patient_id
        self.dataset_logger = dataset_logger

    @classmethod
    def from_patient(cls, patient: "Patient") -> "DicomToNiftiConverter":
        """
        Build a converter using patient.dicom_catalog,
        patient.nifti_dir, and patient._logger.
        """
        from .patients import Patient  # Imported at runtime when method is called
        if not isinstance(patient, Patient):
            raise TypeError(f"Expected Patient instance, got {type(patient)}")
            
        return cls(
            catalog=patient.dicom_catalog,
            nifti_dir=patient.nifti_dir,
            logger=patient._logger,
            patient_id=patient.identifier,
            dataset_logger=patient._dataset_logger if hasattr(patient, '_dataset_logger') else None
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _compute_affine(
        self,
        ds0: pydicom.Dataset,
        ds1: Optional[pydicom.Dataset] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Return a 4 × 4 voxel‑to‑patient affine and the dot product for slice direction check.

        Priority for slice spacing Δs:
        1.  SpacingBetweenSlices (centre‑to‑centre) if > 0
        2.  Distance between the first two ImagePositionPatient vectors
        3.  SliceThickness
        4.  Fallback = 1 mm

        Also logs if slice order appears flipped w.r.t. orientation.
        """
        # ------------------------------------------------------------- #
        # 1.  Extract row/col direction vectors and in‑plane spacing
        # ------------------------------------------------------------- #
        try:
            row = np.asarray(ds0.ImageOrientationPatient[:3], dtype=float)
            col = np.asarray(ds0.ImageOrientationPatient[3:], dtype=float)
            Δr, Δc = map(float, ds0.PixelSpacing)
            origin = np.asarray(ds0.ImagePositionPatient, dtype=float)
        except AttributeError as err:
            self.logger.warning(
                f"[{self.patient_id}] Missing mandatory orientation/spacing tags "
                f"({err}); using identity affine."
            )
            return np.eye(4), 0.0

        # ------------------------------------------------------------- #
        # 2.  Determine slice spacing Δs
        # ------------------------------------------------------------- #
        Δs: float
        ipp1 = None
        if ds1 is not None:
            ipp1 = np.asarray(ds1.ImagePositionPatient, dtype=float)
            Δs = np.linalg.norm(ipp1 - origin)
            self.logger.debug(f"Computed Δs from ΔIPP = {Δs:.3f} mm")
        elif hasattr(ds0, "SpacingBetweenSlices") and float(ds0.SpacingBetweenSlices) > 0:
            Δs = float(ds0.SpacingBetweenSlices)
            self.logger.debug(f"Using SpacingBetweenSlices={Δs:.3f} mm")
        elif hasattr(ds0, "SliceThickness"):
            Δs = float(ds0.SliceThickness)
            self.logger.debug(f"Using SliceThickness={Δs:.3f} mm")
        else:
            Δs = 1.0
            self.logger.warning(
                f"[{self.patient_id}] No slice‑spacing tag present; defaulting to 1 mm."
            )

        # ------------------------------------------------------------- #
        # 3.  Build LPS affine columns
        # ------------------------------------------------------------- #
        # Compute slice normal using right-handed coordinate system
        # DICOM standard specifies row × col for the slice normal
        slice_vec = np.cross(row, col)
        
        # Log all components used to build the affine
        self.logger.debug(
            f"[{self.patient_id}] Affine components:\n"
            f"  Row vector: {row}\n"
            f"  Col vector: {col}\n"
            f"  Slice normal (row × col): {slice_vec}\n"
            f"  Pixel spacing: (Δr={Δr:.3f}, Δc={Δc:.3f})\n"
            f"  Slice spacing: Δs={Δs:.3f}\n"
            f"  Origin: {origin}"
        )

        affine = np.eye(4)
        affine[:3, 0] = row * Δr
        affine[:3, 1] = col * Δc
        affine[:3, 2] = slice_vec * Δs
        affine[:3, 3] = origin

        self.logger.debug(f"[{self.patient_id}] Initial LPS affine matrix:\n{affine}")
        
        # ------------------------------------------------------------- #
        # 4.  Build RAS affine columns
        # TODO(#1): Apparently i am flipping this affine to RAS and then
        # flipping the voxel data so theres an alleged mismatch between
        # the affine and the voxel data. However, the data loads correctly
        # in slicer relative to a direct dicom import. additionally, it
        # visually loads correctly when using sitk; however it doesnt seem
        # to be the case when using nib.load and using get_fdata(). Very 
        # confusing.
        # ------------------------------------------------------------- #
        flip = np.eye(4)
        flip[0, 0] = -1  # L → R
        flip[1, 1] = -1  # P → A
        affine_ras = flip @ affine
        
        self.logger.debug(f"[{self.patient_id}] Final RAS affine matrix:\n{affine_ras}")
        
        return affine_ras

    def _load_series(self, sub_catalog: pd.DataFrame) -> dict[str, np.ndarray]:
        
        sub_catalog = sub_catalog.copy()
        sub_catalog['ipp'] = sub_catalog['imagepositionpatient'].apply(lambda x: np.array(eval(x)))
        sub_catalog['z'] = sub_catalog['ipp'].apply(lambda x: x[2])
        sub_catalog['pixelspacing'] = sub_catalog['pixelspacing'].apply(lambda x: np.array(eval(x)))
        sub_catalog['imageorientation'] = sub_catalog['imageorientation'].apply(lambda x: np.array(eval(x)))
        
        # === Process each timepoint ===
        time_indices = sorted(sub_catalog['time_index'].unique())
        volume_list = []
        
        self.logger.info(f"Processing {len(time_indices)} timepoints")
        
        for t in time_indices:
            sub_catalog_t = sub_catalog[sub_catalog['time_index'] == t].copy()
            sub_catalog_t = sub_catalog_t.sort_values('z', ascending=True)  # Inferior → Superior
            filepaths = sub_catalog_t['filepath'].tolist()
            
            try:
                # Set up SimpleITK reader
                sitk.ProcessObject.SetGlobalWarningDisplay(True)
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(filepaths)
                
                # Also catch Python warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    image3d = reader.Execute()
                    
                    for warning in w:
                        self.logger.warning(f"[{self.patient_id}] Python warning while reading time_index={t}: {warning.message}")
                        if self.dataset_logger:
                            self.dataset_logger.warning(f"Python warning for patient {self.patient_id} at time_index={t}: {warning.message}")
                            
            except Exception as e:
                self.logger.error(f"Error reading DICOM series: {str(e)}")
                if self.dataset_logger:
                    self.dataset_logger.error(f"Error reading DICOM series for patient {self.patient_id}: {str(e)}")
                raise

            vol = sitk.GetArrayFromImage(image3d)  # shape: [Z, Y, X]
            volume_list.append(vol)
            
        # === Stack into 4D array ===
        # TODO(#1): Apparently i am flipping this affine to RAS and then
        # flipping the voxel data so theres an alleged mismatch between
        # the affine and the voxel data. However, the data loads correctly
        # in slicer relative to a direct dicom import. additionally, it
        # visually loads correctly when using sitk; however it doesnt seem
        # to be the case when using nib.load and using get_fdata(). Very 
        # confusing.
        arr4d = np.stack(volume_list, axis=-1)  # shape: [Z, Y, X, T]
        arr4d = np.transpose(arr4d, (2, 1, 0, 3))  # → [X, Y, Z, T]

        # === Compute affine ===
        sub_catalog_0 = sub_catalog[sub_catalog['time_index'] == 0].copy()
        sub_catalog_0 = sub_catalog_0.sort_values('z', ascending=True).reset_index(drop=True)
        
        dcm0 = pydicom.dcmread(sub_catalog_0.iloc[0]['filepath'])
        dcm1 = pydicom.dcmread(sub_catalog_0.iloc[1]['filepath'])

        affine = self._compute_affine(dcm0, dcm1)
        
        return {
            'data': arr4d,
            'affine': affine,
            'header': dcm0
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_3d_cine(
        self,
        *,
        save: bool = True,
        as_numpy: bool = False,
    ) -> Union[nib.Nifti1Image, np.ndarray]:
        """
        • Filter catalog to 3‑D cine (caller may already pass sub‑catalog).  
        • Build volume, create NIfTI, optionally save, and return.
        """
        self.logger.info(f"Building 3D cine volume for patient {self.patient_id}")
        
        # Load series data
        series_data = self._load_series(self.catalog)
        
        # Create NIfTI image
        nii = nib.Nifti1Image(series_data['data'], series_data['affine'])
        nii.set_qform(series_data['affine'], code=1)
        nii.set_sform(series_data['affine'], code=1)
        
        hdr = nii.header
        hdr['dim'][0] = 4
        hdr['dim'][4] = series_data['data'].shape[3]
        hdr['pixdim'][4] = 1.0  # or actual time spacing in seconds
        hdr['xyzt_units'] = 2 | 8  # space in mm, time in seconds
        
        if save:
            output_path = self.nifti_dir / f"3d_cine_{self.patient_id}.nii.gz"
            nib.save(nii, output_path)
            self.logger.info(f"Saved 3D cine NIfTI to {output_path}")
            if self.dataset_logger:
                self.dataset_logger.info(f"Saved 3D cine NIfTI to {output_path}")
                
        return sitk.GetImageFromArray(nii) if as_numpy else nii

    def build_4d_flow(
        self,
        *,
        save: bool = True,
        as_numpy: bool = False,
    ) -> Dict[str, Union[nib.Nifti1Image, np.ndarray]]:
        """
        • Split catalog into magnitude + velocity directions.  
        • Build a 4‑D volume for each component.  
        • Save each NIfTI (mag, vx, vy, vz) and return as dict.
        """
        self.logger.info(f"Building 4D flow volumes for patient {self.patient_id}")
        
        # Split catalog by flow encoding
        mag_catalog = self.catalog[self.catalog['tag_0x0043_0x1030'] == 2]
        vx_catalog = self.catalog[self.catalog['tag_0x0043_0x1030'] == 3]
        vy_catalog = self.catalog[self.catalog['tag_0x0043_0x1030'] == 4]
        vz_catalog = self.catalog[self.catalog['tag_0x0043_0x1030'] == 5]
        
        # Build each component
        components = {
            'mag': mag_catalog,
            'vx': vx_catalog,
            'vy': vy_catalog,
            'vz': vz_catalog
        }
        
        results = {}
        for comp, catalog in components.items():
            if len(catalog) == 0:
                self.logger.warning(f"No {comp} component found for patient {self.patient_id}")
                continue
                
            self.logger.debug(f"Building {comp} component with {len(catalog)} DICOMs")
            series_data = self._load_series(catalog)
            
            # Create NIfTI image
            nii = nib.Nifti1Image(series_data['data'], series_data['affine'])
            nii.set_qform(series_data['affine'], code=1)
            nii.set_sform(series_data['affine'], code=1)
            
            # Set up header with time information
            hdr = nii.header
            hdr['dim'][0] = 4
            hdr['dim'][4] = series_data['data'].shape[3]
            hdr['pixdim'][4] = 1.0  # or actual time spacing in seconds
            hdr['xyzt_units'] = 2 | 8  # space in mm, time in seconds
            
            if save:
                output_path = self.nifti_dir / f"4d_flow_{comp}_{self.patient_id}.nii.gz"
                nib.save(nii, output_path)
                self.logger.info(f"Saved {comp} NIfTI to {output_path}")
                if self.dataset_logger:
                    self.dataset_logger.info(f"Saved {comp} NIfTI to {output_path}")
            
            results[comp] = sitk.GetImageFromArray(nii) if as_numpy else nii
                    
        return results
    
    def build_resampled_per_timepoint(
        self,
        *,
        from_img_path: Path,
        to_reference_path: Path,
        output_dir: Path,
        name_prefix: str,
        mask_output_path: Optional[Path] = None,
    ) -> None:
        """
        Build per-timepoint volumes by resampling from source image to match reference FOV,
        then splitting into individual timepoints.
        
        Args:
            from_img_path: Path to the source 4D image to be resampled
            to_reference_path: Path to the reference image whose FOV to match (can be 3D or 4D)
            output_dir: Directory to save the per-timepoint volumes
            name_prefix: Prefix for the output filenames
            mask_output_path: If provided, saves a binary support mask indicating where
                             source data has coverage in the reference grid
        """
        # load the source and reference images
        if not from_img_path.exists():
            raise ValueError(f"Source image {from_img_path} does not exist")
        else:
            source_img = sitk.ReadImage(str(from_img_path))
        
        if not to_reference_path.exists():
            self.logger.warning(f"Reference image {to_reference_path} does not exist. Naive resampling from source to source will be performed.")
            reference_img = sitk.ReadImage(str(from_img_path))
        else:
            reference_img = sitk.ReadImage(str(to_reference_path))
        
        # log the dimensions of the source and reference images
        self.logger.info(f"Source image dimensions: {source_img.GetSize()}")
        self.logger.info(f"Reference image dimensions: {reference_img.GetSize()}")
        
        # If reference is 4D, extract first timepoint as 3D reference
        if len(reference_img.GetSize()) == 4:
            reference_img = reference_img[:,:,:,0]
            self.logger.info("Extracted first timepoint from 4D reference image")
        
        # Create support mask if requested
        if mask_output_path is not None:
            # Extract first timepoint from source for mask creation
            source_3d_for_mask = source_img[:,:,:,0] if len(source_img.GetSize()) == 4 else source_img
            
            # Create ones image in source space
            support = sitk.Image(source_3d_for_mask.GetSize(), sitk.sitkUInt8)
            support.CopyInformation(source_3d_for_mask)
            support = sitk.Add(support, 1)  # Fill with ones
            
            # Resample to reference using nearest neighbor
            mask_resampler = sitk.ResampleImageFilter()
            mask_resampler.SetReferenceImage(reference_img)
            mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask_resampler.SetTransform(sitk.Transform())
            mask_resampler.SetDefaultPixelValue(0)
            
            mask_img = mask_resampler.Execute(support)
            sitk.WriteImage(mask_img, str(mask_output_path))
            self.logger.info(f"Saved support mask to {mask_output_path}")
        
        # split the source image into 3D timepoints
        source_volumes = [source_img[:,:,:,t] for t in range(source_img.GetSize()[3])]
        self.logger.info(f"Loaded {len(source_volumes)} 3D timepoints from source")
        
        # set up sitk to resample the source volumes to the reference FOV
        resampled_volumes = []
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(sitk.Transform())

        # resample the source volumes to match reference FOV
        for i, source_3d in enumerate(source_volumes):
            resampled = resampler.Execute(source_3d)
            resampled_volumes.append(resampled)
            self.logger.info(f"Resampled timepoint {i} to reference FOV")
            
        # save the resampled volumes
        for i, vol in enumerate(resampled_volumes):
            out_path = output_dir / f'{name_prefix}_frame_{i:02d}.nii.gz'
            sitk.WriteImage(vol, str(out_path))
            self.logger.info(f"Saved {out_path}")
            
        self.logger.info(f"Saved {len(resampled_volumes)} resampled volumes")
        
    def build_simple_per_timepoint(
        self,
        *,
        name: str,
        img_path: Path,
        output_dir: Path,
    ) -> None:
        """
        Build a volume for each timepoint without any resampling.
        
        Args:
            name: Name prefix for the output files
            img_path: Path to the 4D image to split
            output_dir: Directory to save the per-timepoint volumes
        """
        self.logger.info(f"Building per timepoint volumes for patient {self.patient_id}")
        self.logger.info(f"Loading image from {img_path}")
        
        img = sitk.ReadImage(str(img_path))
        self.logger.info(f"Image dimensions: {img.GetSize()}")
        
        # split the image into timepoints
        timepoints = [img[:,:,:,t] for t in range(img.GetSize()[3])]
        self.logger.info(f"Loaded {len(timepoints)} timepoints")
        
        # save the timepoints
        for i, vol in enumerate(timepoints):
            out_path = output_dir / f'{name}_frame_{i:02d}.nii.gz'
            sitk.WriteImage(vol, str(out_path))
            # self.logger.info(f"Saved {out_path}")
            
        self.logger.info(f"Saved {len(timepoints)} timepoints for {name}")
    
    # ------------------------------------------------------------------
    # Downsampling utilities
    # ------------------------------------------------------------------
    @staticmethod
    def create_downsampled_reference_grid(
        source_img: sitk.Image,
        target_size: tuple[int, int, int],
    ) -> sitk.Image:
        """
        Create a reference image grid that preserves physical FOV but with different voxel size.
        
        The origin is adjusted so that physical corners align between source and target.
        In SimpleITK, the origin is the center of the first voxel, so when spacing changes,
        we must shift the origin to keep the same physical extent.
        
        Args:
            source_img: Source 3D image to derive geometry from
            target_size: Target voxel dimensions (X, Y, Z)
            
        Returns:
            Reference image with target_size, computed spacing, adjusted origin, source direction
        """
        src_size = source_img.GetSize()
        src_spacing = source_img.GetSpacing()
        src_origin = source_img.GetOrigin()
        src_direction = source_img.GetDirection()
        
        # Compute physical extent
        extent = [src_size[i] * src_spacing[i] for i in range(3)]
        
        # Compute new spacing to preserve physical extent
        new_spacing = [extent[i] / target_size[i] for i in range(3)]
        
        # Adjust origin so physical corners align
        # In voxel-centered coordinates: corner = origin - spacing/2
        # We want: new_origin - new_spacing/2 = old_origin - old_spacing/2
        # So: new_origin = old_origin + (new_spacing - old_spacing) / 2
        # This must be done in physical coordinates accounting for direction
        spacing_diff = [(new_spacing[i] - src_spacing[i]) / 2.0 for i in range(3)]
        
        # Apply direction matrix to spacing difference (direction is stored as flat 9-element tuple)
        dir_matrix = list(src_direction)
        new_origin = list(src_origin)
        for i in range(3):  # For each physical axis
            for j in range(3):  # Sum over voxel axes
                new_origin[i] += dir_matrix[i * 3 + j] * spacing_diff[j]
        
        # Create reference image
        reference_img = sitk.Image(target_size, sitk.sitkFloat32)
        reference_img.SetSpacing(new_spacing)
        reference_img.SetOrigin(new_origin)
        reference_img.SetDirection(src_direction)
        
        return reference_img
    
    @staticmethod
    def resample_to_target_grid(
        moving_img: sitk.Image,
        reference_img: sitk.Image,
        interpolator: int = sitk.sitkLinear,
        default_value: float = 0.0,
    ) -> sitk.Image:
        """
        Resample a moving image to match a reference grid.
        
        Args:
            moving_img: Image to resample
            reference_img: Reference image defining target grid
            interpolator: SimpleITK interpolator (e.g., sitkLinear, sitkNearestNeighbor)
            default_value: Default pixel value for regions outside the moving image
            
        Returns:
            Resampled image matching reference grid
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_img)
        resampler.SetInterpolator(interpolator)
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(default_value)
        
        return resampler.Execute(moving_img)
    
    def build_downsampled_per_timepoint(
        self,
        *,
        source_dir: Path,
        output_dir: Path,
        reference_img: sitk.Image,
        name_prefix: str,
        interpolator: int = sitk.sitkLinear,
        default_value: float = 0.0,
    ) -> None:
        """
        Resample per-timepoint files from source directory to a target reference grid.
        
        Args:
            source_dir: Directory containing per-timepoint source files
            output_dir: Directory to save resampled per-timepoint files
            reference_img: Reference image defining target grid
            name_prefix: Prefix for output filenames
            interpolator: SimpleITK interpolator (e.g., sitkLinear, sitkNearestNeighbor)
            default_value: Default pixel value for regions outside source
        """
        import re
        
        # List and sort source files by frame number
        source_files = sorted(
            source_dir.glob("*.nii.gz"),
            key=lambda p: int(re.search(r'frame_(\d+)', p.name).group(1))
        )
        
        if not source_files:
            self.logger.warning(f"No source files found in {source_dir}")
            return
        
        self.logger.info(f"Resampling {len(source_files)} timepoints from {source_dir}")
        
        for source_file in source_files:
            # Extract frame number
            match = re.search(r'frame_(\d+)', source_file.name)
            frame_num = int(match.group(1))
            
            # Load, resample, save
            source_img = sitk.ReadImage(str(source_file))
            resampled = self.resample_to_target_grid(
                source_img, reference_img, interpolator, default_value
            )
            
            out_path = output_dir / f'{name_prefix}_frame_{frame_num:02d}.nii.gz'
            sitk.WriteImage(resampled, str(out_path))
        
        self.logger.info(f"Saved {len(source_files)} downsampled timepoints to {output_dir}")