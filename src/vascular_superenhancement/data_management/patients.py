from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging
import pandas as pd
import zipfile
from ..utils.logger import setup_patient_logger
from ..utils.path_config import PathConfig
from .dicom_catalog import catalog_patient_dicoms
import nibabel as nib
import numpy as np
from .dicom_to_nifti import DicomToNiftiConverter
from .nifti_to_dicom import NiftiToDicomConverter

@dataclass
class Patient:
    """A class representing a patient in the vascular superenhancement project.
    
    This class manages the paths and identifiers for a single patient's data.
    
    Attributes:
        path_config (PathConfig): Configuration object containing base paths
        accession_number (Optional[str]): Accession number for the patient
        phonetic_id (Optional[str]): Phonetic ID for the patient
        skip_database_validation (bool): Whether to skip database validation (default: False)
        debug (bool): Whether to enable debug logging (default: False)
        overwrite_images (bool): Whether to overwrite existing NIfTI image files (default: False)
        overwrite_catalogs (bool): Whether to overwrite existing catalog files (default: False)
        overwrite_corrected (Optional[bool]): Override for corrected velocity files. None=use overwrite_images
        overwrite_downsampled (Optional[bool]): Override for downsampled files. None=use overwrite_images
        config (str): Name of the config file to use (default: "default")
        dataset_logger (Optional[logging.Logger]): Logger for dataset-level logging (default: None)
        
    Properties:
        identifier (str): Primary identifier for the patient (accession_number or phonetic_id)
        unzipped_dir (Path): Path to the patient's folder containing unzipped DICOM files
        working_dir (Path): Path to the patient's working directory under patient_data/
        dicom_catalog (Optional[pd.DataFrame]): The patient's DICOM catalog as a DataFrame
        study_key (Optional[str]): Study key from the database
        study_description (Optional[str]): Study description from the database
        three_d_cine_series_number (Optional[str]): 3D Cine series number from the database
        three_d_cine_series_description (Optional[str]): Description of the 3D Cine series
        series_descriptions (List[str]): List of series descriptions from the database
        series_numbers (List[str]): List of series numbers from the database
        
    At least one of accession_number or phonetic_id must be provided.
    """
    path_config: PathConfig
    accession_number: Optional[str] = None
    phonetic_id: Optional[str] = None
    skip_database_validation: bool = False
    debug: bool = False
    overwrite_images: bool = False
    overwrite_catalogs: bool = False
    overwrite_corrected: Optional[bool] = None  # None = use overwrite_images
    overwrite_downsampled: Optional[bool] = None  # None = use overwrite_images
    config: str = "default"
    dataset_logger: Optional[logging.Logger] = None
    
    def __post_init__(self):
        """Validate that at least one identifier is provided and initialize the catalog."""
        if self.accession_number is None and self.phonetic_id is None:
            raise ValueError("At least one of accession_number or phonetic_id must be provided")
        
        # Set up patient-specific logger
        self._logger = setup_patient_logger(
            self.identifier,
            config=self.config,  # Pass the config parameter
            level=logging.DEBUG if self.debug else logging.INFO  # Level depends on debug flag
        )
        
        # Validate against database unless explicitly skipped
        if not self.skip_database_validation:
            self._validate_against_database()
        
        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize catalogs as None - will be loaded on first access
        self._dicom_catalog = None
        self._dicom_catalog_3d_cine = None
        self._dicom_catalog_4d_flow = None
    
    def _validate_against_database(self) -> None:
        """Validate the patient against the database and load additional information."""
        try:
            # Load the database
            db = pd.read_csv(self.path_config.database_path)
            
            # Try to find the patient in the database
            if self.accession_number is not None:
                patient_data = db[db['Accession Number'] == self.accession_number]
            else:
                patient_data = db[
                    (db['Phonetic ID_x'] == self.phonetic_id) | 
                    (db['Phonetic ID_y'] == self.phonetic_id)
                ]
            
            if len(patient_data) == 0:
                raise ValueError(f"Patient {self.identifier} not found in database")
            elif len(patient_data) > 1:
                self._logger.warning(f"Multiple entries found for patient {self.identifier}")
            
            # Store the first matching entry's data
            row = patient_data.iloc[0]
            
            # Store both identifiers
            if self.accession_number is None:
                self.accession_number = row['Accession Number']
            if self.phonetic_id is None:
                # Try both phonetic ID columns
                self.phonetic_id = row['Phonetic ID_x'] if pd.notna(row['Phonetic ID_x']) else row['Phonetic ID_y']
            
            # Store additional information
            self.study_key = row['Study Key']
            self.study_description = row['Study Description']
            
            # Convert 3D Cine series number to numeric type if it exists
            if pd.notna(row['3D Cine series']):
                try:
                    self.three_d_cine_series_number = int(row['3D Cine series'])
                    self._logger.debug(f"Converted 3D Cine series number to integer: {self.three_d_cine_series_number}")
                except (ValueError, TypeError) as e:
                    self._logger.warning(f"Could not convert 3D Cine series number to integer: {str(e)}")
                    self.three_d_cine_series_number = str(row['3D Cine series']).strip()
            else:
                self.three_d_cine_series_number = None
                
            self.series_descriptions = [desc.strip() for desc in row['Series Descriptions'].split(',')] if pd.notna(row['Series Descriptions']) else []
            self.series_numbers = [num.strip() for num in row['SeriesNumbers'].split(',')] if pd.notna(row['SeriesNumbers']) else []
            
            self._logger.debug(f"Successfully initialized {self.identifier} from database")
            
        except Exception as e:
            self._logger.error(f"Error validating against database: {str(e)}")
            raise
    
    def _should_overwrite(self, category: str) -> bool:
        """Determine if files in the given category should be overwritten.
        
        Args:
            category: One of 'base', 'corrected', or 'downsampled'
            
        Returns:
            True if files should be overwritten, False otherwise.
            
        Category mapping:
            - 'base': Uses self.overwrite_images (DICOM-derived NIfTIs, per-timepoint)
            - 'corrected': Uses self.overwrite_corrected if set, else self.overwrite_images
            - 'downsampled': Uses self.overwrite_downsampled if set, else self.overwrite_images
        """
        if category == 'base':
            return self.overwrite_images
        elif category == 'corrected':
            if self.overwrite_corrected is not None:
                return self.overwrite_corrected
            return self.overwrite_images
        elif category == 'downsampled':
            if self.overwrite_downsampled is not None:
                return self.overwrite_downsampled
            return self.overwrite_images
        else:
            raise ValueError(f"Unknown overwrite category: {category}")
    
    @property
    def identifier(self) -> str:
        """Return the primary identifier for the patient.
        
        Prefers phonetic ID if available, otherwise uses accession number.
        """
        return self.phonetic_id if self.phonetic_id is not None else self.accession_number
    
    @property
    def three_d_cine_series_description(self) -> Optional[str]:
        """Return the description of the 3D Cine series.
        
        This is found by matching the 3D Cine series number with the corresponding
        series description from the series_descriptions list.
        """
        if self.three_d_cine_series_number is None:
            return None
            
        try:
            # Debug logging
            self._logger.debug(f"Looking for 3D Cine series number: {self.three_d_cine_series_number}")
            self._logger.debug(f"Available series numbers: {self.series_numbers}")
            self._logger.debug(f"Available series descriptions: {self.series_descriptions}")
            
            # Find the index of the 3D Cine series number in the series_numbers list
            index = self.series_numbers.index(self.three_d_cine_series_number)
            
            self._logger.debug(f"Found index: {index}")
            # Return the corresponding series description
            return self.series_descriptions[index]
        except ValueError:
            self._logger.warning(f"Could not find series number {self.three_d_cine_series_number} in {self.series_numbers}")
            return None
        except IndexError:
            self._logger.warning(f"Found series number but index {index} is out of range for descriptions list")
            return None
    
    @property
    def unzipped_dir(self) -> Path:
        """Return the path to the patient's unzipped DICOM files."""
        return self.path_config.unzipped_dir / self.identifier
    
    @property
    def working_dir(self) -> Path:
        """Return the path to the patient's working directory under patient_data/.
        
        This is where all generated files and data for this patient will be stored.
        """
        return self.path_config.working_dir / "patient_data" / self.identifier
    
    @property
    def nifti_dir(self) -> Path:
        """
        Create (if necessary) and return
        <working_dir>/nifti/  for this patient.
        """
        nifti_path = self.working_dir / "nifti"
        nifti_path.mkdir(parents=True, exist_ok=True)
        # self._logger.debug(f"Created/accessed NIfTI directory at {nifti_path}")
        return nifti_path
    
    @property
    def cine_per_timepoint_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/cine_per_timepoint/ for this patient."""
        folder_name = f"3d_cine_{self.identifier}_per_timepoint"
        cine_per_timepoint_dir = self.nifti_dir / folder_name
        cine_per_timepoint_dir.mkdir(parents=True, exist_ok=True)
        # self._logger.debug(f"Created/accessed NIfTI directory at {cine_per_timepoint_dir}")
        return cine_per_timepoint_dir
    
    @property
    def flow_mag_per_timepoint_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/flow_mag_per_timepoint/ for this patient."""
        folder_name = f"4d_flow_mag_{self.identifier}_per_timepoint"
        flow_mag_per_timepoint_dir = self.nifti_dir / folder_name
        flow_mag_per_timepoint_dir.mkdir(parents=True, exist_ok=True)
        return flow_mag_per_timepoint_dir
    
    @property
    def flow_vx_per_timepoint_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/flow_vx_per_timepoint/ for this patient."""
        folder_name = f"4d_flow_vx_{self.identifier}_per_timepoint"
        flow_vx_per_timepoint_dir = self.nifti_dir / folder_name
        flow_vx_per_timepoint_dir.mkdir(parents=True, exist_ok=True)
        return flow_vx_per_timepoint_dir
    
    @property
    def flow_vy_per_timepoint_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/flow_vy_per_timepoint/ for this patient."""
        folder_name = f"4d_flow_vy_{self.identifier}_per_timepoint"
        flow_vy_per_timepoint_dir = self.nifti_dir / folder_name
        flow_vy_per_timepoint_dir.mkdir(parents=True, exist_ok=True)
        return flow_vy_per_timepoint_dir
    
    @property
    def flow_vz_per_timepoint_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/flow_vz_per_timepoint/ for this patient."""
        folder_name = f"4d_flow_vz_{self.identifier}_per_timepoint"
        flow_vz_per_timepoint_dir = self.nifti_dir / folder_name
        flow_vz_per_timepoint_dir.mkdir(parents=True, exist_ok=True)
        return flow_vz_per_timepoint_dir
    
    @property
    def flow_mag_per_timepoint_full_fov_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/flow_mag_per_timepoint_full_fov/ for this patient."""
        folder_name = f"4d_flow_mag_{self.identifier}_per_timepoint_full_fov"
        flow_mag_per_timepoint_full_fov_dir = self.nifti_dir / folder_name
        flow_mag_per_timepoint_full_fov_dir.mkdir(parents=True, exist_ok=True)
        return flow_mag_per_timepoint_full_fov_dir
    
    @property
    def flow_vx_per_timepoint_full_fov_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/flow_vx_per_timepoint_full_fov/ for this patient."""
        folder_name = f"4d_flow_vx_{self.identifier}_per_timepoint_full_fov"
        flow_vx_per_timepoint_full_fov_dir = self.nifti_dir / folder_name
        flow_vx_per_timepoint_full_fov_dir.mkdir(parents=True, exist_ok=True)
        return flow_vx_per_timepoint_full_fov_dir
    
    @property
    def flow_vy_per_timepoint_full_fov_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/flow_vy_per_timepoint_full_fov/ for this patient."""
        folder_name = f"4d_flow_vy_{self.identifier}_per_timepoint_full_fov"
        flow_vy_per_timepoint_full_fov_dir = self.nifti_dir / folder_name
        flow_vy_per_timepoint_full_fov_dir.mkdir(parents=True, exist_ok=True)
        return flow_vy_per_timepoint_full_fov_dir
    
    @property
    def flow_vz_per_timepoint_full_fov_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/flow_vz_per_timepoint_full_fov/ for this patient."""
        folder_name = f"4d_flow_vz_{self.identifier}_per_timepoint_full_fov"
        flow_vz_per_timepoint_full_fov_dir = self.nifti_dir / folder_name
        flow_vz_per_timepoint_full_fov_dir.mkdir(parents=True, exist_ok=True)
        return flow_vz_per_timepoint_full_fov_dir
    
    @property
    def cine_per_timepoint_full_fov_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/3d_cine_{id}_per_timepoint_full_fov/ for this patient."""
        folder_name = f"3d_cine_{self.identifier}_per_timepoint_full_fov"
        cine_per_timepoint_full_fov_dir = self.nifti_dir / folder_name
        cine_per_timepoint_full_fov_dir.mkdir(parents=True, exist_ok=True)
        return cine_per_timepoint_full_fov_dir
    
    @property
    def flow_speed_per_timepoint_dir(self) -> Path:
        """Create (if necessary) and return
        <working_dir>/nifti/flow_speed_per_timepoint/ for this patient.
        
        Speed volumes are precomputed from sqrt(vx^2 + vy^2 + vz^2) and stored
        here for efficient training (avoids repeated computation)."""
        folder_name = f"4d_flow_speed_{self.identifier}_per_timepoint"
        flow_speed_per_timepoint_dir = self.nifti_dir / folder_name
        flow_speed_per_timepoint_dir.mkdir(parents=True, exist_ok=True)
        return flow_speed_per_timepoint_dir
    
    @property
    def corrected_velocities_dir(self) -> Path:
        """Return the directory containing phase-error-corrected velocity numpy files.
        
        Located at <repository_root>/corrected_velocities/
        """
        return self.path_config.repository_root / "corrected_velocities"
    
    @property
    def corrected_velocity_numpy_path(self) -> Path:
        """Return the path to this patient's corrected velocity numpy file."""
        return self.corrected_velocities_dir / f"{self.identifier}.npy"
    
    @property
    def flow_vx_corr_per_timepoint_dir(self) -> Path:
        """Create (if necessary) and return directory for corrected vx per-timepoint files."""
        folder_name = f"4d_flow_vx_corr_{self.identifier}_per_timepoint"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def flow_vy_corr_per_timepoint_dir(self) -> Path:
        """Create (if necessary) and return directory for corrected vy per-timepoint files."""
        folder_name = f"4d_flow_vy_corr_{self.identifier}_per_timepoint"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def flow_vz_corr_per_timepoint_dir(self) -> Path:
        """Create (if necessary) and return directory for corrected vz per-timepoint files."""
        folder_name = f"4d_flow_vz_corr_{self.identifier}_per_timepoint"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def flow_speed_corr_per_timepoint_dir(self) -> Path:
        """Create (if necessary) and return directory for corrected speed per-timepoint files.
        
        Speed is computed as sqrt(vx_corr^2 + vy_corr^2 + vz_corr^2)."""
        folder_name = f"4d_flow_speed_corr_{self.identifier}_per_timepoint"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    # -------------------------------------------------------------------------
    # Uncorrected data in CORRECTED (unpadded) FOV - per-timepoint directories
    # -------------------------------------------------------------------------
    
    @property
    def flow_mag_per_timepoint_corr_fov_dir(self) -> Path:
        """Directory for magnitude per-timepoint files resampled to corrected FOV."""
        folder_name = f"4d_flow_mag_{self.identifier}_per_timepoint_corr_fov"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def flow_vx_per_timepoint_corr_fov_dir(self) -> Path:
        """Directory for uncorrected vx per-timepoint files resampled to corrected FOV."""
        folder_name = f"4d_flow_vx_{self.identifier}_per_timepoint_corr_fov"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def flow_vy_per_timepoint_corr_fov_dir(self) -> Path:
        """Directory for uncorrected vy per-timepoint files resampled to corrected FOV."""
        folder_name = f"4d_flow_vy_{self.identifier}_per_timepoint_corr_fov"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def flow_vz_per_timepoint_corr_fov_dir(self) -> Path:
        """Directory for uncorrected vz per-timepoint files resampled to corrected FOV."""
        folder_name = f"4d_flow_vz_{self.identifier}_per_timepoint_corr_fov"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def cine_per_timepoint_corr_fov_dir(self) -> Path:
        """Directory for 3D cine per-timepoint files resampled to corrected FOV."""
        folder_name = f"3d_cine_{self.identifier}_per_timepoint_corr_fov"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def cine_mask_corr_fov_path(self) -> Path:
        """Path for 3D cine mask resampled to corrected FOV."""
        return self.nifti_dir / f"3d_cine_{self.identifier}_corr_fov_mask.nii.gz"
    
    @property
    def flow_diff_vx_per_timepoint_dir(self) -> Path:
        """Directory for velocity diff (corrected - uncorrected) vx per-timepoint files."""
        folder_name = f"4d_flow_diff_vx_{self.identifier}_per_timepoint"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def flow_diff_vy_per_timepoint_dir(self) -> Path:
        """Directory for velocity diff (corrected - uncorrected) vy per-timepoint files."""
        folder_name = f"4d_flow_diff_vy_{self.identifier}_per_timepoint"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def flow_diff_vz_per_timepoint_dir(self) -> Path:
        """Directory for velocity diff (corrected - uncorrected) vz per-timepoint files."""
        folder_name = f"4d_flow_diff_vz_{self.identifier}_per_timepoint"
        d = self.nifti_dir / folder_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def velocity_correction_dir(self) -> Path:
        """Create (if necessary) and return directory for velocity correction data.
        
        Contains: delta (corrected - uncorrected), polynomial coefficients, ground truth.
        All data is stored in UNPADDED dimensions matching the corrected velocity numpy."""
        d = self.nifti_dir / "velocity_correction_crop-17.5"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def flow_geometry_dir(self) -> Path:
        """Create (if necessary) and return the flow-measurement directory.

        Located at ``<working_dir>/flow_measurement/`` (a sibling of ``nifti/``).
        Holds the compact, precomputed flow geometry cache for this patient
        (``geometry.npz``: vessel-spline sample points, plane unit normals, the
        cross-section segmentation masks, pixel area, BPM and VENC) plus optional
        QA NIfTIs. The cache is produced once offline from the auto-flow pipeline
        outputs and is consumed by the lightweight in-repo flow evaluator during
        validation; the heavy auto-flow raw outputs themselves live in a separate
        staging directory and are not required at evaluation time.
        """
        d = self.working_dir / "flow_measurement"
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def num_timepoints(self) -> int:
        """Return the number of timepoints for this patient.

        Counted from the full/padded-FOV magnitude per-timepoint directory,
        which is built unconditionally as the foundation for the
        corrected-FOV and downsampled outputs that the active dual-task
        pipeline consumes.
        """
        files = list(self.flow_mag_per_timepoint_full_fov_dir.glob("*.nii.gz"))
        if not files:
            raise ValueError(
                f"No mag per-timepoint files for patient {self.identifier} in "
                f"{self.flow_mag_per_timepoint_full_fov_dir}. "
                "Run build_4d_flow_per_timepoint_full_fov() first."
            )
        return len(files)
    
    def _load_or_create_catalog(self) -> None:
        """Load the DICOM catalog if it exists, otherwise create it.
        
        Checks for both new format (dicom_catalog_{identifier}.csv) and old format
        ({identifier}_dicom_catalog.csv). If old format is found, it will be loaded
        and saved in the new format.
        
        After loading, deduplicates by SOPInstanceUID to handle source data that
        contains byte-identical duplicate files (one of several known PACS export
        artifacts). The deduplicated catalog is persisted back to disk so the
        cleanup happens once per patient.
        """
        new_catalog_path = self.working_dir / f"dicom_catalog_{self.identifier}.csv"
        old_catalog_path = self.working_dir / f"{self.identifier}_dicom_catalog.csv"
        
        # First check for new format
        if new_catalog_path.exists() and not self.overwrite_catalogs:
            try:
                self._dicom_catalog = pd.read_csv(new_catalog_path)
                self._logger.info(f"Loaded existing DICOM catalog for patient {self.identifier}")
            except Exception as e:
                self._logger.error(f"Error reading DICOM catalog: {str(e)}")
                self._dicom_catalog = None
        # Then check for old format
        elif old_catalog_path.exists() and not self.overwrite_catalogs:
            try:
                self._logger.info("Found old format catalog, migrating to new format")
                self._dicom_catalog = pd.read_csv(old_catalog_path)
                # Save in new format
                self._dicom_catalog.to_csv(new_catalog_path, index=False)
                self._logger.info(f"Successfully migrated catalog to new format for patient {self.identifier}")
            except Exception as e:
                self._logger.error(f"Error reading/migrating old format catalog: {str(e)}")
                self._dicom_catalog = None
        else:
            self._create_catalog()
        
        if self._dicom_catalog is not None:
            self._deduplicate_catalog_by_sop_uid(persist_path=new_catalog_path)
    
    def _deduplicate_catalog_by_sop_uid(self, persist_path: Optional[Path] = None) -> None:
        """Drop rows with duplicate SOPInstanceUIDs from the in-memory catalog.
        
        Some source DICOMs contain byte-identical duplicates (same SOP UID,
        same content, different filenames) introduced upstream during PACS
        export or anonymization. These duplicates inflate per-series file
        counts, perturb downstream geometry-consistency filters, and produce
        repeated spatial positions that break affine decomposition.
        
        Keeps the first occurrence per ``sopinstanceuid``. If ``persist_path``
        is provided and any duplicates were dropped, the cleaned catalog is
        written back to that path so the dedup happens once per patient.
        """
        if self._dicom_catalog is None or 'sopinstanceuid' not in self._dicom_catalog.columns:
            return
        
        n_before = len(self._dicom_catalog)
        n_unique = self._dicom_catalog['sopinstanceuid'].nunique(dropna=False)
        n_duplicate = n_before - n_unique
        
        if n_duplicate == 0:
            return
        
        max_copies = int(self._dicom_catalog.groupby('sopinstanceuid').size().max())
        self._logger.warning(
            f"Patient {self.identifier}: found {n_duplicate} duplicate-SOP rows "
            f"(unique={n_unique}, total={n_before}, max copies of any SOP={max_copies}); "
            f"deduplicating by sopinstanceuid (keeping first occurrence)."
        )
        
        self._dicom_catalog = (
            self._dicom_catalog
            .drop_duplicates(subset='sopinstanceuid', keep='first')
            .reset_index(drop=True)
        )
        
        if persist_path is not None:
            try:
                self._dicom_catalog.to_csv(persist_path, index=False)
                self._logger.info(
                    f"Persisted deduplicated catalog ({len(self._dicom_catalog)} rows) "
                    f"to {persist_path}"
                )
            except Exception as e:
                self._logger.error(f"Failed to persist deduplicated catalog: {e}")
    
    def _create_catalog(self) -> None:
        """Create a new DICOM catalog for the patient."""
        self._logger.info(f"Creating new DICOM catalog for patient {self.identifier}")
        success = catalog_patient_dicoms(
            patient_dir=self.unzipped_dir,
            catalog_dir=self.working_dir,
            logger=self._logger,
            overwrite=True  # Always overwrite since we've decided to create
        )
        
        if success:
            try:
                self._dicom_catalog = pd.read_csv(self.working_dir / f"dicom_catalog_{self.identifier}.csv")
                self._logger.info(f"Successfully created DICOM catalog for patient {self.identifier}")
            except Exception as e:
                self._logger.error(f"Error reading newly created catalog: {str(e)}")
                self._dicom_catalog = None
        else:
            self._logger.error(f"Failed to create DICOM catalog for patient {self.identifier}")
            self._dicom_catalog = None
    
    def reload_catalog(self) -> None:
        """Explicitly reload the DICOM catalog."""
        self._logger.info(f"Reloading DICOM catalog for patient {self.identifier}")
        self._load_or_create_catalog()
        # Clear derived catalogs since they depend on the DICOM catalog
        self._dicom_catalog_3d_cine = None
        self._dicom_catalog_4d_flow = None
    
    def clear_catalog(self) -> None:
        """Clear the in-memory catalogs to free up memory."""
        self._logger.info(f"Clearing DICOM catalogs from memory for patient {self.identifier}")
        self._dicom_catalog = None
        self._dicom_catalog_3d_cine = None
        self._dicom_catalog_4d_flow = None
    
    def delete_catalog(self, catalog_type: str) -> bool:
        """Delete a specific catalog from memory and disk.
        
        Args:
            catalog_type: Type of catalog to delete. Must be one of:
                - 'dicom': Base DICOM catalog
                - '3d_cine': 3D Cine catalog
                - '4d_flow': 4D Flow catalog
        
        Returns:
            bool: True if the catalog was successfully deleted, False otherwise
        """
        # Map catalog types to their corresponding attributes and file patterns
        catalog_info = {
            'dicom': {
                'attribute': '_dicom_catalog',
                'file_pattern': 'dicom_catalog_{}.csv'
            },
            '3d_cine': {
                'attribute': '_dicom_catalog_3d_cine',
                'file_pattern': 'dicom_catalog_3d-cine_{}.csv'
            },
            '4d_flow': {
                'attribute': '_dicom_catalog_4d_flow',
                'file_pattern': 'dicom_catalog_4d-flow_{}.csv'
            }
        }
        
        if catalog_type not in catalog_info:
            self._logger.error(f"Invalid catalog type: {catalog_type}. Must be one of {list(catalog_info.keys())}")
            return False
            
        info = catalog_info[catalog_type]
        attribute_name = info['attribute']
        file_pattern = info['file_pattern']
        
        try:
            # Clear from memory
            setattr(self, attribute_name, None)
            self._logger.debug(f"Cleared {catalog_type} catalog from memory")
            
            # Delete file if it exists
            file_path = self.working_dir / file_pattern.format(self.identifier)
            if file_path.exists():
                file_path.unlink()
                self._logger.info(f"Deleted {catalog_type} catalog file: {file_path}")
            else:
                self._logger.debug(f"No file found for {catalog_type} catalog")
                
            return True
            
        except Exception as e:
            self._logger.error(f"Error deleting {catalog_type} catalog: {str(e)}")
            return False
    
    @property
    def dicom_catalog(self) -> Optional[pd.DataFrame]:
        """Return the patient's DICOM catalog as a DataFrame.
        
        The catalog is loaded on first access if it hasn't been loaded yet.
        """
        if self._dicom_catalog is None:
            self._load_or_create_catalog()
        return self._dicom_catalog
    
    @property
    def dicom_catalog_3d_cine(self) -> Optional[pd.DataFrame]:
        """Return the patient's 3D Cine DICOM catalog as a DataFrame.
        
        The catalog is created on first access if it hasn't been created yet.
        """
        self._logger.debug(f"Accessing 3D Cine catalog for patient {self.identifier}")
        
        if self._dicom_catalog_3d_cine is None:
            self._logger.debug("3D Cine catalog not in memory, checking for existing file")
            
            # Check if catalog file exists
            catalog_path = self.working_dir / f"dicom_catalog_3d-cine_{self.identifier}.csv"
            if catalog_path.exists() and not self.overwrite_catalogs:
                try:
                    self._logger.debug(f"Loading existing 3D Cine catalog from {catalog_path}")
                    self._dicom_catalog_3d_cine = pd.read_csv(catalog_path)
                    self._logger.info(f"Successfully loaded existing 3D Cine catalog for patient {self.identifier}")
                    return self._dicom_catalog_3d_cine
                except Exception as e:
                    self._logger.error(f"Error reading existing 3D Cine catalog: {str(e)}")
                    # Continue to create new catalog if reading fails
            
            self._logger.debug("Creating new 3D Cine catalog")
            
            # Get the DICOM catalog first
            catalog = self.dicom_catalog
            if catalog is None:
                self._logger.error("Cannot create 3D Cine catalog: DICOM catalog is None")
                return None
                
            self._logger.debug(f"Found DICOM catalog with {len(catalog)} entries")
            
            # Filter based on series number or description
            if self.three_d_cine_series_number is not None:
                self._logger.debug(f"Filtering by series number: {self.three_d_cine_series_number} (type: {type(self.three_d_cine_series_number)})")
                
                # Debug the series numbers in the catalog
                unique_series = catalog['seriesnumber'].unique()
                self._logger.debug(f"Unique series numbers in catalog: {unique_series.tolist()}")
                self._logger.debug(f"Types of series numbers in catalog: {[type(x) for x in unique_series]}")
                
                # Try to find exact match
                filtered_catalog = catalog[catalog['seriesnumber'] == self.three_d_cine_series_number]
                
                # If no matches, try converting types
                if len(filtered_catalog) == 0:
                    self._logger.debug("No exact matches found, trying type conversion")
                    # Try converting catalog series numbers to string
                    catalog_series_str = catalog['seriesnumber'].astype(str)
                    target_series_str = str(self.three_d_cine_series_number)
                    self._logger.debug(f"Comparing string versions - target: {target_series_str} (type: {type(target_series_str)})")
                    filtered_catalog = catalog[catalog_series_str == target_series_str]
                    
                    if len(filtered_catalog) == 0:
                        self._logger.debug("Still no matches after string conversion, trying numeric conversion")
                        # Try converting to numeric
                        try:
                            catalog_series_num = pd.to_numeric(catalog['seriesnumber'], errors='coerce')
                            target_series_num = float(self.three_d_cine_series_number)
                            self._logger.debug(f"Comparing numeric versions - target: {target_series_num} (type: {type(target_series_num)})")
                            filtered_catalog = catalog[catalog_series_num == target_series_num]
                        except (ValueError, TypeError) as e:
                            self._logger.error(f"Error converting series numbers to numeric: {str(e)}")
            else:
                self._logger.debug(f"Filtering by series description: {self.three_d_cine_series_description}")
                filtered_catalog = catalog[catalog['seriesdescription'] == self.three_d_cine_series_description]
                
            if len(filtered_catalog) == 0:
                self._logger.warning(f"No matching 3D Cine series found in DICOM catalog for patient {self.identifier}")
                return None
                
            self._logger.debug(f"Found {len(filtered_catalog)} matching DICOM files")
            
            # Add time_index and slice_index columns
            filtered_catalog = filtered_catalog.copy()  # Avoid SettingWithCopyWarning
            self._logger.debug("Adding time_index and slice_index columns")
            
            # Log some sample values for debugging
            sample_instances = filtered_catalog['instancenumber'].head(3)
            sample_cardiac = filtered_catalog['cardiacnumberofimages'].head(3)
            self._logger.debug(f"Sample InstanceNumbers: {sample_instances.tolist()}")
            self._logger.debug(f"Sample CardiacNumberOfImages: {sample_cardiac.tolist()}")
            
            filtered_catalog['time_index'] = (filtered_catalog['instancenumber'] - 1) % filtered_catalog['cardiacnumberofimages']
            filtered_catalog['slice_index'] = (filtered_catalog['instancenumber'] - 1) // filtered_catalog['cardiacnumberofimages']
            
            nan_mask = filtered_catalog['time_index'].isna()
            if nan_mask.any():
                dropped_series = sorted(filtered_catalog.loc[nan_mask, 'seriesnumber'].unique())
                self._logger.info(
                    f"Dropping {nan_mask.sum()} 3D Cine files with NaN time_index "
                    f"(series {dropped_series}) for patient {self.identifier}"
                )
                filtered_catalog = filtered_catalog[~nan_mask]
            
            # Log some sample calculated indices
            sample_time = filtered_catalog['time_index'].head(3)
            sample_slice = filtered_catalog['slice_index'].head(3)
            self._logger.debug(f"Sample time indices: {sample_time.tolist()}")
            self._logger.debug(f"Sample slice indices: {sample_slice.tolist()}")
            
            # Save the filtered catalog
            self._logger.debug(f"Saving catalog to {catalog_path}")
            
            try:
                filtered_catalog.to_csv(catalog_path, index=False)
                self._logger.info(f"Successfully saved 3D Cine catalog for patient {self.identifier}")
                self._dicom_catalog_3d_cine = filtered_catalog
            except Exception as e:
                self._logger.error(f"Error saving 3D Cine catalog for patient {self.identifier}: {str(e)}")
                return None
        else:
            self._logger.debug("Returning cached 3D Cine catalog")
                
        return self._dicom_catalog_3d_cine
    
    @property
    def dicom_catalog_4d_flow(self) -> Optional[pd.DataFrame]:
        """Return the patient's 4D Flow DICOM catalog as a DataFrame.
        
        A file is considered 4D Flow if:
        - Tag_0019_10B3 > 1
        OR
        - Tag_0043_1030 > 1 AND Tag_0043_1030 < 6
        
        Files with Tag_0043_1030 = 7 are explicitly excluded.
        
        The catalog is created on first access if it hasn't been created yet.
        """
        self._logger.debug(f"Accessing 4D Flow catalog for patient {self.identifier}")
        
        if self._dicom_catalog_4d_flow is None:
            self._logger.debug("4D Flow catalog not in memory, checking for existing file")
            
            # Check if catalog file exists
            catalog_path = self.working_dir / f"dicom_catalog_4d-flow_{self.identifier}.csv"
            if catalog_path.exists() and not self.overwrite_catalogs:
                try:
                    self._logger.debug(f"Loading existing 4D Flow catalog from {catalog_path}")
                    self._dicom_catalog_4d_flow = pd.read_csv(catalog_path)
                    self._logger.info(f"Successfully loaded existing 4D Flow catalog for patient {self.identifier}")
                    return self._dicom_catalog_4d_flow
                except Exception as e:
                    self._logger.error(f"Error reading existing 4D Flow catalog: {str(e)}")
                    # Continue to create new catalog if reading fails
            
            self._logger.debug("Creating new 4D Flow catalog")
            
            # Get the DICOM catalog first
            catalog = self.dicom_catalog
            if catalog is None:
                self._logger.error("Cannot create 4D Flow catalog: DICOM catalog is None")
                return None
                
            self._logger.debug(f"Found DICOM catalog with {len(catalog)} entries")
            
            # Filter based on 4D Flow criteria
            self._logger.debug("Filtering for 4D Flow files")
            
            # Convert velocity encoding and flow encoding tags to numeric
            velocity_encoding = pd.to_numeric(catalog['tag_0x0019_0x10B3'], errors='coerce')
            flow_encoding = pd.to_numeric(catalog['tag_0x0043_0x1030'], errors='coerce')
            
            # Apply 4D Flow criteria
            is_velocity_encoded = velocity_encoding > 1
            is_flow_encoded = (flow_encoding > 1) & (flow_encoding < 6)
            is_excluded = flow_encoding == 7
            is_4d_flow = (is_velocity_encoded | is_flow_encoded) & ~is_excluded
            
            cardiac_phases = pd.to_numeric(catalog['cardiacnumberofimages'], errors='coerce')
            is_cardiac_gated = cardiac_phases > 0
            is_4d_flow = is_4d_flow & is_cardiac_gated
            
            # Log the number of files matching each criterion
            self._logger.debug(f"Files with velocity encoding > 1: {is_velocity_encoded.sum()}")
            self._logger.debug(f"Files with flow encoding between 1 and 6: {is_flow_encoded.sum()}")
            self._logger.debug(f"Files with flow encoding = 7 (excluded): {is_excluded.sum()}")
            self._logger.debug(f"Files with cardiacnumberofimages <= 0 (excluded): {(~is_cardiac_gated).sum()}")
            self._logger.debug(f"Total 4D Flow files: {is_4d_flow.sum()}")
            
            filtered_catalog = catalog[is_4d_flow]
            
            if len(filtered_catalog) == 0:
                self._logger.warning(f"No 4D Flow files found in DICOM catalog for patient {self.identifier}")
                return None
                
            self._logger.debug(f"Found {len(filtered_catalog)} 4D Flow files")
            
            # --- DualVenc filter (database-driven) ---
            # GE GenIQ research dual-venc protocol stores two complete
            # velocity sets per direction (HighVenc + LowVenc) under series
            # named e.g. "Ax GE 4D FLOW DualVenc - SI Flow HighVenc", plus
            # an "Anatomy" and "Preview" series. Each is tagged as 4D-flow
            # via the standard GE private tags, but the rest of the pipeline
            # assumes a single-venc reconstruction (one mag + one vx/vy/vz
            # per cardiac phase). Dual-venc rows must therefore be dropped
            # before any UID/geometry filtering.
            #
            # Detection: DICOM SeriesDescription is empty in our anonymized
            # exports, so we use the patient_database.csv lookup populated
            # by `_validate_against_database` (`self.series_descriptions`
            # and `self.series_numbers`, parallel lists). Series whose
            # description contains "dualvenc" (case-insensitive) are
            # dropped by `seriesnumber`.
            #
            # Effect:
            #  - Patients with a single-venc 4D flow series acquired in the
            #    same study survive (only their DualVenc series are dropped).
            #  - Patients with ONLY DualVenc 4D flow data end up with an
            #    empty catalog and are caught by the "No 4D Flow files"
            #    guard below; mark them as skip in splits.
            if (
                getattr(self, 'series_descriptions', None)
                and getattr(self, 'series_numbers', None)
                and len(self.series_descriptions) == len(self.series_numbers)
            ):
                dv_series_numbers: set = set()
                dv_descs: set = set()
                for sn, desc in zip(self.series_numbers, self.series_descriptions):
                    if not desc or not sn:
                        continue
                    if 'dualvenc' in desc.lower():
                        sn_str = str(sn).strip()
                        try:
                            dv_series_numbers.add(int(sn_str))
                        except (ValueError, TypeError):
                            self._logger.debug(
                                f"Could not coerce DualVenc series number {sn_str!r} to int"
                            )
                            continue
                        dv_descs.add(desc.strip())
                
                if dv_series_numbers:
                    catalog_sn_int = pd.to_numeric(filtered_catalog['seriesnumber'], errors='coerce')
                    is_dualvenc = catalog_sn_int.isin(dv_series_numbers)
                    n_dv = int(is_dualvenc.sum())
                    if n_dv:
                        present_dv_sns = sorted(
                            int(x) for x in filtered_catalog.loc[is_dualvenc, 'seriesnumber'].dropna().unique()
                        )
                        self._logger.info(
                            f"Dropping {n_dv} DualVenc 4D-flow files "
                            f"(series {present_dv_sns}, descriptions {sorted(dv_descs)}) "
                            f"for patient {self.identifier}; this protocol is not "
                            f"supported by the single-venc pipeline."
                        )
                        filtered_catalog = filtered_catalog[~is_dualvenc]
            
            if len(filtered_catalog) == 0:
                self._logger.warning(
                    f"No non-DualVenc 4D Flow files remain for patient "
                    f"{self.identifier} after DualVenc filtering. Patient should "
                    f"be marked as skip in splits."
                )
                return None
            
            # Coverage-aware phantom/duplicate UID filter:
            # within each series number, when multiple SeriesInstanceUIDs
            # exist, only drop a UID if its cardiac-phase coverage is fully
            # contained within another (larger) UID's coverage. UIDs that
            # add NEW cardiac phases are kept and merged into the same
            # logical series.
            #
            # This handles two distinct GE patterns:
            #  - True duplicate: a "phantom" reference UID that overlaps
            #    one or more cphases of the main UID (drop it).
            #  - Complementary split: the first cardiac phase mag stored
            #    under one UID, cphases 2-N under another (keep both).
            #    A naive "keep largest UID per series" filter would drop
            #    cphase 1 of mag entirely, breaking downstream alignment.
            filtered_catalog = filtered_catalog.copy()
            uids_to_drop = []
            for sn, sn_grp in filtered_catalog.groupby('seriesnumber'):
                if sn_grp['seriesinstanceuid'].nunique() <= 1:
                    continue
                uid_info = []
                for uid, uid_grp in sn_grp.groupby('seriesinstanceuid'):
                    cphases = set(
                        pd.to_numeric(uid_grp['cardiacphasenumber'], errors='coerce')
                        .dropna().astype(int).tolist()
                    )
                    uid_info.append({
                        'uid': uid,
                        'count': len(uid_grp),
                        'cphases': cphases,
                    })
                # Sort by file count descending; greedily accept UIDs that
                # add at least one new cardiac phase, drop the rest.
                uid_info.sort(key=lambda x: -x['count'])
                covered_cphases: set = set()
                for ui in uid_info:
                    new_phases = ui['cphases'] - covered_cphases
                    if new_phases:
                        covered_cphases |= ui['cphases']
                    else:
                        uids_to_drop.append((sn, ui['uid'], ui['count']))
            
            if uids_to_drop:
                drop_uid_set = {uid for _, uid, _ in uids_to_drop}
                drop_mask = filtered_catalog['seriesinstanceuid'].isin(drop_uid_set)
                dropped_count = int(drop_mask.sum())
                drop_series = sorted({sn for sn, _, _ in uids_to_drop})
                self._logger.info(
                    f"Dropping {dropped_count} files from {len(drop_uid_set)} "
                    f"redundant UID(s) in series {drop_series} for patient "
                    f"{self.identifier} (cphase coverage already provided by "
                    f"a larger sibling UID)."
                )
                filtered_catalog = filtered_catalog[~drop_mask]
            
            # --- Geometry-consistency filter ---
            # When multiple 4D flow acquisitions with different geometries
            # exist (e.g., DualVenc + standard, or preview + real), keep
            # only the group whose series form the most complete component
            # set (mag, vx, vy, vz).
            #
            # Geometry is keyed by the number of unique ImagePositionPatient
            # values (i.e. the spatial slice count). This is more robust than
            # file_count / cardiac_n, which gives the wrong answer when a
            # series is missing a cardiac phase: e.g. mag with 19 of 20
            # phases at 152 slices reads as files/cardiac=2888/20=144 yet has
            # the same 152-slice spatial geometry as the velocity components.
            _series_stats = filtered_catalog.groupby('seriesnumber').agg(
                file_count=('filepath', 'count'),
                n_slices=('imagepositionpatient', 'nunique'),
                acq_tag=('tag_0x0043_0x1030', 'first'),
            ).reset_index()

            if _series_stats['n_slices'].nunique() > 1:
                best_n_slices = None
                best_score = (-1, -1)
                for n_slices, grp in _series_stats.groupby('n_slices'):
                    n_components = grp['acq_tag'].nunique()
                    total_files = int(grp['file_count'].sum())
                    score = (n_components, total_files)
                    if score > best_score:
                        best_score = score
                        best_n_slices = n_slices

                keep_sn = set(
                    _series_stats.loc[
                        _series_stats['n_slices'] == best_n_slices, 'seriesnumber'
                    ]
                )
                drop_sn = set(_series_stats['seriesnumber']) - keep_sn
                if drop_sn:
                    self._logger.info(
                        f"Dropping series {sorted(drop_sn)} with incompatible "
                        f"geometry (keeping {sorted(keep_sn)}, "
                        f"n_slices={best_n_slices}) "
                        f"for patient {self.identifier}"
                    )
                    filtered_catalog = filtered_catalog[
                        filtered_catalog['seriesnumber'].isin(keep_sn)
                    ]

            # Add time_index and slice_index columns if they don't exist
            if 'time_index' not in filtered_catalog.columns or 'slice_index' not in filtered_catalog.columns:
                self._logger.debug("Adding time_index and slice_index columns")
                
                # Log some sample values for debugging
                sample_instances = filtered_catalog['instancenumber'].head(3)
                sample_cardiac = filtered_catalog['cardiacnumberofimages'].head(3)
                self._logger.debug(f"Sample InstanceNumbers: {sample_instances.tolist()}")
                self._logger.debug(f"Sample CardiacNumberOfImages: {sample_cardiac.tolist()}")
                
                filtered_catalog['time_index'] = (filtered_catalog['instancenumber'] - 1) % filtered_catalog['cardiacnumberofimages']
                filtered_catalog['slice_index'] = (filtered_catalog['instancenumber'] - 1) // filtered_catalog['cardiacnumberofimages']
                
                # Drop rows where time_index is NaN (raw/phantom series whose
                # cardiacnumberofimages is 0 or NaN, producing invalid indices)
                nan_mask = filtered_catalog['time_index'].isna()
                if nan_mask.any():
                    dropped_series = sorted(filtered_catalog.loc[nan_mask, 'seriesnumber'].unique())
                    self._logger.info(
                        f"Dropping {nan_mask.sum()} files with NaN time_index "
                        f"(series {dropped_series}) for patient {self.identifier}"
                    )
                    filtered_catalog = filtered_catalog[~nan_mask]
                
                # Log some sample calculated indices
                sample_time = filtered_catalog['time_index'].head(3)
                sample_slice = filtered_catalog['slice_index'].head(3)
                self._logger.debug(f"Sample time indices: {sample_time.tolist()}")
                self._logger.debug(f"Sample slice indices: {sample_slice.tolist()}")
            
            # Save the filtered catalog
            self._logger.debug(f"Saving catalog to {catalog_path}")
            
            try:
                filtered_catalog.to_csv(catalog_path, index=False)
                self._logger.info(f"Successfully saved 4D Flow catalog for patient {self.identifier}")
                self._dicom_catalog_4d_flow = filtered_catalog
            except Exception as e:
                self._logger.error(f"Error saving 4D Flow catalog for patient {self.identifier}: {str(e)}")
                return None
        else:
            self._logger.debug("Returning cached 4D Flow catalog")
                
        return self._dicom_catalog_4d_flow
    
    @property
    def venc(self) -> float:
        """Get VENC (velocity encoding) value from the first 4D flow DICOM.
        
        The VENC is stored in Siemens private tag (0x0019, 0x10CC).
        
        Returns:
            VENC value in cm/s
        """
        import pydicom
        
        catalog = self.dicom_catalog_4d_flow
        if catalog is None or catalog.empty:
            raise ValueError(f"No 4D flow DICOM catalog available for patient {self.identifier}")
        
        first_filepath = catalog.iloc[0]['filepath']
        dcm = pydicom.dcmread(first_filepath, stop_before_pixels=True)
        
        try:
            venc_value = float(dcm[0x0019, 0x10CC].value)
            self._logger.debug(f"Read VENC={venc_value} from {first_filepath}")
            return venc_value
        except KeyError:
            raise ValueError(f"VENC tag (0x0019, 0x10CC) not found in DICOM for patient {self.identifier}")
    
    @property
    def bpm(self) -> float:
        """Get heart rate (beats per minute) from the first 4D flow DICOM.

        Reads the standard ``HeartRate`` tag (0x0018, 0x1088). If absent or
        non-positive, falls back to ``NominalInterval`` (0x0018, 0x1062), the
        average R-R interval in milliseconds, converted via ``60000 / interval_ms``.

        BPM is needed to integrate the per-cycle flow curve into a volumetric
        flow rate (L/min) in the flow-measurement evaluator.

        Returns:
            Heart rate in beats per minute.
        """
        import pydicom

        catalog = self.dicom_catalog_4d_flow
        if catalog is None or catalog.empty:
            raise ValueError(f"No 4D flow DICOM catalog available for patient {self.identifier}")

        first_filepath = catalog.iloc[0]['filepath']
        dcm = pydicom.dcmread(first_filepath, stop_before_pixels=True)

        # Preferred: explicit HeartRate tag
        try:
            heart_rate = float(dcm[0x0018, 0x1088].value)
            if heart_rate > 0:
                self._logger.debug(f"Read HeartRate={heart_rate} bpm from {first_filepath}")
                return heart_rate
            self._logger.debug("HeartRate tag present but non-positive; trying NominalInterval")
        except (KeyError, ValueError, TypeError):
            self._logger.debug("HeartRate tag (0x0018, 0x1088) absent; trying NominalInterval")

        # Fallback: derive from NominalInterval (average R-R interval, ms)
        try:
            interval_ms = float(dcm[0x0018, 0x1062].value)
            if interval_ms > 0:
                bpm_value = 60000.0 / interval_ms
                self._logger.debug(
                    f"Derived bpm={bpm_value:.1f} from NominalInterval={interval_ms} ms "
                    f"in {first_filepath}"
                )
                return bpm_value
        except (KeyError, ValueError, TypeError):
            pass

        raise ValueError(
            f"Could not determine BPM for patient {self.identifier}: neither HeartRate "
            f"(0x0018,0x1088) nor a positive NominalInterval (0x0018,0x1062) was found."
        )
    
    def get_3d_cine(self, *, as_numpy: bool = False):
        """
        Specification:
        1. Compute expected path <nifti_dir>/<id>_cine.nii.gz.
        2. If file exists and overwrite_images is False → load and return
        (as np.ndarray if as_numpy True, else nib object).
        3. Else → build converter via `DicomToNiftiConverter.from_patient(self)`,
        set converter.catalog = self.dicom_catalog_3d_cine,
        call build_3d_cine(save=True, as_numpy=False),
        then return result in requested format.
        """
        # Get the expected path
        expected_path = self.nifti_dir / f"3d_cine_{self.identifier}.nii.gz"
        
        # Check if file exists and we shouldn't overwrite
        if expected_path.exists() and not self.overwrite_images:
            self._logger.info(f"Loading existing 3D cine NIfTI from {expected_path}")
            nii = nib.load(expected_path)
            return nii.get_fdata() if as_numpy else nii
        
        # Create converter and set catalog
        converter = DicomToNiftiConverter.from_patient(self)
        converter.catalog = self.dicom_catalog_3d_cine
        
        # Build and get result
        result = converter.build_3d_cine(save=True, as_numpy=as_numpy)
        
        return result
    
    def get_4d_flow(self, *, as_numpy: bool = False):
        """
        Specification:
        1. Define expected paths for 'mag', 'vx', 'vy', 'vz'
        inside self.nifti_dir.
        2. If all exist and overwrite_images is False → load all and return
        dict of images/arrays.
        3. Else → create converter with from_patient(),
        set converter.catalog = self.dicom_catalog_4d_flow,
        call build_4d_flow(save=True, as_numpy=False),
        then load freshly‑saved files and return in requested format.
        """
        # Define expected paths
        expected_paths = {
            'mag': self.nifti_dir / f"4d_flow_mag_{self.identifier}.nii.gz",
            'vx': self.nifti_dir / f"4d_flow_vx_{self.identifier}.nii.gz",
            'vy': self.nifti_dir / f"4d_flow_vy_{self.identifier}.nii.gz",
            'vz': self.nifti_dir / f"4d_flow_vz_{self.identifier}.nii.gz"
        }
        
        # Check if all files exist and we shouldn't overwrite
        if all(p.exists() for p in expected_paths.values()) and not self.overwrite_images:
            self._logger.info(f"Loading existing 4D flow NIfTIs from {self.nifti_dir}")
            results = {}
            for comp, path in expected_paths.items():
                nii = nib.load(path)
                results[comp] = nii.get_fdata() if as_numpy else nii
            return results
        
        # Create converter and set catalog
        converter = DicomToNiftiConverter.from_patient(self)
        converter.catalog = self.dicom_catalog_4d_flow
        
        # Build and get result
        result = converter.build_4d_flow(save=True, as_numpy=as_numpy)
        
        return result
    
    def build_images(self, *, as_numpy: bool = False) -> dict:
        """Build all images for the patient (3D cine and 4D flow).
        
        This method will build both 3D cine and 4D flow images if they don't exist
        or if overwrite_images is True. The images are returned in a dictionary
        with the following structure:
        {
            '3d_cine': nib.Nifti1Image or np.ndarray,
            '4d_flow': {
                'mag': nib.Nifti1Image or np.ndarray,
                'vx': nib.Nifti1Image or np.ndarray,
                'vy': nib.Nifti1Image or np.ndarray,
                'vz': nib.Nifti1Image or np.ndarray
            }
        }
        
        Args:
            as_numpy: If True, return numpy arrays instead of NIfTI images
            
        Returns:
            dict: Dictionary containing all built images
        """
        self._logger.info(f"Building all images for patient {self.identifier}")
        
        # Build 3D cine
        try:
            self._logger.debug("Building 3D cine image")
            cine = self.get_3d_cine(as_numpy=as_numpy)
            self._logger.info(f"Successfully built 3D cine image for patient {self.identifier}")
        except Exception as e:
            self._logger.error(f"Error building 3D cine image for patient {self.identifier}: {e}")
            cine = None
        
        # Build 4D flow
        try:
            self._logger.debug("Building 4D flow images")
            flow = self.get_4d_flow(as_numpy=as_numpy)
            self._logger.info(f"Successfully built 4D flow images for patient {self.identifier}")
        except Exception as e:
            self._logger.error(f"Error building 4D flow images for patient {self.identifier}: {e}")
            flow = None
        
        # Build corrected velocities and correction data (if numpy file exists)
        corr_vel = None
        corr_data = None
        if self.corrected_velocity_numpy_path.exists():
            try:
                self._logger.debug("Building corrected velocity images")
                corr_vel = self.build_corrected_velocities()
                self._logger.info(f"Successfully built corrected velocity images for patient {self.identifier}")
            except Exception as e:
                self._logger.error(f"Error building corrected velocity images for patient {self.identifier}: {e}")
            
            # Build velocity correction data (delta, coefficients, ground truth)
            # TODO: Uncomment after testing corrected velocities and downsampling
            # try:
            #     self._logger.debug("Building velocity correction data")
            #     corr_data = self.build_velocity_correction_data()
            #     self._logger.info(f"Successfully built velocity correction data for patient {self.identifier}")
            # except Exception as e:
            #     self._logger.error(f"Error building velocity correction data for patient {self.identifier}: {e}")
        else:
            self._logger.info(f"No corrected velocities numpy file found for patient {self.identifier}, skipping")
        
        # Combine into result dictionary
        result = {
            '3d_cine': cine,
            '4d_flow': flow,
            'corrected_velocities': corr_vel,
            'velocity_correction_data': corr_data,
        }
        
        self._logger.info(f"Successfully built all images for patient {self.identifier}")
        return result
    
    def build_3d_cine_per_timepoint(self) -> None:
        """Build 3D cine volumes for each timepoint in original FOV."""
        
        self._logger.info(f"Building 3D cine volumes for each timepoint for patient {self.identifier}")
        
        cine_path = self.nifti_dir / f"3d_cine_{self.identifier}.nii.gz"
        output_dir = self.cine_per_timepoint_dir
        
        # if the cine does not exist, raise an error
        if not cine_path.exists():
            raise ValueError(f"3D cine for patient {self.identifier} does not exist")
        
        # the output directory is not empty and overwrite_images is False, log number of files
        if output_dir.exists() and len(list(output_dir.glob('*.nii.gz')))>0 and not self.overwrite_images:
            self._logger.info(f"Output directory {output_dir} already exists and overwrite_images is False, skipping")
            self._logger.info(f"Number of files in output directory: {len(list(output_dir.glob('*.nii.gz')))}")
            return
                
        # build the 3D cine volumes for each timepoint in original FOV
        converter = DicomToNiftiConverter.from_patient(self)
        converter.build_simple_per_timepoint(
            name=f"3d_cine_{self.identifier}",
            img_path=cine_path,
            output_dir=output_dir
        )
        
        self._logger.info(f"Successfully built 3D cine volumes for each timepoint for patient {self.identifier}")
        
    def build_4d_flow_per_timepoint(self) -> None:
        """Build 4D flow volumes for each timepoint, resampled to 3D cine FOV."""        
        
        self._logger.info(f"Building 4D flow volumes for each timepoint and component for patient {self.identifier}")
        
        flow_components = ['mag', 'vx', 'vy', 'vz']
        cine_path = self.nifti_dir / f"3d_cine_{self.identifier}.nii.gz"

        # Map each component to its instance path
        split_dirs = {
            'mag': self.flow_mag_per_timepoint_dir,
            'vx': self.flow_vx_per_timepoint_dir,
            'vy': self.flow_vy_per_timepoint_dir,
            'vz': self.flow_vz_per_timepoint_dir,
        }

        # Pair each flow file with its split output directory
        paths = [
            (
                comp,
                self.nifti_dir / f"4d_flow_{comp}_{self.identifier}.nii.gz",
                split_dirs[comp]
            )
            for comp in flow_components
        ]

        # # Check if 3D cine exists for reference FOV
        # if not cine_path.exists():
        #     raise ValueError(f"3D cine for patient {self.identifier} does not exist - needed for FOV reference")

        # Instantiate converter once
        converter = DicomToNiftiConverter.from_patient(self)

        # Run per-timepoint conversion with resampling to 3D cine FOV
        for comp, flow_path, split_path in paths:
            self._logger.info(f"Working on {flow_path}")
            
            if not flow_path.exists():
                raise ValueError(f"4D flow {comp} for patient {self.identifier} does not exist")
            
            if split_path.exists() and len(list(split_path.glob('*.nii.gz')))>0 and not self.overwrite_images:
                self._logger.info(f"Output directory {split_path} already exists and overwrite_images is False, skipping")
                self._logger.info(f"Number of files in output directory: {len(list(split_path.glob('*.nii.gz')))}")
                continue
            
            # Resample 4D flow component to 3D cine FOV and split into timepoints
            converter.build_resampled_per_timepoint(
                from_img_path=flow_path,      # Source: 4D flow component
                to_reference_path=cine_path,  # Reference: 3D cine FOV
                output_dir=split_path,
                name_prefix=f"4d_flow_{comp}_{self.identifier}"
            )
        
        self._logger.info(f"Successfully built 4D flow volumes for each timepoint and component for patient {self.identifier}")
    
    def build_4d_flow_per_timepoint_full_fov(self) -> None:
        """Build 4D flow volumes for each timepoint in original FOV (no resampling).
        
        This method creates per-timepoint volumes from the original 4D flow NIfTI files
        without resampling, preserving the full field of view. This is needed for inference
        to generate predictions for the entire original volume.
        """        
        self._logger.info(f"Building 4D flow volumes for each timepoint (full FOV) for patient {self.identifier}")
        
        flow_components = ['mag', 'vx', 'vy', 'vz']
        
        # Map each component to its full FOV split directory
        split_dirs = {
            'mag': self.flow_mag_per_timepoint_full_fov_dir,
            'vx': self.flow_vx_per_timepoint_full_fov_dir,
            'vy': self.flow_vy_per_timepoint_full_fov_dir,
            'vz': self.flow_vz_per_timepoint_full_fov_dir,
        }
        
        # Pair each flow file with its split output directory
        paths = [
            (
                comp,
                self.nifti_dir / f"4d_flow_{comp}_{self.identifier}.nii.gz",
                split_dirs[comp]
            )
            for comp in flow_components
        ]
        
        # Instantiate converter once
        converter = DicomToNiftiConverter.from_patient(self)
        
        # Run per-timepoint conversion WITHOUT resampling (use build_simple_per_timepoint)
        for comp, flow_path, split_path in paths:
            self._logger.info(f"Working on {flow_path} (full FOV)")
            
            if not flow_path.exists():
                raise ValueError(f"4D flow {comp} for patient {self.identifier} does not exist")
            
            if split_path.exists() and len(list(split_path.glob('*.nii.gz'))) > 0 and not self.overwrite_images:
                self._logger.info(f"Output directory {split_path} already exists and overwrite_images is False, skipping")
                self._logger.info(f"Number of files in output directory: {len(list(split_path.glob('*.nii.gz')))}")
                continue
            
            # Split into timepoints WITHOUT resampling
            converter.build_simple_per_timepoint(
                name=f"4d_flow_{comp}_{self.identifier}",
                img_path=flow_path,
                output_dir=split_path
            )
        
        self._logger.info(f"Successfully built 4D flow volumes (full FOV) for each timepoint for patient {self.identifier}")
    
    def build_3d_cine_per_timepoint_full_fov(self) -> None:
        """Build 3D cine volumes for each timepoint resampled to flow full FOV grid.
        
        Creates per-timepoint cine volumes in the flow magnitude full FOV grid,
        plus a binary support mask indicating where cine has valid coverage.
        """
        self._logger.info(f"Building 3D cine per timepoint (full FOV) for patient {self.identifier}")
        
        cine_path = self.nifti_dir / f"3d_cine_{self.identifier}.nii.gz"
        output_dir = self.cine_per_timepoint_full_fov_dir
        mask_path = self.nifti_dir / f"3d_cine_{self.identifier}_full_fov_mask.nii.gz"
        reference_path = self.flow_mag_per_timepoint_full_fov_dir / f"4d_flow_mag_{self.identifier}_frame_00.nii.gz"
        
        # Check if cine exists
        if not cine_path.exists():
            raise ValueError(f"3D cine for patient {self.identifier} does not exist")
        
        # Check if reference exists
        if not reference_path.exists():
            raise ValueError(
                f"Reference flow magnitude frame 00 not found: {reference_path}. "
                "Run build_4d_flow_per_timepoint_full_fov first."
            )
        
        # Check idempotency
        existing_frames = list(output_dir.glob("*.nii.gz"))
        if existing_frames and mask_path.exists() and not self.overwrite_images:
            self._logger.info(
                f"Output directory {output_dir} already has {len(existing_frames)} files "
                f"and mask exists. overwrite_images is False, skipping."
            )
            return
        
        # Instantiate converter
        converter = DicomToNiftiConverter.from_patient(self)
        
        # Resample cine to flow reference grid (opposite direction of build_4d_flow_per_timepoint)
        converter.build_resampled_per_timepoint(
            from_img_path=cine_path,           # Source: 4D cine
            to_reference_path=reference_path,   # Reference: flow mag full FOV
            output_dir=output_dir,
            name_prefix=f"3d_cine_{self.identifier}",
            mask_output_path=mask_path #if not mask_path.exists() or self.overwrite_images else None,
        )
        
        self._logger.info(f"Successfully built 3D cine per timepoint (full FOV) for patient {self.identifier}")
    
    def build_downsampled_full_fov_per_timepoint(
        self,
        target_size: tuple[int, int, int] = (128, 128, 64),
        poly_crop_tag: str | None = None,
        output_folder_name: str | None = None,
        reorient_non_axial: bool = False,
    ) -> None:
        """Build downsampled per-timepoint volumes in corrected velocity FOV.
        
        Uses the corrected velocity FOV (unpadded, with shifted affine) as the 
        reference. All data (mag, cine, cine mask) is resampled to this FOV and
        then downsampled to the target size.
        
        When ``reorient_non_axial=True``, sagittal and coronal acquisitions are
        physically reoriented onto a canonical (LR, AP, SI) image-axis layout
        with identity-LPS direction, with the through-plane slab zero-padded
        (LR for sagittal, AP for coronal). Axial acquisitions are unchanged.
        See "Sagittal Patient Integration" plan for full design.

        Args:
            target_size: Target voxel dimensions (X, Y, Z), default (128, 128, 64)
            poly_crop_tag: Optional tag appended to the default output folder
                name to distinguish runs with different polyfit crop settings
                (e.g. "crop-17.5" -> "downsampled_full_fov_128x128x64_crop-17.5").
                Ignored when ``output_folder_name`` is set.
            output_folder_name: Optional explicit folder name override. When set,
                the output is written to ``<nifti_dir>/<output_folder_name>/``
                directly, ignoring ``poly_crop_tag`` and the default
                ``downsampled_full_fov_<size_tag>`` naming. Useful for
                experimental builds that should not overwrite production data.
            reorient_non_axial: When True, sagittal/coronal patients are
                resampled onto a canonical axial-layout reference grid built
                via ``create_axial_aligned_reference_grid``; axial patients
                continue to use the existing ``create_downsampled_reference_grid``
                path. When False (default), all patients use the existing path
                regardless of acquisition orientation (preserves legacy
                behavior).
        """
        import SimpleITK as sitk
        import numpy as np
        import re
        
        size_tag = f"{target_size[0]}x{target_size[1]}x{target_size[2]}"
        self._logger.info(
            f"Building downsampled corrected FOV per timepoint ({size_tag}) for patient {self.identifier}"
        )
        
        # Create output root directory
        if output_folder_name is not None:
            folder_name = output_folder_name
        else:
            folder_name = f"downsampled_full_fov_{size_tag}"
            if poly_crop_tag:
                folder_name = f"{folder_name}_{poly_crop_tag}"
        output_root = self.nifti_dir / folder_name
        output_root.mkdir(parents=True, exist_ok=True)
        
        # Reference source: CORRECTED velocity per-timepoint frame 00 (unpadded FOV)
        reference_source_path = (
            self.flow_vx_corr_per_timepoint_dir / 
            f"4d_flow_vx_corr_{self.identifier}_frame_00.nii.gz"
        )
        
        if not reference_source_path.exists():
            raise ValueError(
                f"Reference corrected velocity frame 00 not found: {reference_source_path}. "
                "Run build_corrected_velocities_per_timepoint first."
            )
        
        # Load corrected velocity as reference. Choice of reference grid depends
        # on acquisition orientation when reorient_non_axial is enabled.
        source_img = sitk.ReadImage(str(reference_source_path))
        orientation = DicomToNiftiConverter.classify_orientation(source_img)
        self._logger.info(
            f"Acquisition orientation for {self.identifier}: {orientation} "
            f"(reorient_non_axial={reorient_non_axial})"
        )

        if reorient_non_axial and orientation == 'oblique':
            self._logger.warning(
                f"Patient {self.identifier} has oblique acquisition (slice "
                f"direction does not align with any patient axis above 0.9 "
                f"cosine). Skipping downsampled build; mark as skip in splits."
            )
            return

        if reorient_non_axial and orientation in ('sagittal', 'coronal'):
            reference_img = DicomToNiftiConverter.create_axial_aligned_reference_grid(
                source_img, target_size, orientation
            )
            self._logger.info(
                f"Using axial-aligned reference grid (identity LPS direction) "
                f"for {orientation} patient {self.identifier}"
            )
        else:
            reference_img = DicomToNiftiConverter.create_downsampled_reference_grid(
                source_img, target_size
            )
        
        self._logger.info(f"Corrected FOV size: {source_img.GetSize()}, spacing: {source_img.GetSpacing()}")
        self._logger.info(f"Target size: {reference_img.GetSize()}, spacing: {reference_img.GetSpacing()}")
        self._logger.info(f"Target direction: {reference_img.GetDirection()}")
        self._logger.info(f"Target origin: {reference_img.GetOrigin()}")
        
        # Save reference grid as debugging artifact
        reference_path = output_root / "reference.nii.gz"
        if not reference_path.exists() or self._should_overwrite('downsampled'):
            sitk.WriteImage(reference_img, str(reference_path))
            self._logger.info(f"Saved reference grid to {reference_path}")

        # Save orientation sidecar so the dataset loader can gate orientation-
        # specific augmentations (e.g. RandomLREdgeDropout axial-only).
        orientation_path = output_root / f"orientation_{self.identifier}.txt"
        if not orientation_path.exists() or self._should_overwrite('downsampled'):
            orientation_path.write_text(orientation + "\n")
            self._logger.info(f"Saved orientation sidecar to {orientation_path}")

        # Build the padding_support_mask BEFORE the per-stream loops below so
        # the mask is available even if a later step fails. The mask is a
        # constant-1 image at the source's affine resampled onto the new
        # reference grid; it is 1 wherever the source covers and 0 in the
        # zero-padded slab regions.
        padding_mask_path = output_root / f"padding_support_mask_{self.identifier}.nii.gz"
        if not padding_mask_path.exists() or self._should_overwrite('downsampled'):
            constant_one = sitk.Image(source_img.GetSize(), sitk.sitkUInt8)
            constant_one.CopyInformation(source_img)
            constant_one += 1  # all-ones at source's affine
            padding_mask = DicomToNiftiConverter.resample_to_target_grid(
                constant_one,
                reference_img,
                interpolator=sitk.sitkNearestNeighbor,
                default_value=0,
            )
            sitk.WriteImage(padding_mask, str(padding_mask_path))
            mask_arr = sitk.GetArrayFromImage(padding_mask)
            coverage = float(mask_arr.mean())
            self._logger.info(
                f"Saved padding_support_mask to {padding_mask_path.name} "
                f"(coverage = {coverage:.3f}; 1.0 = no padding, lower = more padding)"
            )
        
        converter = DicomToNiftiConverter.from_patient(self)
        
        # =====================================================================
        # Process corrected velocities (already in correct FOV, just downsample)
        # =====================================================================
        corr_vel_components = [
            ("4d_flow_vx_corr", self.flow_vx_corr_per_timepoint_dir, sitk.sitkLinear),
            ("4d_flow_vy_corr", self.flow_vy_corr_per_timepoint_dir, sitk.sitkLinear),
            ("4d_flow_vz_corr", self.flow_vz_corr_per_timepoint_dir, sitk.sitkLinear),
        ]
        
        for name, source_dir, interpolator in corr_vel_components:
            output_subdir = output_root / name
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            if not source_dir.exists() or not list(source_dir.glob("*.nii.gz")):
                self._logger.warning(f"Source directory {source_dir} is empty or missing, skipping {name}")
                continue
            
            existing_files = list(output_subdir.glob("*.nii.gz"))
            expected_files = list(source_dir.glob("*.nii.gz"))
            if existing_files and len(existing_files) >= len(expected_files) and not self._should_overwrite('downsampled'):
                self._logger.info(f"Output subdir {output_subdir} already has {len(existing_files)} files, skipping {name}")
                continue
            
            self._logger.info(f"Processing {name} (same FOV, just downsample)...")
            converter.build_downsampled_per_timepoint(
                source_dir=source_dir,
                output_dir=output_subdir,
                reference_img=reference_img,
                name_prefix=f"{name}_{self.identifier}",
                interpolator=interpolator,
                default_value=0.0,
            )
        
        # =====================================================================
        # Process padded data: resample from padded FOV to corrected FOV, then downsample
        # =====================================================================
        # These are in the padded FOV and need to be resampled to corrected FOV
        padded_components = [
            ("4d_flow_mag", self.flow_mag_per_timepoint_full_fov_dir, sitk.sitkLinear),
            ("4d_flow_vx", self.flow_vx_per_timepoint_full_fov_dir, sitk.sitkLinear),
            ("4d_flow_vy", self.flow_vy_per_timepoint_full_fov_dir, sitk.sitkLinear),
            ("4d_flow_vz", self.flow_vz_per_timepoint_full_fov_dir, sitk.sitkLinear),
            ("3d_cine", self.cine_per_timepoint_full_fov_dir, sitk.sitkLinear),
        ]
        
        for name, source_dir, interpolator in padded_components:
            output_subdir = output_root / name
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            if not source_dir.exists() or not list(source_dir.glob("*.nii.gz")):
                self._logger.warning(f"Source directory {source_dir} is empty or missing, skipping {name}")
                continue
            
            existing_files = list(output_subdir.glob("*.nii.gz"))
            expected_files = list(source_dir.glob("*.nii.gz"))
            if existing_files and len(existing_files) >= len(expected_files) and not self._should_overwrite('downsampled'):
                self._logger.info(f"Output subdir {output_subdir} already has {len(existing_files)} files, skipping {name}")
                continue
            
            # Resample from padded FOV -> corrected FOV -> downsampled
            self._logger.info(f"Processing {name} (resample from padded to corrected FOV, then downsample)...")
            converter.build_downsampled_per_timepoint(
                source_dir=source_dir,
                output_dir=output_subdir,
                reference_img=reference_img,
                name_prefix=f"{name}_{self.identifier}",
                interpolator=interpolator,
                default_value=0.0,
            )
        
        # =====================================================================
        # Process cine mask (single 3D file, resample to corrected FOV + downsample)
        # =====================================================================
        cine_mask_path = self.nifti_dir / f"3d_cine_{self.identifier}_full_fov_mask.nii.gz"
        cine_mask_output_path = output_root / f"3d_cine_mask_{self.identifier}.nii.gz"
        
        if cine_mask_path.exists():
            if not cine_mask_output_path.exists() or self._should_overwrite('downsampled'):
                self._logger.info("Processing cine_mask (resample to corrected FOV + downsample)...")
                mask_img = sitk.ReadImage(str(cine_mask_path))
                resampled_mask = DicomToNiftiConverter.resample_to_target_grid(
                    mask_img, reference_img, 
                    interpolator=sitk.sitkNearestNeighbor,
                    default_value=0.0
                )
                sitk.WriteImage(resampled_mask, str(cine_mask_output_path))
                self._logger.info(f"Saved resampled cine mask to {cine_mask_output_path}")
            else:
                self._logger.info(f"Cine mask already exists at {cine_mask_output_path}, skipping")
        else:
            self._logger.warning(f"Cine mask not found at {cine_mask_path}, skipping")
        
        # =====================================================================
        # Compute speed from downsampled corrected velocities
        # =====================================================================
        speed_output_dir = output_root / "4d_flow_speed_corr"
        speed_output_dir.mkdir(parents=True, exist_ok=True)
        
        vx_dir = output_root / "4d_flow_vx_corr"
        vy_dir = output_root / "4d_flow_vy_corr"
        vz_dir = output_root / "4d_flow_vz_corr"
        
        if vx_dir.exists() and vy_dir.exists() and vz_dir.exists():
            vx_files = sorted(vx_dir.glob("*.nii.gz"))
            
            existing_speed_files = list(speed_output_dir.glob("*.nii.gz"))
            if existing_speed_files and len(existing_speed_files) >= len(vx_files) and not self._should_overwrite('downsampled'):
                self._logger.info(f"Speed output dir already has {len(existing_speed_files)} files, skipping")
            else:
                self._logger.info("Computing speed from downsampled corrected velocity components...")
                
                for vx_file in vx_files:
                    match = re.search(r'frame_(\d+)', vx_file.name)
                    if not match:
                        continue
                    frame_num = int(match.group(1))
                    
                    vy_file = vy_dir / f"4d_flow_vy_corr_{self.identifier}_frame_{frame_num:02d}.nii.gz"
                    vz_file = vz_dir / f"4d_flow_vz_corr_{self.identifier}_frame_{frame_num:02d}.nii.gz"
                    speed_file = speed_output_dir / f"4d_flow_speed_corr_{self.identifier}_frame_{frame_num:02d}.nii.gz"
                    
                    if not vy_file.exists() or not vz_file.exists():
                        self._logger.warning(f"Missing velocity component for frame {frame_num}, skipping")
                        continue
                    
                    vx_img = sitk.ReadImage(str(vx_file))
                    vy_img = sitk.ReadImage(str(vy_file))
                    vz_img = sitk.ReadImage(str(vz_file))
                    
                    vx_arr = sitk.GetArrayFromImage(vx_img).astype(np.float32)
                    vy_arr = sitk.GetArrayFromImage(vy_img).astype(np.float32)
                    vz_arr = sitk.GetArrayFromImage(vz_img).astype(np.float32)
                    
                    speed_arr = np.sqrt(vx_arr**2 + vy_arr**2 + vz_arr**2)
                    
                    speed_img = sitk.GetImageFromArray(speed_arr)
                    speed_img.CopyInformation(vx_img)
                    
                    sitk.WriteImage(speed_img, str(speed_file))
                
                self._logger.info(f"Saved {len(vx_files)} corrected speed volumes to {speed_output_dir}")
        else:
            self._logger.warning("Downsampled corrected velocity directories not found, skipping corrected speed computation")
        
        # =====================================================================
        # Compute speed from downsampled uncorrected velocities
        # =====================================================================
        speed_uncorr_output_dir = output_root / "4d_flow_speed"
        speed_uncorr_output_dir.mkdir(parents=True, exist_ok=True)
        
        vx_uncorr_dir = output_root / "4d_flow_vx"
        vy_uncorr_dir = output_root / "4d_flow_vy"
        vz_uncorr_dir = output_root / "4d_flow_vz"
        
        if vx_uncorr_dir.exists() and vy_uncorr_dir.exists() and vz_uncorr_dir.exists():
            vx_uncorr_files = sorted(vx_uncorr_dir.glob("*.nii.gz"))
            
            existing_speed_files = list(speed_uncorr_output_dir.glob("*.nii.gz"))
            if existing_speed_files and len(existing_speed_files) >= len(vx_uncorr_files) and not self._should_overwrite('downsampled'):
                self._logger.info(f"Uncorrected speed output dir already has {len(existing_speed_files)} files, skipping")
            else:
                self._logger.info("Computing speed from downsampled uncorrected velocity components...")
                
                for vx_file in vx_uncorr_files:
                    match = re.search(r'frame_(\d+)', vx_file.name)
                    if not match:
                        continue
                    frame_num = int(match.group(1))
                    
                    vy_file = vy_uncorr_dir / f"4d_flow_vy_{self.identifier}_frame_{frame_num:02d}.nii.gz"
                    vz_file = vz_uncorr_dir / f"4d_flow_vz_{self.identifier}_frame_{frame_num:02d}.nii.gz"
                    speed_file = speed_uncorr_output_dir / f"4d_flow_speed_{self.identifier}_frame_{frame_num:02d}.nii.gz"
                    
                    if not vy_file.exists() or not vz_file.exists():
                        self._logger.warning(f"Missing uncorrected velocity component for frame {frame_num}, skipping")
                        continue
                    
                    vx_img = sitk.ReadImage(str(vx_file))
                    vy_img = sitk.ReadImage(str(vy_file))
                    vz_img = sitk.ReadImage(str(vz_file))
                    
                    vx_arr = sitk.GetArrayFromImage(vx_img).astype(np.float32)
                    vy_arr = sitk.GetArrayFromImage(vy_img).astype(np.float32)
                    vz_arr = sitk.GetArrayFromImage(vz_img).astype(np.float32)
                    
                    speed_arr = np.sqrt(vx_arr**2 + vy_arr**2 + vz_arr**2)
                    
                    speed_img = sitk.GetImageFromArray(speed_arr)
                    speed_img.CopyInformation(vx_img)
                    
                    sitk.WriteImage(speed_img, str(speed_file))
                
                self._logger.info(f"Saved {len(vx_uncorr_files)} uncorrected speed volumes to {speed_uncorr_output_dir}")
        else:
            self._logger.warning("Downsampled uncorrected velocity directories not found, skipping uncorrected speed computation")
        
        # =====================================================================
        # Downsample diff per-timepoint (corrected - uncorrected)
        # =====================================================================
        diff_components = [
            ("4d_flow_diff_vx", self.flow_diff_vx_per_timepoint_dir),
            ("4d_flow_diff_vy", self.flow_diff_vy_per_timepoint_dir),
            ("4d_flow_diff_vz", self.flow_diff_vz_per_timepoint_dir),
        ]
        
        for name, source_dir in diff_components:
            output_subdir = output_root / name
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            if not source_dir.exists() or not list(source_dir.glob("*.nii.gz")):
                self._logger.warning(f"Diff source directory {source_dir} is empty or missing, skipping {name}")
                continue
            
            existing_files = list(output_subdir.glob("*.nii.gz"))
            expected_files = list(source_dir.glob("*.nii.gz"))
            if existing_files and len(existing_files) >= len(expected_files) and not self._should_overwrite('downsampled'):
                self._logger.info(f"Output subdir {output_subdir} already has {len(existing_files)} files, skipping {name}")
                continue
            
            # Diff is already in corrected FOV, just downsample
            self._logger.info(f"Processing {name} (already in corrected FOV, just downsample)...")
            converter.build_downsampled_per_timepoint(
                source_dir=source_dir,
                output_dir=output_subdir,
                reference_img=reference_img,
                name_prefix=f"{name}_{self.identifier}",
                interpolator=sitk.sitkLinear,
                default_value=0.0,
            )
        
        # =====================================================================
        # Reconstruct ground truth correction at downsampled resolution using polyfit
        # =====================================================================
        # Choose the velocity_correction source directory based on the crop tag
        # that produced this downsampled folder. Without this, the hardcoded
        # ``self.velocity_correction_dir`` property would always point at
        # ``velocity_correction_crop-17.5/`` regardless of which crop was just
        # written by Phase 4, silently mixing crops.
        if poly_crop_tag:
            vc_source = self.nifti_dir / f"velocity_correction_{poly_crop_tag}"
        else:
            vc_source = self.nifti_dir / "velocity_correction"
        coefficients_path = vc_source / f"poly_coefficients_{self.identifier}.npz"

        if coefficients_path.exists():
            gt_vx_output = output_root / f"ground_truth_correction_vx_{self.identifier}.nii.gz"
            gt_vy_output = output_root / f"ground_truth_correction_vy_{self.identifier}.nii.gz"
            gt_vz_output = output_root / f"ground_truth_correction_vz_{self.identifier}.nii.gz"
            
            if (gt_vx_output.exists() and gt_vy_output.exists() and gt_vz_output.exists() 
                and not self._should_overwrite('downsampled')):
                self._logger.info("Downsampled ground truth correction already exists, skipping")
            else:
                self._logger.info(
                    f"Reconstructing ground truth correction at downsampled resolution "
                    f"from {vc_source.name}..."
                )
                
                # Load coefficients
                coeff_data = np.load(coefficients_path)
                coefficients = coeff_data['coefficients']  # Shape (n_coeffs, 3)
                n_coeffs = coefficients.shape[0]
                
                # Build polynomial basis at target (downsampled) shape
                # Uses normalized [-1, 1] coordinates, so same coefficients work at any resolution
                target_basis = self._build_polynomial_basis(target_size, n_coeffs)
                
                # Reconstruct correction (stays VENC-normalized, consistent with full-res ground truth)
                ground_truth = self._reconstruct_from_coefficients(coefficients, target_basis, target_size)
                
                # Get affine from downsampled reference grid
                import nibabel as nib
                ref_nib = nib.load(reference_path)
                ds_affine = ref_nib.affine
                
                # Save each component
                self._save_nifti(ground_truth['vx'], ds_affine, gt_vx_output, 
                                "downsampled ground truth correction vx")
                self._save_nifti(ground_truth['vy'], ds_affine, gt_vy_output, 
                                "downsampled ground truth correction vy")
                self._save_nifti(ground_truth['vz'], ds_affine, gt_vz_output, 
                                "downsampled ground truth correction vz")
                
                self._logger.info(f"Saved downsampled ground truth corrections to {output_root}")
            
            # Downsample both masks (nearest-neighbor to stay binary).
            # tissue_mask: training-time tissue/air weighting input.
            # fit_mask: provenance/QA — voxels used during polyfit.
            for mask_kind in ('tissue', 'fit'):
                mask_source = vc_source / f"correction_{mask_kind}_mask_{self.identifier}.nii.gz"
                mask_output = output_root / f"correction_{mask_kind}_mask_{self.identifier}.nii.gz"
                if mask_source.exists() and (not mask_output.exists() or self._should_overwrite('downsampled')):
                    mask_img = sitk.ReadImage(str(mask_source))
                    resampled_mask = sitk.Resample(
                        mask_img, reference_img,
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,
                        0.0,
                        mask_img.GetPixelID(),
                    )
                    sitk.WriteImage(resampled_mask, str(mask_output))
                    self._logger.info(f"Saved downsampled {mask_kind} mask to {mask_output.name}")
        else:
            self._logger.info(f"No polynomial coefficients found at {coefficients_path}, skipping ground truth reconstruction")
        
        self._logger.info(
            f"Successfully built downsampled full FOV per timepoint ({size_tag}) for patient {self.identifier}"
        )
    
    def build_speed_per_timepoint(self) -> None:
        """Compute and save speed volumes from vx, vy, vz for each timepoint.
        
        Speed is computed as sqrt(vx^2 + vy^2 + vz^2) and saved per-timepoint.
        This is used for efficient training (avoids repeated speed computation).
        
        Prerequisites:
            build_4d_flow_per_timepoint() must be run first to create vx, vy, vz volumes.
        """
        import SimpleITK as sitk
        import numpy as np
        
        self._logger.info(f"Building speed volumes for each timepoint for patient {self.identifier}")
        
        output_dir = self.flow_speed_per_timepoint_dir
        
        # Check if already built
        if output_dir.exists() and len(list(output_dir.glob('*.nii.gz'))) > 0 and not self.overwrite_images:
            self._logger.info(f"Output directory {output_dir} already exists and overwrite_images is False, skipping")
            self._logger.info(f"Number of files in output directory: {len(list(output_dir.glob('*.nii.gz')))}")
            return
        
        # Get number of timepoints from existing velocity files
        vx_files = sorted(self.flow_vx_per_timepoint_dir.glob('*.nii.gz'))
        if not vx_files:
            raise ValueError(f"No vx per-timepoint files found for patient {self.identifier}. Run build_4d_flow_per_timepoint first.")
        
        num_timepoints = len(vx_files)
        self._logger.info(f"Computing speed for {num_timepoints} timepoints")
        
        for t in range(num_timepoints):
            vx_path = self.flow_vx_per_timepoint_dir / f'4d_flow_vx_{self.identifier}_frame_{t:02d}.nii.gz'
            vy_path = self.flow_vy_per_timepoint_dir / f'4d_flow_vy_{self.identifier}_frame_{t:02d}.nii.gz'
            vz_path = self.flow_vz_per_timepoint_dir / f'4d_flow_vz_{self.identifier}_frame_{t:02d}.nii.gz'
            speed_path = output_dir / f'4d_flow_speed_{self.identifier}_frame_{t:02d}.nii.gz'
            
            # Load velocity components
            vx_img = sitk.ReadImage(str(vx_path))
            vy_img = sitk.ReadImage(str(vy_path))
            vz_img = sitk.ReadImage(str(vz_path))
            
            # Convert to numpy, compute speed, convert back
            vx = sitk.GetArrayFromImage(vx_img).astype(np.float32)
            vy = sitk.GetArrayFromImage(vy_img).astype(np.float32)
            vz = sitk.GetArrayFromImage(vz_img).astype(np.float32)
            
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            
            # Create image with same metadata as vx
            speed_img = sitk.GetImageFromArray(speed)
            speed_img.CopyInformation(vx_img)
            
            sitk.WriteImage(speed_img, str(speed_path))
        
        self._logger.info(f"Successfully built speed volumes for each timepoint for patient {self.identifier}")
    
    def build_per_timepoint_images(self) -> None:
        """Build basic per-timepoint volumes for 3d cine and 4d flow.

        Splits composite 4D NIfTIs into per-timepoint 3D volumes in the
        full (padded) 4D flow FOV. The legacy cine-FOV per-timepoint
        outputs (``3d_cine_{id}_per_timepoint/``,
        ``4d_flow_{mag,vx,vy,vz}_{id}_per_timepoint/``,
        ``4d_flow_speed_{id}_per_timepoint/``) are no longer built — the
        active dual-task pipeline operates entirely in the full/corrected
        4D flow FOV. The underlying methods (``build_3d_cine_per_timepoint``,
        ``build_4d_flow_per_timepoint``, ``build_speed_per_timepoint``)
        remain on the class for ad-hoc use.

        Outputs:
            Full/Padded FOV (native 4D flow resolution):
                - 4d_flow_mag/vx/vy/vz_{id}_per_timepoint_full_fov/
                - 3d_cine_{id}_per_timepoint_full_fov/
                - 3d_cine_{id}_full_fov_mask.nii.gz

        Note:
            For corrected velocity processing, call build_corrected_velocity_pipeline()
            For downsampling, call build_downsampled_full_fov_per_timepoint()
        """
        self._logger.info(f"Building per-timepoint volumes for patient {self.identifier}")

        # 4D flow per-timepoint (full/padded FOV, no resampling)
        try:
            self.build_4d_flow_per_timepoint_full_fov()
            self._logger.info(f"Successfully built 4d flow per timepoint (full FOV) for patient {self.identifier}")
        except Exception as e:
            self._logger.error(f"Error building 4d flow per timepoint (full FOV) for patient {self.identifier}: {e}")

        # 3D cine per-timepoint (resampled to full/padded FOV)
        try:
            self.build_3d_cine_per_timepoint_full_fov()
            self._logger.info(f"Successfully built 3d cine per timepoint (full FOV) for patient {self.identifier}")
        except Exception as e:
            self._logger.error(f"Error building 3d cine per timepoint (full FOV) for patient {self.identifier}: {e}")

        self._logger.info(f"Completed build_per_timepoint_images for patient {self.identifier}")
    
    def build_corrected_velocity_pipeline(self) -> None:
        """Build all corrected velocity per-timepoint data in the corrected (unpadded) FOV.
        
        This method processes corrected velocities and creates all per-timepoint
        intermediate files needed for velocity correction training.
        
        Prerequisites:
            - build_corrected_velocities() must be run first (creates composite NIfTIs)
            - build_per_timepoint_images() must be run first (creates full FOV per-timepoint)
        
        Outputs:
            Corrected velocities (native corrected FOV):
                - 4d_flow_vx/vy/vz_corr_{id}_per_timepoint/
                - 4d_flow_speed_corr_{id}_per_timepoint/
            
            Uncorrected resampled to corrected FOV:
                - 4d_flow_mag/vx/vy/vz_{id}_per_timepoint_corr_fov/
            
            Cine resampled to corrected FOV:
                - 3d_cine_{id}_per_timepoint_corr_fov/
                - 3d_cine_{id}_corr_fov_mask.nii.gz
            
            Diff (corrected - uncorrected):
                - 4d_flow_diff_vx/vy/vz_{id}_per_timepoint/
        
        Note:
            After running this, call build_velocity_correction_data() for polynomial fitting.
        """
        self._logger.info(f"Building corrected velocity pipeline for patient {self.identifier}")
        
        # Check prerequisites
        vx_corr_path = self.nifti_dir / f"4d_flow_vx_corr_{self.identifier}.nii.gz"
        if not vx_corr_path.exists():
            raise FileNotFoundError(
                f"Corrected velocity NIfTI not found: {vx_corr_path}. "
                "Run build_corrected_velocities() first."
            )
        
        # 1. Corrected velocities per-timepoint
        try:
            self.build_corrected_velocities_per_timepoint()
            self._logger.info(f"Successfully built corrected velocities per timepoint")
        except Exception as e:
            self._logger.error(f"Error building corrected velocities per timepoint: {e}")
            raise
        
        # 2. Corrected speed per-timepoint
        try:
            self.build_corrected_speed_per_timepoint()
            self._logger.info(f"Successfully built corrected speed per timepoint")
        except Exception as e:
            self._logger.error(f"Error building corrected speed per timepoint: {e}")
            raise
        
        # 3. Uncorrected velocities + magnitude resampled to corrected FOV
        try:
            self.build_uncorrected_per_timepoint_corr_fov()
            self._logger.info(f"Successfully built uncorrected per-timepoint in corrected FOV")
        except Exception as e:
            self._logger.error(f"Error building uncorrected per-timepoint in corrected FOV: {e}")
            raise
        
        # 4. Cine + mask resampled to corrected FOV
        try:
            self.build_cine_per_timepoint_corr_fov()
            self._logger.info(f"Successfully built cine per-timepoint in corrected FOV")
        except Exception as e:
            self._logger.error(f"Error building cine per-timepoint in corrected FOV: {e}")
            raise
        
        # 5. Diff (corrected - uncorrected) per-timepoint
        try:
            self.build_diff_per_timepoint()
            self._logger.info(f"Successfully built diff per-timepoint")
        except Exception as e:
            self._logger.error(f"Error building diff per-timepoint: {e}")
            raise
        
        self._logger.info(f"Completed build_corrected_velocity_pipeline for patient {self.identifier}")
    
    def build_corrected_velocities(self) -> dict[str, Path]:
        """Process phase-error-corrected velocity fields and save as NIfTI.
        
        The corrected velocities from the numpy files are smaller than the original
        DICOM velocities because they don't include the partial phase FOV padding.
        This method saves the corrected data in its native (unpadded) dimensions
        with an affine that has the origin shifted to account for the missing
        padding. This places the data at the correct physical location.
        
        Prerequisites:
            - get_4d_flow() must be run first to create velocity NIfTIs from DICOM
            - Corrected velocity numpy file must exist at corrected_velocity_numpy_path
        
        Returns:
            Dictionary mapping component names to output paths
        """
        self._logger.info(f"Building corrected velocities for patient {self.identifier}")
        
        if not self.corrected_velocity_numpy_path.exists():
            raise FileNotFoundError(
                f"Corrected velocities not found: {self.corrected_velocity_numpy_path}"
            )
        
        output_paths = {
            'vx_corr': self.nifti_dir / f"4d_flow_vx_corr_{self.identifier}.nii.gz",
            'vy_corr': self.nifti_dir / f"4d_flow_vy_corr_{self.identifier}.nii.gz",
            'vz_corr': self.nifti_dir / f"4d_flow_vz_corr_{self.identifier}.nii.gz",
        }
        
        if all(p.exists() for p in output_paths.values()) and not self._should_overwrite('corrected'):
            self._logger.info("Corrected velocities already exist, skipping")
            return output_paths
        
        # Load reference NIfTI for affine and padded shape
        ref_nifti_path = self.nifti_dir / f"4d_flow_vx_{self.identifier}.nii.gz"
        if not ref_nifti_path.exists():
            raise FileNotFoundError(
                f"Reference velocity NIfTI not found: {ref_nifti_path}. "
                "Run get_4d_flow() first."
            )
        
        ref_nifti = nib.load(ref_nifti_path)
        padded_affine = ref_nifti.affine
        padded_shape = ref_nifti.shape  # (X, Y, Z, T)
        self._logger.info(f"Reference (padded) NIfTI shape: {padded_shape}")
        
        # Load corrected velocities
        # Shape: (T, 3, Z, Y, X) where channels 0,1,2 = vx, vy, vz
        corr_vel = np.load(self.corrected_velocity_numpy_path).astype(np.float32)
        self._logger.info(f"Corrected numpy shape: {corr_vel.shape}")
        
        T_corr, C_corr, Z_corr, Y_corr, X_corr = corr_vel.shape
        if C_corr != 3:
            raise ValueError(f"Expected 3 velocity channels, got {C_corr}")
        
        # Compute padding amounts (what was cropped from the corrected data)
        pad_X = padded_shape[0] - X_corr
        pad_Y = padded_shape[1] - Y_corr
        pad_Z = padded_shape[2] - Z_corr
        
        if pad_X < 0 or pad_Y < 0 or pad_Z < 0:
            raise ValueError(
                f"Corrected dimensions larger than reference: "
                f"Corrected ({X_corr}, {Y_corr}, {Z_corr}) vs Padded ({padded_shape[:3]})"
            )
        
        pad_before = (pad_X // 2, pad_Y // 2, pad_Z // 2)
        
        self._logger.info(
            f"Unpadded shape: ({X_corr}, {Y_corr}, {Z_corr}), pad_before: {pad_before}"
        )
        
        # Verify timepoints match
        if T_corr != padded_shape[3]:
            self._logger.warning(f"Timepoint mismatch! Corrected={T_corr}, Reference={padded_shape[3]}")
        
        # Compute unpadded affine: shift origin by pad_before voxels
        unpadded_affine = self._compute_unpadded_affine(padded_affine, pad_before)
        
        self._logger.debug(
            f"Origin shift: {unpadded_affine[:3, 3] - padded_affine[:3, 3]}"
        )
        
        # Process each velocity component
        component_map = {0: 'vx_corr', 1: 'vy_corr', 2: 'vz_corr'}
        
        for channel_idx, comp_name in component_map.items():
            # Extract component: shape (T, Z, Y, X)
            vel_component = corr_vel[:, channel_idx, :, :, :]
            
            # Negate vz (downloaded data has wrong direction)
            if channel_idx == 2:
                vel_component = -vel_component
                self._logger.debug("Negated vz component")
            
            # Transpose to NIfTI convention: (T, Z, Y, X) -> (X, Y, Z, T)
            vel_nifti = np.transpose(vel_component, (3, 2, 1, 0))
            
            # Create NIfTI with unpadded affine (NO PADDING)
            nii = nib.Nifti1Image(vel_nifti, unpadded_affine)
            nii.set_qform(unpadded_affine, code=1)
            nii.set_sform(unpadded_affine, code=1)
            
            hdr = nii.header
            hdr['dim'][0] = 4
            hdr['dim'][4] = vel_nifti.shape[3]
            hdr['pixdim'][4] = 1.0
            hdr['xyzt_units'] = 2 | 8  # mm + seconds
            
            output_path = output_paths[comp_name]
            nib.save(nii, output_path)
            self._logger.info(f"Saved {comp_name} to {output_path}, shape={vel_nifti.shape}")
        
        return output_paths
    
    def build_corrected_velocities_per_timepoint(self) -> None:
        """Build per-timepoint volumes from corrected velocities in full FOV (no resampling).
        
        Prerequisites:
            - build_corrected_velocities() must be run first to create corrected velocity NIfTIs
        """
        self._logger.info(f"Building corrected velocity per-timepoint volumes (full FOV) for patient {self.identifier}")
        
        # Map each corrected velocity component to its paths
        components = {
            'vx_corr': (
                self.nifti_dir / f"4d_flow_vx_corr_{self.identifier}.nii.gz",
                self.flow_vx_corr_per_timepoint_dir,
            ),
            'vy_corr': (
                self.nifti_dir / f"4d_flow_vy_corr_{self.identifier}.nii.gz",
                self.flow_vy_corr_per_timepoint_dir,
            ),
            'vz_corr': (
                self.nifti_dir / f"4d_flow_vz_corr_{self.identifier}.nii.gz",
                self.flow_vz_corr_per_timepoint_dir,
            ),
        }
        
        # Instantiate converter once
        converter = DicomToNiftiConverter.from_patient(self)
        
        for comp_name, (flow_path, split_dir) in components.items():
            self._logger.info(f"Working on {comp_name} (full FOV)")
            
            if not flow_path.exists():
                self._logger.warning(
                    f"Corrected velocity {comp_name} not found at {flow_path}, skipping. "
                    "Run build_corrected_velocities() first."
                )
                continue
            
            # Check idempotency
            if split_dir.exists() and len(list(split_dir.glob('*.nii.gz'))) > 0 and not self._should_overwrite('corrected'):
                self._logger.info(f"Output directory {split_dir} already exists, skipping")
                self._logger.info(f"Number of files: {len(list(split_dir.glob('*.nii.gz')))}")
                continue
            
            # Split into per-timepoint volumes WITHOUT resampling (full FOV)
            converter.build_simple_per_timepoint(
                name=f"4d_flow_{comp_name}_{self.identifier}",
                img_path=flow_path,
                output_dir=split_dir,
            )
        
        self._logger.info(f"Successfully built corrected velocity per-timepoint volumes (full FOV) for patient {self.identifier}")
    
    def build_corrected_speed_per_timepoint(self) -> None:
        """Compute and save corrected speed volumes from vx_corr, vy_corr, vz_corr per-timepoint.
        
        Speed is computed as sqrt(vx_corr^2 + vy_corr^2 + vz_corr^2).
        
        Prerequisites:
            build_corrected_velocities_per_timepoint() must be run first.
        """
        import SimpleITK as sitk
        
        self._logger.info(f"Building corrected speed per-timepoint for patient {self.identifier}")
        
        output_dir = self.flow_speed_corr_per_timepoint_dir
        
        # Check if already built
        if output_dir.exists() and len(list(output_dir.glob('*.nii.gz'))) > 0 and not self._should_overwrite('corrected'):
            self._logger.info(f"Output directory {output_dir} already exists, skipping")
            self._logger.info(f"Number of files: {len(list(output_dir.glob('*.nii.gz')))}")
            return
        
        # Get number of timepoints from vx_corr files
        vx_files = sorted(self.flow_vx_corr_per_timepoint_dir.glob('*.nii.gz'))
        if not vx_files:
            self._logger.warning(
                f"No vx_corr per-timepoint files found. "
                "Run build_corrected_velocities_per_timepoint() first."
            )
            return
        
        num_timepoints = len(vx_files)
        self._logger.info(f"Computing corrected speed for {num_timepoints} timepoints")
        
        for t in range(num_timepoints):
            vx_path = self.flow_vx_corr_per_timepoint_dir / f'4d_flow_vx_corr_{self.identifier}_frame_{t:02d}.nii.gz'
            vy_path = self.flow_vy_corr_per_timepoint_dir / f'4d_flow_vy_corr_{self.identifier}_frame_{t:02d}.nii.gz'
            vz_path = self.flow_vz_corr_per_timepoint_dir / f'4d_flow_vz_corr_{self.identifier}_frame_{t:02d}.nii.gz'
            speed_path = output_dir / f'4d_flow_speed_corr_{self.identifier}_frame_{t:02d}.nii.gz'
            
            if not vx_path.exists() or not vy_path.exists() or not vz_path.exists():
                self._logger.warning(f"Missing corrected velocity component for frame {t}, skipping")
                continue
            
            # Load velocity components
            vx_img = sitk.ReadImage(str(vx_path))
            vy_img = sitk.ReadImage(str(vy_path))
            vz_img = sitk.ReadImage(str(vz_path))
            
            # Convert to numpy, compute speed, convert back
            vx = sitk.GetArrayFromImage(vx_img).astype(np.float32)
            vy = sitk.GetArrayFromImage(vy_img).astype(np.float32)
            vz = sitk.GetArrayFromImage(vz_img).astype(np.float32)
            
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            
            # Create image with same metadata as vx
            speed_img = sitk.GetImageFromArray(speed)
            speed_img.CopyInformation(vx_img)
            
            sitk.WriteImage(speed_img, str(speed_path))
        
        self._logger.info(f"Successfully built corrected speed per-timepoint for patient {self.identifier}")
    
    def build_uncorrected_per_timepoint_corr_fov(self) -> None:
        """Resample uncorrected velocity and magnitude per-timepoint files to corrected FOV.
        
        Takes the per-timepoint files in padded FOV (from build_4d_flow_per_timepoint_full_fov)
        and resamples them to the corrected (unpadded) FOV using the corrected velocity files
        as reference.
        
        Outputs:
            - 4d_flow_mag_{id}_per_timepoint_corr_fov/
            - 4d_flow_vx_{id}_per_timepoint_corr_fov/
            - 4d_flow_vy_{id}_per_timepoint_corr_fov/
            - 4d_flow_vz_{id}_per_timepoint_corr_fov/
        
        Prerequisites:
            - build_corrected_velocities_per_timepoint() must be run first (for reference FOV)
            - build_4d_flow_per_timepoint_full_fov() must be run first (for source data)
        """
        import SimpleITK as sitk
        
        self._logger.info(f"Building uncorrected per-timepoint in corrected FOV for patient {self.identifier}")
        
        # Source (padded FOV) and output (corrected FOV) directories
        components = {
            'mag': (self.flow_mag_per_timepoint_full_fov_dir, self.flow_mag_per_timepoint_corr_fov_dir),
            'vx': (self.flow_vx_per_timepoint_full_fov_dir, self.flow_vx_per_timepoint_corr_fov_dir),
            'vy': (self.flow_vy_per_timepoint_full_fov_dir, self.flow_vy_per_timepoint_corr_fov_dir),
            'vz': (self.flow_vz_per_timepoint_full_fov_dir, self.flow_vz_per_timepoint_corr_fov_dir),
        }
        
        # Use corrected vx as reference FOV
        ref_dir = self.flow_vx_corr_per_timepoint_dir
        ref_files = sorted(ref_dir.glob("*.nii.gz"))
        if not ref_files:
            raise FileNotFoundError(
                f"No corrected vx per-timepoint files found in {ref_dir}. "
                "Run build_corrected_velocities_per_timepoint() first."
            )
        
        num_timepoints = len(ref_files)
        self._logger.info(f"Processing {num_timepoints} timepoints")
        
        for comp, (source_dir, output_dir) in components.items():
            # Check if already built
            existing_files = list(output_dir.glob("*.nii.gz"))
            if len(existing_files) >= num_timepoints and not self._should_overwrite('corrected'):
                self._logger.info(f"{comp} already has {len(existing_files)} files in corr_fov, skipping")
                continue
            
            # Check source exists
            source_files = sorted(source_dir.glob("*.nii.gz"))
            if not source_files:
                self._logger.warning(f"No source files in {source_dir}, skipping {comp}")
                continue
            
            self._logger.info(f"Resampling {comp} to corrected FOV...")
            
            for t in range(num_timepoints):
                # Reference file (corrected FOV)
                ref_file = ref_dir / f"4d_flow_vx_corr_{self.identifier}_frame_{t:02d}.nii.gz"
                
                # Source file (padded FOV)
                source_file = source_dir / f"4d_flow_{comp}_{self.identifier}_frame_{t:02d}.nii.gz"
                
                # Output file
                output_file = output_dir / f"4d_flow_{comp}_{self.identifier}_frame_{t:02d}.nii.gz"
                
                if not ref_file.exists():
                    self._logger.warning(f"Reference file not found: {ref_file}, skipping frame {t}")
                    continue
                if not source_file.exists():
                    self._logger.warning(f"Source file not found: {source_file}, skipping frame {t}")
                    continue
                
                # Load and resample
                ref_img = sitk.ReadImage(str(ref_file))
                source_img = sitk.ReadImage(str(source_file))
                
                resampled = DicomToNiftiConverter.resample_to_target_grid(
                    moving_img=source_img,
                    reference_img=ref_img,
                    interpolator=sitk.sitkLinear,
                    default_value=0.0,
                )
                
                sitk.WriteImage(resampled, str(output_file))
        
        self._logger.info(f"Successfully built uncorrected per-timepoint in corrected FOV for patient {self.identifier}")
    
    def build_cine_per_timepoint_corr_fov(self) -> None:
        """Resample 3D cine per-timepoint files and mask to corrected FOV.
        
        Takes the per-timepoint cine files in full (padded) FOV and resamples them
        to the corrected (unpadded) FOV using the corrected velocity files as reference.
        Also resamples the cine mask to the corrected FOV.
        
        Outputs:
            - 3d_cine_{id}_per_timepoint_corr_fov/
            - 3d_cine_{id}_corr_fov_mask.nii.gz
        
        Prerequisites:
            - build_corrected_velocities_per_timepoint() must be run first (for reference FOV)
            - build_3d_cine_per_timepoint_full_fov() must be run first (for source data)
        """
        import SimpleITK as sitk
        
        self._logger.info(f"Building cine per-timepoint in corrected FOV for patient {self.identifier}")
        
        # Use corrected vx as reference FOV
        ref_dir = self.flow_vx_corr_per_timepoint_dir
        ref_files = sorted(ref_dir.glob("*.nii.gz"))
        if not ref_files:
            raise FileNotFoundError(
                f"No corrected vx per-timepoint files found in {ref_dir}. "
                "Run build_corrected_velocities_per_timepoint() first."
            )
        
        num_timepoints = len(ref_files)
        ref_img = sitk.ReadImage(str(ref_files[0]))  # Reference for mask and all frames
        
        # =========================================================================
        # Resample cine per-timepoint files
        # =========================================================================
        source_dir = self.cine_per_timepoint_full_fov_dir
        output_dir = self.cine_per_timepoint_corr_fov_dir
        
        # Check if already built
        existing_files = list(output_dir.glob("*.nii.gz"))
        if len(existing_files) >= num_timepoints and not self._should_overwrite('corrected'):
            self._logger.info(f"Cine already has {len(existing_files)} files in corr_fov, skipping")
        else:
            # Check source exists
            source_files = sorted(source_dir.glob("*.nii.gz"))
            if not source_files:
                self._logger.warning(f"No source cine files in {source_dir}, skipping")
            else:
                self._logger.info(f"Resampling {num_timepoints} cine frames to corrected FOV...")
                
                for t in range(num_timepoints):
                    # Reference file (corrected FOV)
                    ref_file = ref_dir / f"4d_flow_vx_corr_{self.identifier}_frame_{t:02d}.nii.gz"
                    
                    # Source file (full FOV)
                    source_file = source_dir / f"3d_cine_{self.identifier}_frame_{t:02d}.nii.gz"
                    
                    # Output file
                    output_file = output_dir / f"3d_cine_{self.identifier}_frame_{t:02d}.nii.gz"
                    
                    if not ref_file.exists():
                        self._logger.warning(f"Reference file not found: {ref_file}, skipping frame {t}")
                        continue
                    if not source_file.exists():
                        self._logger.warning(f"Source file not found: {source_file}, skipping frame {t}")
                        continue
                    
                    # Load and resample
                    ref_img_t = sitk.ReadImage(str(ref_file))
                    source_img = sitk.ReadImage(str(source_file))
                    
                    resampled = DicomToNiftiConverter.resample_to_target_grid(
                        moving_img=source_img,
                        reference_img=ref_img_t,
                        interpolator=sitk.sitkLinear,
                    )
                    
                    sitk.WriteImage(resampled, str(output_file))
                
                self._logger.info(f"Successfully resampled cine per-timepoint to corrected FOV")
        
        # =========================================================================
        # Resample cine mask (single 3D file)
        # =========================================================================
        cine_mask_path = self.nifti_dir / f"3d_cine_{self.identifier}_full_fov_mask.nii.gz"
        output_mask_path = self.cine_mask_corr_fov_path
        
        if output_mask_path.exists() and not self._should_overwrite('corrected'):
            self._logger.info(f"Cine mask in corr_fov already exists, skipping")
        elif not cine_mask_path.exists():
            self._logger.warning(f"Cine mask not found at {cine_mask_path}, skipping")
        else:
            self._logger.info("Resampling cine mask to corrected FOV...")
            mask_img = sitk.ReadImage(str(cine_mask_path))
            
            resampled_mask = DicomToNiftiConverter.resample_to_target_grid(
                moving_img=mask_img,
                reference_img=ref_img,
                interpolator=sitk.sitkNearestNeighbor,  # Nearest neighbor for mask
            )
            
            sitk.WriteImage(resampled_mask, str(output_mask_path))
            self._logger.info(f"Saved resampled cine mask to {output_mask_path}")
        
        self._logger.info(f"Successfully built cine per-timepoint in corrected FOV for patient {self.identifier}")
    
    def build_diff_per_timepoint(self) -> None:
        """Compute velocity diff (corrected - uncorrected) per-timepoint.
        
        Uses corrected velocities and uncorrected velocities (both in corrected FOV)
        to compute the diff.
        
        Outputs:
            - 4d_flow_diff_vx_{id}_per_timepoint/
            - 4d_flow_diff_vy_{id}_per_timepoint/
            - 4d_flow_diff_vz_{id}_per_timepoint/
        
        Prerequisites:
            - build_corrected_velocities_per_timepoint() must be run first
            - build_uncorrected_per_timepoint_corr_fov() must be run first
        """
        import SimpleITK as sitk
        
        self._logger.info(f"Building velocity diff per-timepoint for patient {self.identifier}")
        
        # Source directories
        corr_dirs = {
            'vx': self.flow_vx_corr_per_timepoint_dir,
            'vy': self.flow_vy_corr_per_timepoint_dir,
            'vz': self.flow_vz_corr_per_timepoint_dir,
        }
        uncorr_dirs = {
            'vx': self.flow_vx_per_timepoint_corr_fov_dir,
            'vy': self.flow_vy_per_timepoint_corr_fov_dir,
            'vz': self.flow_vz_per_timepoint_corr_fov_dir,
        }
        diff_dirs = {
            'vx': self.flow_diff_vx_per_timepoint_dir,
            'vy': self.flow_diff_vy_per_timepoint_dir,
            'vz': self.flow_diff_vz_per_timepoint_dir,
        }
        
        # Get number of timepoints from corrected vx
        corr_vx_files = sorted(corr_dirs['vx'].glob("*.nii.gz"))
        if not corr_vx_files:
            raise FileNotFoundError(
                f"No corrected vx per-timepoint files found. "
                "Run build_corrected_velocities_per_timepoint() first."
            )
        num_timepoints = len(corr_vx_files)
        
        for comp in ['vx', 'vy', 'vz']:
            corr_dir = corr_dirs[comp]
            uncorr_dir = uncorr_dirs[comp]
            diff_dir = diff_dirs[comp]
            
            # Check if already built
            existing_files = list(diff_dir.glob("*.nii.gz"))
            if len(existing_files) >= num_timepoints and not self._should_overwrite('corrected'):
                self._logger.info(f"diff_{comp} already has {len(existing_files)} files, skipping")
                continue
            
            # Check uncorrected exists
            uncorr_files = list(uncorr_dir.glob("*.nii.gz"))
            if not uncorr_files:
                raise FileNotFoundError(
                    f"No uncorrected {comp} per-timepoint files in corrected FOV. "
                    "Run build_uncorrected_per_timepoint_corr_fov() first."
                )
            
            self._logger.info(f"Computing diff for {comp}...")
            
            for t in range(num_timepoints):
                corr_file = corr_dir / f"4d_flow_{comp}_corr_{self.identifier}_frame_{t:02d}.nii.gz"
                uncorr_file = uncorr_dir / f"4d_flow_{comp}_{self.identifier}_frame_{t:02d}.nii.gz"
                diff_file = diff_dir / f"4d_flow_diff_{comp}_{self.identifier}_frame_{t:02d}.nii.gz"
                
                if not corr_file.exists() or not uncorr_file.exists():
                    self._logger.warning(f"Missing file for frame {t}, skipping")
                    continue
                
                # Load both
                corr_img = sitk.ReadImage(str(corr_file))
                uncorr_img = sitk.ReadImage(str(uncorr_file))
                
                # Compute diff
                corr_arr = sitk.GetArrayFromImage(corr_img).astype(np.float32)
                uncorr_arr = sitk.GetArrayFromImage(uncorr_img).astype(np.float32)
                diff_arr = corr_arr - uncorr_arr
                
                # Save
                diff_img = sitk.GetImageFromArray(diff_arr)
                diff_img.CopyInformation(corr_img)
                sitk.WriteImage(diff_img, str(diff_file))
                
                if t == 0:
                    self._logger.info(f"  {comp} frame 0: diff range=[{diff_arr.min():.2f}, {diff_arr.max():.2f}]")
        
        self._logger.info(f"Successfully built velocity diff per-timepoint for patient {self.identifier}")
    
    # =========================================================================
    # Velocity correction helper methods
    # =========================================================================
    
    @staticmethod
    def _compute_unpadded_affine(
        padded_affine: np.ndarray,
        pad_before: tuple[int, int, int],
    ) -> np.ndarray:
        """Compute affine for unpadded volume.
        
        When cropping by pad_before voxels from each side, the origin shifts
        by pad_before * spacing in physical coordinates. The affine encodes
        direction and spacing in its first 3 columns.
        
        Args:
            padded_affine: 4x4 affine from the padded NIfTI
            pad_before: (pad_x, pad_y, pad_z) voxels cropped from start of each axis
        
        Returns:
            4x4 affine for the unpadded volume
        """
        unpadded_affine = padded_affine.copy()
        
        # New origin = old_origin + sum(direction_vector_i * pad_i)
        # The first 3 columns are direction vectors scaled by spacing
        for i in range(3):
            unpadded_affine[:3, 3] += padded_affine[:3, i] * pad_before[i]
        
        return unpadded_affine
    
    @staticmethod
    def _build_polynomial_basis(
        shape: tuple[int, int, int],
        n_coeffs: int = 20,
    ) -> np.ndarray:
        """Build 3rd order polynomial basis matrix with normalized coordinates.
        
        Coordinates are normalized to [-1, 1] along each axis, which:
        - Keeps polynomial terms bounded (max magnitude = 1)
        - Makes coefficients comparable across patients with different FOV sizes
        - Improves numerical stability of the least squares solve
        
        Args:
            shape: (X, Y, Z) dimensions
            n_coeffs: Number of polynomial terms (default 20 for 3rd order)
        
        Returns:
            Basis matrix of shape (n_voxels, n_coeffs)
        """
        X, Y, Z = shape
        # Use normalized coordinates [-1, 1] instead of raw voxel indices
        r, c, s = np.meshgrid(
            np.linspace(-1, 1, X, dtype=np.float64),
            np.linspace(-1, 1, Y, dtype=np.float64),
            np.linspace(-1, 1, Z, dtype=np.float64),
            indexing='ij'
        )
        
        n_voxels = X * Y * Z
        basis = np.zeros((n_voxels, n_coeffs), dtype=np.float64)
        
        # 3rd order terms
        basis[:, 0] = np.ravel(r**3)
        basis[:, 1] = np.ravel(c**3)
        basis[:, 2] = np.ravel(s**3)
        basis[:, 3] = np.ravel(c * r**2)
        basis[:, 4] = np.ravel(s * r**2)
        basis[:, 5] = np.ravel(r * c**2)
        basis[:, 6] = np.ravel(s * c**2)
        basis[:, 7] = np.ravel(r * s**2)
        basis[:, 8] = np.ravel(c * s**2)
        basis[:, 9] = np.ravel(r * c * s)
        # 2nd order terms
        basis[:, 10] = np.ravel(r**2)
        basis[:, 11] = np.ravel(c**2)
        basis[:, 12] = np.ravel(s**2)
        basis[:, 13] = np.ravel(r * c)
        basis[:, 14] = np.ravel(r * s)
        basis[:, 15] = np.ravel(c * s)
        # 1st order terms
        basis[:, 16] = np.ravel(r)
        basis[:, 17] = np.ravel(c)
        basis[:, 18] = np.ravel(s)
        # constant
        basis[:, 19] = 1.0
        
        return basis
    
    @staticmethod
    def _create_magnitude_mask(
        magnitude: np.ndarray,
        threshold_fraction: float = 0.10,
        smooth_sigma: float = 1.5,
        shrink_margin: int = 4,
        normalization_percentile: float = 99.0,
        shrink_fraction: float | None = None,
        rethreshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create two binary masks from a magnitude volume.

        The first mask is a pure tissue/air classification (unshrunk). The
        second is the subset of tissue voxels that survive a from-edge crop;
        this is the set of voxels actually used during the polynomial fit.

        Normalizes magnitude by a high percentile (robust to outliers) before
        applying threshold so the threshold is consistent across patients
        regardless of raw magnitude scale.

        Args:
            magnitude: Magnitude volume, shape (X, Y, Z) or (X, Y, Z, T).
            threshold_fraction: Fraction of normalized magnitude to use as the
                tissue/air threshold. Voxels with normalized magnitude <
                threshold are classified as air. Default 0.10 (tuned for
                chest 4D-flow FOVs; the legacy 0.05 was too permissive and
                let bronchial vessels seed the mask inside lungs).
            smooth_sigma: Gaussian smoothing sigma applied before the second
                threshold (used to fill small holes and remove specks).
                Default 1.5 (tuned to avoid bridging chest-wall / mediastinum
                signal across lung air gaps; the legacy 3.0 caused the mask
                to "fill in" the lungs entirely).
            shrink_margin: Fixed-pixel margin to shrink from edges when
                ``shrink_fraction`` is None.
            normalization_percentile: Percentile used for magnitude
                normalization (default 99, outlier-robust alternative to max).
            shrink_fraction: Fraction of each axis to exclude per side (e.g.
                0.175 = 17.5% per side). When set, overrides ``shrink_margin``.
            rethreshold: Threshold applied to the Gaussian-smoothed binary
                mask. Default 0.5 (lung-tight); legacy was 0.333.

        Returns:
            Tuple ``(tissue_mask, fit_mask)`` both of shape (X, Y, Z) and dtype
            float32:
              * ``tissue_mask``: 1 in tissue, 0 in air, no edge cropping.
                Suitable for tissue/air loss weighting at training time.
              * ``fit_mask``: ``tissue_mask`` intersected with the from-edge
                crop. The polynomial fit consumes only voxels where
                ``fit_mask > 0``.
        """
        from scipy.ndimage import gaussian_filter

        if magnitude.ndim == 4:
            mag = np.mean(magnitude, axis=-1)
        else:
            mag = magnitude

        norm_value = np.percentile(mag, normalization_percentile)
        if norm_value > 0:
            mag_normalized = mag / norm_value
        else:
            mag_normalized = mag

        tissue_mask = (mag_normalized > threshold_fraction).astype(np.float32)
        tissue_mask = gaussian_filter(tissue_mask, sigma=smooth_sigma)
        tissue_mask = (tissue_mask > rethreshold).astype(np.float32)

        if shrink_fraction is not None:
            margin_x = int(mag.shape[0] * shrink_fraction)
            margin_y = int(mag.shape[1] * shrink_fraction)
            margin_z = int(mag.shape[2] * shrink_fraction)
        else:
            margin_x = margin_y = margin_z = shrink_margin

        fit_mask = tissue_mask.copy()
        if margin_x > 0 or margin_y > 0 or margin_z > 0:
            shrunk = np.zeros_like(fit_mask)
            shrunk[margin_x:-margin_x if margin_x else None,
                   margin_y:-margin_y if margin_y else None,
                   margin_z:-margin_z if margin_z else None] = \
                fit_mask[margin_x:-margin_x if margin_x else None,
                         margin_y:-margin_y if margin_y else None,
                         margin_z:-margin_z if margin_z else None]
            fit_mask = shrunk

        return tissue_mask, fit_mask
    
    @staticmethod
    def _fit_polynomial_coefficients(
        delta: np.ndarray,
        basis: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Fit polynomial coefficients to velocity delta.
        
        Args:
            delta: Shape (T, 3, Z, Y, X)
            basis: Shape (n_voxels, n_coeffs)
            mask: Shape (X, Y, Z)
        
        Returns:
            Median coefficients across timepoints, shape (n_coeffs, 3)
        """
        T = delta.shape[0]
        n_coeffs = basis.shape[1]
        
        # Get valid voxel indices
        valid_idx = np.where(mask.ravel() > 0)[0]
        basis_valid = basis[valid_idx, :]
        
        # Precompute pseudo-inverse: (X^T X)^-1 X^T
        xtx_inv_xt = np.linalg.pinv(basis_valid.T @ basis_valid) @ basis_valid.T
        
        # Fit for each timepoint and component
        all_coeffs = np.zeros((T, n_coeffs, 3), dtype=np.float64)
        
        for t in range(T):
            for comp in range(3):
                # delta[t, comp, z, y, x] -> transpose to (x, y, z) and flatten
                delta_vol = np.transpose(delta[t, comp], (2, 1, 0))
                y_valid = delta_vol.ravel()[valid_idx].astype(np.float64)
                all_coeffs[t, :, comp] = xtx_inv_xt @ y_valid
        
        return np.median(all_coeffs, axis=0)
    
    @staticmethod
    def _reconstruct_from_coefficients(
        coefficients: np.ndarray,
        basis: np.ndarray,
        shape: tuple[int, int, int],
    ) -> dict[str, np.ndarray]:
        """Reconstruct correction volume from polynomial coefficients.
        
        Args:
            coefficients: Shape (n_coeffs, 3)
            basis: Shape (n_voxels, n_coeffs)
            shape: (X, Y, Z)
        
        Returns:
            Dictionary with keys 'vx', 'vy', 'vz', each with shape (X, Y, Z)
        """
        X, Y, Z = shape
        comp_names = ['vx', 'vy', 'vz']
        ground_truth = {}
        
        for i, name in enumerate(comp_names):
            reconstructed = basis @ coefficients[:, i]
            ground_truth[name] = reconstructed.reshape(X, Y, Z).astype(np.float32)
        
        return ground_truth
    
    def _save_nifti(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        path: Path,
        description: str = "",
    ) -> None:
        """Save array as NIfTI with proper header setup.
        
        Args:
            data: Array to save (3D or 4D)
            affine: 4x4 affine matrix
            path: Output path
            description: Optional description for logging
        """
        nii = nib.Nifti1Image(data.astype(np.float32), affine)
        nii.set_qform(affine, code=1)
        nii.set_sform(affine, code=1)
        
        hdr = nii.header
        hdr['dim'][0] = data.ndim
        if data.ndim >= 4:
            hdr['dim'][4] = data.shape[3]
            hdr['pixdim'][4] = 1.0
        hdr['xyzt_units'] = 2 | 8  # mm + seconds
        
        nib.save(nii, path)
        self._logger.info(f"Saved {description}: {path.name}, shape={data.shape}")
    
    # =========================================================================
    # Main velocity correction method
    # =========================================================================
    
    def build_velocity_correction_data(
        self,
        n_coeffs: int = 20,
        mag_threshold_fraction: float = 0.10,
        shrink_margin: int = 4,
        normalization_percentile: float = 99.0,
        shrink_fraction: float | None = None,
    ) -> dict[str, Path]:
        """Compute velocity correction data: polynomial coefficients and ground truth.
        
        This method loads per-timepoint intermediate files and computes polynomial 
        fit coefficients for velocity correction.
        
        Steps:
        1. Load diff files (corrected - uncorrected) per-timepoint
        2. Normalize delta by VENC for stable fitting
        3. Create magnitude mask to exclude air voxels
        4. Fit polynomial coefficients to the normalized delta
        5. Generate ground truth correction volume from median coefficients
        
        Prerequisites:
            - build_corrected_velocity_pipeline() must be run first to create:
                - 4d_flow_diff_vx/vy/vz_{id}_per_timepoint/ (diff files)
                - 4d_flow_mag_{id}_per_timepoint_corr_fov/ (magnitude in corrected FOV)
        
        Args:
            n_coeffs: Number of polynomial coefficients (default 20 for 3rd order)
            mag_threshold_fraction: Fraction of normalized magnitude for air/tissue threshold.
                                   Magnitude is normalized by percentile before thresholding.
            shrink_margin: Fixed-pixel margin to shrink mask from edges (used when
                          shrink_fraction is None)
            normalization_percentile: Percentile to normalize magnitude by (default 99).
                                     Using percentile instead of max for outlier robustness.
            shrink_fraction: Fraction of each axis to exclude per side (e.g. 0.175 =
                            17.5% per side). When set, overrides shrink_margin and
                            outputs to velocity_correction_crop-{pct}/ instead.
        
        Returns:
            Dictionary of output paths
        """
        import SimpleITK as sitk
        
        self._logger.info(f"Building velocity correction data for patient {self.identifier}")
        
        if shrink_fraction is not None:
            pct_tag = f"{shrink_fraction * 100:g}"
            dir_name = f"velocity_correction_crop-{pct_tag}"
        else:
            dir_name = "velocity_correction"
        output_dir = self.nifti_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Primary output paths.
        # tissue_mask:  unshrunk tissue/air classification — used by training
        #               loss for tissue/air weighting.
        # fit_mask:     shrunk subset (tissue ∩ from-edge crop) — the voxels
        #               that the polynomial fit was actually performed on.
        output_paths = {
            'ground_truth_vx': output_dir / f"ground_truth_correction_vx_{self.identifier}.nii.gz",
            'ground_truth_vy': output_dir / f"ground_truth_correction_vy_{self.identifier}.nii.gz",
            'ground_truth_vz': output_dir / f"ground_truth_correction_vz_{self.identifier}.nii.gz",
            'tissue_mask': output_dir / f"correction_tissue_mask_{self.identifier}.nii.gz",
            'fit_mask': output_dir / f"correction_fit_mask_{self.identifier}.nii.gz",
            'coefficients': output_dir / f"poly_coefficients_{self.identifier}.npz",
        }
        
        # Check idempotency
        if all(p.exists() for p in output_paths.values()) and not self._should_overwrite('corrected'):
            self._logger.info("Velocity correction data already exists, skipping")
            return output_paths
        
        # =====================================================================
        # 1. Verify prerequisite files exist (built by build_per_timepoint_images)
        # =====================================================================
        diff_vx_dir = self.flow_diff_vx_per_timepoint_dir
        diff_vy_dir = self.flow_diff_vy_per_timepoint_dir
        diff_vz_dir = self.flow_diff_vz_per_timepoint_dir
        mag_corr_fov_dir = self.flow_mag_per_timepoint_corr_fov_dir
        
        # Check if diff files exist
        diff_vx_files = sorted(diff_vx_dir.glob("*.nii.gz"))
        if not diff_vx_files:
            raise FileNotFoundError(
                f"No diff vx files found in {diff_vx_dir}. "
                "Run build_per_timepoint_images() first to create required intermediate files."
            )
        
        # =====================================================================
        # 2. Load diff files and construct 4D delta arrays
        # =====================================================================
        self._logger.info("Step 1: Loading per-timepoint diff files...")
        
        num_timepoints = len(diff_vx_files)
        
        # Load first file with nibabel to get shape and affine (single load)
        first_diff_nib = nib.load(diff_vx_files[0])
        unpadded_affine = first_diff_nib.affine
        X, Y, Z = first_diff_nib.shape  # nibabel shape is (X, Y, Z)
        unpadded_shape = (X, Y, Z)
        
        self._logger.info(f"Spatial shape: {unpadded_shape}, Timepoints: {num_timepoints}")
        
        # Allocate 4D arrays: (T, Z, Y, X) for consistency with _fit_polynomial_coefficients
        delta_vx = np.zeros((num_timepoints, Z, Y, X), dtype=np.float32)
        delta_vy = np.zeros((num_timepoints, Z, Y, X), dtype=np.float32)
        delta_vz = np.zeros((num_timepoints, Z, Y, X), dtype=np.float32)
        
        # Load all timepoints
        for t in range(num_timepoints):
            vx_file = diff_vx_dir / f"4d_flow_diff_vx_{self.identifier}_frame_{t:02d}.nii.gz"
            vy_file = diff_vy_dir / f"4d_flow_diff_vy_{self.identifier}_frame_{t:02d}.nii.gz"
            vz_file = diff_vz_dir / f"4d_flow_diff_vz_{self.identifier}_frame_{t:02d}.nii.gz"
            
            delta_vx[t] = sitk.GetArrayFromImage(sitk.ReadImage(str(vx_file))).astype(np.float32)
            delta_vy[t] = sitk.GetArrayFromImage(sitk.ReadImage(str(vy_file))).astype(np.float32)
            delta_vz[t] = sitk.GetArrayFromImage(sitk.ReadImage(str(vz_file))).astype(np.float32)
        
        # =====================================================================
        # 3. Normalize delta by VENC
        # =====================================================================
        venc = self.venc
        self._logger.info(f"Step 3: Normalizing by VENC={venc} cm/s")
        
        delta_vx = delta_vx / venc
        delta_vy = delta_vy / venc
        delta_vz = delta_vz / venc
        
        self._logger.info(
            f"Normalized delta ranges: "
            f"vx=[{delta_vx.min():.4f}, {delta_vx.max():.4f}], "
            f"vy=[{delta_vy.min():.4f}, {delta_vy.max():.4f}], "
            f"vz=[{delta_vz.min():.4f}, {delta_vz.max():.4f}]"
        )
        
        # =====================================================================
        # 5. Create mask from magnitude (already in corrected FOV)
        # =====================================================================
        self._logger.info("Step 5: Creating magnitude mask...")
        
        # Load magnitude per-timepoint files (already resampled to corrected FOV)
        mag_files = sorted(mag_corr_fov_dir.glob("*.nii.gz"))
        
        if not mag_files:
            raise FileNotFoundError(
                f"No magnitude files in corrected FOV found in {mag_corr_fov_dir}. "
                "build_uncorrected_per_timepoint_corr_fov() should have created these."
            )
        
        # Load all timepoints and compute mean for masking
        mag_arrays = []
        for mag_file in mag_files:
            mag_img = sitk.ReadImage(str(mag_file))
            mag_arrays.append(sitk.GetArrayFromImage(mag_img).astype(np.float32))
        
        # Stack to (T, Z, Y, X) and transpose to (X, Y, Z, T) for masking function
        mag_4d = np.stack(mag_arrays, axis=0)  # (T, Z, Y, X)
        mag_unpad = np.transpose(mag_4d, (3, 2, 1, 0))  # (X, Y, Z, T)
        
        tissue_mask, fit_mask = self._create_magnitude_mask(
            mag_unpad,
            threshold_fraction=mag_threshold_fraction,
            shrink_margin=shrink_margin,
            normalization_percentile=normalization_percentile,
            shrink_fraction=shrink_fraction,
        )
        self._save_nifti(
            tissue_mask, unpadded_affine, output_paths['tissue_mask'],
            "correction tissue mask (unshrunk, for training loss weighting)",
        )
        self._save_nifti(
            fit_mask, unpadded_affine, output_paths['fit_mask'],
            "correction fit mask (tissue ∩ from-edge crop, voxels used in polyfit)",
        )

        n_tissue = int(np.sum(tissue_mask > 0))
        n_fit = int(np.sum(fit_mask > 0))
        n_voxels = int(np.prod(unpadded_shape))
        self._logger.info(
            f"Tissue voxels: {n_tissue} / {n_voxels}  "
            f"({100.0 * n_tissue / n_voxels:.1f}%); "
            f"fit voxels: {n_fit} / {n_voxels}  ({100.0 * n_fit / n_voxels:.1f}%)"
        )

        # Polynomial fit uses the (possibly shrunk) fit mask only.
        mask = fit_mask

        if n_fit < n_coeffs:
            self._logger.warning(f"Not enough valid voxels ({n_fit}) for polynomial fit")
            return output_paths
        
        # =====================================================================
        # 6. Build polynomial basis and fit coefficients
        # =====================================================================
        self._logger.info("Step 6: Fitting polynomial coefficients...")
        
        basis = self._build_polynomial_basis(unpadded_shape, n_coeffs)
        
        # Stack delta into (T, 3, Z, Y, X) format
        delta_stacked = np.stack([delta_vx, delta_vy, delta_vz], axis=1)
        
        coefficients = self._fit_polynomial_coefficients(delta_stacked, basis, mask)
        self._logger.info(f"Computed median coefficients, shape: {coefficients.shape}")
        
        # Save coefficients with metadata
        np.savez(
            output_paths['coefficients'],
            coefficients=coefficients,
            venc=venc,
            original_shape=np.array(unpadded_shape),
        )
        self._logger.info(f"Saved coefficients to {output_paths['coefficients'].name}")
        
        # =====================================================================
        # 7. Generate ground truth correction volumes (one per component)
        # =====================================================================
        self._logger.info("Step 7: Generating ground truth correction volumes...")
        
        ground_truth = self._reconstruct_from_coefficients(coefficients, basis, unpadded_shape)
        
        for comp in ['vx', 'vy', 'vz']:
            self._save_nifti(
                ground_truth[comp], 
                unpadded_affine, 
                output_paths[f'ground_truth_{comp}'], 
                f"ground truth correction {comp}"
            )
        
        # =====================================================================
        # 8. Compute error metrics: ground truth vs raw diffs
        # =====================================================================
        self._logger.info("Step 8: Computing error metrics (ground truth vs raw diffs)...")
        
        # delta_vx/vy/vz are in (T, Z, Y, X) order, VENC-normalized
        # ground_truth['vx'] etc are in (X, Y, Z) order, VENC-normalized
        # Need to compare at each timepoint
        
        comp_deltas = {'vx': delta_vx, 'vy': delta_vy, 'vz': delta_vz}
        
        for comp in ['vx', 'vy', 'vz']:
            gt = ground_truth[comp]  # (X, Y, Z)
            raw_deltas = comp_deltas[comp]  # (T, Z, Y, X)
            
            # Compute per-timepoint metrics
            maes = []
            rmses = []
            correlations = []
            
            for t in range(num_timepoints):
                # Get raw delta for this timepoint, transpose to (X, Y, Z)
                raw_t = np.transpose(raw_deltas[t], (2, 1, 0))  # (Z, Y, X) -> (X, Y, Z)
                
                # Apply mask to compare only tissue voxels
                gt_masked = gt[mask > 0]
                raw_masked = raw_t[mask > 0]
                
                # Mean Absolute Error
                mae = np.mean(np.abs(gt_masked - raw_masked))
                maes.append(mae)
                
                # Root Mean Square Error
                rmse = np.sqrt(np.mean((gt_masked - raw_masked) ** 2))
                rmses.append(rmse)
                
                # Pearson correlation
                if np.std(gt_masked) > 0 and np.std(raw_masked) > 0:
                    corr = np.corrcoef(gt_masked.flatten(), raw_masked.flatten())[0, 1]
                else:
                    corr = 0.0
                correlations.append(corr)
            
            # Log summary statistics
            mean_mae = np.mean(maes)
            mean_rmse = np.mean(rmses)
            mean_corr = np.mean(correlations)
            std_mae = np.std(maes)
            
            self._logger.info(
                f"  {comp}: MAE={mean_mae:.4f}±{std_mae:.4f}, "
                f"RMSE={mean_rmse:.4f}, Corr={mean_corr:.4f} "
                f"(across {num_timepoints} timepoints, {n_fit} voxels)"
            )
        
        self._logger.info(f"Successfully built velocity correction data for patient {self.identifier}")
        return output_paths
    
    def __str__(self) -> str:
        """Return a string representation of the patient."""
        return f"Patient({self.identifier})"
    
    def write_predictions_to_dicoms(
        self,
        prediction_dir: Path,
        output_dir: Optional[Path] = None,
        timepoint: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Write NIfTI predictions back to DICOM format.
        
        This method replaces magnitude pixel data in DICOM files with predicted
        magnitude values while preserving velocity data and all metadata.
        
        Args:
            prediction_dir: Directory containing prediction NIfTI files
            output_dir: Directory to save modified DICOM files. If None, creates
                       a 'dicom_predictions' directory at the same level as prediction_dir
            timepoint: Specific timepoint to process (0-based). If None, processes all timepoints
            overwrite: Whether to overwrite existing DICOM files
        """
        # Convert prediction_dir to Path
        prediction_dir = Path(prediction_dir)
        
        if output_dir is None:
            # Create dicom_predictions folder at the same level as prediction_dir
            output_dir = prediction_dir.parent / "dicom_predictions"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._logger.info(
            f"Writing predictions to DICOMs for patient {self.identifier}"
        )
        
        # Create converter
        converter = NiftiToDicomConverter.from_patient(self)
        
        if timepoint is not None:
            # Process single timepoint
            self._logger.info(f"Processing timepoint {timepoint}")
            converter.write_predictions_to_dicoms(
                prediction_dir=prediction_dir,
                output_dir=output_dir,
                timepoint=timepoint,
                overwrite=overwrite,
            )
        else:
            # Process all timepoints
            num_timepoints = self.num_timepoints
            self._logger.info(f"Processing all {num_timepoints} timepoints")
            converter.write_all_timepoints_to_dicoms(
                prediction_dir=prediction_dir,
                output_dir=output_dir,
                num_timepoints=num_timepoints,
                overwrite=overwrite,
            )
        
        self._logger.info(
            f"Completed writing predictions to DICOMs. Output directory: {output_dir}"
        )
        
        # Zip the output directory
        zip_path = output_dir.parent / f"{self.identifier}_dicom_predictions.zip"
        self._logger.info(f"Creating ZIP archive: {zip_path}")
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files in output_dir to the zip
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        # Use relative path from output_dir to maintain directory structure
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file_path, arcname=arcname)
                        self._logger.debug(f"Added {file_path.name} to ZIP")
            
            self._logger.info(
                f"Successfully created ZIP archive: {zip_path} "
                f"({zip_path.stat().st_size / (1024*1024):.2f} MB)"
            )
        except Exception as e:
            self._logger.error(
                f"Error creating ZIP archive: {str(e)}",
                exc_info=True
            )
            raise
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the patient."""
        return (f"Patient(path_config={self.path_config}, "
                f"accession_number={self.accession_number}, "
                f"phonetic_id={self.phonetic_id}, "
                f"skip_database_validation={self.skip_database_validation}, "
                f"debug={self.debug}, "
                f"overwrite_images={self.overwrite_images}, "
                f"overwrite_catalogs={self.overwrite_catalogs})") 