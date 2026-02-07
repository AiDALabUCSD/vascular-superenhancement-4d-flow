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
    
    @property
    def velocity_correction_dir(self) -> Path:
        """Create (if necessary) and return directory for velocity correction data.
        
        Contains: delta (corrected - uncorrected), polynomial coefficients, ground truth.
        All data is stored in UNPADDED dimensions matching the corrected velocity numpy."""
        d = self.nifti_dir / "velocity_correction"
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    @property
    def num_timepoints(self) -> int:
        """Return the number of timepoints for this patient."""
        
        # check if there are the same number of files in all the timepoint directories
        cine_files = list(self.cine_per_timepoint_dir.glob("*.nii.gz"))
        flow_mag_files = list(self.flow_mag_per_timepoint_dir.glob("*.nii.gz"))
        flow_vx_files = list(self.flow_vx_per_timepoint_dir.glob("*.nii.gz"))
        flow_vy_files = list(self.flow_vy_per_timepoint_dir.glob("*.nii.gz"))
        flow_vz_files = list(self.flow_vz_per_timepoint_dir.glob("*.nii.gz"))
        
        # Get counts for each component
        counts = {
            'cine': len(cine_files),
            'flow_mag': len(flow_mag_files),
            'flow_vx': len(flow_vx_files),
            'flow_vy': len(flow_vy_files),
            'flow_vz': len(flow_vz_files)
        }
        
        
        # Check if all counts are the same
        if not all(count == counts['flow_mag'] for count in counts.values()):
            
            if counts['cine'] == 0:
                self._logger.info(f"Patient {self.identifier} does not have cine data")
            else:
                self._logger.warning(
                    f"Inconsistent number of timepoints for patient {self.identifier}:\n"
                    f"Cine: {counts['cine']}\n"
                    f"Flow Mag: {counts['flow_mag']}\n"
                    f"Flow Vx: {counts['flow_vx']}\n"
                    f"Flow Vy: {counts['flow_vy']}\n"
                    f"Flow Vz: {counts['flow_vz']}"
                )
            
                # TODO(#2): Some patients actually dont have the same number of timepoints
                # for cine and flow components. We are currently skipping these patients.
                # Might neet to fix by either ensuring the data is correct or interpolating
                # over time
                raise ValueError("Inconsistent number of timepoints across components")
        
        return len(flow_mag_files)
    
    def _load_or_create_catalog(self) -> None:
        """Load the DICOM catalog if it exists, otherwise create it.
        
        Checks for both new format (dicom_catalog_{identifier}.csv) and old format
        ({identifier}_dicom_catalog.csv). If old format is found, it will be loaded
        and saved in the new format.
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
            
            # Log the number of files matching each criterion
            self._logger.debug(f"Files with velocity encoding > 1: {is_velocity_encoded.sum()}")
            self._logger.debug(f"Files with flow encoding between 1 and 6: {is_flow_encoded.sum()}")
            self._logger.debug(f"Files with flow encoding = 7 (excluded): {is_excluded.sum()}")
            self._logger.debug(f"Total 4D Flow files: {is_4d_flow.sum()}")
            
            filtered_catalog = catalog[is_4d_flow]
            
            if len(filtered_catalog) == 0:
                self._logger.warning(f"No 4D Flow files found in DICOM catalog for patient {self.identifier}")
                return None
                
            self._logger.debug(f"Found {len(filtered_catalog)} 4D Flow files")
            
            # Add time_index and slice_index columns if they don't exist
            filtered_catalog = filtered_catalog.copy()  # Avoid SettingWithCopyWarning
            if 'time_index' not in filtered_catalog.columns or 'slice_index' not in filtered_catalog.columns:
                self._logger.debug("Adding time_index and slice_index columns")
                
                # Log some sample values for debugging
                sample_instances = filtered_catalog['instancenumber'].head(3)
                sample_cardiac = filtered_catalog['cardiacnumberofimages'].head(3)
                self._logger.debug(f"Sample InstanceNumbers: {sample_instances.tolist()}")
                self._logger.debug(f"Sample CardiacNumberOfImages: {sample_cardiac.tolist()}")
                
                filtered_catalog['time_index'] = (filtered_catalog['instancenumber'] - 1) % filtered_catalog['cardiacnumberofimages']
                filtered_catalog['slice_index'] = (filtered_catalog['instancenumber'] - 1) // filtered_catalog['cardiacnumberofimages']
                
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
    ) -> None:
        """Build downsampled per-timepoint volumes in corrected velocity FOV.
        
        Uses the corrected velocity FOV (unpadded, with shifted affine) as the 
        reference. All data (mag, cine, cine mask) is resampled to this FOV and
        then downsampled to the target size.
        
        Args:
            target_size: Target voxel dimensions (X, Y, Z), default (128, 128, 64)
        """
        import SimpleITK as sitk
        import numpy as np
        import re
        
        size_tag = f"{target_size[0]}x{target_size[1]}x{target_size[2]}"
        self._logger.info(
            f"Building downsampled corrected FOV per timepoint ({size_tag}) for patient {self.identifier}"
        )
        
        # Create output root directory
        output_root = self.nifti_dir / f"downsampled_full_fov_{size_tag}"
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
        
        # Load corrected velocity as reference and create downsampled reference grid
        source_img = sitk.ReadImage(str(reference_source_path))
        reference_img = DicomToNiftiConverter.create_downsampled_reference_grid(
            source_img, target_size
        )
        
        self._logger.info(f"Corrected FOV size: {source_img.GetSize()}, spacing: {source_img.GetSpacing()}")
        self._logger.info(f"Target size: {reference_img.GetSize()}, spacing: {reference_img.GetSpacing()}")
        
        # Save reference grid as debugging artifact
        reference_path = output_root / "reference.nii.gz"
        if not reference_path.exists() or self._should_overwrite('downsampled'):
            sitk.WriteImage(reference_img, str(reference_path))
            self._logger.info(f"Saved reference grid to {reference_path}")
        
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
        """Build per-timepoint volumes for 3d cine, 4d flow, speed, and full FOV variants."""
        
        self._logger.info(f"Building per-timepoint volumes for patient {self.identifier}")
        
        # build the per-timepoint volumes for 3d cine (original FOV)
        try:
            self.build_3d_cine_per_timepoint()
            self._logger.info(f"Successfully built 3d cine per timepoint for patient {self.identifier}")
        except Exception as e:
            self._logger.error(f"Error building 3d cine per timepoint for patient {self.identifier}: {e}")
        
        # build the per-timepoint volumes for 4d flow (resampled to cine FOV)
        try:
            self.build_4d_flow_per_timepoint()
            self._logger.info(f"Successfully built 4d flow per timepoint for patient {self.identifier}")
        except Exception as e:
            self._logger.error(f"Error building 4d flow per timepoint for patient {self.identifier}: {e}")
        
        # build the per-timepoint volumes for speed (derived from vx, vy, vz)
        try:
            self.build_speed_per_timepoint()
            self._logger.info(f"Successfully built speed per timepoint for patient {self.identifier}")
        except Exception as e:
            self._logger.error(f"Error building speed per timepoint for patient {self.identifier}: {e}")
        
        # build the per-timepoint volumes for 4d flow (full FOV, no resampling)
        try:
            self.build_4d_flow_per_timepoint_full_fov()
            self._logger.info(f"Successfully built 4d flow per timepoint (full FOV) for patient {self.identifier}")
        except Exception as e:
            self._logger.error(f"Error building 4d flow per timepoint (full FOV) for patient {self.identifier}: {e}")
        
        # build the per-timepoint volumes for 3d cine (resampled to flow full FOV)
        try:
            self.build_3d_cine_per_timepoint_full_fov()
            self._logger.info(f"Successfully built 3d cine per timepoint (full FOV) for patient {self.identifier}")
        except Exception as e:
            self._logger.error(f"Error building 3d cine per timepoint (full FOV) for patient {self.identifier}: {e}")
        
        
        
        # build corrected velocities per-timepoint (if corrected velocity NIfTIs exist)
        vx_corr_path = self.nifti_dir / f"4d_flow_vx_corr_{self.identifier}.nii.gz"
        if vx_corr_path.exists():
            try:
                self.build_corrected_velocities_per_timepoint()
                self._logger.info(f"Successfully built corrected velocities per timepoint for patient {self.identifier}")
            except Exception as e:
                self._logger.error(f"Error building corrected velocities per timepoint for patient {self.identifier}: {e}")
            
            # build corrected speed per-timepoint
            try:
                self.build_corrected_speed_per_timepoint()
                self._logger.info(f"Successfully built corrected speed per timepoint for patient {self.identifier}")
            except Exception as e:
                self._logger.error(f"Error building corrected speed per timepoint for patient {self.identifier}: {e}")
        else:
            self._logger.info(f"No corrected velocity NIfTIs found for patient {self.identifier}, skipping per-timepoint processing")
            
        # build downsampled full FOV per timepoint
        try:
            self.build_downsampled_full_fov_per_timepoint()
            self._logger.info(f"Successfully built downsampled full FOV per timepoint for patient {self.identifier}")
        except Exception as e:
            self._logger.error(f"Error building downsampled full FOV per timepoint for patient {self.identifier}: {e}")
    
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
        """Build 3rd order polynomial basis matrix.
        
        Args:
            shape: (X, Y, Z) dimensions
            n_coeffs: Number of polynomial terms (default 20 for 3rd order)
        
        Returns:
            Basis matrix of shape (n_voxels, n_coeffs)
        """
        X, Y, Z = shape
        r, c, s = np.meshgrid(
            np.arange(X, dtype=np.float64),
            np.arange(Y, dtype=np.float64),
            np.arange(Z, dtype=np.float64),
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
        threshold_fraction: float = 0.1,
        smooth_sigma: float = 3.0,
        shrink_margin: int = 4,
    ) -> np.ndarray:
        """Create binary mask excluding air voxels.
        
        Args:
            magnitude: Magnitude volume, shape (X, Y, Z) or (X, Y, Z, T)
            threshold_fraction: Fraction of max to use as threshold
            smooth_sigma: Gaussian smoothing sigma
            shrink_margin: Margin to shrink from edges
        
        Returns:
            Binary mask, shape (X, Y, Z)
        """
        from scipy.ndimage import gaussian_filter
        
        # Mean across time if 4D
        if magnitude.ndim == 4:
            mag = np.mean(magnitude, axis=-1)
        else:
            mag = magnitude
        
        # Threshold
        mask = (mag > threshold_fraction * mag.max()).astype(np.float32)
        
        # Smooth
        mask = gaussian_filter(mask, sigma=smooth_sigma)
        mask = (mask > 0.333).astype(np.float32)
        
        # Shrink from edges
        if shrink_margin > 0:
            shrunk = np.zeros_like(mask)
            shrunk[shrink_margin:-shrink_margin,
                   shrink_margin:-shrink_margin,
                   shrink_margin:-shrink_margin] = mask[shrink_margin:-shrink_margin,
                                                         shrink_margin:-shrink_margin,
                                                         shrink_margin:-shrink_margin]
            mask = shrunk
        
        return mask
    
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
    ) -> np.ndarray:
        """Reconstruct correction volume from polynomial coefficients.
        
        Args:
            coefficients: Shape (n_coeffs, 3)
            basis: Shape (n_voxels, n_coeffs)
            shape: (X, Y, Z)
        
        Returns:
            Ground truth volume, shape (X, Y, Z, 3)
        """
        X, Y, Z = shape
        ground_truth = np.zeros((X, Y, Z, 3), dtype=np.float32)
        
        for comp in range(3):
            reconstructed = basis @ coefficients[:, comp]
            ground_truth[:, :, :, comp] = reconstructed.reshape(X, Y, Z)
        
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
        mag_threshold: float = 0.1,
        shrink_margin: int = 4,
    ) -> dict[str, Path]:
        """Compute velocity correction data: delta, polynomial coefficients, and ground truth.
        
        This method:
        1. Loads corrected velocity NIfTI (unpadded, with shifted affine)
        2. Resamples uncorrected velocity/mag to the corrected FOV (automatic unpadding)
        3. Computes delta = corrected - uncorrected
        4. Fits polynomial coefficients to the delta
        5. Generates ground truth correction volume from median coefficients
        
        All output is stored in the corrected velocity FOV (unpadded dimensions).
        
        Args:
            n_coeffs: Number of polynomial coefficients (default 20 for 3rd order)
            mag_threshold: Threshold for masking low-magnitude (air) voxels (fraction of max)
            shrink_margin: Margin to shrink mask from edges to avoid boundary artifacts
        
        Returns:
            Dictionary of output paths
        """
        import SimpleITK as sitk
        
        self._logger.info(f"Building velocity correction data for patient {self.identifier}")
        
        output_dir = self.velocity_correction_dir
        
        # Output paths - volumetric data as NIfTI, metadata as numpy
        output_paths = {
            'delta_vx': output_dir / f"delta_vx_{self.identifier}.nii.gz",
            'delta_vy': output_dir / f"delta_vy_{self.identifier}.nii.gz",
            'delta_vz': output_dir / f"delta_vz_{self.identifier}.nii.gz",
            'mag_unpadded': output_dir / f"mag_unpadded_{self.identifier}.nii.gz",
            'ground_truth': output_dir / f"ground_truth_correction_{self.identifier}.nii.gz",
            'mask': output_dir / f"correction_mask_{self.identifier}.nii.gz",
            'coefficients': output_dir / f"poly_coefficients_{self.identifier}.npy",
        }
        
        # Check idempotency
        if all(p.exists() for p in output_paths.values()) and not self._should_overwrite('corrected'):
            self._logger.info("Velocity correction data already exists, skipping")
            return output_paths
        
        # =====================================================================
        # 1. Load corrected velocity NIfTI (already unpadded with correct affine)
        # =====================================================================
        vx_corr_path = self.nifti_dir / f"4d_flow_vx_corr_{self.identifier}.nii.gz"
        vy_corr_path = self.nifti_dir / f"4d_flow_vy_corr_{self.identifier}.nii.gz"
        vz_corr_path = self.nifti_dir / f"4d_flow_vz_corr_{self.identifier}.nii.gz"
        
        for p in [vx_corr_path, vy_corr_path, vz_corr_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Corrected velocity NIfTI not found: {p}. "
                    "Run build_corrected_velocities() first."
                )
        
        # Load corrected velocities as SimpleITK images (reference geometry)
        vx_corr_sitk = sitk.ReadImage(str(vx_corr_path))
        vy_corr_sitk = sitk.ReadImage(str(vy_corr_path))
        vz_corr_sitk = sitk.ReadImage(str(vz_corr_path))
        
        # Get corrected arrays
        vx_corr = sitk.GetArrayFromImage(vx_corr_sitk).astype(np.float32)  # (T, Z, Y, X)
        vy_corr = sitk.GetArrayFromImage(vy_corr_sitk).astype(np.float32)
        vz_corr = sitk.GetArrayFromImage(vz_corr_sitk).astype(np.float32)
        
        self._logger.info(f"Corrected velocity shape (T,Z,Y,X): {vx_corr.shape}")
        
        # Get unpadded dimensions from corrected image
        T, Z_unpad, Y_unpad, X_unpad = vx_corr.shape
        unpadded_shape = (X_unpad, Y_unpad, Z_unpad)
        
        # Get unpadded affine from the corrected NIfTI
        vx_corr_nii = nib.load(vx_corr_path)
        unpadded_affine = vx_corr_nii.affine
        
        self._logger.info(f"Unpadded shape (X,Y,Z): {unpadded_shape}")
        
        # =====================================================================
        # 2. Resample uncorrected velocity/mag to corrected FOV (automatic unpadding)
        # =====================================================================
        vx_uncorr_path = self.nifti_dir / f"4d_flow_vx_{self.identifier}.nii.gz"
        vy_uncorr_path = self.nifti_dir / f"4d_flow_vy_{self.identifier}.nii.gz"
        vz_uncorr_path = self.nifti_dir / f"4d_flow_vz_{self.identifier}.nii.gz"
        mag_path = self.nifti_dir / f"4d_flow_mag_{self.identifier}.nii.gz"
        
        for p in [vx_uncorr_path, vy_uncorr_path, vz_uncorr_path, mag_path]:
            if not p.exists():
                raise FileNotFoundError(f"Uncorrected velocity/mag NIfTI not found: {p}")
        
        # Resample uncorrected to corrected FOV
        # Using NearestNeighbor since voxels should align exactly (just cropping)
        vx_uncorr_sitk = sitk.ReadImage(str(vx_uncorr_path))
        vy_uncorr_sitk = sitk.ReadImage(str(vy_uncorr_path))
        vz_uncorr_sitk = sitk.ReadImage(str(vz_uncorr_path))
        mag_sitk = sitk.ReadImage(str(mag_path))
        
        self._logger.info(f"Uncorrected (padded) shape: {vx_uncorr_sitk.GetSize()}")
        
        # Resample to corrected geometry
        vx_uncorr_resampled = sitk.Resample(
            vx_uncorr_sitk, vx_corr_sitk,
            sitk.Transform(), sitk.sitkNearestNeighbor, 0.0
        )
        vy_uncorr_resampled = sitk.Resample(
            vy_uncorr_sitk, vy_corr_sitk,
            sitk.Transform(), sitk.sitkNearestNeighbor, 0.0
        )
        vz_uncorr_resampled = sitk.Resample(
            vz_uncorr_sitk, vz_corr_sitk,
            sitk.Transform(), sitk.sitkNearestNeighbor, 0.0
        )
        mag_resampled = sitk.Resample(
            mag_sitk, vx_corr_sitk,
            sitk.Transform(), sitk.sitkNearestNeighbor, 0.0
        )
        
        # Get arrays
        vx_uncorr = sitk.GetArrayFromImage(vx_uncorr_resampled).astype(np.float32)
        vy_uncorr = sitk.GetArrayFromImage(vy_uncorr_resampled).astype(np.float32)
        vz_uncorr = sitk.GetArrayFromImage(vz_uncorr_resampled).astype(np.float32)
        mag_unpad = sitk.GetArrayFromImage(mag_resampled).astype(np.float32)
        
        self._logger.info(f"Resampled uncorrected shape: {vx_uncorr.shape}")
        
        # Save unpadded magnitude as NIfTI (transpose from sitk array order)
        # sitk array is (T, Z, Y, X), NIfTI expects (X, Y, Z, T)
        mag_unpad_nifti = np.transpose(mag_unpad, (3, 2, 1, 0))
        self._save_nifti(mag_unpad_nifti, unpadded_affine, output_paths['mag_unpadded'], "unpadded magnitude")
        
        # =====================================================================
        # 3. Compute delta = corrected - uncorrected
        # =====================================================================
        delta_vx = vx_corr - vx_uncorr
        delta_vy = vy_corr - vy_uncorr
        delta_vz = vz_corr - vz_uncorr
        
        self._logger.info(
            f"Delta ranges: vx=[{delta_vx.min():.2f}, {delta_vx.max():.2f}], "
            f"vy=[{delta_vy.min():.2f}, {delta_vy.max():.2f}], "
            f"vz=[{delta_vz.min():.2f}, {delta_vz.max():.2f}]"
        )
        
        # Save delta components as NIfTI (transpose from sitk array order)
        for name, delta_arr in [('vx', delta_vx), ('vy', delta_vy), ('vz', delta_vz)]:
            delta_nifti = np.transpose(delta_arr, (3, 2, 1, 0))  # (T,Z,Y,X) -> (X,Y,Z,T)
            self._save_nifti(delta_nifti, unpadded_affine, output_paths[f'delta_{name}'], f"delta {name}")
        
        # =====================================================================
        # 4. Create mask and build polynomial basis
        # =====================================================================
        # Use the mean magnitude (in NIfTI order) for masking
        mask = self._create_magnitude_mask(mag_unpad_nifti, mag_threshold, 3.0, shrink_margin)
        self._save_nifti(mask, unpadded_affine, output_paths['mask'], "correction mask")
        
        n_valid = int(np.sum(mask > 0))
        n_voxels = int(np.prod(unpadded_shape))
        self._logger.info(f"Valid voxels for fitting: {n_valid} / {n_voxels}")
        
        if n_valid < n_coeffs:
            self._logger.warning(f"Not enough valid voxels ({n_valid}) for polynomial fit")
            return output_paths
        
        basis = self._build_polynomial_basis(unpadded_shape, n_coeffs)
        
        # =====================================================================
        # 5. Fit polynomial coefficients
        # =====================================================================
        # Stack delta into (T, 3, Z, Y, X) format for _fit_polynomial_coefficients
        delta_stacked = np.stack([delta_vx, delta_vy, delta_vz], axis=1)
        
        coefficients = self._fit_polynomial_coefficients(delta_stacked, basis, mask)
        self._logger.info(f"Computed median coefficients, shape: {coefficients.shape}")
        
        # Save coefficients as numpy (not spatial data)
        np.save(output_paths['coefficients'], coefficients)
        self._logger.info(f"Saved coefficients to {output_paths['coefficients'].name}")
        
        # =====================================================================
        # 6. Generate ground truth correction volume
        # =====================================================================
        ground_truth = self._reconstruct_from_coefficients(coefficients, basis, unpadded_shape)
        self._save_nifti(ground_truth, unpadded_affine, output_paths['ground_truth'], "ground truth correction")
        
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