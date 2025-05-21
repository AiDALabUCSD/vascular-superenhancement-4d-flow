from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import logging
import pandas as pd
from ..utils.logger import setup_patient_logger
from ..utils.path_config import PathConfig
from .dicom_catalog import catalog_patient_dicoms
import nibabel as nib
from .dicom_to_nifti import DicomToNiftiConverter

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
    config: str = "default"
    dataset_logger: Optional[logging.Logger] = None
    
    def __post_init__(self):
        """Validate that at least one identifier is provided and initialize the catalog."""
        if self.accession_number is None and self.phonetic_id is None:
            raise ValueError("At least one of accession_number or phonetic_id must be provided")
        
        # Set up patient-specific logger with separate file and console levels
        self._logger = setup_patient_logger(
            self.identifier,
            config=self.config,  # Pass the config parameter
            file_level=logging.DEBUG,  # Always log debug to file
            console_level=logging.DEBUG if self.debug else logging.INFO  # Console level depends on debug flag
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
        if not all(count == counts['cine'] for count in counts.values()):
            self._logger.error(
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
        
        return len(cine_files)
    
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
                self._logger.info(f"Found old format catalog, migrating to new format")
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
        self._logger.debug("Building 3D cine image")
        cine = self.get_3d_cine(as_numpy=as_numpy)
        
        # Build 4D flow
        self._logger.debug("Building 4D flow images")
        flow = self.get_4d_flow(as_numpy=as_numpy)
        
        # Combine into result dictionary
        result = {
            '3d_cine': cine,
            '4d_flow': flow
        }
        
        self._logger.info(f"Successfully built all images for patient {self.identifier}")
        return result
    
    def build_3d_cine_per_timepoint(self) -> None:
        """Build 3D cine volumes for each timepoint."""
        
        self._logger.info(f"Building 3D cine volumes for each timepoint for patient {self.identifier}")
        
        cine_path = self.nifti_dir / f"3d_cine_{self.identifier}.nii.gz"
        fmag_path = self.nifti_dir / f"4d_flow_mag_{self.identifier}.nii.gz"
        output_dir = self.cine_per_timepoint_dir
        
        # if the cine or flow mag do not exist, raise an error
        if not cine_path.exists() or not fmag_path.exists():
            raise ValueError(f"3D cine or flow mag for patient {self.identifier} do not exist")
        
        # the output directory is not empty and overwrite_images is False, log number of files
        if output_dir.exists() and len(list(output_dir.glob('*.nii.gz')))>0 and not self.overwrite_images:
            self._logger.info(f"Output directory {output_dir} already exists and overwrite_images is False, skipping")
            self._logger.info(f"Number of files in output directory: {len(list(output_dir.glob('*.nii.gz')))}")
            return
                
        # build the 3D cine volumes for each timepoint
        converter = DicomToNiftiConverter.from_patient(self)
        converter.build_3d_cine_per_timepoint(
            from_cine_path=cine_path,
            to_flow_mag_path=fmag_path,
            output_dir=output_dir
        )
        
        self._logger.info(f"Successfully built 3D cine volumes for each timepoint for patient {self.identifier}")
    def build_4d_flow_per_timepoint(self) -> None:
        """Build 4D flow volumes for each timepoint."""        
        
        self._logger.info(f"Building 4D flow volumes for each timepoint and component for patient {self.identifier}")
        
        flow_components = ['mag', 'vx', 'vy', 'vz']

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

        # Instantiate converter once
        converter = DicomToNiftiConverter.from_patient(self)

        # Run per-timepoint conversion
        for comp,flow_path, split_path in paths:
            self._logger.info(f"Working on {flow_path}")
            
            if not flow_path.exists():
                raise ValueError(f"4D flow {comp} for patient {self.identifier} do not exist")
            
            if split_path.exists() and len(list(split_path.glob('*.nii.gz')))>0 and not self.overwrite_images:
                self._logger.info(f"Output directory {split_path} already exists and overwrite_images is False, skipping")
                self._logger.info(f"Number of files in output directory: {len(list(split_path.glob('*.nii.gz')))}")
                continue
            
            converter.build_per_timepoint(
                name=f"4d_flow_{comp}_{self.identifier}",
                img_path=flow_path,
                output_dir=split_path
            )
        
        self._logger.info(f"Successfully built 4D flow volumes for each timepoint and component for patient {self.identifier}")
    
    def build_per_timepoint_images(self) -> None:
        """Build per-timepoint volumes for 3d cine and 4d flow using build_3d_cine_per_timepoint and build_4d_flow_per_timepoint"""
        
        self._logger.info(f"Building per-timepoint volumes for 3d cine and 4d flow for patient {self.identifier}")
        
        # build the per-timepoint volumes for 3d cine
        self.build_3d_cine_per_timepoint()
        
        # build the per-timepoint volumes for 4d flow
        self.build_4d_flow_per_timepoint()
        
        self._logger.info(f"Successfully built per-timepoint volumes for 3d cine and 4d flow for patient {self.identifier}")
    
    def __str__(self) -> str:
        """Return a string representation of the patient."""
        return f"Patient({self.identifier})"
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the patient."""
        return (f"Patient(path_config={self.path_config}, "
                f"accession_number={self.accession_number}, "
                f"phonetic_id={self.phonetic_id}, "
                f"skip_database_validation={self.skip_database_validation}, "
                f"debug={self.debug}, "
                f"overwrite_images={self.overwrite_images}, "
                f"overwrite_catalogs={self.overwrite_catalogs})") 