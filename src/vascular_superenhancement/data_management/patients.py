from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import logging
import pandas as pd
from ..utils.logger import setup_patient_logger
from ..utils.path_config import PathConfig
from .dicom_catalog import catalog_patient_dicoms

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
        overwrite (bool): Whether to overwrite existing catalogs (default: False)
        
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
    overwrite: bool = False
    
    def __post_init__(self):
        """Validate that at least one identifier is provided and initialize the catalog."""
        if self.accession_number is None and self.phonetic_id is None:
            raise ValueError("At least one of accession_number or phonetic_id must be provided")
        
        # Set up patient-specific logger
        self._logger = setup_patient_logger(
            self.identifier,
            level=logging.DEBUG if self.debug else logging.INFO
        )
        
        # Validate against database unless explicitly skipped
        if not self.skip_database_validation:
            self._validate_against_database()
        
        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize catalogs as None - will be loaded on first access
        self._dicom_catalog = None
        self._dicom_catalog_3d_cine = None
    
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
            
            self._logger.info(f"Successfully loaded patient information from database")
            
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
    
    def _load_or_create_catalog(self, overwrite: Optional[bool] = None) -> None:
        """Load the DICOM catalog if it exists, otherwise create it.
        
        Args:
            overwrite: If provided, overrides self.overwrite for this call only.
                     If None, uses self.overwrite.
        """
        catalog_path = self.working_dir / f"dicom_catalog_{self.identifier}.csv"
        
        # Use provided overwrite if specified, otherwise use object's overwrite
        should_overwrite = overwrite if overwrite is not None else self.overwrite
        
        if catalog_path.exists() and not should_overwrite:
            try:
                self._dicom_catalog = pd.read_csv(catalog_path)
                self._logger.info(f"Loaded existing DICOM catalog for patient {self.identifier}")
            except Exception as e:
                self._logger.error(f"Error reading DICOM catalog: {str(e)}")
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
    
    def reload_catalog(self, overwrite: Optional[bool] = None) -> None:
        """Explicitly reload the DICOM catalog.
        
        Args:
            overwrite: If provided, overrides self.overwrite for this call only.
                     If None, uses self.overwrite.
        """
        self._logger.info(f"Reloading DICOM catalog for patient {self.identifier}")
        self._load_or_create_catalog(overwrite=overwrite)
        # Clear 3D Cine catalog since it depends on the DICOM catalog
        self._dicom_catalog_3d_cine = None
    
    def clear_catalog(self) -> None:
        """Clear the in-memory catalogs to free up memory."""
        self._logger.info(f"Clearing DICOM catalogs from memory for patient {self.identifier}")
        self._dicom_catalog = None
        self._dicom_catalog_3d_cine = None
    
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
            if catalog_path.exists() and not self.overwrite:
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
                f"overwrite={self.overwrite})") 