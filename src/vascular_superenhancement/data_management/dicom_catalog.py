import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import pydicom
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from ..utils.logger import setup_patient_logger, TqdmLoggingHandler
from ..utils.path_config import load_path_config

def get_dicom_tag_value(ds: pydicom.Dataset, tag: tuple) -> Any:
    """
    Safely get a DICOM tag value, returning None if the tag doesn't exist.
    
    Args:
        ds: DICOM dataset
        tag: DICOM tag tuple (group, element)
        
    Returns:
        The tag value or None if the tag doesn't exist
    """
    try:
        return ds[tag].value
    except KeyError:
        return None

def catalog_dicom_file(file_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Catalog a single DICOM file and extract relevant metadata.
    
    Args:
        file_path: Path to the DICOM file
        logger: Logger instance for logging messages
        
    Returns:
        Dictionary containing the extracted metadata
    """
    try:
        ds = pydicom.dcmread(file_path)
        
        # Extract required metadata
        metadata = {
            'filepath': str(file_path),
            'patientid': get_dicom_tag_value(ds, (0x0010, 0x0020)),
            'studyinstanceuid': get_dicom_tag_value(ds, (0x0020, 0x000D)),
            'seriesinstanceuid': get_dicom_tag_value(ds, (0x0020, 0x000E)),
            'seriesnumber': get_dicom_tag_value(ds, (0x0020, 0x0011)),
            'sopinstanceuid': get_dicom_tag_value(ds, (0x0008, 0x0018)),
            'modality': get_dicom_tag_value(ds, (0x0008, 0x0060)),
            'studydate': get_dicom_tag_value(ds, (0x0008, 0x0020)),
            'seriesdescription': get_dicom_tag_value(ds, (0x0008, 0x103E)),
            'instancenumber': get_dicom_tag_value(ds, (0x0020, 0x0013)),
            'imagepositionpatient': get_dicom_tag_value(ds, (0x0020, 0x0032)),
            'imageorientation': get_dicom_tag_value(ds, (0x0020, 0x0037)),
            'pixelspacing': get_dicom_tag_value(ds, (0x0028, 0x0030)),
            'slicethickness': get_dicom_tag_value(ds, (0x0018, 0x0050)),
            'tag_0x0019_0x10B3': get_dicom_tag_value(ds, (0x0019, 0x10B3)),
            'tag_0x0043_0x1030': get_dicom_tag_value(ds, (0x0043, 0x1030)),
            'numberoftemporalpositions': get_dicom_tag_value(ds, (0x0020, 0x0105)),
            'cardiacphasenumber': get_dicom_tag_value(ds, (0x0019, 0x10D7)),
            'cardiacnumberofimages': get_dicom_tag_value(ds, (0x0018, 0x1090)),
            'acquisitionnumber': get_dicom_tag_value(ds, (0x0020, 0x0012)),
            'imagesinacquisition': get_dicom_tag_value(ds, (0x0020, 0x1002)),
            'stackid': get_dicom_tag_value(ds, (0x0020, 0x9056)),
            'instackpositionnumber': get_dicom_tag_value(ds, (0x0020, 0x9057)),
            'slicelocation': get_dicom_tag_value(ds, (0x0020, 0x1041)),
            'locationsinacquisition': get_dicom_tag_value(ds, (0x0021, 0x104F)),
            'num3dslabs': get_dicom_tag_value(ds, (0x0021, 0x1056)),
            'locsper3dslab': get_dicom_tag_value(ds, (0x0021, 0x1057))
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error reading DICOM file {file_path}: {str(e)}")
        return None

def find_dicom_files(directory: Path) -> List[Path]:
    """
    Recursively find all DICOM files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of paths to DICOM files
    """
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            try:
                # Try to read the file as DICOM
                pydicom.dcmread(file_path, stop_before_pixels=True)
                dicom_files.append(file_path)
            except:
                continue
    return dicom_files

def catalog_patient_dicoms(patient_dir: Path, catalog_dir: Path, logger: logging.Logger, overwrite: bool = False) -> bool:
    """
    Catalog all DICOM files for a single patient.
    
    Args:
        patient_dir: Directory containing patient's DICOM files
        catalog_dir: Directory where to save the catalog
        logger: Logger instance for logging messages
        overwrite: Whether to overwrite existing catalog files
        
    Returns:
        bool: True if cataloging was successful, False otherwise
    """
    try:
        # Check if catalog file already exists
        catalog_file = catalog_dir / f"dicom_catalog_{patient_dir.name}.csv"
        if catalog_file.exists() and not overwrite:
            logger.info(f"Catalog for patient {patient_dir.name} already exists. Skipping.")
            return True
            
        # Find all DICOM files
        dicom_files = find_dicom_files(patient_dir)
        if not dicom_files:
            logger.warning(f"No DICOM files found in {patient_dir}")
            return False
            
        # Catalog each file with progress bar
        metadata_list = []
        for file_path in tqdm(dicom_files, desc=f"Processing {patient_dir.name}", leave=False):
            metadata = catalog_dicom_file(file_path, logger)
            if metadata:
                metadata_list.append(metadata)
                
        if not metadata_list:
            logger.warning(f"No valid DICOM files found in {patient_dir}")
            return False
            
        # Create DataFrame and save to CSV
        df = pd.DataFrame(metadata_list)
        df.to_csv(catalog_file, index=False)
        
        logger.info(f"Successfully cataloged {len(metadata_list)} DICOM files for patient {patient_dir.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error cataloging DICOM files for patient {patient_dir.name}: {str(e)}")
        return False

def _process_patient_directory(args: tuple) -> bool:
    """
    Helper function for parallel processing of patient directories.
    
    Args:
        args: Tuple containing (patient_dir, catalog_dir, overwrite, log_file, position)
        
    Returns:
        bool: True if cataloging was successful, False otherwise
    """
    patient_dir, catalog_dir, overwrite, log_file, position = args
    
    # Get path configuration for log directories
    path_config = load_path_config()
    main_log_dir = path_config.working_dir / "logs"
    patient_log_dir = main_log_dir / "patients"
    patient_log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create patient-specific logger
    patient_logger = logging.getLogger(f"vascular_superenhancement.patient.{patient_dir.name}")
    patient_logger.setLevel(logging.INFO)
    
    # Create process logger
    process_logger = logging.getLogger(f"dicom_catalog_{mp.current_process().name}")
    process_logger.setLevel(logging.INFO)
    
    # Configure both loggers
    for logger, log_path in [
        (process_logger, log_file),
        (patient_logger, patient_log_dir / f"{patient_dir.name}.log")
    ]:
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Add console handler that works with tqdm
        console_handler = TqdmLoggingHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    
    patient_logger.info(f"Starting DICOM cataloging for patient {patient_dir.name}")
    
    try:
        # Find all DICOM files first
        dicom_files = find_dicom_files(patient_dir)
        if not dicom_files:
            patient_logger.warning(f"No DICOM files found in {patient_dir}")
            return False
        
        # Create progress bar for DICOM file processing
        pbar = tqdm(total=len(dicom_files), desc=f"Patient {patient_dir.name}", position=position, leave=True)
        
        # Set the progress bar for the patient logger's TqdmLoggingHandler
        for handler in patient_logger.handlers:
            if isinstance(handler, TqdmLoggingHandler):
                handler.set_last_progress_bar(pbar)
        
        # Catalog each file with progress bar
        metadata_list = []
        for file_path in dicom_files:
            metadata = catalog_dicom_file(file_path, patient_logger)
            if metadata:
                metadata_list.append(metadata)
            pbar.update(1)
        
        if not metadata_list:
            patient_logger.warning(f"No valid DICOM files found in {patient_dir}")
            pbar.close()
            return False
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(metadata_list)
        catalog_file = catalog_dir / f"{patient_dir.name}_dicom_catalog.csv"
        df.to_csv(catalog_file, index=False)
        
        patient_logger.info(f"Successfully cataloged {len(metadata_list)} DICOM files for patient {patient_dir.name}")
        pbar.close()
        return True
        
    except Exception as e:
        patient_logger.error(f"Error during DICOM cataloging for patient {patient_dir.name}: {str(e)}")
        pbar.close()
        return False

def catalog_all_patients(source_dir: Path, catalog_dir: Path, logger: logging.Logger, overwrite: bool = False, num_workers: int = None, log_file: str = "dicom_catalog.log") -> None:
    """
    Catalog DICOM files for all patients in the source directory using parallel processing.
    
    Args:
        source_dir: Directory containing patient folders
        catalog_dir: Directory where to save the catalogs
        logger: Logger instance for logging messages
        overwrite: Whether to overwrite existing catalog files
        num_workers: Number of worker processes to use. If None, uses CPU count - 1
        log_file: Path to the log file for all processes
    """
    # Get list of patient directories
    patient_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not patient_dirs:
        logger.warning(f"No patient directories found in {source_dir}")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    logger.info(f"Starting parallel DICOM cataloging with {num_workers} workers")
    
    # Create main progress bar for overall patient completion
    main_pbar = tqdm(total=len(patient_dirs), desc="Overall Progress", position=0, leave=True)
    
    # Set the main progress bar for the main logger's TqdmLoggingHandler
    for handler in logger.handlers:
        if isinstance(handler, TqdmLoggingHandler):
            handler.set_last_progress_bar(main_pbar)
    
    # Prepare arguments for parallel processing, including position for each patient's progress bar
    # Start patient progress bars at position 1 to leave room for main progress bar
    process_args = [
        (patient_dir, catalog_dir, overwrite, log_file, i + 1)
        for i, patient_dir in enumerate(patient_dirs)
    ]
    
    # Use multiprocessing Pool
    with mp.Pool(processes=num_workers) as pool:
        # Process each patient
        results = []
        for result in pool.imap(_process_patient_directory, process_args):
            results.append(result)
            main_pbar.update(1)
    
    # Close the main progress bar
    main_pbar.close()
    
    # Log summary
    successful = sum(1 for r in results if r)
    logger.info(f"Completed processing {len(patient_dirs)} patients. Successfully cataloged: {successful}") 