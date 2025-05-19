#!/usr/bin/env python3
"""
CLI script to build all patient images (3D cine and 4D flow) from DICOM catalogs.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Optional
import logging
from ..utils.path_config import load_path_config
from ..data_management.patients import Patient
from ..utils.logger import setup_dataset_logger, setup_patient_logger
from tqdm import tqdm

def process_patient(
    patient_id: str,
    config: str,
    overwrite_images: bool,
    overwrite_catalogs: bool,
    dataset_logger: logging.Logger,
    debug: bool = False,
) -> bool:
    """Process a single patient's images.
    
    Args:
        patient_id: ID of the patient to process (could be accession number or phonetic ID)
        config: Name of the config file to use
        overwrite_images: Whether to overwrite existing images
        overwrite_catalogs: Whether to overwrite existing catalogs
        dataset_logger: Logger for dataset-level logging
        debug: Whether to enable debug logging
    """
    # Set up patient-specific logger
    logger = setup_patient_logger(
        patient_id, 
        config=config,
        file_level=logging.DEBUG,  # Always log debug to file
        console_level=logging.DEBUG if debug else logging.INFO  # Console level depends on debug flag
    )
    
    try:
        # Load path configuration
        path_config = load_path_config(config)
        
        # Create patient object - let the Patient class determine which identifier to use
        patient = Patient(
            path_config=path_config,
            phonetic_id=patient_id,  # Try phonetic_id first
            debug=debug,
            overwrite_images=overwrite_images,
            overwrite_catalogs=overwrite_catalogs,
            config=config,  # Pass the config parameter
            dataset_logger=dataset_logger  # Pass the dataset logger
        )
        
        # Build images
        logger.info(f"Building images for patient {patient_id}")
        patient.build_images(as_numpy=False)
        patient.build_per_timepoint_images()
        logger.info(f"Successfully built images for patient {patient_id}")
        dataset_logger.info(f"Successfully processed patient {patient_id}")
        
    except Exception as e:
        logger.error(f"Error processing patient {patient_id}: {str(e)}")
        dataset_logger.error(f"Failed to process patient {patient_id}: {str(e)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Build all patient images (3D cine and 4D flow) from DICOM catalogs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Name of the config file to use (without .yaml extension)",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite existing image files if they exist",
    )
    parser.add_argument(
        "--overwrite-catalogs",
        action="store_true",
        help="Overwrite existing catalog files if they exist",
    )
    parser.add_argument(
        "--max-processors",
        type=int,
        default=None,
        help="Maximum number of processors to use. If not specified, uses CPU count - 1",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    try:
        # Load path configuration first
        config = load_path_config(args.config)
        
        # Set up dataset-level logger
        logger = setup_dataset_logger(
            "build_patients", 
            config=args.config,
            level=logging.DEBUG if args.debug else logging.INFO
        )
        
        logger.info(f"Starting image building using config: {args.config}")
        logger.info(f"Repository root: {config.repository_root}")
        if args.overwrite_images:
            logger.info("Overwrite images mode: ON - existing image files will be overwritten")
        else:
            logger.info("Overwrite images mode: OFF - existing image files will be skipped")
        if args.overwrite_catalogs:
            logger.info("Overwrite catalogs mode: ON - existing catalog files will be overwritten")
        else:
            logger.info("Overwrite catalogs mode: OFF - existing catalog files will be skipped")
        if args.debug:
            logger.info("Debug mode: ON - detailed logging enabled")
        
        # Get list of patient IDs from unzipped directory
        unzipped_dir = config.unzipped_dir
        if not unzipped_dir.exists():
            raise FileNotFoundError(f"Unzipped directory not found: {unzipped_dir}")
            
        # Get patient IDs from unzipped directory
        patient_ids = [d.name for d in unzipped_dir.iterdir() if d.is_dir()]
        if not patient_ids:
            raise ValueError(f"No patient directories found in {unzipped_dir}")
            
        logger.info(f"Found {len(patient_ids)} patients to process")
        
        # Set up multiprocessing
        num_workers = args.max_processors or max(1, mp.cpu_count() - 2)
        logger.info(f"Using {num_workers} worker processes")
        
        # Create a pool of workers
        with mp.Pool(num_workers) as pool:
            # Create a list of tasks
            tasks = [
                (patient_id, args.config, args.overwrite_images, args.overwrite_catalogs, logger, args.debug)
                for patient_id in patient_ids
            ]
            
            # Process patients with progress bar
            with tqdm(total=len(patient_ids), desc="Building patient images") as pbar:
                for _ in pool.starmap(process_patient, tasks):
                    pbar.update()
        
        logger.info("Image building completed successfully")
        
    except Exception as e:
        logger.error(f"Error during image building: {str(e)}")
        raise

if __name__ == "__main__":
    main() 