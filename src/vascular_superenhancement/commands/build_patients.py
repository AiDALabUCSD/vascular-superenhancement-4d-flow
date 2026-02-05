#!/usr/bin/env python3
"""
CLI script to build all patient images (3D cine and 4D flow) from DICOM catalogs.
"""

import argparse
import multiprocessing as mp
import subprocess
from pathlib import Path
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
    overwrite_corrected: bool | None,
    overwrite_downsampled: bool | None,
    dataset_logger: logging.Logger,
    debug: bool = False,
) -> bool:
    """Process a single patient's images.
    
    Args:
        patient_id: ID of the patient to process (could be accession number or phonetic ID)
        config: Name of the config file to use
        overwrite_images: Whether to overwrite existing images
        overwrite_catalogs: Whether to overwrite existing catalogs
        overwrite_corrected: Whether to overwrite corrected velocity files (None=use overwrite_images)
        overwrite_downsampled: Whether to overwrite downsampled files (None=use overwrite_images)
        dataset_logger: Logger for dataset-level logging
        debug: Whether to enable debug logging
    """
    # Set up patient-specific logger
    logger = setup_patient_logger(
        patient_id, 
        config=config,
        level=logging.DEBUG if debug else logging.INFO  # Level depends on debug flag
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
            overwrite_corrected=overwrite_corrected,
            overwrite_downsampled=overwrite_downsampled,
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


def run_sync(logger: logging.Logger) -> bool:
    """Run the sync_to_nas.py script to backup data.
    
    Args:
        logger: Logger for logging sync status
        
    Returns:
        bool: True if sync was successful, False otherwise
    """
    # Find the sync script relative to this file
    script_dir = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
    sync_script = script_dir / "sync_to_nas.py"
    
    if not sync_script.exists():
        logger.warning(f"Sync script not found at {sync_script}, skipping sync")
        return False
    
    try:
        logger.info("Starting sync to NAS...")
        result = subprocess.run(
            ["python", str(sync_script)],
            capture_output=True,
            text=True,
            cwd=sync_script.parent.parent,  # Run from repo root
        )
        
        if result.returncode == 0:
            logger.info("Sync completed successfully")
            return True
        else:
            logger.error(f"Sync failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Sync stderr: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running sync: {str(e)}")
        return False

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
        "--overwrite-corrected",
        action="store_true",
        help="Overwrite existing corrected velocity files (4D NIfTIs, per-timepoint, speed)",
    )
    parser.add_argument(
        "--overwrite-downsampled",
        action="store_true",
        help="Overwrite existing downsampled training data files",
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
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync to NAS after all patients complete",
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
        if args.overwrite_corrected:
            logger.info("Overwrite corrected mode: ON - corrected velocity files will be overwritten")
        if args.overwrite_downsampled:
            logger.info("Overwrite downsampled mode: ON - downsampled files will be overwritten")
        if args.debug:
            logger.info("Debug mode: ON - detailed logging enabled")
        
        # Get list of patient IDs from unzipped directory
        unzipped_dir = config.unzipped_dir
        if not unzipped_dir.exists():
            raise FileNotFoundError(f"Unzipped directory not found: {unzipped_dir}")
            
        # Get patient IDs from unzipped directory
        patient_ids = sorted([d.name for d in unzipped_dir.iterdir() if d.is_dir()])
        if not patient_ids:
            raise ValueError(f"No patient directories found in {unzipped_dir}")
            
        logger.info(f"Found {len(patient_ids)} patients to process")
        
        # Set up multiprocessing
        num_workers = args.max_processors or max(1, mp.cpu_count() - 2)
        logger.info(f"Using {num_workers} worker processes")
        
        # Create a pool of workers
        with mp.Pool(num_workers) as pool:
            # Create a list of tasks
            # For overwrite_corrected/downsampled: True if flag set, None otherwise (uses overwrite_images)
            overwrite_corrected = True if args.overwrite_corrected else None
            overwrite_downsampled = True if args.overwrite_downsampled else None
            tasks = [
                (patient_id, args.config, args.overwrite_images, args.overwrite_catalogs, 
                 overwrite_corrected, overwrite_downsampled, logger, args.debug)
                for patient_id in patient_ids
            ]
            
            # Process patients with progress bar
            with tqdm(total=len(patient_ids), desc="Building patient images") as pbar:
                for _ in pool.starmap(process_patient, tasks):
                    pbar.update()
        
        logger.info("Image building completed successfully")
        
        # Sync to NAS after all patients complete
        if args.sync:
            logger.info("All patients processed, starting sync to NAS...")
            run_sync(logger)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error during image building: {str(e)}")
        raise

if __name__ == "__main__":
    main() 