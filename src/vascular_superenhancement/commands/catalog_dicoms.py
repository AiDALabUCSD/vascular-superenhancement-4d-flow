#!/usr/bin/env python3
"""
CLI script to catalog DICOM files for all patients in the unzipped directory.
"""

import argparse
from pathlib import Path
from ..utils.path_config import load_path_config
from ..data_management.dicom_catalog import catalog_all_patients
from ..utils.logger import setup_dataset_logger
import multiprocessing as mp

def main():
    parser = argparse.ArgumentParser(
        description="Catalog DICOM files for all patients in the unzipped directory."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Name of the config file to use (without .yaml extension)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing catalog files if they exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes to use. If not specified, uses CPU count - 1",
    )
    
    args = parser.parse_args()
    
    try:
        # Load path configuration first
        config = load_path_config(args.config)
        
        # Set up logger with the specified config
        logger = setup_dataset_logger("dicom_catalog", config=args.config)
        
        logger.info(f"Starting DICOM cataloging using config: {args.config}")
        logger.info(f"Unzipped directory: {config.unzipped_dir}")
        logger.info(f"Repository root: {config.repository_root}")
        if args.overwrite:
            logger.info("Overwrite mode: ON - existing catalog files will be overwritten")
        else:
            logger.info("Overwrite mode: OFF - existing catalog files will be skipped")
        
        # Create catalog directory in repository root
        catalog_dir = config.repository_root / "dicom_catalogs"
        catalog_dir.mkdir(parents=True, exist_ok=True)
        
        # Catalog DICOM files
        catalog_all_patients(
            config.unzipped_dir, 
            catalog_dir, 
            logger, 
            args.overwrite,
            args.num_workers,
            config=args.config
        )
        
        logger.info("DICOM cataloging completed successfully")
        
    except Exception as e:
        logger.error(f"Error during DICOM cataloging: {str(e)}")
        raise

if __name__ == "__main__":
    main() 