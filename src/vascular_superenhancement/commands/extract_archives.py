#!/usr/bin/env python3
"""
CLI script to extract all archives from the zipped directory to the unzipped directory.
"""

import argparse
from pathlib import Path
from ..utils.path_config import load_path_config
from ..data_management.archives import process_all_archives
from ..utils.logger import setup_dataset_logger

logger = setup_dataset_logger("archives")

def main():
    parser = argparse.ArgumentParser(
        description="Extract all archives from the zipped directory to the unzipped directory."
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
        help="Overwrite existing extracted files if they exist",
    )
    
    args = parser.parse_args()
    
    try:
        # Load path configuration
        config = load_path_config(args.config)
        
        logger.info(f"Starting archive extraction using config: {args.config}")
        logger.info(f"Zipped directory: {config.zipped_dir}")
        logger.info(f"Unzipped directory: {config.unzipped_dir}")
        if args.overwrite:
            logger.info("Overwrite mode: ON - existing files will be overwritten")
        else:
            logger.info("Overwrite mode: OFF - existing files will be skipped")
        
        # Process all archives
        process_all_archives(config.zipped_dir, config.unzipped_dir, args.overwrite, logger)
        
        logger.info("Archive extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Error during archive extraction: {str(e)}")
        raise

if __name__ == "__main__":
    main() 