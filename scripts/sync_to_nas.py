#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.utils.logger import setup_sync_logger
from datetime import datetime

def sync_directories(source: Path, destination: Path, logger, path_config) -> bool:
    """
    Sync source directory to destination using rsync.
    
    Args:
        source: Source directory path
        destination: Destination directory path
        logger: Logger instance
        path_config: Path configuration object
        
    Returns:
        bool: True if sync was successful, False otherwise
    """
    try:
        # Ensure destination directory exists
        destination.mkdir(parents=True, exist_ok=True)
        
        # Construct rsync command
        cmd = [
            "rsync",
            "-av",  # Archive mode, verbose
            "--delete",  # Delete files in dest that don't exist in source
            f"{source}/",  # Trailing slash to copy contents
            str(destination)
        ]
        
        # Run rsync
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse rsync output to check for real changes
            output_lines = result.stdout.strip().split('\n')
            # Look for actual file changes, ignoring directory entries and summary lines
            changed_files = [line for line in output_lines 
                           if line.strip() 
                           and not line.startswith('sending incremental file list')
                           and not line.endswith('/')
                           and not line.startswith('sent ')
                           and not line.startswith('total size')]
            
            if changed_files:  # Only log if there are actual file changes
                logger.info(f"=== Sync started at {datetime.now().strftime('%I:%M:%S %p')} ===")
                logger.info(f"From: {source}")
                logger.info(f"To: {destination}")
                logger.info("Changed files:")
                for file in changed_files:
                    logger.info(f"  {file}")
                logger.info("=== Sync completed successfully ===\n")
            return True
        else:
            logger.error(f"Sync failed with error:\n{result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error during sync: {str(e)}")
        return False

def main():
    # Setup logger
    logger = setup_sync_logger()
    
    try:
        # Load path configuration
        path_config = load_path_config()
        
        # Get source and destination paths
        # Source is the working directory inside the repository
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        source = repo_root / "working_dir"
        
        # Destination is the project directory
        destination = path_config.base_data_dir / "projects" / path_config.project_name
        
        # Perform sync
        success = sync_directories(source, destination, logger, path_config)
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 