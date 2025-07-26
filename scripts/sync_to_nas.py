#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.utils.logger import setup_sync_logger
from datetime import datetime
import os
import time
import re
from tqdm import tqdm
import logging

def format_size(size_bytes):
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def parse_rsync_progress(line):
    """Parse rsync progress output to extract transfer information."""
    # Match patterns like "1,234,567 100% 1.23MB/s 0:00:45"
    match = re.search(r'(\d+(?:,\d+)*)\s+(\d+)%\s+([\d.]+)([KMG]B/s)\s+(\d+):(\d+):(\d+)', line)
    if match:
        bytes_transferred = int(match.group(1).replace(',', ''))
        percent = int(match.group(2))
        speed = float(match.group(3))
        speed_unit = match.group(4)
        hours, minutes, seconds = map(int, match.group(5, 6, 7))
        return {
            'bytes': bytes_transferred,
            'percent': percent,
            'speed': f"{speed} {speed_unit}",
            'time_remaining': f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        }
    return None

def check_mount_point(path: Path, logger) -> bool:
    """Check if a path is a valid mount point."""
    try:
        # Get the mount point of the path
        result = subprocess.run(['mountpoint', '-q', str(path)], capture_output=True)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking mount point: {str(e)}")
        return False

def get_total_files_and_size(source: Path) -> tuple[int, int]:
    """Get total number of files and total size to sync."""
    try:
        # Use find with -ls to get file sizes
        result = subprocess.run(
            ['find', str(source), '-type', 'f', '-ls'],
            capture_output=True,
            text=True
        )
        
        if not result.stdout.strip():
            return 0, 0
            
        # Parse the output to get file sizes
        total_size = 0
        file_count = 0
        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 7:  # -ls output format has size in 7th column
                try:
                    size = int(parts[6])
                    total_size += size
                    file_count += 1
                except (ValueError, IndexError):
                    continue
                    
        return file_count, total_size
    except Exception:
        return 0, 0

def ensure_destination_structure(source: Path, destination: Path, logger) -> bool:
    """Ensure destination directory structure exists and is writable."""
    try:
        # Create all parent directories
        destination.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions by creating a test file
        test_file = destination / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()  # Remove test file
        except Exception as e:
            logger.error(f"Destination directory is not writable: {str(e)}")
            return False
            
        # Check if we can create subdirectories
        test_dir = destination / ".test_dir"
        try:
            test_dir.mkdir()
            test_dir.rmdir()
        except Exception as e:
            logger.error(f"Cannot create subdirectories in destination: {str(e)}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error setting up destination structure: {str(e)}")
        return False

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
        # Log the start of sync operation
        logger.info(f"Starting sync operation at {datetime.now().strftime('%I:%M:%S %p')}")
        logger.info(f"Source directory: {source}")
        logger.info(f"Destination directory: {destination}")
        
        # Check if source directory exists
        if not source.exists():
            logger.error(f"Source directory does not exist: {source}")
            return False
            
        # Check if source directory is readable
        if not os.access(source, os.R_OK):
            logger.error(f"Source directory is not readable: {source}")
            return False
            
        # Ensure destination structure is properly set up
        if not ensure_destination_structure(source, destination, logger):
            return False

        # First do a dry run to see what would be synced
        dry_run_cmd = [
            "rsync",
            "-avn",  # Archive mode, verbose, dry run
            "--delete",  # Delete files in dest that don't exist in source
            "--ignore-errors",  # Continue even if some files fail
            "--force",  # Force deletion of directories
            "--no-perms",  # Don't transfer permissions
            "--no-owner",  # Don't transfer owner
            "--no-group",  # Don't transfer group
            "--exclude=logs/sync.log",
            "--include=.*",  # Include hidden files and directories
            "--stats",  # Show transfer statistics
            f"{source}/",  # Trailing slash to copy contents
            str(destination)
        ]
        
        print("\nChecking for files that need syncing...")
        print(f"Running command: {' '.join(dry_run_cmd)}")
        dry_run = subprocess.run(dry_run_cmd, capture_output=True, text=True)
        
        # Print the full dry run output for debugging
        print("\nDry run output:")
        print(dry_run.stdout)
        if dry_run.stderr:
            print("\nDry run errors:")
            print(dry_run.stderr)
        
        # Count files that would be transferred
        files_to_transfer = [line for line in dry_run.stdout.split('\n') if line.startswith('>f')]
        files_to_delete = [line for line in dry_run.stdout.split('\n') if line.startswith('*deleting')]
        
        # Check if there are any files to transfer based on the stats output
        has_files_to_transfer = False
        for line in dry_run.stdout.split('\n'):
            if "Number of regular files transferred:" in line:
                num_files = line.split(":")[1].strip()
                if num_files != "0":
                    has_files_to_transfer = True
                    break
        
        if not has_files_to_transfer and not files_to_delete:
            print("No files need to be synced - everything is up to date!")
            return True
            
        print(f"\nFound {len(files_to_transfer)} files to transfer")
        if files_to_delete:
            print(f"Found {len(files_to_delete)} files to delete")
        
        # Get total number of files and size to sync
        total_files, total_size = get_total_files_and_size(source)
        size_str = format_size(total_size)
        logger.info(f"Found {total_files} files ({size_str}) to sync")
        print(f"Total size to sync: {size_str}")
        
        # Construct rsync command with additional options for better error handling
        cmd = [
            "rsync",
            "-av",  # Archive mode, verbose (removed -n for dry run)
            "--delete",  # Delete files in dest that don't exist in source
            "--ignore-errors",  # Continue even if some files fail
            "--force",  # Force deletion of directories
            "--no-perms",  # Don't transfer permissions
            "--no-owner",  # Don't transfer owner
            "--no-group",  # Don't transfer group
            "--progress",  # Show progress during transfer
            "--stats",  # Show transfer statistics
            "--exclude=logs/sync.log",
            "--include=.*",  # Include hidden files and directories
            f"{source}/",  # Trailing slash to copy contents
            str(destination)
        ]
        
        print("\nStarting sync... This may take a while for large transfers.")
        
        # Run rsync with timeout and progress monitoring
        start_time = time.time()
        files_processed = 0
        bytes_processed = 0
        current_file = None
        
        # Create progress bar
        pbar = tqdm(
            total=len(files_to_transfer),
            desc="Syncing files",
            unit="files",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor the process output
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break
                    
                # Read output without blocking
                output = process.stdout.readline()
                if output:
                    # Track file changes
                    if output.startswith('>f') or output.startswith('*deleting'):
                        file_path = output.split(' ', 1)[1].strip()
                        if file_path != current_file:
                            files_processed += 1
                            current_file = file_path
                            pbar.update(1)
                            pbar.set_postfix(file=os.path.basename(file_path))
                    
                time.sleep(0.1)  # Small delay to prevent CPU spinning
                
            # Get remaining output
            stdout, stderr = process.communicate()
            
            # Close progress bar
            pbar.close()
            
            # Log any errors from stderr
            if stderr:
                logger.error(stderr)
                print("\nErrors occurred during sync:")
                print(stderr)
                
            if process.returncode == 0:
                duration = time.time() - start_time
                completion_msg = f"Sync completed successfully in {duration:.1f} seconds"
                logger.info(completion_msg)
                print(f"\n{completion_msg}")
                return True
            else:
                error_msg = f"Sync failed with error code {process.returncode}"
                logger.error(error_msg)
                print(f"\n{error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Error running rsync: {str(e)}"
            logger.error(error_msg)
            print(f"\n{error_msg}")
            return False
            
    except Exception as e:
        error_msg = f"Error during sync: {str(e)}"
        logger.error(error_msg)
        print(f"\n{error_msg}")
        return False

def setup_sync_logger():
    """Setup logger for sync operations."""
    # Create logs directory if it doesn't exist
    log_dir = Path("working_dir/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger("sync")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logging
    logger.handlers = []
    
    # Create file handler
    log_file = log_dir / "sync.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    
    # Create formatter and add it to the handlers
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    # Setup logger
    logger = setup_sync_logger()
    
    # Print immediate feedback
    print("Starting backup process...")
    print("Logs will be written to working_dir/logs/sync.log")
    
    logger.info("Starting backup script")
    
    try:
        # Load path configuration
        print("Loading configuration...")
        logger.info("Loading path configuration")
        path_config = load_path_config()
        
        # Get source and destination paths
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        source = Path("/home/ayeluru/vascular-superenhancement-4d-flow/working_dir")
        destination = Path("/mnt/yeluru/mnt/fourier/projects/vascular-superenhancement-4d-flow")
        
        print(f"\nSource: {source}")
        print(f"Destination: {destination}")
        
        # Verify source exists and has content
        if not source.exists():
            print(f"Error: Source directory does not exist: {source}")
            sys.exit(1)
            
        source_files = list(source.rglob("*"))
        print(f"\nFound {len(source_files)} files in source directory")
        if len(source_files) > 0:
            print("First few files:")
            for f in source_files[:5]:
                print(f"  {f}")
        
        # Verify destination exists
        if not destination.exists():
            print(f"Creating destination directory: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Source path: {source}")
        logger.info(f"Destination path: {destination}")
        logger.info(f"Logs are being written to: {Path('working_dir/logs/sync.log').resolve()}")
        
        # Perform sync
        print("\nStarting sync process...")
        success = sync_directories(source, destination, logger, path_config)
        
        if not success:
            print("\nSync failed! Check the log file for details.")
            logger.error("Sync operation failed")
            sys.exit(1)
        else:
            print("\nSync completed successfully!")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 