import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
import logging

def detect_archive_type(file_path: Path) -> Optional[str]:
    """
    Detect the type of archive file based on its extension.
    
    Args:
        file_path: Path to the archive file
        
    Returns:
        str: Archive type ('zip', 'tar', 'tar.gz', 'tgz') or None if not recognized
    """
    # Get all suffixes and join them
    suffixes = ''.join(file_path.suffixes).lower()
    
    if suffixes == '.zip':
        return 'zip'
    elif suffixes == '.tar':
        return 'tar'
    elif suffixes in ['.tar.gz', '.tgz']:
        return 'tar.gz'
    return None

def extract_zip(source: Path, destination: Path) -> None:
    """
    Extract a ZIP archive.
    
    Args:
        source: Path to the ZIP file
        destination: Path where to extract the contents
        
    Raises:
        zipfile.BadZipFile: If the file is not a valid ZIP file
        zipfile.LargeZipFile: If the ZIP file requires ZIP64 functionality
        FileNotFoundError: If the source file does not exist
        PermissionError: If there are permission issues
    """
    try:
        # First check if file exists and is readable
        if not source.exists():
            raise FileNotFoundError(f"ZIP file not found: {source}")
        if not os.access(source, os.R_OK):
            raise PermissionError(f"No read permission for ZIP file: {source}")
            
        # Try to open and test the ZIP file
        with zipfile.ZipFile(source, 'r') as zip_ref:
            # Test if the ZIP file is valid by trying to read the file list
            file_list = zip_ref.namelist()
            if not file_list:
                raise zipfile.BadZipFile(f"ZIP file appears to be empty: {source}")
            
            # Extract all files
            zip_ref.extractall(destination)
            
    except zipfile.BadZipFile as e:
        raise zipfile.BadZipFile(f"Invalid ZIP file format: {source}. Error: {str(e)}")
    except zipfile.LargeZipFile as e:
        raise zipfile.LargeZipFile(f"ZIP file requires ZIP64 functionality: {source}. Error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error extracting ZIP file {source}: {str(e)}")

def extract_tar(source: Path, destination: Path) -> None:
    """
    Extract a TAR archive.
    
    Args:
        source: Path to the TAR file
        destination: Path where to extract the contents
    """
    with tarfile.open(source, 'r:*') as tar_ref:
        tar_ref.extractall(destination)

def extract_archive(source: Path, destination: Path, overwrite: bool, logger: logging.Logger) -> bool:
    """
    Extract an archive file based on its type.
    
    Args:
        source: Path to the archive file
        destination: Path where to extract the contents
        overwrite: Whether to overwrite existing files (default: False)
        logger: Logger instance for logging messages
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Log basic file information
        logger.info(f"Attempting to extract archive: {source}")
        logger.info(f"File size: {source.stat().st_size} bytes")
        logger.info(f"File permissions: {oct(source.stat().st_mode)[-3:]}")
        
        archive_type = detect_archive_type(source)
        if archive_type is None:
            logger.error(f"Unsupported archive type for file: {source}")
            logger.error(f"File extension: {''.join(source.suffixes)}")
            return False
            
        # Check if destination exists and handle overwrite
        if destination.exists():
            if not overwrite:
                logger.warning(f"Destination {destination} already exists. Skipping extraction.")
                return True
            else:
                logger.info(f"Destination {destination} exists. Overwriting due to --overwrite flag.")
                shutil.rmtree(destination)
        
        # Create destination directory
        destination.mkdir(parents=True, exist_ok=True)
        
        try:
            if archive_type == 'zip':
                extract_zip(source, destination)
            elif archive_type in ['tar', 'tar.gz']:
                extract_tar(source, destination)
        except Exception as e:
            logger.error(f"Error during extraction of {source}: {str(e)}")
            logger.error(f"Archive type: {archive_type}")
            return False
            
        logger.info(f"Successfully extracted {source} to {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {source}: {str(e)}")
        return False

def get_patient_name(archive_path: Path) -> str:
    """
    Extract the patient name from the archive filename.
    
    Args:
        archive_path: Path to the archive file
        
    Returns:
        str: Patient name (filename without extension)
    """
    return archive_path.stem

def process_all_archives(zipped_dir: Path, unzipped_dir: Path, overwrite: bool, logger: logging.Logger) -> None:
    """
    Process all archive files in the zipped directory.
    
    Args:
        zipped_dir: Directory containing the archive files
        unzipped_dir: Directory where to extract the contents
        overwrite: Whether to overwrite existing files (default: False)
        logger: Logger instance for logging messages
    """
    # Create unzipped directory if it doesn't exist
    unzipped_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each archive file
    for archive_file in zipped_dir.glob('*'):
        if not archive_file.is_file():
            continue
            
        patient_name = get_patient_name(archive_file)
        patient_dir = unzipped_dir / patient_name
        
        logger.info(f"Processing archive for patient {patient_name}")
        success = extract_archive(archive_file, patient_dir, overwrite, logger)
        
        if not success:
            logger.warning(f"Failed to process archive for patient {patient_name}") 