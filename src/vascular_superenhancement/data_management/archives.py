import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from vascular_superenhancement.utils.logger import setup_dataset_logger

logger = setup_dataset_logger("archives")

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
    """
    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(destination)

def extract_tar(source: Path, destination: Path) -> None:
    """
    Extract a TAR archive.
    
    Args:
        source: Path to the TAR file
        destination: Path where to extract the contents
    """
    with tarfile.open(source, 'r:*') as tar_ref:
        tar_ref.extractall(destination)

def extract_archive(source: Path, destination: Path, overwrite: bool = False) -> bool:
    """
    Extract an archive file based on its type.
    
    Args:
        source: Path to the archive file
        destination: Path where to extract the contents
        overwrite: Whether to overwrite existing files (default: False)
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        archive_type = detect_archive_type(source)
        if archive_type is None:
            logger.error(f"Unsupported archive type for file: {source}")
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
        
        if archive_type == 'zip':
            extract_zip(source, destination)
        elif archive_type in ['tar', 'tar.gz']:
            extract_tar(source, destination)
            
        logger.info(f"Successfully extracted {source} to {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting {source}: {str(e)}")
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

def process_all_archives(zipped_dir: Path, unzipped_dir: Path, overwrite: bool = False) -> None:
    """
    Process all archive files in the zipped directory.
    
    Args:
        zipped_dir: Directory containing the archive files
        unzipped_dir: Directory where to extract the contents
        overwrite: Whether to overwrite existing files (default: False)
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
        success = extract_archive(archive_file, patient_dir, overwrite)
        
        if not success:
            logger.warning(f"Failed to process archive for patient {patient_name}") 