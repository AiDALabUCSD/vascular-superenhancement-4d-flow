#!/usr/bin/env python3
"""
CLI script to write NIfTI predictions back to DICOM format.

Uses hydra config to determine prediction directory and output location.
"""

from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import pandas as pd

from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.data_management.patients import Patient

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.1",
    config_path=str((Path(__file__).resolve().parents[3] / "hydra_configs").as_posix()),
    config_name="config"
)
def main(cfg: DictConfig):
    """Write predictions to DICOM format for a patient or all test patients.
    
    Args:
        cfg: Hydra config containing nifti_to_dicom settings
    """
    # Check for required parameters
    if not cfg.nifti_to_dicom.get('patient_id') and not cfg.nifti_to_dicom.get('all_patients'):
        logger.error("Either patient_id or all_patients must be provided")
        raise ValueError("Either patient_id or all_patients must be provided")
    
    # Load path configuration
    path_config = load_path_config(cfg.path_config.path_config_name)
    
    # Construct base output directory from hydra config
    output_dir = Path(cfg.inference.output_dir)
    
    if cfg.nifti_to_dicom.get('all_patients'):
        logger.info("Writing DICOMs for all test patients")
        
        # Read splits CSV to get test patient IDs
        df = pd.read_csv(cfg.data.splits_path)
        patient_ids = df[df.split == 'test'].patient_id.tolist()
        
        logger.info(f"Found {len(patient_ids)} test patients")
        
        for patient_id in patient_ids:
            logger.info(f"Starting DICOM writing for patient_id: {patient_id}")
            try:
                # Create patient object
                patient = Patient(
                    path_config=path_config,
                    phonetic_id=patient_id,
                    debug=False
                )
                
                # Construct prediction directory
                prediction_dir = output_dir / patient_id / "predictions"
                
                if not prediction_dir.exists():
                    logger.warning(f"Prediction directory not found for {patient_id}: {prediction_dir}")
                    continue
                
                logger.info(f"Prediction directory: {prediction_dir}")
                
                # Write predictions to DICOMs
                zip_path = patient.write_predictions_to_dicoms(
                    prediction_dir=prediction_dir,
                    output_dir=None,  # Will create dicom_predictions at same level
                    timepoint=None,   # Process all timepoints
                    overwrite=cfg.nifti_to_dicom.overwrite
                )
                
                if zip_path:
                    logger.info(f"Successfully created DICOM archive: {zip_path}")
                logger.info(f"Successfully wrote DICOMs for patient {patient_id}")
                
            except Exception as e:
                logger.error(f"Error writing DICOMs for patient {patient_id}: {str(e)}", exc_info=True)
                continue
        
        logger.info("Completed DICOM writing for all test patients")
    
    else:
        patient_id = cfg.nifti_to_dicom.patient_id
        logger.info(f"Writing DICOMs for patient: {patient_id}")
        
        # Create patient object
        patient = Patient(
            path_config=path_config,
            phonetic_id=patient_id,
            debug=False
        )
        
        # Construct prediction directory
        prediction_dir = output_dir / patient_id / "predictions"
        
        if not prediction_dir.exists():
            logger.error(f"Prediction directory not found: {prediction_dir}")
            raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
        
        logger.info(f"Prediction directory: {prediction_dir}")
        
        # Write predictions to DICOMs
        try:
            zip_path = patient.write_predictions_to_dicoms(
                prediction_dir=prediction_dir,
                output_dir=None,  # Will create dicom_predictions at same level
                timepoint=None,   # Process all timepoints
                overwrite=cfg.nifti_to_dicom.overwrite
            )
            
            if zip_path:
                logger.info(f"Successfully created DICOM archive: {zip_path}")
            logger.info(f"Successfully wrote DICOMs for patient {patient_id}")
            
        except Exception as e:
            logger.error(f"Error writing DICOMs for patient {patient_id}: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    main()