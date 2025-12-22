#!/usr/bin/env python3
"""
CLI script to write NIfTI predictions back to DICOM format.

Uses hydra config to determine prediction directory and output location.
"""

from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

from ..utils.path_config import load_path_config
from ..data_management.patients import Patient

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.1",
    config_path=str((Path(__file__).resolve().parents[3] / "hydra_configs").as_posix()),
    config_name="config"
)
def main(cfg: DictConfig):
    """Write predictions to DICOM format for a patient.
    
    Args:
        cfg: Hydra config containing inference settings
    """
    # Check for required parameters
    if not cfg.inference.get('patient_id'):
        logger.error("patient_id is required but not provided")
        raise ValueError("patient_id is required")
    
    patient_id = cfg.inference.patient_id
    logger.info(f"Writing DICOMs for patient: {patient_id}")
    
    # Load path configuration
    path_config = load_path_config(cfg.path_config.path_config_name)
    
    # Create patient object
    patient = Patient(
        path_config=path_config,
        phonetic_id=patient_id,
        debug=False
    )
    
    # Construct prediction directory from hydra config
    # Format: output_dir / patient_id / "predictions"
    output_dir = Path(cfg.inference.output_dir)
    prediction_dir = output_dir / patient_id / "predictions"
    
    if not prediction_dir.exists():
        logger.error(f"Prediction directory not found: {prediction_dir}")
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    
    logger.info(f"Prediction directory: {prediction_dir}")
    
    # Write predictions to DICOMs
    # output_dir=None will create dicom_predictions at same level as prediction_dir
    try:
        patient.write_predictions_to_dicoms(
            prediction_dir=prediction_dir,
            output_dir=None,  # Will create dicom_predictions at same level
            timepoint=None,   # Process all timepoints
            overwrite=cfg.inference.get('overwrite', False)
        )
        logger.info(f"Successfully wrote DICOMs for patient {patient_id}")
    except Exception as e:
        logger.error(f"Error writing DICOMs for patient {patient_id}: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
