"""Detector for TrOCR (Transformer-based Optical Character Recognition) models.

This module identifies TrOCR models based on their configuration files.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

from detectors.base import BaseDetector

logger = logging.getLogger(__name__)


class TrOCRDetector(BaseDetector):
    """Detects TrOCR models based on their configuration."""

    def __init__(self):
        super().__init__()
        self.model_type = "ocr"
        self.model_family = "vision-language"

    def detect(self, model_path: Path, config: Optional[Dict[str, Any]] = None) -> bool:
        """Detects if the given model is a TrOCR model.

        Args:
            model_path: The path to the model directory.
            config: The loaded model configuration (config.json) if already available.

        Returns:
            True if the model is detected as TrOCR, False otherwise.
        """
        if config is None:
            config_path = model_path / "config.json"
            if not config_path.exists():
                return False
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                return False

        # TrOCR models typically have a `processor_class` and `architectures`
        # that indicate VisionEncoderDecoderModel
        architectures = config.get("architectures", [])
        processor_class = config.get("processor_class")

        is_trocr = (
            "VisionEncoderDecoderModel" in architectures and
            processor_class == "TrOCRProcessor"
        )

        if is_trocr:
            logger.info(f"Detected TrOCR model at {model_path}")
            self.model_id = model_path.name # Or get from config if available
            self.model_info = {
                "model_id": self.model_id,
                "model_type": self.model_type,
                "model_family": self.model_family,
                "architectures": architectures,
                "processor_class": processor_class
            }
            return True
        
        return False
