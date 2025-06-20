"""
Model detector v2 - API-first with specialized parsers
"""

import logging
from typing import Optional
from huggingface_hub import model_info as hf_model_info

from .detectors_v2 import (
    ModelInfo,
    TransformersDetector,
    DiffusersDetector,
    SentenceTransformersDetector,
    GGUFDetector,
    TimmDetector,
    AudioDetector,
    BagelDetector,
    FluxDetector,
    CosmosDetector,
    ControlNetDetector
)
from .utils import get_hf_token, fetch_model_files_with_sizes, fetch_model_config


logger = logging.getLogger(__name__)


class ModelDetectorV2:
    """Main detector that orchestrates specialized detectors"""

    def __init__(self):
        self.detectors = [
            GGUFDetector(),  # Check GGUF first (most specific)
            BagelDetector(),  # BAGEL models
            ControlNetDetector(),  # ControlNet models (before FLUX/diffusers)
            FluxDetector(),  # FLUX models (before general diffusers)
            CosmosDetector(),  # NVIDIA Cosmos models
            DiffusersDetector(),
            SentenceTransformersDetector(),
            TimmDetector(),
            AudioDetector(),
            TransformersDetector(),  # Default fallback
        ]

    def detect(self, model_id: str) -> Optional[ModelInfo]:
        """
        Detect model using API-first approach

        1. Get all info from HuggingFace API
        2. Find appropriate detector based on library_name/tags
        3. Let detector parse type-specific configs
        """
        try:
            # Step 1: Get API info
            info = self._fetch_api_info(model_id)
            if not info:
                return None

            # Step 2: Find appropriate detector
            for detector in self.detectors:
                if detector.can_handle(info):
                    logger.info(f"Using {detector.name} for {model_id}")
                    # Step 3: Let detector enrich the info
                    return detector.detect(info)

            # No specific detector found - return basic info
            logger.warning(f"No specific detector for {model_id}, returning basic info")
            return info

        except Exception as e:
            logger.error(f"Failed to detect {model_id}: {e}")
            return None

    def _fetch_api_info(self, model_id: str) -> Optional[ModelInfo]:
        """Fetch all available info from API"""
        try:
            token = get_hf_token()

            # Get model metadata from API
            api_info = hf_model_info(model_id, token=token)

            # Create ModelInfo with API data
            info = ModelInfo(
                model_id=model_id,
                library_name=getattr(api_info, 'library_name', None),
                pipeline_tag=getattr(api_info, 'pipeline_tag', None),
                tags=getattr(api_info, 'tags', []),
                downloads=getattr(api_info, 'downloads', 0),
                likes=getattr(api_info, 'likes', 0),
                created_at=getattr(api_info, 'created_at', None),
                sha=getattr(api_info, 'sha', None),
                private=getattr(api_info, 'private', False),
                gated=getattr(api_info, 'gated', None)
            )

            # Get file info
            file_sizes = fetch_model_files_with_sizes(model_id)
            info.files = list(file_sizes.keys())
            info.file_sizes = file_sizes
            info.size_gb = round(sum(file_sizes.values()), 1)

            # Try to get config.json
            config = fetch_model_config(model_id, "config.json")
            if config:
                info.config = config
            else:
                # Try llm_config.json as fallback (some models like BAGEL use this)
                config = fetch_model_config(model_id, "llm_config.json")
                if config:
                    info.config = config

            # For diffusers, also get model_index.json
            if info.library_name == 'diffusers' or 'model_index.json' in info.files:
                model_index = fetch_model_config(model_id, "model_index.json")
                if model_index:
                    info.model_index = model_index

            return info

        except Exception as e:
            logger.error(f"Failed to fetch API info for {model_id}: {e}")
            return None
