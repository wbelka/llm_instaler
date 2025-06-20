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

            # Step 1: Inventory files - get complete file listing ("map of territory")
            file_sizes = fetch_model_files_with_sizes(model_id)
            info.files = list(file_sizes.keys())
            info.file_sizes = file_sizes
            info.size_gb = round(sum(file_sizes.values()), 1)

            # Step 2: Determine model type by indicator files (order matters!)
            
            # Scenario 1: Composite model (e.g., Diffusers) - check model_index.json first
            if 'model_index.json' in info.files:
                model_index = fetch_model_config(model_id, "model_index.json")
                if model_index:
                    info.model_index = model_index
                    logger.debug(f"Found model_index.json - composite model detected")
            
            # Scenario 2: Standard Transformers model - check config.json
            config = fetch_model_config(model_id, "config.json")
            if config:
                info.config = config
                logger.debug(f"Found config.json - standard model configuration")
            
            # Scenario 3: Model with non-standard config name
            elif not info.model_index:  # Only if no model_index.json found
                # Try alternative config names
                alternative_configs = ['llm_config.json', 'model_config.json']
                for alt_config in alternative_configs:
                    if alt_config in info.files:
                        config = fetch_model_config(model_id, alt_config)
                        if config and self._is_architecture_config(config):
                            info.config = config
                            logger.debug(f"Found {alt_config} with architecture info")
                            break
            
            # Scenario 4: Unrecognized model
            if not info.config and not info.model_index:
                logger.warning(f"No standard configuration files found for {model_id}")

            return info

        except Exception as e:
            logger.error(f"Failed to fetch API info for {model_id}: {e}")
            return None
    
    def _is_architecture_config(self, config: dict) -> bool:
        """Check if config describes model architecture (not API config)"""
        # Architecture configs have these keys
        architecture_keys = [
            'model_type', 'hidden_size', 'vocab_size', 'num_layers',
            'num_attention_heads', 'architectures', 'torch_dtype',
            'num_hidden_layers', 'intermediate_size', 'max_position_embeddings'
        ]
        
        # API configs have these keys
        api_keys = ['api_key', 'base_url', 'endpoint', 'api_version']
        
        # If any API key found, it's not an architecture config
        if any(key in config for key in api_keys):
            return False
        
        # If at least 2 architecture keys found, it's likely an architecture config
        found_keys = sum(1 for key in architecture_keys if key in config)
        return found_keys >= 2
