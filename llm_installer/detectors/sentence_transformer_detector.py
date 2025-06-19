"""
Detector for sentence transformer models (embedding models)
"""

from typing import Dict, List, Optional, Any
from .base import BaseDetector, ModelProfile


class SentenceTransformerDetector(BaseDetector):
    """Detector for sentence-transformers embedding models"""

    def detect(self, model_id: str, config: Dict[str, Any],
               files: List[str]) -> Optional[ModelProfile]:
        """
        Detect sentence-transformers models used for embeddings
        """
        # Check for sentence-transformers specific files
        st_indicators = [
            'sentence_bert_config.json',
            'modules.json',
            '1_Pooling/config.json',
            '2_Normalize/config.json'
        ]

        if any(indicator in files for indicator in st_indicators):
            return self._create_sentence_transformer_profile(model_id, config, files)

        # Check config for sentence-transformers indicators
        if self._is_sentence_transformer_config(config):
            return self._create_sentence_transformer_profile(model_id, config, files)

        # Check model card or name patterns
        if self._check_model_name_pattern(model_id):
            return self._create_sentence_transformer_profile(model_id, config, files)

        return None

    def _is_sentence_transformer_config(self, config: Dict[str, Any]) -> bool:
        """Check if config indicates a sentence transformer model"""
        # Check for typical sentence transformer architectures
        sentence_transformer_archs = [
            'BertModel',
            'RobertaModel',
            'DistilBertModel',
            'MPNetModel',
            'XLMRobertaModel',
            'AlbertModel',
            'ElectraModel',
            'DebertaModel',
            'DebertaV2Model'
        ]

        architectures = config.get('architectures', [])
        if any(arch in architectures for arch in sentence_transformer_archs):
            # These architectures are used for embeddings when not for classification
            task_specific_heads = [
                'ForSequenceClassification',
                'ForTokenClassification',
                'ForQuestionAnswering',
                'ForMaskedLM',
                'ForCausalLM'
            ]

            # If it has a task-specific head, it's not a sentence transformer
            if not any(head in str(architectures) for head in task_specific_heads):
                return True

        # Check for embedding-specific config
        if config.get('task', '') == 'feature-extraction':
            return True

        return False

    def _check_model_name_pattern(self, model_id: str) -> bool:
        """Check if model name suggests it's a sentence transformer"""
        embedding_patterns = [
            'sentence-transformers/',
            '-embeddings',
            'embedding',
            'e5-',
            'bge-',
            'gte-',
            'instructor-',
            'all-MiniLM',
            'all-mpnet',
            'paraphrase-',
            'msmarco-',
            'sbert',
            'labse',
            'use-',  # Universal Sentence Encoder
            'simcse',
            'contriever'
        ]

        model_lower = model_id.lower()
        return any(pattern in model_lower for pattern in embedding_patterns)

    def _create_sentence_transformer_profile(self, model_id: str, config: Dict[str, Any],
                                             files: List[str]) -> ModelProfile:
        """Create profile for sentence transformer model"""
        profile = ModelProfile(
            model_type="sentence-transformer",
            model_id=model_id,
            library="sentence-transformers",
            task="feature-extraction",
            architecture=self._get_embedding_architecture(config)
        )

        # These models are typically small
        profile.estimated_size_gb = 0.5
        profile.estimated_memory_gb = 2.0

        # Check for specific embedding model types
        model_lower = model_id.lower()

        if 'e5' in model_lower:
            profile.metadata = {'family': 'E5'}
            if 'large' in model_lower:
                profile.estimated_size_gb = 1.5
                profile.estimated_memory_gb = 4.0
        elif 'bge' in model_lower:
            profile.metadata = {'family': 'BGE'}
        elif 'gte' in model_lower:
            profile.metadata = {'family': 'GTE'}
        elif 'instructor' in model_lower:
            profile.metadata = {'family': 'Instructor', 'requires_instruction': True}
        elif 'minilm' in model_lower:
            profile.metadata = {'family': 'MiniLM'}
            profile.estimated_size_gb = 0.1
            profile.estimated_memory_gb = 1.0

        # Set requirements
        profile.special_requirements = [
            'torch',
            'transformers',
            'sentence-transformers',
            'numpy',
            'scikit-learn',  # Often used with embeddings
            'faiss-cpu'  # For similarity search
        ]

        # Check for ONNX optimization
        if any('onnx' in f for f in files):
            profile.special_requirements.append('onnxruntime')
            profile.metadata = profile.metadata or {}
            profile.metadata['onnx_optimized'] = True

        # Embedding models typically don't support these
        profile.supports_vllm = False
        profile.supports_tensorrt = False  # Could support it but not common

        return profile

    def _get_embedding_architecture(self, config: Dict[str, Any]) -> Optional[str]:
        """Get the embedding model architecture"""
        architectures = config.get('architectures', [])
        if architectures and isinstance(architectures, list):
            return architectures[0]

        # Try model_type
        model_type = config.get('model_type', '')
        if model_type:
            return model_type

        return "SentenceTransformer"
