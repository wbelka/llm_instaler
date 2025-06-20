"""
Detector for sentence-transformers models
"""

from .base import BaseDetector, ModelInfo


class SentenceTransformersDetector(BaseDetector):
    """Detector for sentence-transformers embedding models"""

    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this is a sentence-transformers model"""
        # Explicit library
        if info.library_name == 'sentence-transformers':
            return True

        # Has sentence-transformers tag
        if 'sentence-transformers' in info.tags:
            return True

        # Has sentence-transformers specific files
        st_files = ['sentence_bert_config.json', 'modules.json']
        if any(f in info.files for f in st_files):
            return True

        # Pipeline tag for embeddings
        if info.pipeline_tag in ['sentence-similarity', 'feature-extraction']:
            # Additional check for ST patterns
            if any(tag in info.tags for tag in ['embeddings', 'sentence-similarity']):
                return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect sentence-transformers specific information"""
        info.model_type = 'sentence-transformer'
        info.task = info.pipeline_tag or 'feature-extraction'

        # Try to get sentence_bert_config
        if 'sentence_bert_config.json' in info.files:
            from ..utils import fetch_model_config
            st_config = fetch_model_config(info.model_id, 'sentence_bert_config.json')
            if st_config:
                info.metadata['sentence_transformer_config'] = st_config
                # Max sequence length
                if 'max_seq_length' in st_config:
                    info.metadata['max_seq_length'] = st_config['max_seq_length']

        # Architecture from config or tags
        if info.config and 'model_type' in info.config:
            info.architecture = info.config['model_type']
        else:
            # Try to infer from model name/tags
            arch_patterns = {
                'mpnet': 'MPNet',
                'minilm': 'MiniLM',
                'e5': 'E5',
                'bge': 'BGE',
                'gte': 'GTE',
                'instructor': 'Instructor',
                'roberta': 'RoBERTa',
                'bert': 'BERT',
                'distilbert': 'DistilBERT'
            }

            model_lower = info.model_id.lower()
            for pattern, arch in arch_patterns.items():
                if pattern in model_lower:
                    info.architecture = arch
                    break

        # Check for multilingual support
        if any(tag in info.tags for tag in ['multilingual', 'multi-language', '100-languages']):
            info.metadata['multilingual'] = True

        # Requirements
        info.special_requirements = [
            'sentence-transformers',
            'transformers',
            'torch',
            'numpy',
            'scikit-learn'
        ]

        # Add FAISS for similarity search
        if info.size_gb > 0.5:  # Larger models benefit from FAISS
            info.special_requirements.append('faiss-cpu')

        # ONNX support
        if any('onnx' in f for f in info.files):
            info.special_requirements.append('onnxruntime')
            info.metadata['onnx_optimized'] = True

        # Quantization support (limited for embeddings)
        info.supports_quantization = ['fp32', 'fp16']
        if info.metadata.get('onnx_optimized'):
            info.supports_quantization.append('onnx')

        # Default dtype
        if info.config and 'torch_dtype' in info.config:
            info.default_dtype = info.config['torch_dtype']
        else:
            info.default_dtype = 'float32'  # Embeddings often use fp32 for precision
        
        # vLLM doesn't support embedding models
        info.metadata['supports_vllm'] = False
        
        # Limited TensorRT support for embeddings
        info.metadata['supports_tensorrt'] = False

        return info
