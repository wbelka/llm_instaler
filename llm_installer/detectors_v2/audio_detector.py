"""
Detector for audio models (pyannote, speechbrain, etc.)
"""

from .base import BaseDetector, ModelInfo


class AudioDetector(BaseDetector):
    """Detector for audio processing models"""

    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this is an audio model"""
        # Audio libraries
        audio_libs = ['pyannote-audio', 'speechbrain', 'asteroid', 'nemo_toolkit']
        if info.library_name in audio_libs:
            return True

        # Audio tags
        audio_tags = ['audio', 'speech', 'asr', 'whisper', 'wav2vec2',
                     'speaker-diarization', 'audio-classification']
        if any(tag in info.tags for tag in audio_tags):
            return True

        # Audio pipeline tags
        audio_pipelines = ['automatic-speech-recognition', 'audio-classification',
                          'audio-to-audio', 'voice-activity-detection',
                          'text-to-speech', 'text-to-audio']
        if info.pipeline_tag in audio_pipelines:
            return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect audio-specific information"""
        info.model_type = 'audio'

        # Task detection
        if info.pipeline_tag:
            info.task = info.pipeline_tag
        else:
            # Infer from tags/library
            if 'asr' in info.tags or 'speech-recognition' in info.tags:
                info.task = 'automatic-speech-recognition'
            elif 'speaker-diarization' in info.tags:
                info.task = 'speaker-diarization'
            elif 'audio-classification' in info.tags:
                info.task = 'audio-classification'
            else:
                info.task = 'audio-processing'

        # Architecture detection
        info.architecture = self._detect_audio_architecture(info)

        # Library-specific handling
        if info.library_name == 'pyannote-audio':
            info.metadata['framework'] = 'pyannote'
            info.special_requirements = [
                'pyannote.audio',
                'torch',
                'torchaudio',
                'soundfile',
                'speechbrain'
            ]
        elif info.library_name == 'speechbrain':
            info.metadata['framework'] = 'speechbrain'
            info.special_requirements = [
                'speechbrain',
                'torch',
                'torchaudio',
                'transformers'
            ]
        else:
            # Default audio requirements
            info.special_requirements = [
                'transformers',
                'torch',
                'torchaudio',
                'librosa',
                'soundfile'
            ]

        # Whisper models
        if 'whisper' in info.model_id.lower() or 'whisper' in info.tags:
            info.metadata['model_family'] = 'Whisper'
            info.special_requirements.append('openai-whisper')

        # Wav2Vec2 models
        elif 'wav2vec2' in info.model_id.lower() or 'wav2vec2' in info.tags:
            info.metadata['model_family'] = 'Wav2Vec2'

        # Check for language support
        languages = []
        for tag in info.tags:
            # Common language tags
            if tag in ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']:
                languages.append(tag)
        if languages:
            info.metadata['languages'] = languages

        # Sampling rate
        if info.config:
            if 'sampling_rate' in info.config:
                info.metadata['sampling_rate'] = info.config['sampling_rate']
            elif 'sample_rate' in info.config:
                info.metadata['sampling_rate'] = info.config['sample_rate']

        # Quantization support (limited for audio)
        info.supports_quantization = ['fp32', 'fp16']

        # Default dtype
        if info.config and 'torch_dtype' in info.config:
            info.default_dtype = info.config['torch_dtype']
        else:
            info.default_dtype = 'float32'
        
        # Audio models don't typically support vLLM
        info.metadata['supports_vllm'] = False
        
        # Limited TensorRT support for audio
        info.metadata['supports_tensorrt'] = False

        return info

    def _detect_audio_architecture(self, info: ModelInfo) -> str:
        """Detect audio model architecture"""
        # From config
        if info.config and 'architectures' in info.config:
            return info.config['architectures'][0]

        # From model name/tags
        model_lower = info.model_id.lower()

        if 'whisper' in model_lower:
            # Whisper sizes
            if 'tiny' in model_lower:
                return 'Whisper-tiny'
            elif 'base' in model_lower:
                return 'Whisper-base'
            elif 'small' in model_lower:
                return 'Whisper-small'
            elif 'medium' in model_lower:
                return 'Whisper-medium'
            elif 'large' in model_lower:
                return 'Whisper-large'
            else:
                return 'Whisper'

        elif 'wav2vec2' in model_lower:
            return 'Wav2Vec2'

        elif 'hubert' in model_lower:
            return 'HuBERT'

        elif 'wavlm' in model_lower:
            return 'WavLM'

        elif 'speecht5' in model_lower:
            return 'SpeechT5'

        # Library specific
        if info.library_name == 'pyannote-audio':
            if 'segmentation' in model_lower:
                return 'Segmentation'
            elif 'embedding' in model_lower:
                return 'Embedding'
            elif 'diarization' in model_lower:
                return 'SpeakerDiarization'

        return 'Audio-Model'
