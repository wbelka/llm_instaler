"""Handler for audio models (Whisper, TTS, music generation, etc.).

This handler manages models for audio processing, including
speech-to-text, text-to-speech, and audio generation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from handlers.base import BaseHandler

logger = logging.getLogger(__name__)


class AudioHandler(BaseHandler):
    """Handler for audio processing models."""
    
    def get_dependencies(self) -> List[str]:
        """Get Python dependencies for audio models."""
        base_deps = [
            'torch>=2.0.0',
            'torchaudio>=2.0.0',
            'numpy',
            'scipy',
            'librosa>=0.10.0',
            'soundfile>=0.12.0'
        ]
        
        # Model-specific dependencies
        model_id = self.model_id.lower()
        
        if 'whisper' in model_id:
            base_deps.extend([
                'openai-whisper',
                'tiktoken',
                'numba'
            ])
        elif 'bark' in model_id:
            base_deps.append('bark')
        elif 'musicgen' in model_id:
            base_deps.extend([
                'audiocraft',
                'xformers'
            ])
        elif 'seamless' in model_id:
            base_deps.append('fairseq2')
        
        # Add transformers for many audio models
        if 'transformers' not in base_deps:
            base_deps.append('transformers>=4.30.0')
        
        return base_deps
    
    def get_system_dependencies(self) -> List[str]:
        """Get system dependencies for audio processing."""
        deps = []
        
        # Audio processing libraries
        deps.extend([
            'ffmpeg',
            'libsndfile1',
            'sox'
        ])
        
        # CUDA for GPU acceleration
        if self.model_info.get('requires_gpu', True):
            deps.append('cuda>=11.7')
        
        return deps
    
    def analyze(self) -> 'ModelRequirements':
        """Analyze audio model requirements."""
        from core.checker import ModelRequirements
        
        requirements = ModelRequirements()
        requirements.model_type = self.model_type
        requirements.model_family = self._determine_audio_family()
        requirements.primary_library = self._determine_primary_library()
        requirements.base_dependencies = self.get_dependencies()
        
        # Memory requirements
        model_size_gb = self._estimate_model_size()
        requirements.disk_space_gb = model_size_gb * 2
        requirements.memory_requirements = {
            "min": max(8, model_size_gb * 2),
            "recommended": max(16, model_size_gb * 3),
            "gpu_min": max(8, model_size_gb * 1.5),
            "gpu_recommended": max(12, model_size_gb * 2)
        }
        
        # Capabilities
        requirements.capabilities = {
            "supports_streaming": True,
            "supports_batch_inference": True,
            "supports_cpu_inference": True,
            "supports_quantization": False,
            "input_formats": ["wav", "mp3", "flac", "m4a"],
            "output_formats": ["wav", "mp3"],
            "sample_rates": [16000, 22050, 44100, 48000]
        }
        
        return requirements
    
    def _determine_audio_family(self) -> str:
        """Determine the audio model family."""
        model_id = self.model_id.lower()
        
        if 'whisper' in model_id or 'wav2vec' in model_id:
            return 'speech-recognition'
        elif 'tts' in model_id or 'bark' in model_id or 'vall-e' in model_id:
            return 'text-to-speech'
        elif 'musicgen' in model_id or 'audiogen' in model_id:
            return 'music-generation'
        elif 'seamless' in model_id:
            return 'speech-translation'
        else:
            return 'audio-processing'
    
    def _determine_primary_library(self) -> str:
        """Determine the primary library for the model."""
        model_id = self.model_id.lower()
        
        if 'whisper' in model_id:
            return 'whisper'
        elif 'bark' in model_id:
            return 'bark'
        elif 'musicgen' in model_id:
            return 'audiocraft'
        else:
            return 'transformers'
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in GB."""
        if 'model_size_gb' in self.model_info:
            return self.model_info['model_size_gb']
        
        # Estimate based on model name
        model_id = self.model_id.lower()
        
        if 'whisper' in model_id:
            if 'large' in model_id:
                return 3.0
            elif 'medium' in model_id:
                return 1.5
            elif 'small' in model_id:
                return 0.5
            else:
                return 0.1  # tiny/base
        elif 'musicgen' in model_id:
            if 'large' in model_id:
                return 10.0
            elif 'medium' in model_id:
                return 5.0
            else:
                return 2.0
        else:
            return 1.0  # Default
    
    def load_model(self, model_path: str, **kwargs):
        """Load audio model with optimal settings."""
        model_family = self._determine_audio_family()
        primary_lib = self._determine_primary_library()
        
        if primary_lib == 'whisper':
            return self._load_whisper_model(model_path, **kwargs)
        elif primary_lib == 'bark':
            return self._load_bark_model(model_path, **kwargs)
        elif primary_lib == 'audiocraft':
            return self._load_audiocraft_model(model_path, **kwargs)
        else:
            return self._load_transformers_audio(model_path, **kwargs)
    
    def _load_whisper_model(self, model_path: str, **kwargs):
        """Load Whisper model."""
        import whisper
        
        # Whisper has specific model loading
        model_size = model_path.split('/')[-1].split('-')[-1]
        model = whisper.load_model(model_size, device=kwargs.get('device', 'auto'))
        
        return model, None
    
    def _load_bark_model(self, model_path: str, **kwargs):
        """Load Bark TTS model."""
        from bark import MODELS, preload_models
        
        # Bark loads models differently
        preload_models(
            text_use_gpu=kwargs.get('device', 'auto') != 'cpu',
            coarse_use_gpu=kwargs.get('device', 'auto') != 'cpu',
            fine_use_gpu=kwargs.get('device', 'auto') != 'cpu',
            codec_use_gpu=kwargs.get('device', 'auto') != 'cpu'
        )
        
        return MODELS, None
    
    def _load_audiocraft_model(self, model_path: str, **kwargs):
        """Load AudioCraft model (MusicGen, AudioGen)."""
        from audiocraft.models import MusicGen
        
        model = MusicGen.get_pretrained(model_path)
        return model, None
    
    def _load_transformers_audio(self, model_path: str, **kwargs):
        """Load audio model using transformers."""
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        device = kwargs.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        dtype = kwargs.get('dtype', 'auto')
        if dtype == 'auto':
            torch_dtype = torch.float16 if device == 'cuda' else torch.float32
        else:
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(dtype, torch.float32)
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        return model, processor
    
    def get_inference_params(self) -> Dict[str, Any]:
        """Get default inference parameters."""
        model_family = self._determine_audio_family()
        
        if model_family == 'speech-recognition':
            return {
                'language': None,  # Auto-detect
                'task': 'transcribe',
                'temperature': 0.0,
                'no_speech_threshold': 0.6,
                'condition_on_previous_text': True
            }
        elif model_family == 'text-to-speech':
            return {
                'voice_preset': 'v2/en_speaker_0',
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.95
            }
        elif model_family == 'music-generation':
            return {
                'duration': 10.0,
                'temperature': 1.0,
                'top_k': 250,
                'top_p': 0.0,
                'cfg_scale': 3.0
            }
        else:
            return {}
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get default training parameters."""
        return {
            'learning_rate': 1e-5,
            'batch_size': 8,
            'num_epochs': 10,
            'warmup_steps': 500,
            'gradient_accumulation_steps': 2
        }
    
    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate audio model files."""
        model_path = Path(model_path)
        
        # Check for basic config
        if not (model_path / 'config.json').exists():
            # Some audio models don't have config.json
            # Check for model-specific files
            model_files = list(model_path.glob('*.pt')) + \
                         list(model_path.glob('*.pth')) + \
                         list(model_path.glob('*.bin')) + \
                         list(model_path.glob('*.safetensors'))
            
            if not model_files:
                return False, "No model files found"
        
        return True, None
    
    def generate_audio(self, text: str = None, prompt: str = None,
                      model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Generate audio (TTS or music generation)."""
        import base64
        import numpy as np
        import soundfile as sf
        from io import BytesIO
        
        model_family = self._determine_audio_family()
        
        if model_family == 'text-to-speech':
            if not text:
                raise ValueError("Text required for TTS")
            
            # Generate speech
            if hasattr(model, 'generate_audio'):
                audio_array = model.generate_audio(text, **kwargs)
            else:
                # Use processor if available
                inputs = processor(text, return_tensors="pt")
                with torch.no_grad():
                    audio_array = model.generate(**inputs, **kwargs)
            
            sample_rate = kwargs.get('sample_rate', 22050)
            
        elif model_family == 'music-generation':
            if not prompt:
                raise ValueError("Prompt required for music generation")
            
            # Generate music
            if hasattr(model, 'generate'):
                with torch.no_grad():
                    audio_array = model.generate([prompt], **kwargs)[0, 0].cpu().numpy()
            else:
                raise NotImplementedError("Music generation not implemented for this model")
            
            sample_rate = kwargs.get('sample_rate', 32000)
        
        else:
            raise ValueError(f"Audio generation not supported for {model_family}")
        
        # Convert to audio file
        buffer = BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        buffer.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(buffer.read()).decode()
        
        return {
            'audio': audio_base64,
            'metadata': {
                'sample_rate': sample_rate,
                'duration': len(audio_array) / sample_rate,
                'format': 'wav'
            }
        }
    
    def process_audio(self, audio: str, task: str = "transcribe",
                     model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process audio for various tasks."""
        import base64
        import numpy as np
        import soundfile as sf
        from io import BytesIO
        
        # Decode audio from base64
        audio_data = base64.b64decode(audio)
        audio_array, sample_rate = sf.read(BytesIO(audio_data))
        
        if task == "transcribe":
            # Speech-to-text
            if hasattr(model, 'transcribe'):
                # Whisper-style model
                result = model.transcribe(audio_array, **kwargs)
                return {'text': result['text'], 'language': result.get('language')}
            else:
                # Transformers-style model
                inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
                with torch.no_grad():
                    predicted_ids = model.generate(**inputs)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                return {'text': transcription}
        
        elif task == "translate":
            # Speech translation
            kwargs['task'] = 'translate'
            if hasattr(model, 'transcribe'):
                result = model.transcribe(audio_array, **kwargs)
                return {'text': result['text'], 'language': result.get('language')}
            else:
                raise NotImplementedError("Translation not implemented for this model")
        
        else:
            raise ValueError(f"Unknown audio processing task: {task}")
    
    def get_supported_modes(self) -> List[str]:
        """Get supported modes."""
        model_family = self._determine_audio_family()
        
        if model_family == 'speech-recognition':
            return ['transcribe', 'translate', 'auto']
        elif model_family == 'text-to-speech':
            return ['tts', 'auto']
        elif model_family == 'music-generation':
            return ['music', 'auto']
        else:
            return ['auto']
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get mode descriptions."""
        return {
            'auto': 'Automatic mode selection',
            'transcribe': 'Speech to text transcription',
            'translate': 'Speech translation',
            'tts': 'Text to speech synthesis',
            'music': 'Music generation from text'
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities."""
        capabilities = super().get_model_capabilities()
        model_family = self._determine_audio_family()
        
        if model_family == 'speech-recognition':
            capabilities.update({
                'input_modalities': ['audio'],
                'output_modalities': ['text'],
                'supports_streaming': True,
                'languages': ['multi']  # Most modern models support multiple languages
            })
        elif model_family == 'text-to-speech':
            capabilities.update({
                'input_modalities': ['text'],
                'output_modalities': ['audio'],
                'supports_streaming': True
            })
        elif model_family == 'music-generation':
            capabilities.update({
                'input_modalities': ['text'],
                'output_modalities': ['audio'],
                'supports_streaming': False
            })
        
        return capabilities