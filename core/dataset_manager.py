"""Dataset management for training.

Supports multiple formats and automatic preprocessing for different model types.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset loading and preprocessing for training."""
    
    # Supported formats
    SUPPORTED_FORMATS = {
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.csv': 'csv',
        '.txt': 'text',
        '.parquet': 'parquet',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.xml': 'xml',
        '.tsv': 'tsv'
    }
    
    # Format detection patterns
    FORMAT_PATTERNS = {
        'alpaca': ['instruction', 'input', 'output'],
        'sharegpt': ['conversations', 'from', 'value'],
        'openai': ['messages', 'role', 'content'],
        'completion': ['prompt', 'completion'],
        'qa': ['question', 'answer'],
        'text': ['text'],
        'chat': ['user', 'assistant'],
        'multi_turn': ['turns'],
        'vision': ['image', 'caption'],
        'vision_qa': ['image', 'question', 'answer'],
        # New formats
        'dolly': ['instruction', 'context', 'response'],
        'oasst': ['message_id', 'parent_id', 'text', 'role'],
        'vicuna': ['id', 'conversations'],
        'wizardlm': ['instruction', 'output'],
        'orca': ['system_prompt', 'question', 'answer'],
        'guanaco': ['text'],  # But with special formatting
        'lima': ['conversations'],  # High quality format
        'anthropic': ['human', 'assistant'],
        'claude': ['Human', 'Assistant'],
        'mistral': ['text'],  # With special tokens
        'chatml': ['messages'],  # ChatML format
        'llama_chat': ['system', 'user', 'assistant'],
        'function_calling': ['functions', 'function_call'],
        'code_alpaca': ['instruction', 'input', 'output', 'lang']
    }
    
    def __init__(self, model_type: str = "", model_family: str = ""):
        """Initialize dataset manager.
        
        Args:
            model_type: Type of model (for format detection)
            model_family: Family of model (for preprocessing)
        """
        self.model_type = model_type
        self.model_family = model_family
        self.dataset_cache = {}
    
    def load_dataset(
        self,
        data_path: Union[str, Path, List[Union[str, Path]]],
        format: str = "auto",
        validation_split: float = 0.1,
        max_examples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """Load dataset from file(s) or directory.
        
        Args:
            data_path: Path to data file(s) or directory
            format: Dataset format (auto-detect if 'auto')
            validation_split: Fraction for validation
            max_examples: Maximum examples to load
            shuffle: Whether to shuffle data
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data)
        """
        # Handle multiple paths
        if isinstance(data_path, list):
            all_data = []
            for path in data_path:
                data = self._load_single_source(path, format)
                all_data.extend(data)
        else:
            all_data = self._load_single_source(data_path, format)
        
        # Limit examples if specified
        if max_examples and len(all_data) > max_examples:
            if shuffle:
                random.seed(seed)
                random.shuffle(all_data)
            all_data = all_data[:max_examples]
        
        # Split into train/validation
        if validation_split > 0:
            if shuffle and not max_examples:  # Don't shuffle twice
                random.seed(seed)
                random.shuffle(all_data)
            
            val_size = int(len(all_data) * validation_split)
            val_data = all_data[:val_size]
            train_data = all_data[val_size:]
        else:
            train_data = all_data
            val_data = []
        
        logger.info(f"Loaded {len(train_data)} training examples, {len(val_data)} validation examples")
        
        return train_data, val_data
    
    def _load_single_source(self, data_path: Union[str, Path], format: str) -> List[Dict]:
        """Load data from a single source."""
        data_path = Path(data_path)
        
        if data_path.is_dir():
            return self._load_from_directory(data_path, format)
        elif data_path.is_file():
            return self._load_from_file(data_path, format)
        else:
            # Try HuggingFace dataset
            return self._load_from_huggingface(str(data_path), format)
    
    def _load_from_directory(self, dir_path: Path, format: str) -> List[Dict]:
        """Load all supported files from directory."""
        all_data = []
        
        # Check for format-specific subdirectories
        format_dirs = {
            'alpaca': dir_path / 'alpaca',
            'chat': dir_path / 'chat',
            'completion': dir_path / 'completion',
            'qa': dir_path / 'qa',
            'vision': dir_path / 'vision'
        }
        
        # Load from format-specific dirs if they exist
        for fmt, fmt_dir in format_dirs.items():
            if fmt_dir.exists():
                logger.info(f"Loading {fmt} format data from {fmt_dir}")
                for file_path in fmt_dir.glob('*'):
                    if file_path.suffix in self.SUPPORTED_FORMATS:
                        data = self._load_from_file(file_path, fmt)
                        all_data.extend(data)
        
        # Load from root directory
        for file_path in dir_path.glob('*'):
            if file_path.is_file() and file_path.suffix in self.SUPPORTED_FORMATS:
                data = self._load_from_file(file_path, format)
                all_data.extend(data)
        
        return all_data
    
    def _load_from_file(self, file_path: Path, format: str) -> List[Dict]:
        """Load data from a single file."""
        logger.info(f"Loading {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Handle datasets with 'data' or 'examples' key
                    for key in ['data', 'examples', 'items', 'samples']:
                        if key in data:
                            data = data[key]
                            break
                    else:
                        # Single example
                        data = [data]
        
        elif suffix == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        
        elif suffix == '.csv':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
        
        elif suffix == '.txt':
            # Plain text, each line is an example
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append({'text': line.strip()})
        
        elif suffix == '.parquet':
            # Use pandas if available
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)
                data = df.to_dict('records')
            except ImportError:
                logger.warning("pandas not available, skipping parquet file")
                return []
        
        elif suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        # Check for common keys
                        for key in ['data', 'examples', 'items', 'samples', 'conversations']:
                            if key in data:
                                data = data[key]
                                break
                        else:
                            data = [data]
                    elif not isinstance(data, list):
                        data = [data]
            except ImportError:
                logger.warning("PyYAML not available, skipping YAML file")
                return []
        
        elif suffix == '.toml':
            try:
                import toml
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = toml.load(f)
                    if isinstance(data, dict):
                        # Check for data section
                        for key in ['data', 'examples', 'dataset']:
                            if key in data:
                                data = data[key]
                                break
                        else:
                            data = [data]
                    elif not isinstance(data, list):
                        data = [data]
            except ImportError:
                logger.warning("toml not available, skipping TOML file")
                return []
        
        elif suffix == '.xml':
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(file_path)
                root = tree.getroot()
                data = []
                # Simple XML parsing - can be extended
                for child in root:
                    item = {elem.tag: elem.text for elem in child}
                    data.append(item)
            except Exception as e:
                logger.warning(f"Failed to parse XML: {e}")
                return []
        
        elif suffix == '.tsv':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                data = list(reader)
        
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return []
        
        # Auto-detect format if needed
        if format == "auto" and data:
            format = self._detect_format(data[0])
            logger.info(f"Auto-detected format: {format}")
        
        # Preprocess based on format
        processed_data = []
        for item in data:
            processed = self._preprocess_item(item, format)
            if processed:
                processed_data.append(processed)
        
        return processed_data
    
    def _load_from_huggingface(self, dataset_name: str, format: str) -> List[Dict]:
        """Load dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            
            logger.info(f"Loading HuggingFace dataset: {dataset_name}")
            
            # Parse dataset name (might include config)
            parts = dataset_name.split(':')
            if len(parts) > 1:
                name, config = parts[0], parts[1]
                dataset = load_dataset(name, config)
            else:
                dataset = load_dataset(dataset_name)
            
            # Convert to list of dicts
            data = []
            for split in ['train', 'training']:
                if split in dataset:
                    data.extend(dataset[split])
                    break
            
            # Auto-detect format
            if format == "auto" and data:
                format = self._detect_format(data[0])
                logger.info(f"Auto-detected format: {format}")
            
            # Preprocess
            processed_data = []
            for item in data:
                processed = self._preprocess_item(dict(item), format)
                if processed:
                    processed_data.append(processed)
            
            return processed_data
            
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace dataset: {e}")
            return []
    
    def _detect_format(self, sample: Dict) -> str:
        """Auto-detect dataset format from sample."""
        # Check each format pattern
        for format_name, required_keys in self.FORMAT_PATTERNS.items():
            if all(key in sample or any(key in str(v) for v in sample.values()) 
                   for key in required_keys):
                return format_name
        
        # Check for nested structures
        for key, value in sample.items():
            if isinstance(value, list) and value:
                if isinstance(value[0], dict):
                    # Might be conversations format
                    if 'role' in value[0] or 'from' in value[0]:
                        return 'sharegpt'
        
        # Default based on model type
        if 'vision' in self.model_type or 'multimodal' in self.model_family:
            return 'vision'
        
        return 'text'
    
    def _preprocess_item(self, item: Dict, format: str) -> Optional[Dict]:
        """Preprocess item based on format."""
        try:
            if format == 'alpaca':
                return self._preprocess_alpaca(item)
            elif format == 'sharegpt':
                return self._preprocess_sharegpt(item)
            elif format == 'openai':
                return self._preprocess_openai(item)
            elif format == 'completion':
                return self._preprocess_completion(item)
            elif format == 'qa':
                return self._preprocess_qa(item)
            elif format == 'chat':
                return self._preprocess_chat(item)
            elif format == 'text':
                return self._preprocess_text(item)
            elif format == 'vision':
                return self._preprocess_vision(item)
            elif format == 'vision_qa':
                return self._preprocess_vision_qa(item)
            # New formats
            elif format == 'dolly':
                return self._preprocess_dolly(item)
            elif format == 'oasst':
                return self._preprocess_oasst(item)
            elif format == 'vicuna':
                return self._preprocess_vicuna(item)
            elif format == 'wizardlm':
                return self._preprocess_wizardlm(item)
            elif format == 'orca':
                return self._preprocess_orca(item)
            elif format == 'anthropic':
                return self._preprocess_anthropic(item)
            elif format == 'claude':
                return self._preprocess_claude(item)
            elif format == 'chatml':
                return self._preprocess_chatml(item)
            elif format == 'llama_chat':
                return self._preprocess_llama_chat(item)
            elif format == 'code_alpaca':
                return self._preprocess_code_alpaca(item)
            else:
                # Try generic preprocessing
                return self._preprocess_generic(item)
        except Exception as e:
            logger.warning(f"Failed to preprocess item: {e}")
            return None
    
    def _preprocess_alpaca(self, item: Dict) -> Dict:
        """Preprocess Alpaca format."""
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction
        
        return {
            'prompt': prompt,
            'completion': output,
            'format': 'alpaca'
        }
    
    def _preprocess_sharegpt(self, item: Dict) -> Dict:
        """Preprocess ShareGPT format."""
        conversations = item.get('conversations', [])
        
        messages = []
        for conv in conversations:
            role = conv.get('from', 'user')
            if role == 'human':
                role = 'user'
            elif role == 'gpt':
                role = 'assistant'
            
            messages.append({
                'role': role,
                'content': conv.get('value', '')
            })
        
        return {
            'messages': messages,
            'format': 'sharegpt'
        }
    
    def _preprocess_openai(self, item: Dict) -> Dict:
        """Preprocess OpenAI format."""
        messages = item.get('messages', [])
        
        # Ensure proper format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'role': msg.get('role', 'user'),
                'content': msg.get('content', '')
            })
        
        return {
            'messages': formatted_messages,
            'format': 'openai'
        }
    
    def _preprocess_completion(self, item: Dict) -> Dict:
        """Preprocess completion format."""
        return {
            'prompt': item.get('prompt', ''),
            'completion': item.get('completion', ''),
            'format': 'completion'
        }
    
    def _preprocess_qa(self, item: Dict) -> Dict:
        """Preprocess Q&A format."""
        question = item.get('question', '')
        answer = item.get('answer', '')
        context = item.get('context', '')
        
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}"
        else:
            prompt = f"Question: {question}"
        
        return {
            'prompt': prompt,
            'completion': f"Answer: {answer}",
            'format': 'qa'
        }
    
    def _preprocess_chat(self, item: Dict) -> Dict:
        """Preprocess chat format."""
        messages = []
        
        # Handle various chat formats
        if 'user' in item and 'assistant' in item:
            messages.append({'role': 'user', 'content': item['user']})
            messages.append({'role': 'assistant', 'content': item['assistant']})
        elif 'human' in item and 'assistant' in item:
            messages.append({'role': 'user', 'content': item['human']})
            messages.append({'role': 'assistant', 'content': item['assistant']})
        elif 'input' in item and 'output' in item:
            messages.append({'role': 'user', 'content': item['input']})
            messages.append({'role': 'assistant', 'content': item['output']})
        
        return {
            'messages': messages,
            'format': 'chat'
        }
    
    def _preprocess_text(self, item: Dict) -> Dict:
        """Preprocess plain text format."""
        text = item.get('text', '')
        
        # For plain text, we'll use it as completion
        return {
            'prompt': '',
            'completion': text,
            'format': 'text'
        }
    
    def _preprocess_vision(self, item: Dict) -> Dict:
        """Preprocess vision format."""
        return {
            'image': item.get('image', ''),
            'caption': item.get('caption', item.get('text', '')),
            'format': 'vision'
        }
    
    def _preprocess_vision_qa(self, item: Dict) -> Dict:
        """Preprocess vision Q&A format."""
        return {
            'image': item.get('image', ''),
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'format': 'vision_qa'
        }
    
    def _preprocess_dolly(self, item: Dict) -> Dict:
        """Preprocess Dolly format."""
        instruction = item.get('instruction', '')
        context = item.get('context', '')
        response = item.get('response', '')
        
        if context:
            prompt = f"{instruction}\n\nContext: {context}"
        else:
            prompt = instruction
            
        return {
            'prompt': prompt,
            'completion': response,
            'format': 'dolly'
        }
    
    def _preprocess_oasst(self, item: Dict) -> Dict:
        """Preprocess OpenAssistant format."""
        # OASST has tree structure, we'll flatten it
        messages = []
        role = item.get('role', 'user')
        text = item.get('text', '')
        
        if role == 'prompter':
            role = 'user'
        
        messages.append({
            'role': role,
            'content': text
        })
        
        return {
            'messages': messages,
            'format': 'oasst'
        }
    
    def _preprocess_vicuna(self, item: Dict) -> Dict:
        """Preprocess Vicuna format."""
        conversations = item.get('conversations', [])
        messages = []
        
        for i, conv in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({
                'role': role,
                'content': conv
            })
        
        return {
            'messages': messages,
            'format': 'vicuna'
        }
    
    def _preprocess_wizardlm(self, item: Dict) -> Dict:
        """Preprocess WizardLM format."""
        return {
            'prompt': item.get('instruction', ''),
            'completion': item.get('output', ''),
            'format': 'wizardlm'
        }
    
    def _preprocess_orca(self, item: Dict) -> Dict:
        """Preprocess Orca format."""
        system = item.get('system_prompt', '')
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': question})
        messages.append({'role': 'assistant', 'content': answer})
        
        return {
            'messages': messages,
            'format': 'orca'
        }
    
    def _preprocess_anthropic(self, item: Dict) -> Dict:
        """Preprocess Anthropic format."""
        messages = []
        
        if 'human' in item and 'assistant' in item:
            messages.append({'role': 'user', 'content': item['human']})
            messages.append({'role': 'assistant', 'content': item['assistant']})
        
        return {
            'messages': messages,
            'format': 'anthropic'
        }
    
    def _preprocess_claude(self, item: Dict) -> Dict:
        """Preprocess Claude format."""
        messages = []
        
        if 'Human' in item and 'Assistant' in item:
            messages.append({'role': 'user', 'content': item['Human']})
            messages.append({'role': 'assistant', 'content': item['Assistant']})
        
        return {
            'messages': messages,
            'format': 'claude'
        }
    
    def _preprocess_chatml(self, item: Dict) -> Dict:
        """Preprocess ChatML format."""
        messages = item.get('messages', [])
        
        # ChatML format is already in the right structure
        return {
            'messages': messages,
            'format': 'chatml'
        }
    
    def _preprocess_llama_chat(self, item: Dict) -> Dict:
        """Preprocess Llama Chat format."""
        messages = []
        
        if 'system' in item:
            messages.append({'role': 'system', 'content': item['system']})
        if 'user' in item:
            messages.append({'role': 'user', 'content': item['user']})
        if 'assistant' in item:
            messages.append({'role': 'assistant', 'content': item['assistant']})
        
        return {
            'messages': messages,
            'format': 'llama_chat'
        }
    
    def _preprocess_code_alpaca(self, item: Dict) -> Dict:
        """Preprocess Code Alpaca format."""
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        lang = item.get('lang', 'python')
        
        if input_text:
            prompt = f"{instruction}\n\nInput:\n```{lang}\n{input_text}\n```"
        else:
            prompt = instruction
        
        if lang and output:
            completion = f"```{lang}\n{output}\n```"
        else:
            completion = output
            
        return {
            'prompt': prompt,
            'completion': completion,
            'format': 'code_alpaca'
        }
    
    def _preprocess_generic(self, item: Dict) -> Dict:
        """Generic preprocessing for unknown formats."""
        # Try to find text-like fields
        text_fields = ['text', 'content', 'message', 'sentence', 'paragraph']
        
        for field in text_fields:
            if field in item:
                return {
                    'prompt': '',
                    'completion': str(item[field]),
                    'format': 'generic'
                }
        
        # If nothing found, convert to string
        return {
            'prompt': '',
            'completion': json.dumps(item, ensure_ascii=False),
            'format': 'generic'
        }
    
    def prepare_for_model(self, data: List[Dict], tokenizer: Any) -> List[Dict]:
        """Prepare data for specific model type using tokenizer."""
        prepared = []
        
        for item in data:
            if 'messages' in item:
                # Multi-turn conversation
                text = tokenizer.apply_chat_template(
                    item['messages'],
                    tokenize=False,
                    add_generation_prompt=False
                )
                prepared.append({'text': text})
            
            elif 'prompt' in item and 'completion' in item:
                # Prompt-completion pairs
                if hasattr(tokenizer, 'apply_chat_template'):
                    # Use chat template if available
                    messages = [
                        {'role': 'user', 'content': item['prompt']},
                        {'role': 'assistant', 'content': item['completion']}
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                else:
                    # Simple concatenation
                    text = f"{item['prompt']}\n{item['completion']}"
                
                prepared.append({'text': text})
            
            else:
                # Fallback
                prepared.append(item)
        
        return prepared