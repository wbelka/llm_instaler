# Алгоритм детекции типов моделей в LLM Installer

## Обзор

Система детекции определяет тип модели НЕ по её названию, а по структуре и метаданным. Это позволяет автоматически поддерживать новые модели без изменения кода.

## Пошаговый алгоритм

### Шаг 1: Загрузка метаданных (ModelChecker.check_model)

```python
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"  # пример
```

1. **Запрос к HuggingFace API**:
   - Получение информации о модели (tags, pipeline_tag, library_name)
   - Получение списка файлов в репозитории
   - Скачивание конфигурационных файлов

2. **Формирование структуры данных**:
```python
model_data = {
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "files": [
        {"path": "config.json", "size": 1234},
        {"path": "model.safetensors", "size": 15000000000},
        {"path": "tokenizer.json", "size": 5678},
        # ... другие файлы
    ],
    "config_data": {
        "config.json": {
            "model_type": "qwen2_5_vl",
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "vision_config": {...},
            "hidden_size": 3584,
            # ... остальные поля
        },
        # Если есть другие конфиги (model_index.json для diffusers)
    },
    "tags": ["vision", "multimodal", "text-generation"],
    "pipeline_tag": "text-generation",  # или None
    "library_name": "transformers"  # или "diffusers", "timm", etc
}
```

### Шаг 2: Запуск цепочки детекторов

```python
# DetectorRegistry содержит упорядоченный список детекторов
detectors = [
    ConfigDetector(priority=100),      # Высший приоритет
    DiffusersDetector(priority=90),    
    AudioDetector(priority=80),
    EmbeddingDetector(priority=70),
    # ... другие детекторы
]
```

### Шаг 3: Проверка каждого детектора

Для каждого детектора в порядке убывания приоритета:

```python
for detector in detectors:
    if detector.matches(model_data):
        analysis = detector.analyze(model_data)
        model_data.update(analysis)
        break  # Первый подходящий детектор определяет тип
```

### Шаг 4: Логика детекторов

#### ConfigDetector (приоритет 100)
**matches()**: Проверяет наличие config.json в model_data["config_data"]

**analyze()**: 
1. Извлекает config.json
2. Читает model_type и architectures
3. Проверяет маппинги:
   ```python
   ARCHITECTURE_HANDLERS = {
       "Qwen2_5_VLForConditionalGeneration": "multimodal",
       "LlamaForCausalLM": "language-model",
       "StableDiffusionPipeline": "image-generation",
       # ...
   }
   
   MODEL_TYPE_MAPPING = {
       "qwen2_5_vl": "multimodal",
       "llama": "language-model",
       "whisper": "audio-model",
       # ...
   }
   ```
4. Если не нашел в маппингах, анализирует структуру:
   - Есть `vision_config` или `image_token_id` → multimodal
   - Есть `audio_config` или `mel_bins` → audio-model
   - Есть только `vocab_size` и `hidden_size` → language-model

#### DiffusersDetector (приоритет 90)
**matches()**: 
- Есть model_index.json
- Или есть папки unet/, vae/
- Или library_name == "diffusers"

**analyze()**: 
- Читает pipeline класс из model_index.json
- Определяет тип: text-to-image, text-to-video, inpainting, etc

#### AudioDetector (приоритет 80)
**matches()**: 
- Есть feature_extractor_type в конфиге
- Или есть audio_processor
- Или model_type содержит "whisper", "wav2vec2", etc

### Шаг 5: Результат анализа

Детектор возвращает:
```python
{
    "model_type": "qwen2_5_vl",              # Точный тип модели
    "model_family": "multimodal",            # Семейство (для выбора handler)
    "architecture_type": "Qwen2_5_VLForConditionalGeneration",
    "primary_library": "transformers",       # Основная библиотека
    "trust_remote_code": True,               # Нужен ли trust_remote_code
    "capabilities": {
        "supports_images": True,
        "supports_vision": True,
        "max_context_length": 128000,
        # ...
    },
    "special_requirements": ["flash-attn"],  # Особые зависимости
}
```

### Шаг 6: Fallback механизм

Если ни один детектор не сработал:
```python
def _infer_model_type(model_data):
    # Пытаемся определить по pipeline_tag из HuggingFace
    pipeline_tag = model_data.get("pipeline_tag", "")
    
    if "text-generation" in pipeline_tag:
        return {
            "model_type": "transformer",
            "model_family": "language-model"
        }
    elif "text-to-image" in pipeline_tag:
        return {
            "model_type": "diffusion",
            "model_family": "image-generation"
        }
    # ... и т.д.
    
    # Если совсем ничего не нашли
    return {
        "model_type": "unknown",
        "model_family": "unknown"
    }
```

## Пример работы для Qwen2.5-VL

1. **Загрузка**: Получаем config.json с model_type="qwen2_5_vl"
2. **ConfigDetector.matches()**: Да, есть config.json → True
3. **ConfigDetector.analyze()**:
   - model_type = "qwen2_5_vl"
   - Проверяем ARCHITECTURE_HANDLERS["Qwen2_5_VLForConditionalGeneration"] = "multimodal"
   - Находим vision_config в конфиге → подтверждаем multimodal
   - trust_remote_code = True (для qwen моделей)
4. **Результат**: model_family = "multimodal" → будет использован MultimodalHandler или QwenVLHandler

## Важные принципы

1. **Детекция НЕ по имени**: Не проверяем название модели, только структуру
2. **Приоритетность**: Более специфичные детекторы имеют высший приоритет
3. **Первый подходящий**: Используется первый детектор, который вернул matches()=True
4. **Расширяемость**: Новые детекторы автоматически регистрируются через registry
5. **Fallback**: Всегда есть запасной вариант через pipeline_tag

## Добавление поддержки новой модели

1. Если модель имеет уникальную структуру → создать новый детектор
2. Если модель похожа на существующую → добавить в маппинги
3. Создать соответствующий handler в handlers/
4. Система автоматически начнет определять и поддерживать новый тип