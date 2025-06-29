# LLM Installer v2: Постановка задачи

## Общее описание проекта

### Проблема
Установка и запуск LLM моделей с HuggingFace требует:
- Ручной установки множества зависимостей
- Написания кода для загрузки и запуска модели
- Понимания особенностей каждой конкретной модели
- Настройки параметров под железо пользователя
- Организации файлов и окружений

### Решение
LLM Installer v2 - это набор скриптов, который автоматизирует весь процесс от проверки совместимости до запуска модели.

### Ключевые возможности
1. **Проверка совместимости** - анализ модели БЕЗ скачивания весов
2. **Автоматическая установка** - загрузка модели и всех зависимостей
3. **Изоляция окружений** - каждая модель в своей папке со своим venv
4. **Универсальные скрипты** - копирование готовых скриптов, которые работают с любой моделью
5. **Простой запуск** - одна команда для старта модели

### Требования
- Поддержка Linux и macOS (НЕ Windows)
- Python >= 3.8
- Работа с моделями из HuggingFace и локальными файлами
- Поддержка различных типов моделей:
  - Языковые модели (transformers)
  - Генерация изображений (diffusers)
  - Мультимодальные модели (vision-language)
  - Эмбеддинг модели (sentence-transformers)
  - Аудио модели (speech, music)
  - Computer Vision модели
  - Специализированные архитектуры

### Пример использования
```bash
# Проверка совместимости
./llm-installer check meta-llama/Llama-3-8B

# Установка
./llm-installer install meta-llama/Llama-3-8B --quantization 4bit

# Запуск установленной модели
cd ~/LLM/models/meta-llama_Llama-3-8B
./start.sh
```

## Step 1: Создание базовой инфраструктуры

### Цель
Создать основу проекта: структуру директорий, базовые скрипты, систему конфигурации и первичные проверки.

### Концепция работы инсталлятора

**Что делает инсталлятор при установке модели:**

1. **Создает изолированную директорию** для модели
2. **Загружает файлы модели** с HuggingFace
3. **Анализирует модель** и создает `model_info.json` с полной информацией
4. **Создает виртуальное окружение** для модели
5. **Устанавливает зависимости** на основе типа модели (определяет хандлер)
6. **Копирует универсальные скрипты** из `~/LLM/installer/scripts/` в папку модели
7. **Создает конфигурацию** с путями и настройками

После установки структура модели:
```
~/LLM/models/meta-llama_Llama-3-8B/
├── model/              # Файлы модели
├── .venv/              # Изолированное окружение
├── model_info.json     # Детальная информация о модели
├── config.yaml         # Конфигурация запуска
├── start.sh           # Копия универсального скрипта
├── train.sh           # Копия универсального скрипта
├── serve_api.py       # Копия универсального скрипта
└── serve_ui.py        # Копия универсального скрипта
```

Скрипты универсальные - они читают `model_info.json` и используют библиотеки инсталлятора для работы с конкретной моделью.

### Что нужно сделать

#### 1.1 Структура проекта
Создать следующую структуру:
```
llm-installer/                  # Репозиторий (устанавливается в ~/LLM/installer)
├── llm-installer              # Главный bash скрипт (точка входа)
├── setup.sh                   # Скрипт первичной настройки
├── requirements.txt           # Зависимости инсталлятора
├── config.yaml.example        # Пример конфигурации
├── core/                      # Ядро системы
│   ├── __init__.py
│   ├── installer.py          # Основная логика установки
│   ├── checker.py            # Логика проверки моделей
│   ├── config.py             # Работа с конфигурацией
│   └── utils.py              # Вспомогательные функции
├── handlers/                  # Обработчики для разных типов моделей
│   ├── __init__.py
│   ├── base.py              # Базовый класс для всех хандлеров
│   ├── transformer.py        # Хандлер для языковых моделей
│   ├── diffusion.py          # Хандлер для генерации изображений
│   ├── multimodal.py         # Хандлер для мультимодальных моделей
│   ├── embedding.py          # Хандлер для эмбеддингов
│   ├── vision.py             # Хандлер для computer vision
│   ├── audio.py              # Хандлер для аудио моделей
│   └── specialized.py        # Хандлер для специальных архитектур
├── detectors/                 # Детекторы типов моделей
│   ├── __init__.py
│   ├── base.py              # Базовый класс детектора
│   ├── config_detector.py    # Определение по config.json
│   ├── file_detector.py      # Определение по файлам в репозитории
│   ├── architecture_detector.py # Определение по архитектуре
│   └── registry.py           # Реестр всех детекторов
├── scripts/                   # Универсальные скрипты для моделей
│   ├── start.sh              # Универсальный скрипт запуска (executable)
│   ├── train.sh              # Универсальный скрипт обучения (executable)
│   ├── serve_api.py          # Универсальный API сервер
│   ├── serve_ui.py           # Универсальный веб-интерфейс
│   └── model_loader.py       # Универсальный загрузчик моделей
└── README.md                  # Документация
```

#### 1.2 Скрипт setup.sh
Должен выполнять:
- Проверку операционной системы (отказ если Windows)
- Проверку версии Python (>= 3.8)
- Проверку доступного места на диске
- Проверку CUDA/nvidia-smi (если есть GPU)
- Создание директорий:
  - `~/LLM/models` - для установленных моделей
  - `~/LLM/cache` - для кеша HuggingFace
  - `~/LLM/logs` - для логов
- Создание виртуального окружения инсталлятора в `.venv`
- Установку Python зависимостей из requirements.txt
- Копирование config.yaml.example в config.yaml
- Добавление алиаса в ~/.bashrc или ~/.zshrc для удобства

**Важно**: Универсальные скрипты из директории `scripts/` остаются в инсталляторе. При установке модели они будут копироваться в её директорию.

#### 1.3 Главный скрипт llm-installer
Bash wrapper который:
- Проверяет наличие .venv
- Активирует виртуальное окружение
- Добавляет путь к модулям в PYTHONPATH
- Передает все аргументы в Python скрипт
- Обрабатывает коды возврата
- Показывает help если нет аргументов

**Важно**: Путь к инсталлятору должен быть доступен из установленных моделей для импорта хандлеров.

Поддерживаемые команды:
- `check <model>` - проверить совместимость
- `install <model> [options]` - установить модель
- `list` - показать установленные модели
- `config` - показать текущую конфигурацию

#### 1.4 Конфигурационный файл config.yaml
```yaml
# Токены и авторизация
huggingface_token: ""  # Можно переопределить через HF_TOKEN

# Пути (относительно home)
models_dir: "~/LLM/models"
cache_dir: "~/LLM/cache"
logs_dir: "~/LLM/logs"

# Настройки установки
default_device: "auto"  # auto, cuda, cpu
max_download_workers: 4
resume_downloads: true

# Системные требования
min_disk_space_gb: 50
warn_disk_space_gb: 100

# Логирование
log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
log_rotation: "10MB"
```

#### 1.5 Python requirements.txt
```
huggingface_hub>=0.20.0
requests>=2.28.0
pyyaml>=6.0
click>=8.1.0        # для CLI
rich>=13.0.0        # для красивого вывода
psutil>=5.9.0       # для проверки системы
safetensors>=0.4.0  # для анализа файлов
```

#### 1.6 Модульная структура core/

**core/installer.py**
- Основная логика установки моделей
- Использует детекторы для определения типа
- Вызывает соответствующий хандлер
- Управляет процессом загрузки

**core/checker.py**
- Проверка совместимости без загрузки
- Анализ метаданных модели
- Формирование отчета о требованиях

**core/config.py**
- Загрузка и валидация конфигурации
- Обработка переменных окружения
- Расширение путей (~)

**core/utils.py**
- Вспомогательные функции
- Работа с файловой системой
- Форматирование вывода
- Системные проверки

#### 1.6 Базовая структура installer.py
Основные функции:
- `parse_arguments()` - обработка аргументов командной строки
- `load_config()` - загрузка конфигурации с учетом переменных окружения
- `check_system()` - проверка системных требований
- `setup_logging()` - настройка логирования
- Диспетчер команд (check, install, list, config)

#### 1.7 Модульная архитектура

**Базовые классы (base.py файлы)**

Handler Base Class:
```python
class BaseHandler:
    def __init__(self, model_info):
        self.model_info = model_info
    
    def get_dependencies(self) -> List[str]:
        """Возвращает список зависимостей для модели"""
        raise NotImplementedError
    
    def load_model(self, model_path: str):
        """Загружает модель с оптимальными параметрами"""
        raise NotImplementedError
    
    def get_inference_params(self) -> Dict:
        """Возвращает параметры для инференса"""
        raise NotImplementedError
    
    def get_training_params(self) -> Dict:
        """Возвращает параметры для обучения"""
        raise NotImplementedError
```

Detector Base Class:
```python
class BaseDetector:
    def matches(self, model_info: Dict) -> bool:
        """Проверяет, подходит ли детектор для модели"""
        raise NotImplementedError
    
    def analyze(self, model_info: Dict) -> Dict:
        """Анализирует модель и возвращает дополнительную информацию"""
        raise NotImplementedError
```

**Определение типа модели**

Детекторы работают по цепочке, проверяя метаданные модели:
- Структура config.json (architectures, model_type, task_specific_params)
- Наличие специфичных файлов (model_index.json для diffusers)
- Структура репозитория (наличие unet/, vae/ папок)
- Наличие специальных процессоров (image_processor, audio_processor)

Пример детекции:
- Мультимодальная модель: наличие vision_config + text_config в config.json
- Diffusion модель: наличие model_index.json с компонентами pipeline
- Audio модель: наличие feature_extractor_type или audio_processor
- Embedding модель: архитектура содержит только encoder, нет decoder

**Важно**: Детекция НЕ привязана к названию модели, только к её структуре и метаданным.

**Реестр детекторов (registry.py)**
- Автоматическая регистрация всех детекторов
- Приоритеты детекторов
- Цепочка обработки

**Примеры хандлеров**

TransformerHandler:
- Работает с языковыми моделями
- Определяет chat templates и токенизаторы по config.json
- Настраивает параметры генерации текста

MultimodalHandler:
- Работает с моделями обрабатывающими и текст, и изображения
- Определяется по наличию vision_config в конфигурации
- Создает интерфейс для мультимодального ввода

DiffusionHandler:
- Работает с моделями генерации изображений
- Определяется по наличию model_index.json
- Настраивает schedulers и pipelines

EmbeddingHandler:
- Работает с моделями для векторных представлений
- Определяется по отсутствию декодера в архитектуре
- Оптимизация для batch processing

AudioHandler:
- Работает с аудио моделями
- Определяется по наличию audio processor в конфигурации
- Настраивает параметры для работы со звуком

SpecializedHandler:
- Для новых архитектур не подходящих под другие категории
- Определяется по уникальным полям в config.json
- Кастомная логика загрузки

**Расширяемость**
Для добавления нового типа модели нужно:
1. Создать новый детектор в `detectors/`
2. Создать новый хандлер в `handlers/`
3. Детектор и хандлер автоматически подключатся

**Примечание**: Список хандлеров выше - это примеры. Система спроектирована так, чтобы легко добавлять поддержку любых новых типов моделей, которые появляются в экосистеме ML.

**Использование хандлеров в универсальных скриптах**

Универсальные скрипты работают для любой модели, потому что:
1. Читают `model_info.json` чтобы понять тип модели
2. Импортируют соответствующий хандлер из инсталлятора
3. Хандлер знает как правильно загрузить и работать с моделью

Пример универсального serve_api.py:
```python
# Читаем информацию о модели
with open('model_info.json') as f:
    model_info = json.load(f)

# Импортируем нужный хандлер
sys.path.append(os.path.expanduser('~/LLM/installer'))
from handlers import get_handler

# Хандлер знает как работать с этой моделью
handler = get_handler(model_info)
model = handler.load_model('./model')

# Универсальный API работает с любой моделью
app = create_universal_api(model, handler)
```

Это позволит:
- Обновлять логику работы с моделями централизованно
- Не дублировать код между моделями
- Легко добавлять поддержку новых типов моделей

#### 1.8 Обработка токена HuggingFace
Приоритет (от высшего к низшему):
1. Переменная окружения `HF_TOKEN`
2. Значение `huggingface_token` из config.yaml
3. Работа без токена (только публичные модели)

### Результат Step 1
После выполнения этого шага должно быть:

1. **Работающий инсталлятор** который можно установить через:
   ```bash
   git clone <repo> ~/LLM/installer
   cd ~/LLM/installer
   ./setup.sh
   ```

2. **Базовый CLI** который:
   - Показывает help
   - Проверяет систему при запуске
   - Читает конфигурацию
   - Логирует действия

3. **Модульная архитектура**:
   - Базовые классы для хандлеров и детекторов
   - Реестр для автоматической регистрации модулей
   - Структура директорий для будущих хандлеров
   - Универсальные скрипты в `scripts/`

4. **Подготовленная инфраструктура**:
   - Все необходимые директории созданы
   - Конфигурация настроена
   - Виртуальное окружение готово
   - Система проверена

**Важно**: В Step 1 создается только базовая инфраструктура. Конкретные хандлеры (TransformerHandler, MultimodalHandler и т.д.) будут добавляться в последующих шагах по мере необходимости.

### Критерии успеха Step 1
- [ ] setup.sh успешно устанавливает инсталлятор
- [ ] llm-installer показывает help и версию
- [ ] Конфигурация читается корректно
- [ ] HF_TOKEN имеет приоритет над config.yaml
- [ ] Логи пишутся в указанную директорию
- [ ] Проверки системы работают (Python, диск, GPU)
- [ ] На Windows выдается понятная ошибка
- [ ] Базовые классы Handler и Detector созданы
- [ ] Реестр детекторов работает
- [ ] Создан хотя бы один пример хандлера (например, TransformerHandler)
- [ ] Подготовлена структура для добавления других хандлеров
- [ ] Универсальные скрипты созданы и готовы к копированию

### Примеры команд после Step 1
```bash
# Проверка установки
./llm-installer --version
./llm-installer --help

# Просмотр конфигурации
./llm-installer config

# Попытка проверки (пока заглушка)
./llm-installer check meta-llama/Llama-3-8B
> "Check functionality will be implemented in Step 2"

# Список моделей (пока пустой)
./llm-installer list
> "No models installed yet"
```
# LLM Installer v2: Step 2 - Техническое задание для команды check

## Общее описание

Команда `check` анализирует модель с HuggingFace БЕЗ загрузки весов и выдает детальный отчет о совместимости с системой пользователя.

## Входные данные

Команда принимает один аргумент - идентификатор модели на HuggingFace:
- Формат: `owner/model-name`
- Примеры: 
  - `meta-llama/Llama-3-8B`
  - `mistralai/Mistral-7B-v0.1`
  - `runwayml/stable-diffusion-v1-5`

## Алгоритм работы

### 1. Получение информации о модели через HuggingFace API

**1.1 Инвентаризация файлов:**
- Использовать `huggingface_hub.HfApi()` для получения списка всех файлов
- Получить размер каждого файла (без загрузки)
- Сохранить структуру директорий (включая поддиректории)

**1.2 Получение метаданных:**
- Теги модели (tags)
- Pipeline tags (для определения задачи)
- Library name
- Model card (если нужна дополнительная информация)

### 2. Загрузка и анализ конфигурационных файлов

**2.1 Порядок поиска конфигураций (строго в этом порядке):**

1. **model_index.json** 
   - Если найден → это композитная модель (например, Diffusers)
   - Содержит `_class_name` и ссылки на компоненты
   - Прекратить поиск других конфигов в корне

2. **config.json**
   - Основной конфигурационный файл для большинства моделей
   - Содержит архитектуру и параметры

3. **Альтернативные имена** (если нет стандартных):
   - `llm_config.json`
   - `model_config.json`
   - Другие `*config.json` файлы
   - **Важно**: проверить содержимое на наличие архитектурных ключей

**2.2 Валидация альтернативных конфигов:**
- Проверить наличие ключей: `model_type`, `hidden_size`, `num_layers`, `vocab_size`
- Игнорировать конфиги с ключами: `api_key`, `base_url`, `endpoint`

**2.3 Рекурсивный поиск в поддиректориях:**
- Для композитных моделей искать `config.json` в каждой поддиректории
- Собрать все конфигурации компонентов (unet/, vae/, text_encoder/)

### 3. Определение типа модели

**3.1 Первичное определение по API tags:**
- `text-generation` → языковая модель
- `text-to-image` → diffusion модель  
- `automatic-speech-recognition` → аудио модель
- `feature-extraction` → embedding модель
- `text-classification` → классификатор

**3.2 Уточнение по структуре конфигов:**

**Для model_index.json:**
- Тип: композитная модель
- Семейство: diffusers
- Pipeline класс из `_class_name`

**Для config.json проверять:**
- `vision_config` + `text_config` → мультимодальная модель
- `is_encoder_decoder: true` → seq2seq модель
- `architectures` содержит:
  - `*ForCausalLM` → генеративная языковая
  - `*ForMaskedLM` → encoder модель
  - `*ForSequenceClassification` → классификатор
- `model_type` для определения архитектуры

**Специальные архитектуры (по model_type):**
- `mamba`, `jamba` → State Space Models (требуют специальные библиотеки)
- `rwkv` → RWKV архитектура
- `whisper` → аудио модель
- `clip` → мультимодальная модель

**3.3 Обработка неизвестных архитектур:**
- Если не удалось определить тип → вернуть ошибку
- Сообщение: "Неподдерживаемая архитектура модели"

### 4. Сбор требований модели

**4.1 Определение необходимых библиотек:**

Базовые правила:
- Если есть `model_index.json` → `diffusers`
- Если `architectures` в config.json → `transformers`
- Если `feature_extractor_type` → дополнительно нужны аудио библиотеки

Дополнительные зависимости по типу:
- Diffusion модели: `accelerate`, `xformers` (опционально)
- Аудио модели: `librosa`, `soundfile`
- Vision модели: `pillow`, `opencv-python`

Специфичные зависимости по архитектуре:
- `model_type: "mamba"` → `mamba-ssm`, `causal-conv1d`
- `model_type: "rwkv"` → `rwkv`
- Наличие `use_flash_attention_2` в конфиге → `flash-attn`
- `model_type: "jamba"` → `mamba-ssm`, `causal-conv1d`

Проверка специальных полей в config.json:
- `_attn_implementation` → может указывать на flash attention
- `use_cache` → может требовать дополнительную память
- Любые custom extensions в конфиге

**4.2 Анализ специальных требований:**
- Проверить `torch_dtype` в конфиге
- Проверить наличие `quantization_config`
- Определить поддерживаемые форматы (fp32, fp16, bf16, int8, int4)

**4.3 Предупреждения о сложных зависимостях:**
Некоторые библиотеки требуют компиляции или специальную установку:
- `flash-attn` → требует CUDA и компиляцию
- `mamba-ssm` → требует специфичные версии CUDA
- `xformers` → опциональна, но улучшает производительность

В отчете помечать такие зависимости как:
- `mamba-ssm (requires CUDA compilation)`
- `flash-attn (optional, requires CUDA 11.6+)`

### 5. Оценка требований к ресурсам

**5.1 Дисковое пространство:**
- Сумма размеров всех файлов модели (из API)
- Добавить 3GB на виртуальное окружение
- Итого: размер модели + 3GB

**5.2 Оценка памяти для инференса:**

Базовый расчет по размеру файлов весов:
- Определить какие файлы содержат веса (.bin, .safetensors, .pt, .pth)
- Суммировать их размеры

Коэффициенты для разных dtype:
- int4: размер файлов × 1.2
- int8: размер файлов × 1.3
- float16/bfloat16: размер файлов × 1.5  
- float32: размер файлов × 2.0

Если есть `quantization_config`:
- Показать два варианта: текущий размер и размер после квантования

**5.3 Оценка памяти для обучения:**

LoRA:
- float16: память инференса × 1.3
- float32: память инференса × 1.5

Full fine-tuning:
- float16: память инференса × 4
- float32: память инференса × 5

### 6. Анализ совместимости с системой

**6.1 Сбор информации о системе:**
- CPU: количество ядер
- RAM: общий объем и доступный
- GPU: модель и объем VRAM (если есть)
- Доступное дисковое пространство

**6.2 Проверка совместимости:**

Для каждой конфигурации (dtype) проверить:
- Хватает ли места на диске
- Хватает ли RAM/VRAM для инференса
- Определить оптимальное устройство (CPU/CUDA)

Правила:
- Если требуется больше VRAM чем есть → проверить влезет ли в RAM
- Если не влезает никуда → пометить как несовместимую

### 7. Формирование отчета

**7.1 Структура отчета:**

```
Model Compatibility Report: [model-id]
============================================================
Model Type: [тип из детекторов]
Library: [основная библиотека]  
Architecture: [архитектура]
Task: [задача из tags]

Storage Requirements:
  - Model files: X.X GB
  - Virtual environment: ~3 GB
  - Total disk space needed: ~X.X GB

Memory Requirements (RAM/VRAM):
  - Default configuration ([dtype]): X.X GB
  - [другие варианты если есть]

Model Capabilities:
  - [список из tags и конфига]

Special Requirements:
  - [список необходимых библиотек]

Примеры для разных моделей:
- Обычная transformer модель: `transformers, torch, accelerate`
- Mamba модель: `transformers, torch, accelerate, mamba-ssm, causal-conv1d`
- Diffusion модель: `diffusers, transformers, torch, accelerate, xformers`
- Модель с Flash Attention: `transformers, torch, accelerate, flash-attn`

------------------------------------------------------------
Hardware Compatibility Check
------------------------------------------------------------
Your Hardware:
  - CPU: X cores, Y threads
  - RAM: X.X GB total, Y.Y GB available
  - GPU(s): [если есть]
    • [название GPU]
      VRAM: X.X GB total, Y.Y GB free

Compatibility Analysis:
Quantization Memory       Can Run?   Device     Notes
------------------------------------------------------------
[dtype]      X.X GB       [✓/✗]      [device]   [notes]

Training Compatibility:
Method          Memory     CPU      GPU      Device    
------------------------------------------------------------
LoRA ([dtype])  X.X GB     [✓/✗]    [✓/✗]    [device]
Full ([dtype])  X.X GB     [✓/✗]    [✓/✗]    [device]

============================================================
```

**7.2 Определение Model Capabilities:**
- Из pipeline_tags API
- Из специфичных полей конфига
- Примеры: "Text Generation", "Image To Image", "Multimodal"

**7.3 Notes в таблице совместимости:**
- Оставлять пустым если все хорошо
- "Requires quantization" если только с квантованием влезает
- "Close to limit" если остается < 1GB свободной памяти

### 8. Обработка ошибок

**8.1 Модель не найдена:**
- Сообщение: "Model not found on HuggingFace"

**8.2 Приватная модель:**
- Сообщение: "This is a private model. Please set HF_TOKEN"

**8.3 Неподдерживаемая архитектура:**
- Сообщение: "Unsupported model architecture"

**8.4 Отсутствуют конфигурационные файлы:**
- Сообщение: "No configuration files found"

**8.5 Специфичные требования:**
- Если модель требует сложные в установке библиотеки, показать предупреждение
- Пример: "Warning: This model requires mamba-ssm which needs CUDA compilation"

### 9. Сохранение результатов

После успешной проверки сохранить:
- Полный отчет в `~/LLM/logs/checks/[model-id].txt`
- JSON с метаданными для последующего использования в install

## Дополнительные детали реализации

### Маппинг архитектур к специальным зависимостям

Создать словарь соответствий:
```
model_type → special_dependencies
"mamba" → ["mamba-ssm", "causal-conv1d"]
"jamba" → ["mamba-ssm", "causal-conv1d"]  
"rwkv" → ["rwkv"]
"whisper" → ["openai-whisper"]
```

Также проверять поля в config.json:
- `_attn_implementation: "flash_attention_2"` → добавить `flash-attn`
- `use_flash_attn: true` → добавить `flash-attn`

### Отображение в отчете

В секции Special Requirements показывать:
- Базовые библиотеки первыми
- Специфичные библиотеки с пометками о сложности установки
- Опциональные библиотеки отдельно

Пример:
```
Special Requirements:
  Base:
  - transformers
  - torch  
  - accelerate
  
  Architecture-specific:
  - mamba-ssm (requires CUDA compilation)
  - causal-conv1d (requires CUDA)
  
  Optional:
  - flash-attn (for faster inference, requires CUDA 11.6+)
```

## Критерии успешной реализации

1. Команда работает БЕЗ загрузки файлов весов
2. Правильно определяет тип модели по иерархии конфигов
3. Точно оценивает требования к ресурсам
4. Корректно определяет совместимость с системой
5. Генерирует понятный и информативный отчет
6. Обрабатывает все типы ошибок
7. Работает со всеми основными типами моделей (transformer, diffusion, multimodal, audio)
8. Правильно определяет специфичные зависимости (mamba-ssm, flash-attn и т.д.)


# LLM Installer v2: Step 3 - Техническое задание для команды install

## Общее описание

Команда `install` загружает модель с HuggingFace и создает полностью автономное окружение для её запуска.

## Входные данные

Команда принимает один обязательный аргумент:
```bash
./llm-installer install <model-id>
```

Примеры:
- `./llm-installer install meta-llama/Llama-3-8B`
- `./llm-installer install runwayml/stable-diffusion-v1-5`

## Предварительные проверки

### 1. Использование результатов check

**1.1 Проверить наличие сохраненного результата:**
- Искать файл `~/LLM/logs/checks/<model-id>.json`
- Если есть и свежий (< 24 часа) - использовать
- Если нет - запустить check автоматически

**1.2 Валидация результатов check:**
- Проверить что модель совместима
- Если несовместима - показать отчет и отказать в установке
- Если совместима с ограничениями - показать предупреждение

### 2. Проверка дискового пространства

- Получить требуемое место из результатов check
- Проверить доступное место в ~/LLM/models
- Если недостаточно - показать сколько нужно и прервать

## Процесс установки

### 3. Создание структуры директорий

**3.1 Создать папку модели:**
```
~/LLM/models/<safe_model_name>/
├── model/          # Для файлов модели
├── logs/           # Для логов работы
└── scripts/        # Для скопированных скриптов (опционально)
```

**3.2 Правила именования:**
- Заменить `/` на `_` в имени модели
- Пример: `meta-llama/Llama-3-8B` → `meta-llama_Llama-3-8B`

**3.3 Обработка существующей установки:**
- Если папка существует - спросить подтверждение на переустановку
- При подтверждении - удалить старую установку полностью

### 4. Логирование процесса

**4.1 Создать install.log:**
- Путь: `~/LLM/models/<model_name>/install.log`
- Записывать все этапы установки с timestamp
- При ошибке - записать полный traceback

### 5. Загрузка модели

**5.1 Параметры загрузки:**
```python
# НЕ использовать symlinks, делать полную копию
local_dir_use_symlinks = False

# Путь для загрузки
local_dir = "~/LLM/models/<model_name>/model"

# Использовать кеш для ускорения
cache_dir = "~/LLM/cache"

# Многопоточная загрузка
max_workers = 4

# Возобновление прерванной загрузки
resume_download = True
```

**5.2 Отображение прогресса:**
```
Downloading model: meta-llama/Llama-3-8B
File 1/45: config.json (2.1 KB) ✓
File 2/45: model-00001-of-00004.safetensors (4.9 GB)
  Progress: 2.3 GB / 4.9 GB (47%) - 25 MB/s - ETA: 1m 44s
```

**5.3 Обработка ошибок загрузки:**
- При сетевой ошибке - retry до 3 раз
- При прерывании пользователем - очистить частичные файлы
- Записать ошибку в install.log

### 6. Создание виртуального окружения

**6.1 Создание venv:**
```bash
cd ~/LLM/models/<model_name>
python3 -m venv .venv
```

**6.2 Обновление pip:**
```bash
.venv/bin/pip install --upgrade pip setuptools wheel
```

### 7. Установка зависимостей

**7.1 Базовые зависимости из model_info:**
- Основная библиотека (transformers/diffusers)
- torch с правильным CUDA индексом
- accelerate

**7.2 Специфичные зависимости:**
Использовать маппинг из check:
- mamba модели → mamba-ssm, causal-conv1d
- flash attention модели → flash-attn
- И т.д.

**7.3 Обработка ошибок установки:**

Для обычных пакетов - просто fail.

Для специальных пакетов (mamba-ssm, flash-attn):
```
ERROR: Failed to install mamba-ssm

This model requires mamba-ssm which needs additional setup:

1. Ensure CUDA toolkit is installed:
   nvcc --version

2. Install build dependencies:
   - Ubuntu/Debian: sudo apt-get install build-essential
   - macOS: xcode-select --install

3. Try manual installation:
   cd ~/LLM/models/<model_name>
   source .venv/bin/activate
   pip install mamba-ssm --no-cache-dir

After fixing, run install again.
Installation aborted.
```

### 8. Копирование универсальных скриптов

**8.1 Скрипты для копирования:**
- `~/LLM/installer/scripts/start.sh` → `start.sh`
- `~/LLM/installer/scripts/train.sh` → `train.sh`
- `~/LLM/installer/scripts/serve_api.py` → `serve_api.py`
- `~/LLM/installer/scripts/model_loader.py` → `model_loader.py`
- `~/LLM/installer/scripts/*.html` → `*.html` (все веб-интерфейсы)

**8.2 Установка прав:**
- chmod +x для .sh скриптов

**8.3 Создание model_info.json:**
- Скопировать результаты check в папку модели
- Добавить дополнительные поля:
  - `install_date`: timestamp установки
  - `installer_version`: версия инсталлятора
  - `model_path`: "./model"

### 9. Тестирование установки

**9.1 Простой тест загрузки:**

Создать временный test_load.py:
```python
import sys
sys.path.append("~/LLM/installer")

# Попытаться загрузить модель
try:
    from model_loader import load_model
    model_info = json.load(open("model_info.json"))
    
    print("Loading model...")
    model, tokenizer = load_model(model_info, device="cpu", load_in_8bit=True)
    
    print("Model loaded successfully!")
    
    # Простой тест для языковых моделей
    if model_info.get("model_family") == "language-model":
        inputs = tokenizer("Hello", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=5)
        result = tokenizer.decode(outputs[0])
        print(f"Test generation: {result}")
        
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)
```

**9.2 Обработка результатов теста:**
- Успех → продолжить
- Ошибка импорта → проблема с зависимостями
- Ошибка загрузки → проблема с моделью или памятью
- Записать результат в install.log

### 10. Финализация установки

**10.1 Создать SUCCESS маркер:**
- Файл `.install_complete` с датой установки
- Используется для проверки завершенности установки

**10.2 Показать итоговую информацию:**
```
✓ Model installed successfully!

Location: ~/LLM/models/meta-llama_Llama-3-8B
Size: 13.5 GB
Virtual environment: 3.1 GB

To start the model:
  cd ~/LLM/models/meta-llama_Llama-3-8B
  ./start.sh

To train/fine-tune:
  ./train.sh --data your_data.json

Logs saved to: install.log
```

## Обработка особых случаев

### 11. Прерывание установки

**11.1 Ctrl+C во время загрузки:**
- Поймать сигнал
- Показать: "Installation interrupted. Cleaning up..."
- Удалить частично загруженные файлы
- Сохранить состояние в install.log

**11.2 Возобновление:**
- НЕ поддерживается
- Показать: "Previous installation incomplete. Starting fresh..."
- Удалить всю папку и начать заново

### 12. Модели с дополнительными файлами

**12.1 Tokenizer файлы:**
- Обязательно загрузить все файлы токенизатора
- Проверить наличие: tokenizer.json, tokenizer_config.json

**12.2 Дополнительные процессоры:**
- Для мультимодальных: preprocessor_config.json
- Для аудио: feature_extractor_config.json

### 13. Проверки после установки

**13.1 Валидация структуры:**
- Проверить что все обязательные файлы на месте
- Размеры файлов соответствуют ожидаемым
- model_info.json читается корректно

**13.2 Проверка окружения:**
- .venv/bin/python существует
- Основные пакеты импортируются

## Структура финальной установки

```
~/LLM/models/meta-llama_Llama-3-8B/
├── model/                  # Все файлы модели
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── .venv/                  # Виртуальное окружение
├── logs/                   # Директория для логов работы
├── model_info.json        # Информация о модели из check
├── install.log            # Лог установки
├── .install_complete      # Маркер успешной установки
├── start.sh              # Скрипт запуска (executable)
├── train.sh              # Скрипт обучения (executable)
├── serve_api.py          # API сервер
├── serve_ui.py           # Веб интерфейс
└── model_loader.py       # Загрузчик модели
```

## Критерии успешной реализации

1. Модель полностью загружается в изолированную папку
2. Виртуальное окружение создается без конфликтов
3. Все зависимости устанавливаются корректно
4. Специальные зависимости обрабатываются с понятными инструкциями
5. Прогресс загрузки отображается информативно
6. Все ошибки логируются и показываются пользователю
7. Тест загрузки проходит успешно
8. Универсальные скрипты готовы к использованию
9. При прерывании установка корректно очищается
10. install.log содержит полную историю установки


# LLM Installer v2: Step 4 - Техническое задание для универсальных скриптов

## Общее описание

Создание набора универсальных скриптов, которые работают с любым типом модели. Скрипты копируются в папку модели при установке и используют информацию из `model_info.json` для правильной работы.

## Список скриптов

1. **start.sh** - главный скрипт запуска
2. **serve_api.py** - универсальный API сервер
3. **model_loader.py** - модуль загрузки моделей
4. **train.sh** - скрипт для обучения
5. **serve_ui.html** - веб интерфейс (обязательный)

## Детальная спецификация

### 1. start.sh - Главный скрипт запуска

**Назначение:** Простой запуск модели одной командой

**Функциональность:**
- Активация виртуального окружения
- Проверка что установка завершена (.install_complete)
- Установка переменных окружения
- Запуск API сервера с параметрами по умолчанию
- Обработка аргументов командной строки

**Структура:**
```bash
#!/bin/bash
# Универсальный скрипт запуска модели

# 1. Определение путей
# 2. Активация venv
# 3. Проверки
# 4. Установка переменных (CUDA_VISIBLE_DEVICES, etc)
# 5. Запуск serve_api.py с аргументами
```

**Поддерживаемые аргументы:**
- `--port PORT` - порт для API (default: 8000)
- `--host HOST` - хост для binding (default: 0.0.0.0)
- `--dtype DTYPE` - тип данных (default: auto)
- `--device DEVICE` - устройство (default: auto)
- `--no-browser` - не открывать браузер автоматически
- Все остальные передаются в serve_api.py

**Автоматические действия:**
- После успешного запуска открывает браузер с UI
- Показывает URL для доступа: `http://localhost:8000`

### 2. serve_api.py - Универсальный API сервер

**Назначение:** FastAPI сервер, адаптирующийся под тип модели

**Ключевые компоненты:**

**2.1 Инициализация:**
- Чтение model_info.json
- Определение пути к инсталлятору
- Импорт нужного хандлера
- Загрузка модели через хандлер

**2.2 Универсальные эндпоинты:**
```
GET /health - проверка статуса
GET /info - информация о модели
POST /generate - основной эндпоинт (адаптируется под тип)
GET /generate/stream - SSE стриминг для текстовых моделей
WS /ws - WebSocket для real-time коммуникации
GET / - веб интерфейс (serve_ui.html)
GET /static/* - статические файлы (если нужны)
```

**2.3 Адаптация под тип модели:**

Для языковых моделей:
- `/generate` принимает prompt или messages
- `/generate/stream` - Server-Sent Events для стриминга токенов
- Поддержка streaming через параметр `stream: true`
- Параметры: temperature, max_tokens, top_p, stop_sequences
- **Для reasoning моделей**:
  - Параметр `reasoning_mode: true`
  - Разделение ответа на `thinking` и `answer`
  - Отдельный стриминг для каждой части

Реализация стриминга:
```python
@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    async def event_generator():
        # Для reasoning моделей
        if model_info.get("supports_reasoning") and request.reasoning_mode:
            async for chunk in model.generate_stream_reasoning(request.prompt):
                yield {
                    "data": json.dumps({
                        "type": chunk.type,  # "thinking" or "answer"
                        "token": chunk.token,
                        "finished": False
                    })
                }
        else:
            # Обычный стриминг
            async for token in model.generate_stream(request.prompt):
                yield {
                    "data": json.dumps({"token": token, "finished": False})
                }
        yield {"data": json.dumps({"finished": True})}
    
    return EventSourceResponse(event_generator())
```

Для diffusion моделей:
- `/generate` принимает prompt и параметры изображения
- `/ws` - WebSocket для передачи промежуточных результатов
- Возвращает base64 изображение или URL
- Параметры: steps, guidance_scale, size, seed, scheduler

Для embedding моделей:
- `/generate` принимает тексты для векторизации
- Возвращает векторы и метаданные
- Поддержка батчей до 100 текстов
- Параметры: normalize, return_tensors

Для multimodal моделей:
- `/generate` принимает multipart/form-data
- Поддержка текста + изображений
- Стриминг для текстовых ответов
- Автоматическая обработка различных модальностей

**2.4 Обработка ошибок:**
- Нехватка памяти → понятное сообщение
- Неверный формат запроса → примеры правильного
- Ошибка модели → логирование + user-friendly ответ

### 3. model_loader.py - Универсальный загрузчик

**Назначение:** Модуль для загрузки любого типа модели

**Основные функции:**

**3.1 get_handler(model_info):**
- Определяет тип модели из model_info
- Импортирует соответствующий хандлер из инсталлятора
- Возвращает инстанс хандлера

**3.2 load_model(model_info, **kwargs):**
- Получает хандлер
- Вызывает handler.load_model() с параметрами
- Обрабатывает device placement
- Возвращает модель и дополнительные компоненты (tokenizer, processor)

**3.3 Обработка параметров:**
- dtype: auto-определение или явное указание
- device: auto, cuda, cpu, mps
- quantization: загрузка с квантованием если указано
- memory optimization: gradient checkpointing, cpu offload

**3.4 Поддержка reasoning моделей:**
Хандлер должен определять и обрабатывать модели с reasoning:
- Определение по model_type или специальным полям в config
- Разделение генерации на thinking и answer фазы
- Правильная обработка специальных токенов (если есть)
- Передача информации о поддержке в model_info.json

### 4. train.sh - Скрипт обучения

**Назначение:** Универсальный запуск обучения/fine-tuning

**Функциональность:**
- Активация окружения
- Проверка наличия данных
- Запуск train.py с параметрами
- Мониторинг процесса

**Поддерживаемые аргументы:**
- `--data PATH` - путь к данным для обучения
- `--output PATH` - куда сохранять чекпоинты
- `--method METHOD` - lora, qlora, full (default: lora)
- `--epochs N` - количество эпох
- `--batch-size N` - размер батча
- `--learning-rate LR` - learning rate

**Примечание:** Сам train.py будет реализован позже, сейчас train.sh может показывать заглушку.

### 5. serve_ui.html - Веб интерфейс (обязательный)

**Назначение:** Полнофункциональный веб UI для взаимодействия с моделью

**Архитектура:**
- Единый HTML файл с встроенным CSS и JavaScript
- Автоопределение типа модели через API
- Адаптивный интерфейс под тип модели
- Поддержка всех возможностей модели

**Функциональность по типам моделей:**

**Для языковых моделей:**
- Chat интерфейс с историей сообщений
- Стриминг ответов в реальном времени
- Настройки генерации (temperature, max_tokens, top_p)
- Поддержка system prompts
- Экспорт/импорт истории чата
- Счетчик токенов
- **Режим reasoning** (для моделей с поддержкой):
  - Отображение процесса рассуждений
  - Разделение thinking/answer
  - Счетчик reasoning токенов отдельно

**Для diffusion моделей:**
- Поле для prompt и negative prompt
- Настройки генерации (steps, guidance_scale, размер)
- Предпросмотр в реальном времени (если модель поддерживает)
- Галерея сгенерированных изображений
- Скачивание результатов
- Поддержка img2img (загрузка изображения)

**Для embedding моделей:**
- Поле для ввода текстов (поддержка батчей)
- Визуализация векторов (t-SNE/UMAP)
- Поиск по сходству
- Экспорт векторов в различных форматах

**Для multimodal моделей:**
- Загрузка изображений drag&drop
- Текстовый ввод
- Комбинированный вывод

**Технические требования:**
- WebSocket соединение для стриминга
- Graceful fallback на polling если WebSocket недоступен
- Responsive дизайн
- Темная/светлая тема
- Сохранение настроек в localStorage
- **Автоопределение возможностей через /info endpoint**

**Определение поддержки reasoning:**
```javascript
// При загрузке UI
fetch('/info')
  .then(res => res.json())
  .then(info => {
    if (info.capabilities.supports_reasoning) {
      // Показать checkbox "Reasoning mode"
      // Добавить отдельные области для thinking/answer
      // Включить счетчики для обеих частей
    }
  });
```

**Стриминг реализация:**
```javascript
// Для текстовых моделей - Server-Sent Events
const eventSource = new EventSource('/generate/stream');
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Обработка reasoning моделей
    if (data.type === 'thinking') {
        appendToThinking(data.token);
        updateThinkingTokenCount();
    } else if (data.type === 'answer') {
        appendToAnswer(data.token);
        updateAnswerTokenCount();
    } else {
        // Обычные модели
        appendToOutput(data.token);
    }
};

// Для diffusion - WebSocket для промежуточных результатов
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const preview = JSON.parse(event.data);
    updatePreview(preview.image);
};
```

## Используемые библиотеки

### Для API сервера (serve_api.py):
```
fastapi>=0.109.0         # Основной веб-фреймворк
uvicorn>=0.27.0          # ASGI сервер
websockets>=12.0         # WebSocket поддержка
sse-starlette>=2.0.0     # Server-Sent Events для стриминга
pydantic>=2.5.0          # Валидация данных
python-multipart>=0.0.6  # Для загрузки файлов
pillow>=10.0.0          # Работа с изображениями
numpy>=1.24.0           # Обработка данных
aiofiles>=23.0.0        # Асинхронная работа с файлами
```

### Для model_loader.py:
```
# Основные ML библиотеки (устанавливаются при install)
torch                    # PyTorch
transformers            # Hugging Face Transformers
diffusers               # Для diffusion моделей
accelerate              # Оптимизация загрузки
safetensors             # Безопасная загрузка весов
```

### Для веб-интерфейса (serve_ui.html):
```
# Встроенные в HTML (через CDN):
- Нет внешних зависимостей
- Чистый JavaScript (ES6+)
- CSS3 с CSS Grid/Flexbox
- Optional: можно добавить через CDN:
  - marked.js для рендеринга Markdown
  - highlight.js для подсветки кода
  - plotly.js для визуализации embeddings
```

### Дополнительные утилиты:
```
psutil>=5.9.0           # Мониторинг системных ресурсов
rich>=13.0.0            # Красивый вывод в консоль
python-dotenv>=1.0.0    # Загрузка переменных окружения
```

### Импорт хандлеров

Все скрипты должны уметь импортировать модули инсталлятора:
```python
import sys
import os

# Добавляем путь к инсталлятору
installer_path = os.path.expanduser("~/LLM/installer")
sys.path.insert(0, installer_path)

# Теперь можем импортировать
from handlers import get_handler
from core.utils import setup_logging
```

### Использование model_info.json

Структура model_info.json (создается при install):
```json
{
  "model_id": "meta-llama/Llama-3-8B",
  "model_type": "transformer",
  "model_family": "language-model",
  "architecture_type": "decoder-only",
  "primary_library": "transformers",
  "model_path": "./model",
  "install_date": "2024-01-01T12:00:00",
  "installer_version": "0.1.0",
  "requirements": {
    "dtype": "float16",
    "estimated_vram_gb": 16.0
  },
  "capabilities": {
    "supports_reasoning": false,  // true для o1-style моделей
    "supports_streaming": true,
    "supports_system_prompt": true,
    "max_context_length": 4096
  },
  "special_config": {
    // Специфичные параметры модели
  }
}
```

## Обработка ошибок

### Общие принципы:
1. Все ошибки логируются в `logs/server.log`
2. Пользователю показываются понятные сообщения
3. При критических ошибках - graceful shutdown

### Типичные ошибки:

**Нехватка памяти:**
```
Error: Out of memory
Required: 16.0 GB VRAM
Available: 8.0 GB VRAM

Try running with quantization:
./start.sh --load-in-8bit
```

**Отсутствие model_info.json:**
```
Error: model_info.json not found
This model was not properly installed.
Please run: llm-installer install <model-id>
```

**Несовместимый хандлер:**
```
Error: No handler found for model type: unknown
This model type is not yet supported.
```

## Логирование

### Структура логов:
```
logs/
├── server.log      # Основной лог API сервера
├── training.log    # Логи обучения (когда реализуем)
└── archive/        # Ротированные логи
```

### Формат логов:
```
2024-01-01 12:00:00 INFO: Starting API server
2024-01-01 12:00:01 INFO: Loading model from ./model
2024-01-01 12:00:15 INFO: Model loaded successfully
2024-01-01 12:00:16 INFO: Server running on http://0.0.0.0:8000
```

## Производительность

### Оптимизации:
1. Модель загружается один раз при старте
2. Кеширование токенизатора
3. Batch processing где возможно
4. Async endpoints для I/O операций

### Мониторинг:
- Endpoint `/metrics` для получения статистики
- Время генерации
- Использование памяти
- Количество запросов

## Обработка reasoning моделей

### Определение поддержки reasoning

Поддержка reasoning определяется на этапе `check` (Step 2) и сохраняется в `model_info.json`:
- По model_type (например, "o1", "reasoning-llm")
- По наличию специальных токенов в tokenizer
- По архитектуре модели
- По тегам на HuggingFace

### API адаптация для reasoning

Дополнительные параметры в запросах:
```json
{
  "prompt": "Solve this step by step",
  "reasoning_mode": true,
  "max_thinking_tokens": 10000,
  "max_answer_tokens": 2000,
  "show_thinking": true
}
```

### UI адаптация для reasoning

- Checkbox для включения reasoning mode
- Раздельное отображение thinking и answer
- Отдельные счетчики токенов
- Возможность скрыть/показать thinking
- Сохранение обеих частей при экспорте

## Тестирование скриптов

### Минимальный тест каждого скрипта:

**start.sh:**
- Запускается без ошибок
- API сервер поднимается
- Отвечает на /health
- Браузер открывается автоматически

**serve_api.py:**
- Загружает модель
- Обрабатывает базовый запрос
- Корректно обрабатывает ошибки
- Стриминг работает
- Reasoning mode работает (если поддерживается)

**model_loader.py:**
- Импортируется без ошибок
- Находит правильный хандлер
- Загружает тестовую модель
- Определяет capabilities правильно

### Минимальный тест каждого скрипта:

**start.sh:**
- Запускается без ошибок
- API сервер поднимается
- Отвечает на /health

**serve_api.py:**
- Загружает модель
- Обрабатывает базовый запрос
- Корректно обрабатывает ошибки

**model_loader.py:**
- Импортируется без ошибок
- Находит правильный хандлер
- Загружает тестовую модель

**Примеры интерфейса:**

Для Chat (LLM):
```
┌─────────────────────────────────────────┐
│ Model: Llama-3-8B    [Settings ⚙] [🌙]  │
├─────────────────────────────────────────┤
│ System: You are a helpful assistant     │
├─────────────────────────────────────────┤
│ User: Hello!                            │
│ Assistant: Hi! How can I help you? ▊    │
│                                         │
├─────────────────────────────────────────┤
│ [Message...                       ][Send]│
│ Tokens: 45/4096  Temp: 0.7  [Export 📥] │
└─────────────────────────────────────────┘
```

Для Chat с Reasoning:
```
┌─────────────────────────────────────────┐
│ Model: o1-mini    [Settings ⚙] [🌙]     │
├─────────────────────────────────────────┤
│ User: Solve 25 * 37 step by step       │
├─────────────────────────────────────────┤
│ 🤔 Thinking... (850 tokens)             │
│ ┌─[Show thinking]───────────────────┐   │
│ │ I need to multiply 25 by 37...    │   │
│ │ 25 * 30 = 750                     │   │
│ │ 25 * 7 = 175                      │   │
│ │ 750 + 175 = 925                   │   │
│ └───────────────────────────────────┘   │
│                                         │
│ Assistant: To solve 25 × 37:           │
│ 1. Break down: 25 × (30 + 7)           │
│ 2. Calculate: 750 + 175 = 925          │
│ Answer: 925                             │
├─────────────────────────────────────────┤
│ □ Reasoning mode  [Message...    ][Send]│
│ Thinking: 850  Answer: 75  [Export 📥]  │
└─────────────────────────────────────────┘
```

Для Image Generation:
```
┌─────────────────────────────────────────┐
│ Model: Stable-Diffusion-v1.5      [🌙]  │
├─────────────────────────────────────────┤
│ Prompt: [A beautiful landscape...     ] │
│ Negative: [blurry, low quality...     ] │
│                                         │
│ Steps: [====] 50   CFG: [===] 7.5      │
│ Size: [512x512 ▼]  Seed: [Random]      │
│                                         │
│ [Generate] [Interrupt] [Clear Gallery]  │
├─────────────────────────────────────────┤
│ [Generated Image Preview]               │
│ [Progress Bar: 45/50 steps]             │
│                                         │
│ Gallery: [img1][img2][img3][img4]       │
└─────────────────────────────────────────┘
```

1. Скрипты работают с любым типом модели
2. Корректно читают model_info.json
3. Успешно импортируют хандлеры из инсталлятора
4. API сервер запускается и отвечает на запросы
5. Веб-интерфейс полностью функционален для всех типов моделей
6. Стриминг работает корректно (SSE для текста, WebSocket для превью)
7. Ошибки обрабатываются gracefully
8. Логи пишутся в правильное место
9. Производительность приемлемая
10. Документированы все endpoints и параметры
11. Скрипты готовы к копированию при установке
12. Нет хардкода специфичного для конкретных моделей
13. UI автоматически адаптируется под возможности модели

# LLM Installer v2: Step 5 - Техническое задание для обучения моделей

## Общее описание

Реализация системы fine-tuning установленных моделей с использованием LoRA/QLoRA, автоматическим определением параметров и интеллектуальным алгоритмом предотвращения переобучения.

## Архитектура системы обучения

### Компоненты

1. **train.sh** - универсальный скрипт запуска обучения
2. **train_lora.py** - основной скрипт обучения с auto-learning алгоритмом
3. **dataset_manager.py** - универсальный загрузчик данных
4. **training_monitor.py** - мониторинг и визуализация

### Структура файлов после обучения

```
~/LLM/models/meta-llama_Llama-3-8B/
├── model/                      # Оригинальная модель
├── lora/                       # Активный LoRA адаптер
│   ├── adapter_config.json    # Конфигурация LoRA
│   ├── adapter_model.bin      # Веса адаптера
│   └── training_state.json    # Состояние для продолжения
├── checkpoints/                # Промежуточные сохранения
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   ├── best/                  # Лучшая модель по валидации
│   └── last/                  # Последний checkpoint
├── logs/
│   └── training/              # Логи для TensorBoard
│       ├── events.out.tfevents.xxx
│       └── training_report.json
└── training_history.json      # История всех обучений
```

## Детальная спецификация

### 1. train.sh - Скрипт запуска

**Функциональность:**
- Активация виртуального окружения
- Проверка наличия данных
- Запуск train_lora.py с параметрами
- Автоматический запуск TensorBoard
- Обработка прерываний (Ctrl+C)

**Поддерживаемые аргументы:**
```bash
# Основные
--data PATH                # Путь к данным (обязательный)
--output PATH             # Куда сохранять (default: ./lora)
--method [lora|qlora]     # Метод обучения (default: lora)

# Параметры обучения
--epochs N                # Максимум эпох (default: auto)
--batch-size N            # Размер батча (default: auto)
--learning-rate LR        # Learning rate (default: auto)
--lora-r N               # LoRA rank (default: auto)
--lora-alpha N           # LoRA alpha (default: 2*r)

# Режимы
--circular               # Circular training для маленьких датасетов
--resume                 # Продолжить с последнего checkpoint
--resume-from PATH       # Продолжить с конкретного checkpoint

# Мониторинг
--no-tensorboard         # Не запускать TensorBoard
--tensorboard-port PORT  # Порт для TensorBoard (default: 6006)

# Auto-learning параметры
--patience N             # Early stopping patience (default: 3)
--overfitting-threshold F # Порог переобучения (default: 0.1)
--min-evaluations N      # Минимум оценок перед проверкой (default: 5)
```

### 2. train_lora.py - Основной скрипт обучения

**2.1 Инициализация:**
```python
# Структура основного скрипта
1. Загрузка model_info.json
2. Определение оптимальных параметров
3. Инициализация модели и LoRA
4. Загрузка и подготовка данных
5. Настройка auto-learning алгоритма
6. Запуск обучения с мониторингом
```

**2.2 Автоматическое определение параметров:**

**LoRA Rank:**
```python
def get_optimal_lora_rank(model_size_b):
    if model_size_b < 1:
        return 8
    elif model_size_b < 7:
        return 16
    elif model_size_b < 13:
        return 32
    else:
        return 64
```

**Learning Rate:**
```python
def get_optimal_learning_rate(model_size_b, dataset_size):
    base_lr = 5e-5
    
    # Уменьшить для больших моделей
    if model_size_b > 7:
        base_lr *= 0.5
    if model_size_b > 13:
        base_lr *= 0.3
        
    # Увеличить для больших датасетов
    if dataset_size > 10000:
        base_lr *= 1.5
        
    return base_lr
```

**Batch Size:**
```python
def get_optimal_batch_size(available_memory_gb):
    # Эмпирические значения
    if available_memory_gb < 8:
        return 1
    elif available_memory_gb < 16:
        return 2
    elif available_memory_gb < 24:
        return 4
    else:
        return 8
```

### 3. data_loader.py - Универсальный загрузчик данных

**3.1 Поддерживаемые форматы:**

1. **Alpaca Format** (JSON):
```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  }
]
```

2. **ShareGPT Format** (JSON):
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Hello"},
      {"from": "gpt", "value": "Hi! How can I help?"}
    ]
  }
]
```

3. **OpenAI Chat Format** (JSON):
```json
[
  {
    "messages": [
      {"role": "system", "content": "You are helpful"},
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi!"}
    ]
  }
]
```

4. **Plain Text** (.txt):
```
Simply raw text for continued pretraining.
Each paragraph separated by newlines.
```

5. **CSV Format**:
```csv
question,answer
"What is 2+2?","4"
"Capital of France?","Paris"
```

6. **JSONL Format**:
```jsonl
{"prompt": "Hello", "completion": "Hi!"}
{"prompt": "How are you?", "completion": "I'm good!"}
```

**3.2 Автоматическое определение формата:**
```python
def detect_format(file_path):
    # По расширению
    ext = Path(file_path).suffix.lower()
    
    if ext == '.csv':
        return 'csv'
    elif ext == '.txt':
        return 'plain_text'
    elif ext in ['.json', '.jsonl']:
        # Анализ содержимого
        with open(file_path, 'r') as f:
            first_line = f.readline()
            # Логика определения по структуре
```

**3.3 Подготовка данных:**
- Автоматическое разделение на train/validation (90/10)
- Применение chat template из токенизатора
- Padding и truncation до max_length
- Создание attention masks

### 4. Auto-Learning Algorithm

**4.1 Основные компоненты:**

**Отслеживание метрик:**
```python
class TrainingMetrics:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.evaluations = 0
```

**Анализ тренда:**
```python
def calculate_trend(values, window=5):
    """Расчет тренда через линейную регрессию"""
    if len(values) < window:
        return 0.0
    
    recent = values[-window:]
    x = np.arange(len(recent))
    slope, _ = np.polyfit(x, recent, 1)
    
    return slope
```

**4.2 Логика принятия решений:**

**Определение переобучения:**
```python
def check_overfitting(metrics, config):
    # Недостаточно данных
    if metrics.evaluations < config.min_evaluations:
        return False, "insufficient_data"
    
    # Анализ тренда валидации
    val_trend = calculate_trend(metrics.val_losses)
    
    # Анализ разрыва train/val
    recent_train = np.mean(metrics.train_losses[-5:])
    recent_val = np.mean(metrics.val_losses[-5:])
    gap = (recent_val - recent_train) / recent_train
    
    # Решение
    if val_trend > 0.001 and gap > config.overfitting_threshold:
        if gap > 0.5:
            return True, "severe_overfitting"
        else:
            return True, "moderate_overfitting"
    
    return False, "no_overfitting"
```

**4.3 Визуальная обратная связь:**
```
📊 Evaluation #7 - Steps: 700/5000
├─ Train Loss: 0.5234 (↓ 0.0123)
├─ Val Loss: 0.6123 (↓ 0.0234)
├─ Trend: ↓ Improving
├─ Gap: 0.089 (17%)
└─ Decision: ✅ Continue training

💾 Saved checkpoint: checkpoints/checkpoint-700
```

### 5. Интеграция с TensorBoard

**5.1 Автоматический запуск:**
```bash
# В train.sh
if [ "$NO_TENSORBOARD" != "true" ]; then
    echo "Starting TensorBoard on port $TB_PORT..."
    tensorboard --logdir ./logs/training --port $TB_PORT &
    TB_PID=$!
    echo "TensorBoard: http://localhost:$TB_PORT"
fi
```

**5.2 Логируемые метрики:**
- loss/train
- loss/validation
- learning_rate
- overfitting_score
- gradient_norm
- tokens_per_second

**5.3 Визуализации:**
- График train vs validation loss
- Learning rate schedule
- Gradient flow
- Attention weights (опционально)

### 6. Обработка checkpoint'ов

**6.1 Сохранение состояния:**
```json
{
  "epoch": 3,
  "global_step": 1500,
  "best_val_loss": 0.523,
  "optimizer_state": "optimizer.pt",
  "scheduler_state": {...},
  "training_args": {...},
  "metrics_history": [...]
}
```

**6.2 Продолжение обучения:**
- Загрузка состояния модели
- Восстановление optimizer и scheduler
- Продолжение с правильного шага
- Добавление к существующей истории

### 7. Circular Training

**Для маленьких датасетов:**
- Данные повторяются циклически
- Более толерантные пороги переобучения
- Специальная логика для early stopping
- Трекинг количества полных проходов

**Активация:**
```bash
./train.sh --data small_dataset.json --circular
```

### 8. Интеграция с start.sh

**Автоматическая загрузка LoRA:**
```bash
# В start.sh
if [ -f "./lora/adapter_model.bin" ]; then
    echo "Found LoRA adapter, loading with modifications..."
    EXTRA_ARGS="--load-lora ./lora"
fi
```

### 9. Отчет о тренировке

**training_report.json:**
```json
{
  "model_id": "meta-llama/Llama-3-8B",
  "training_completed": "2024-01-15T10:30:00",
  "duration_hours": 2.5,
  "final_metrics": {
    "train_loss": 0.234,
    "val_loss": 0.456,
    "best_val_loss": 0.445,
    "best_epoch": 15
  },
  "stop_reason": "early_stopping_patience",
  "lora_config": {
    "r": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "v_proj"]
  },
  "dataset_info": {
    "format": "alpaca",
    "train_examples": 9000,
    "val_examples": 1000
  }
}
```

## Обработка ошибок

### Типичные ошибки и решения:

**Out of Memory:**
```
Error: CUDA out of memory

Suggestions:
1. Reduce batch size: --batch-size 1
2. Use gradient accumulation: --gradient-accumulation 4
3. Enable QLoRA: --method qlora
4. Reduce LoRA rank: --lora-r 8
```

**Неподдерживаемый формат данных:**
```
Error: Unable to detect data format

Supported formats:
- Alpaca JSON: [{"instruction": ..., "output": ...}]
- ShareGPT: [{"conversations": [...]}]
- Plain text: .txt files
- CSV: question,answer format

Please check your data format.
```

**Diverging Loss:**
```
Warning: Loss is increasing rapidly!

Possible causes:
1. Learning rate too high
2. Bad data quality
3. Model instability

Automatically reducing learning rate...
```

## Критерии успешной реализации

1. Универсальный загрузчик работает со всеми форматами
2. Auto-learning предотвращает переобучение
3. TensorBoard запускается автоматически
4. Параметры определяются автоматически но переопределяются
5. Checkpoint'ы сохраняются и восстанавливаются корректно
6. LoRA автоматически загружается при запуске модели
7. Circular training работает для маленьких датасетов
8. Ошибки обрабатываются с полезными подсказками
9. Отчеты генерируются автоматически
10. Производительность приемлемая для разных размеров моделей