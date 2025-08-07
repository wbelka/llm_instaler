# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- CPU offloading support via `--cpu-offload` flag for running large models
- Centralized dependency management in `core/dependencies.py`
- Improved memory management with configurable GPU/CPU allocation
- English documentation and code comments

### Changed
- Refactored handlers to use centralized dependency system
- Updated `get_quantization_config` to support CPU offload parameter
- Enhanced `get_memory_config` with CPU offload strategies
- Translated all documentation to English

### Fixed
- Missing `accelerate` dependency in refactored code
- Improved error handling for missing dependencies

## [2.0.0] - Previous Release

### Added
- Model compatibility checking without downloading weights
- Automatic installation with isolated environments
- Support for multiple model types (transformers, diffusers, multimodal)
- Quantization support (8-bit, 4-bit)
- Web UI and API server
- Training and fine-tuning support
- Handler system for different model architectures

### Security
- Input validation and sanitization in API
- Rate limiting
- CORS configuration
- Secure default settings