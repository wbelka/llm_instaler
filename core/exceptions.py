"""Custom exceptions for LLM Installer.

This module defines specific exception types to replace broad exception handling
and provide better error context.
"""


class LLMInstallerError(Exception):
    """Base exception for all LLM Installer errors."""
    pass


class ModelNotFoundError(LLMInstallerError):
    """Raised when a model cannot be found on HuggingFace."""
    pass


class ModelAccessError(LLMInstallerError):
    """Raised when model access is denied (e.g., gated models)."""
    pass


class CompatibilityError(LLMInstallerError):
    """Raised when model is not compatible with the system."""
    pass


class InstallationError(LLMInstallerError):
    """Raised when model installation fails."""
    pass


class DependencyError(InstallationError):
    """Raised when dependency installation fails."""
    pass


class VirtualEnvironmentError(InstallationError):
    """Raised when virtual environment creation fails."""
    pass


class DownloadError(InstallationError):
    """Raised when model download fails."""
    pass


class DiskSpaceError(InstallationError):
    """Raised when there's insufficient disk space."""
    pass


class HandlerError(LLMInstallerError):
    """Raised when handler operations fail."""
    pass


class HandlerNotFoundError(HandlerError):
    """Raised when no handler is available for a model type."""
    pass


class ConfigurationError(LLMInstallerError):
    """Raised when configuration is invalid or missing."""
    pass


class ValidationError(LLMInstallerError):
    """Raised when input validation fails."""
    pass


class NetworkError(LLMInstallerError):
    """Raised when network operations fail."""
    pass


class CUDAError(LLMInstallerError):
    """Raised when CUDA-related operations fail."""
    pass


class QuantizationError(LLMInstallerError):
    """Raised when quantization operations fail."""
    pass


class ModelLoadError(LLMInstallerError):
    """Raised when model loading fails."""
    pass


class GenerationError(LLMInstallerError):
    """Raised when model generation fails."""
    pass


class TokenizerError(LLMInstallerError):
    """Raised when tokenizer operations fail."""
    pass