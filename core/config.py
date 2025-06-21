"""Configuration management module for LLM Installer.

This module handles loading and validation of configuration from YAML files
and environment variables, with proper path expansion and type checking.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


class Config:
    """Configuration manager for LLM Installer.

    Handles loading configuration from YAML files, environment variables,
    and provides validated access to all settings.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, looks for
                        config.yaml in the installer directory.
        """
        self.config_path = config_path or self._find_config_file()
        self._config = self._load_config()
        self._validate_config()

    def _find_config_file(self) -> Path:
        """Find configuration file in standard locations.

        Returns:
            Path to configuration file.

        Raises:
            ConfigError: If no configuration file is found.
        """
        # Try current directory first
        local_config = Path("config.yaml")
        if local_config.exists():
            return local_config

        # Try installer directory
        installer_dir = Path(__file__).parent.parent
        installer_config = installer_dir / "config.yaml"
        if installer_config.exists():
            return installer_config

        # Try example file as fallback
        example_config = installer_dir / "config.yaml.example"
        if example_config.exists():
            return example_config

        raise ConfigError(
            "No configuration file found. Please create config.yaml "
            "from config.yaml.example"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary with configuration values.

        Raises:
            ConfigError: If file cannot be loaded or parsed.
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
                return config
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in configuration file: {e}")
        except IOError as e:
            raise ConfigError(f"Cannot read configuration file: {e}")

    def _validate_config(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigError: If configuration is invalid.
        """
        # Required fields
        required_fields = [
            'models_dir', 'cache_dir', 'logs_dir',
            'default_device', 'max_download_workers',
            'min_disk_space_gb', 'log_level'
        ]

        for field in required_fields:
            if field not in self._config:
                raise ConfigError(f"Required configuration field missing: {field}")

        # Validate types
        if not isinstance(self._config['max_download_workers'], int):
            raise ConfigError("max_download_workers must be an integer")

        if not isinstance(self._config['min_disk_space_gb'], (int, float)):
            raise ConfigError("min_disk_space_gb must be a number")

        # Validate values
        valid_devices = ['auto', 'cuda', 'cpu', 'mps']
        if self._config['default_device'] not in valid_devices:
            raise ConfigError(
                f"default_device must be one of: {', '.join(valid_devices)}"
            )

        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if self._config['log_level'] not in valid_log_levels:
            raise ConfigError(
                f"log_level must be one of: {', '.join(valid_log_levels)}"
            )

    def _expand_path(self, path: str) -> Path:
        """Expand user home directory and environment variables in path.

        Args:
            path: Path string that may contain ~ or environment variables.

        Returns:
            Expanded Path object.
        """
        # Expand ~ to home directory
        expanded = os.path.expanduser(path)
        # Expand environment variables
        expanded = os.path.expandvars(expanded)
        return Path(expanded)

    @property
    def huggingface_token(self) -> Optional[str]:
        """Get HuggingFace token with environment variable priority.

        Returns:
            Token string or None if not configured.
        """
        # Environment variable takes priority
        env_token = os.environ.get('HF_TOKEN')
        if env_token:
            return env_token

        # Fall back to config file
        return self._config.get('huggingface_token') or None

    @property
    def models_dir(self) -> Path:
        """Get models directory path.

        Returns:
            Expanded Path object for models directory.
        """
        return self._expand_path(self._config['models_dir'])

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path.

        Returns:
            Expanded Path object for cache directory.
        """
        return self._expand_path(self._config['cache_dir'])

    @property
    def logs_dir(self) -> Path:
        """Get logs directory path.

        Returns:
            Expanded Path object for logs directory.
        """
        return self._expand_path(self._config['logs_dir'])

    @property
    def default_device(self) -> str:
        """Get default device setting.

        Returns:
            Device string (auto, cuda, cpu, mps).
        """
        return self._config['default_device']

    @property
    def max_download_workers(self) -> int:
        """Get maximum number of download workers.

        Returns:
            Number of workers for parallel downloads.
        """
        return self._config['max_download_workers']

    @property
    def resume_downloads(self) -> bool:
        """Check if download resumption is enabled.

        Returns:
            True if downloads should be resumed.
        """
        return self._config.get('resume_downloads', True)

    @property
    def min_disk_space_gb(self) -> float:
        """Get minimum required disk space in GB.

        Returns:
            Minimum disk space requirement.
        """
        return float(self._config['min_disk_space_gb'])

    @property
    def warn_disk_space_gb(self) -> float:
        """Get disk space warning threshold in GB.

        Returns:
            Disk space warning threshold.
        """
        return float(self._config.get('warn_disk_space_gb', 100))

    @property
    def log_level(self) -> str:
        """Get logging level.

        Returns:
            Logging level string.
        """
        return self._config['log_level']

    @property
    def log_rotation(self) -> str:
        """Get log rotation setting.

        Returns:
            Log rotation size string (e.g., "10MB").
        """
        return self._config.get('log_rotation', '10MB')

    @property
    def version(self) -> str:
        """Get installer version.

        Returns:
            Version string.
        """
        return "2.0.0"

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary.

        Returns:
            Dictionary with all configuration values.
        """
        # Create a copy with expanded paths
        result = self._config.copy()

        # Add computed properties
        result['huggingface_token'] = self.huggingface_token
        result['models_dir'] = str(self.models_dir)
        result['cache_dir'] = str(self.cache_dir)
        result['logs_dir'] = str(self.logs_dir)

        return result


# Global configuration instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance.

    Args:
        config_path: Optional path to configuration file.

    Returns:
        Global Config instance.
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_path)

    return _config_instance


def reset_config() -> None:
    """Reset global configuration instance.

    Used mainly for testing.
    """
    global _config_instance
    _config_instance = None
