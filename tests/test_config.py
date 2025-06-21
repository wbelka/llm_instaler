"""Tests for configuration module."""

import pytest
from pathlib import Path
import tempfile
import yaml

from core.config import Config, ConfigError, get_config, reset_config


def test_config_load_example():
    """Test loading the example configuration file."""
    # Use the example config file
    config_path = Path(__file__).parent.parent / "config.yaml.example"

    # Load config
    config = Config(config_path)

    # Check basic values
    assert config.models_dir == Path.home() / "LLM" / "models"
    assert config.cache_dir == Path.home() / "LLM" / "cache"
    assert config.logs_dir == Path.home() / "LLM" / "logs"
    assert config.default_device == "auto"
    assert config.max_download_workers == 4
    assert config.resume_downloads is True
    assert config.min_disk_space_gb == 50
    assert config.warn_disk_space_gb == 100
    assert config.log_level == "INFO"
    assert config.log_rotation == "10MB"


def test_config_missing_required_field():
    """Test that missing required fields raise ConfigError."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'models_dir': '~/models'}, f)  # Missing other required fields
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigError) as exc_info:
            Config(temp_path)
        assert "Required configuration field missing" in str(exc_info.value)
    finally:
        temp_path.unlink()


def test_config_invalid_device():
    """Test that invalid device values raise ConfigError."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'models_dir': '~/models',
            'cache_dir': '~/cache',
            'logs_dir': '~/logs',
            'default_device': 'invalid_device',  # Invalid
            'max_download_workers': 4,
            'min_disk_space_gb': 50,
            'log_level': 'INFO'
        }, f)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigError) as exc_info:
            Config(temp_path)
        assert "default_device must be one of" in str(exc_info.value)
    finally:
        temp_path.unlink()


def test_config_path_expansion():
    """Test that paths are properly expanded."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'models_dir': '~/test/models',
            'cache_dir': '$HOME/test/cache',
            'logs_dir': '~/test/logs',
            'default_device': 'auto',
            'max_download_workers': 4,
            'min_disk_space_gb': 50,
            'log_level': 'INFO'
        }, f)
        temp_path = Path(f.name)

    try:
        config = Config(temp_path)

        # Check that ~ is expanded
        assert str(config.models_dir).startswith(str(Path.home()))
        assert str(config.cache_dir).startswith(str(Path.home()))

        # Check that paths end with expected values
        assert config.models_dir.name == 'models'
        assert config.cache_dir.name == 'cache'
        assert config.logs_dir.name == 'logs'
    finally:
        temp_path.unlink()


def test_huggingface_token_priority(monkeypatch):
    """Test that HF_TOKEN environment variable takes priority."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'huggingface_token': 'config_token',
            'models_dir': '~/models',
            'cache_dir': '~/cache',
            'logs_dir': '~/logs',
            'default_device': 'auto',
            'max_download_workers': 4,
            'min_disk_space_gb': 50,
            'log_level': 'INFO'
        }, f)
        temp_path = Path(f.name)

    try:
        # Test without environment variable
        config = Config(temp_path)
        assert config.huggingface_token == 'config_token'

        # Test with environment variable
        monkeypatch.setenv('HF_TOKEN', 'env_token')
        config = Config(temp_path)
        assert config.huggingface_token == 'env_token'
    finally:
        temp_path.unlink()


def test_get_config_singleton():
    """Test that get_config returns the same instance."""
    reset_config()  # Ensure clean state

    config1 = get_config()
    config2 = get_config()

    assert config1 is config2


def test_config_to_dict():
    """Test converting config to dictionary."""
    config_path = Path(__file__).parent.parent / "config.yaml.example"
    config = Config(config_path)

    config_dict = config.to_dict()

    # Check that dict contains expected keys
    assert 'models_dir' in config_dict
    assert 'cache_dir' in config_dict
    assert 'logs_dir' in config_dict
    assert 'huggingface_token' in config_dict

    # Check that paths are strings
    assert isinstance(config_dict['models_dir'], str)
    assert isinstance(config_dict['cache_dir'], str)
