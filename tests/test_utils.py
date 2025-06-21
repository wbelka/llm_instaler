"""Tests for utility functions."""

import pytest
from pathlib import Path

from core.utils import (
    format_size, format_time, safe_model_name,
    check_system_requirements, calculate_model_size
)


def test_format_size():
    """Test size formatting function."""
    assert format_size(0) == "0.0 B"
    assert format_size(1024) == "1.0 KB"
    assert format_size(1024 * 1024) == "1.0 MB"
    assert format_size(1024 * 1024 * 1024) == "1.0 GB"
    assert format_size(1536 * 1024 * 1024) == "1.5 GB"
    assert format_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"


def test_format_time():
    """Test time formatting function."""
    assert format_time(0) == "0s"
    assert format_time(30) == "30s"
    assert format_time(60) == "1m 0s"
    assert format_time(90) == "1m 30s"
    assert format_time(3600) == "1h 0m"
    assert format_time(3661) == "1h 1m"
    assert format_time(7320) == "2h 2m"


def test_safe_model_name():
    """Test model name sanitization."""
    assert safe_model_name("meta-llama/Llama-3-8B") == "meta-llama_Llama-3-8B"
    assert safe_model_name("facebook/bart-large") == "facebook_bart-large"
    assert safe_model_name("model-without-slash") == "model-without-slash"
    assert safe_model_name("org/model/variant") == "org_model_variant"


def test_check_system_requirements():
    """Test system requirements checking."""
    system_info = check_system_requirements()

    # Check that all expected keys are present
    assert 'os' in system_info
    assert 'python_version' in system_info
    assert 'cpu_count' in system_info
    assert 'total_memory_gb' in system_info
    assert 'available_memory_gb' in system_info
    assert 'disk_space_gb' in system_info
    assert 'cuda_available' in system_info
    assert 'gpu_info' in system_info

    # Check types
    assert isinstance(system_info['cpu_count'], int)
    assert isinstance(system_info['total_memory_gb'], float)
    assert isinstance(system_info['available_memory_gb'], float)
    assert isinstance(system_info['cuda_available'], bool)
    assert isinstance(system_info['gpu_info'], list)

    # Check disk space structure
    assert 'home' in system_info['disk_space_gb']
    assert 'total' in system_info['disk_space_gb']['home']
    assert 'free' in system_info['disk_space_gb']['home']


def test_calculate_model_size():
    """Test model size calculation."""
    # Test with empty list
    assert calculate_model_size([]) == 0.0

    # Test with single file
    files = [{'size': 1024 * 1024 * 1024}]  # 1 GB
    assert calculate_model_size(files) == 1.0

    # Test with multiple files
    files = [
        {'size': 1024 * 1024 * 1024},  # 1 GB
        {'size': 512 * 1024 * 1024},   # 0.5 GB
        {'size': 256 * 1024 * 1024}    # 0.25 GB
    ]
    assert calculate_model_size(files) == 1.75

    # Test with missing size field
    files = [
        {'size': 1024 * 1024 * 1024},
        {'name': 'file_without_size.txt'}
    ]
    assert calculate_model_size(files) == 1.0
