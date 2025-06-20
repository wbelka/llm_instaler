"""Utility functions for LLM Installer.

This module provides common utilities for logging, system checks,
formatting, and file operations.
"""

import os
import sys
import logging
import shutil
import psutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from logging.handlers import RotatingFileHandler
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.syntax import Syntax

from core.config import get_config


# Global console for rich output
console = Console()


def setup_logging(log_file: Optional[str] = None, debug_mode: bool = False) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_file: Optional specific log file path.
        debug_mode: Enable debug logging if True.
        
    Returns:
        Configured logger instance.
    """
    config = get_config()
    
    # Determine log level
    if debug_mode:
        log_level = logging.DEBUG
    else:
        log_level_str = config.log_level
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger('llm_installer')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler with detailed format
    if log_file is None:
        # Use default log file
        log_dir = config.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"llm_installer_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Parse rotation size
    rotation_str = config.log_rotation
    if rotation_str.endswith('MB'):
        max_bytes = int(rotation_str[:-2]) * 1024 * 1024
    elif rotation_str.endswith('KB'):
        max_bytes = int(rotation_str[:-2]) * 1024
    else:
        max_bytes = 10 * 1024 * 1024  # Default 10MB
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements and capabilities.
    
    Returns:
        Dictionary with system information.
    """
    system_info = {
        'os': sys.platform,
        'python_version': sys.version.split()[0],
        'cpu_count': psutil.cpu_count(),
        'total_memory_gb': psutil.virtual_memory().total / (1024**3),
        'available_memory_gb': psutil.virtual_memory().available / (1024**3),
        'disk_space_gb': {},
        'cuda_available': False,
        'cuda_version': None,
        'gpu_info': []
    }
    
    # Check disk space for home directory
    home_stat = shutil.disk_usage(Path.home())
    system_info['disk_space_gb']['home'] = {
        'total': home_stat.total / (1024**3),
        'free': home_stat.free / (1024**3)
    }
    
    # Check CUDA availability
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        system_info['cuda_available'] = True
        
        # Parse GPU info
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                if len(parts) >= 2:
                    system_info['gpu_info'].append({
                        'name': parts[0],
                        'memory_mb': int(parts[1].replace(' MiB', ''))
                    })
        
        # Get CUDA version
        cuda_result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            check=True
        )
        for line in cuda_result.stdout.split('\n'):
            if 'CUDA Version' in line:
                cuda_version = line.split('CUDA Version:')[1].split()[0]
                system_info['cuda_version'] = cuda_version
                break
                
    except (subprocess.CalledProcessError, FileNotFoundError):
        # nvidia-smi not available or failed
        pass
    
    # Check for Apple Silicon
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.optional.arm64'],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip() == '1':
                system_info['apple_silicon'] = True
                system_info['gpu_info'].append({
                    'name': 'Apple Silicon GPU',
                    'memory_mb': 'shared'
                })
        except subprocess.CalledProcessError:
            system_info['apple_silicon'] = False
    
    return system_info


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        Formatted string (e.g., "1.5 GB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted string (e.g., "1h 23m 45s").
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def safe_model_name(model_id: str) -> str:
    """Convert model ID to safe directory name.
    
    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3-8B").
        
    Returns:
        Safe name for directory (e.g., "meta-llama_Llama-3-8B").
    """
    return model_id.replace('/', '_')


def print_system_info(system_info: Dict[str, Any]) -> None:
    """Print system information in a formatted table.
    
    Args:
        system_info: Dictionary from check_system_requirements().
    """
    table = Table(title="System Information", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    # OS and Python
    os_name = {
        'linux': 'Linux',
        'darwin': 'macOS',
        'win32': 'Windows'
    }.get(system_info['os'], system_info['os'])
    
    table.add_row("Operating System", os_name)
    table.add_row("Python Version", system_info['python_version'])
    
    # Hardware
    table.add_row("CPU Cores", str(system_info['cpu_count']))
    table.add_row("Total Memory", f"{system_info['total_memory_gb']:.1f} GB")
    table.add_row("Available Memory", f"{system_info['available_memory_gb']:.1f} GB")
    
    # Disk space
    home_space = system_info['disk_space_gb']['home']
    table.add_row("Disk Space (Free)", f"{home_space['free']:.1f} GB")
    
    # GPU
    if system_info['cuda_available']:
        table.add_row("CUDA Version", system_info['cuda_version'] or "Unknown")
        for i, gpu in enumerate(system_info['gpu_info']):
            gpu_mem = f"{gpu['memory_mb']} MB" if isinstance(gpu['memory_mb'], int) else gpu['memory_mb']
            table.add_row(f"GPU {i}", f"{gpu['name']} ({gpu_mem})")
    elif system_info.get('apple_silicon'):
        table.add_row("GPU", "Apple Silicon (Metal)")
    else:
        table.add_row("GPU", "Not available")
    
    console.print(table)


def create_progress_bar() -> Progress:
    """Create a rich progress bar for downloads and operations.
    
    Returns:
        Configured Progress instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    )


def print_error(message: str, exception: Optional[Exception] = None) -> None:
    """Print an error message with optional exception details.
    
    Args:
        message: Error message to display.
        exception: Optional exception to include.
    """
    error_panel = Panel(
        message,
        title="[red]Error[/red]",
        border_style="red"
    )
    console.print(error_panel)
    
    if exception and console.is_terminal:
        console.print(f"\n[dim]Details: {str(exception)}[/dim]")


def print_warning(message: str) -> None:
    """Print a warning message.
    
    Args:
        message: Warning message to display.
    """
    console.print(f"[yellow]   Warning:[/yellow] {message}")


def print_success(message: str) -> None:
    """Print a success message.
    
    Args:
        message: Success message to display.
    """
    console.print(f"[green][/green] {message}")


def print_info(message: str) -> None:
    """Print an info message.
    
    Args:
        message: Info message to display.
    """
    console.print(f"[blue]9[/blue]  {message}")


def confirm_action(prompt: str, default: bool = False) -> bool:
    """Ask user for confirmation.
    
    Args:
        prompt: Question to ask.
        default: Default answer if user just presses Enter.
        
    Returns:
        True if user confirms, False otherwise.
    """
    if default:
        prompt += " [Y/n]"
        valid_yes = ['y', 'yes', '']
    else:
        prompt += " [y/N]"
        valid_yes = ['y', 'yes']
    
    response = console.input(f"{prompt}: ").lower().strip()
    return response in valid_yes


def run_command(
    command: List[str],
    description: str,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None
) -> Tuple[bool, str]:
    """Run a shell command with progress indication.
    
    Args:
        command: Command and arguments as list.
        description: Description for progress display.
        cwd: Working directory for command.
        env: Environment variables.
        
    Returns:
        Tuple of (success, output).
    """
    with console.status(description):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env,
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr or str(e)


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists, creating if necessary.
    
    Args:
        path: Directory path to ensure exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def get_cache_dir() -> Path:
    """Get the cache directory path.
    
    Returns:
        Path to cache directory.
    """
    config = get_config()
    cache_dir = config.cache_dir
    ensure_directory(cache_dir)
    return cache_dir


def get_models_dir() -> Path:
    """Get the models directory path.
    
    Returns:
        Path to models directory.
    """
    config = get_config()
    models_dir = config.models_dir
    ensure_directory(models_dir)
    return models_dir


def calculate_model_size(files_info: List[Dict[str, Any]]) -> float:
    """Calculate total size of model files in GB.
    
    Args:
        files_info: List of file information dictionaries.
        
    Returns:
        Total size in GB.
    """
    total_bytes = sum(f.get('size', 0) for f in files_info)
    return total_bytes / (1024**3)