"""Secure subprocess execution utilities.

This module provides secure wrappers for subprocess execution to prevent
command injection and other security issues.
"""

import subprocess
import shlex
import logging
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import re

logger = logging.getLogger(__name__)


class SecureSubprocessError(Exception):
    """Raised when subprocess execution fails or is insecure."""
    pass


def validate_command(command: List[str]) -> None:
    """Validate command arguments for security.
    
    Args:
        command: Command as list of arguments
        
    Raises:
        SecureSubprocessError: If command contains unsafe elements
    """
    if not command:
        raise SecureSubprocessError("Empty command")
    
    # Check for shell metacharacters that could lead to injection
    dangerous_chars = set(';&|<>$`(){}[]!*?~')
    
    for arg in command:
        if not isinstance(arg, str):
            raise SecureSubprocessError(f"Command argument must be string, got {type(arg)}")
        
        # Check for dangerous characters (except in paths which might contain spaces)
        if any(char in arg for char in dangerous_chars):
            # Allow certain patterns that are safe
            if not (arg.startswith('--') or arg.startswith('-') or Path(arg).exists()):
                raise SecureSubprocessError(f"Potentially dangerous command argument: {arg}")


def validate_pip_package(package: str) -> bool:
    """Validate a pip package specification.
    
    Args:
        package: Package specification (e.g., 'torch==2.0.0', 'numpy>=1.20')
        
    Returns:
        True if valid, False otherwise
    """
    # Pattern for valid package names according to PEP 508
    # Package name can contain letters, numbers, hyphens, underscores, and dots
    # Version specifier can contain ==, >=, <=, >, <, ~=, !=
    pattern = r'^[a-zA-Z0-9\-_.]+(\[[\w,]+\])?(([<>=~!]=?[\d\w.*+\-]+)(,([<>=~!]=?[\d\w.*+\-]+))*)?$'
    
    if not re.match(pattern, package):
        logger.warning(f"Invalid package specification: {package}")
        return False
    
    return True


def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """Validate and normalize a file path.
    
    Args:
        path: Path to validate
        must_exist: If True, path must exist
        
    Returns:
        Normalized Path object
        
    Raises:
        SecureSubprocessError: If path is invalid
    """
    try:
        path_obj = Path(path).resolve()
        
        # Check for path traversal attempts
        if '..' in str(path):
            raise SecureSubprocessError(f"Path traversal detected: {path}")
        
        if must_exist and not path_obj.exists():
            raise SecureSubprocessError(f"Path does not exist: {path}")
        
        return path_obj
        
    except Exception as e:
        raise SecureSubprocessError(f"Invalid path: {path} - {e}")


def secure_run(
    command: List[str],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    check: bool = True,
    timeout: Optional[int] = None,
    **kwargs
) -> subprocess.CompletedProcess:
    """Securely run a subprocess command.
    
    Args:
        command: Command as list of arguments (no shell expansion)
        cwd: Working directory (will be validated)
        env: Environment variables (will be sanitized)
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise on non-zero exit
        timeout: Command timeout in seconds
        **kwargs: Additional arguments for subprocess.run
        
    Returns:
        CompletedProcess instance
        
    Raises:
        SecureSubprocessError: If command is invalid or execution fails
    """
    # Validate command
    validate_command(command)
    
    # Validate working directory if provided
    if cwd is not None:
        cwd = validate_path(cwd, must_exist=True)
    
    # Sanitize environment if provided
    if env is not None:
        # Create a clean environment with only safe variables
        safe_env = {}
        dangerous_vars = {'LD_PRELOAD', 'LD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES'}
        
        for key, value in env.items():
            if key in dangerous_vars:
                logger.warning(f"Skipping potentially dangerous environment variable: {key}")
                continue
            
            # Ensure key and value are strings without shell metacharacters
            if not isinstance(key, str) or not isinstance(value, str):
                logger.warning(f"Skipping non-string environment variable: {key}")
                continue
                
            if any(char in key + value for char in ';&|<>$`(){}[]'):
                logger.warning(f"Skipping environment variable with shell metacharacters: {key}")
                continue
                
            safe_env[key] = value
        
        env = safe_env
    
    # Never use shell=True
    if kwargs.get('shell', False):
        raise SecureSubprocessError("Shell execution is not allowed for security reasons")
    
    kwargs['shell'] = False
    
    # Set default timeout if not specified
    if timeout is None:
        timeout = 300  # 5 minutes default
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            check=check,
            timeout=timeout,
            text=True,
            **kwargs
        )
        
        return result
        
    except subprocess.TimeoutExpired as e:
        raise SecureSubprocessError(f"Command timed out after {timeout} seconds: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        raise SecureSubprocessError(
            f"Command failed with exit code {e.returncode}: {' '.join(command)}\n"
            f"stdout: {e.stdout}\nstderr: {e.stderr}"
        )
    except Exception as e:
        raise SecureSubprocessError(f"Command execution failed: {e}")


def secure_pip_install(
    pip_path: Union[str, Path],
    packages: List[str],
    index_url: Optional[str] = None,
    upgrade: bool = False,
    timeout: int = 600
) -> subprocess.CompletedProcess:
    """Securely install pip packages.
    
    Args:
        pip_path: Path to pip executable
        packages: List of package specifications
        index_url: Optional PyPI index URL
        upgrade: Whether to upgrade packages
        timeout: Installation timeout in seconds
        
    Returns:
        CompletedProcess instance
        
    Raises:
        SecureSubprocessError: If installation fails or packages are invalid
    """
    # Validate pip path
    pip_path = validate_path(pip_path, must_exist=True)
    
    # Validate all packages
    valid_packages = []
    for package in packages:
        if validate_pip_package(package):
            valid_packages.append(package)
        else:
            raise SecureSubprocessError(f"Invalid package specification: {package}")
    
    if not valid_packages:
        raise SecureSubprocessError("No valid packages to install")
    
    # Build command
    command = [str(pip_path), "install"]
    
    if upgrade:
        command.append("--upgrade")
    
    # Validate index URL if provided
    if index_url:
        # Only allow HTTPS URLs from trusted domains
        trusted_domains = [
            'https://pypi.org/',
            'https://download.pytorch.org/',
            'https://pypi.python.org/',
            'https://test.pypi.org/'
        ]
        
        if not any(index_url.startswith(domain) for domain in trusted_domains):
            raise SecureSubprocessError(f"Untrusted index URL: {index_url}")
        
        command.extend(["--index-url", index_url])
    
    command.extend(valid_packages)
    
    # Run installation - packages are already validated, so use subprocess directly
    # to avoid double validation of package specifications with version operators
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            timeout=timeout,
            text=True,
            shell=False  # Never use shell
        )
        return result
    except subprocess.TimeoutExpired as e:
        raise SecureSubprocessError(f"Pip install timed out after {timeout} seconds")
    except subprocess.CalledProcessError as e:
        raise SecureSubprocessError(
            f"Pip install failed with exit code {e.returncode}\n"
            f"stdout: {e.stdout}\nstderr: {e.stderr}"
        )
    except Exception as e:
        raise SecureSubprocessError(f"Pip install failed: {e}")


def secure_python_run(
    python_path: Union[str, Path],
    script_path: Union[str, Path],
    args: Optional[List[str]] = None,
    cwd: Optional[Union[str, Path]] = None,
    timeout: int = 300
) -> subprocess.CompletedProcess:
    """Securely run a Python script.
    
    Args:
        python_path: Path to Python executable
        script_path: Path to Python script
        args: Optional script arguments
        cwd: Working directory
        timeout: Execution timeout in seconds
        
    Returns:
        CompletedProcess instance
        
    Raises:
        SecureSubprocessError: If execution fails
    """
    # Validate paths
    python_path = validate_path(python_path, must_exist=True)
    script_path = validate_path(script_path, must_exist=True)
    
    # Build command
    command = [str(python_path), str(script_path)]
    
    # Add arguments if provided
    if args:
        # Validate each argument
        for arg in args:
            if not isinstance(arg, str):
                raise SecureSubprocessError(f"Script argument must be string, got {type(arg)}")
            command.append(arg)
    
    # Run script
    return secure_run(command, cwd=cwd, capture_output=True, check=True, timeout=timeout)


def quote_path(path: Union[str, Path]) -> str:
    """Safely quote a path for use in commands.
    
    Args:
        path: Path to quote
        
    Returns:
        Quoted path string
    """
    # Convert to string and use shlex.quote for safe quoting
    return shlex.quote(str(path))