"""Context managers for proper resource cleanup in LLM Installer."""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Iterator

logger = logging.getLogger(__name__)


@contextmanager
def temporary_directory(prefix: str = "llm_installer_") -> Iterator[Path]:
    """Create a temporary directory that's automatically cleaned up.
    
    Args:
        prefix: Prefix for the temporary directory name
        
    Yields:
        Path to the temporary directory
    """
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        logger.debug(f"Created temporary directory: {temp_dir}")
        yield temp_dir
    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


@contextmanager
def atomic_write(file_path: Path, mode: str = 'w') -> Iterator:
    """Write to a file atomically by writing to a temp file first.
    
    Args:
        file_path: Target file path
        mode: File open mode
        
    Yields:
        File handle to write to
    """
    temp_fd = None
    temp_path = None
    
    try:
        # Create temp file in same directory for atomic rename
        dir_path = file_path.parent
        dir_path.mkdir(parents=True, exist_ok=True)
        
        temp_fd, temp_path = tempfile.mkstemp(
            dir=str(dir_path),
            prefix=f".{file_path.name}.",
            suffix=".tmp"
        )
        
        # Open the file descriptor with proper mode
        if 'b' in mode:
            file_mode = 'wb'
        else:
            file_mode = 'w'
            
        with os.fdopen(temp_fd, file_mode) as f:
            temp_fd = None  # os.fdopen takes ownership
            yield f
        
        # Atomic rename
        Path(temp_path).replace(file_path)
        temp_path = None
        
    finally:
        # Clean up if something went wrong
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except Exception:
                pass
                
        if temp_path and Path(temp_path).exists():
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@contextmanager
def safe_model_download(download_path: Path) -> Iterator[Path]:
    """Context manager for safe model downloads with cleanup on failure.
    
    Args:
        download_path: Path where model will be downloaded
        
    Yields:
        Path to download directory
    """
    created = False
    
    try:
        if not download_path.exists():
            download_path.mkdir(parents=True, exist_ok=True)
            created = True
            logger.debug(f"Created download directory: {download_path}")
        
        yield download_path
        
    except Exception:
        # Clean up on failure if we created the directory
        if created and download_path.exists():
            try:
                shutil.rmtree(download_path)
                logger.info(f"Cleaned up failed download directory: {download_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up download directory: {e}")
        raise


@contextmanager
def file_lock(lock_path: Path, timeout: int = 300) -> Iterator[None]:
    """Simple file-based lock for preventing concurrent operations.
    
    Args:
        lock_path: Path to lock file
        timeout: Maximum time to wait for lock (seconds)
        
    Yields:
        None when lock is acquired
        
    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    import time
    
    lock_acquired = False
    start_time = time.time()
    
    try:
        while True:
            try:
                # Try to create lock file exclusively
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                lock_acquired = True
                logger.debug(f"Acquired lock: {lock_path}")
                break
                
            except FileExistsError:
                # Lock exists, check timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Could not acquire lock {lock_path} within {timeout} seconds")
                
                # Check if lock file is stale (older than timeout)
                try:
                    lock_stat = lock_path.stat()
                    if time.time() - lock_stat.st_mtime > timeout:
                        logger.warning(f"Removing stale lock file: {lock_path}")
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    # Lock was released, try again
                    continue
                
                # Wait a bit before retrying
                time.sleep(0.1)
        
        yield
        
    finally:
        if lock_acquired:
            try:
                lock_path.unlink(missing_ok=True)
                logger.debug(f"Released lock: {lock_path}")
            except Exception as e:
                logger.warning(f"Failed to remove lock file {lock_path}: {e}")


@contextmanager
def environment_variables(**kwargs) -> Iterator[None]:
    """Temporarily set environment variables.
    
    Args:
        **kwargs: Environment variables to set
        
    Yields:
        None with environment variables set
    """
    old_values = {}
    
    try:
        # Save old values and set new ones
        for key, value in kwargs.items():
            old_values[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        
        yield
        
    finally:
        # Restore old values
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextmanager 
def safe_chdir(path: Path) -> Iterator[None]:
    """Safely change directory and restore on exit.
    
    Args:
        path: Directory to change to
        
    Yields:
        None with directory changed
    """
    old_cwd = Path.cwd()
    
    try:
        os.chdir(path)
        logger.debug(f"Changed directory to: {path}")
        yield
        
    finally:
        os.chdir(old_cwd)
        logger.debug(f"Restored directory to: {old_cwd}")