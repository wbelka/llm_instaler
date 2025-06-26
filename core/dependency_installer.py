"""Dependency installation utilities for LLM Installer.

This module handles the complex logic of installing dependencies,
split from the main installer for better maintainability.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from core.secure_subprocess import secure_pip_install, secure_run, SecureSubprocessError
from core.exceptions import DependencyError
from core.utils import print_info, print_warning


logger = logging.getLogger(__name__)


class DependencyInstaller:
    """Handles dependency installation for models."""
    
    def __init__(self, pip_path: Path, log_func):
        """Initialize dependency installer.
        
        Args:
            pip_path: Path to pip executable
            log_func: Function to log installation messages
        """
        self.pip_path = pip_path
        self._log_install = log_func
    
    def install_base_dependencies(
        self,
        base_deps: List[str],
        special_deps: List[str],
        device_preference: str = "auto"
    ) -> bool:
        """Install base dependencies.
        
        Args:
            base_deps: List of base dependencies
            special_deps: List of special dependencies
            device_preference: Device preference for CUDA packages
            
        Returns:
            True if successful, False otherwise
        """
        if not base_deps:
            return True
        
        try:
            # Ensure transformers is always installed for diffusion models
            # This logic should be in the handler/detector
            
            print_info(f"Installing base dependencies: {', '.join(base_deps)}")
            
            # Separate CUDA and PyPI packages
            cuda_packages, pypi_packages = self._separate_cuda_packages(base_deps)
            
            # Install PyPI packages
            if not self._install_pypi_packages(pypi_packages):
                return False
            
            # Install CUDA packages
            if not self._install_cuda_packages(cuda_packages):
                return False
            
            return True
            
        except SecureSubprocessError as e:
            self._log_install("ERROR", f"Failed to install base deps: {e}")
            raise DependencyError(f"Failed to install base dependencies: {e}") from e
    
    def install_optional_dependencies(
        self,
        optional_deps: List[str],
        device_preference: str = "auto"
    ) -> None:
        """Install optional dependencies.
        
        Args:
            optional_deps: List of optional dependencies
            device_preference: Device preference
        """
        if not optional_deps:
            return
        
        print_info(f"Installing optional dependencies: {', '.join(optional_deps)}")
        
        # Get torch info once for all dependencies
        torch_info = self._get_torch_info()
        
        for dep in optional_deps:
            try:
                # Use compatible version for flash-attn if needed
                if "flash-attn" in dep and dep == "flash-attn":
                    dep = self._get_compatible_flash_attn_version(torch_info)
                
                # Build install command
                install_args = []
                
                # Special build options for flash-attn
                if "flash-attn" in dep:
                    install_args.append("--no-build-isolation")
                
                secure_pip_install(
                    self.pip_path,
                    [dep],
                    index_url=torch_info.get("index_url") if torch_info.get("cuda_version") else None,
                    timeout=1200  # Longer timeout for building from source
                )
                
                self._log_install("INFO", f"Installed optional dependency: {dep}")
                
            except SecureSubprocessError:
                # Optional dependencies can fail silently
                self._log_install("INFO", f"Skipped optional dependency: {dep}")
                
                # Try special handling for xformers
                if dep == "xformers":
                    self._install_xformers_with_cuda()
    
    def install_server_dependencies(self, device_preference: str = "auto") -> bool:
        """Install server and training dependencies.
        
        Args:
            device_preference: Device preference
            
        Returns:
            True if successful, False otherwise
        """
        server_deps = [
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "websockets>=12.0",
            "sse-starlette>=2.0.0",
            "pydantic>=2.5.0",
            "python-multipart>=0.0.6",
            "pillow>=10.0.0",
            "numpy>=1.24.0",
            "aiofiles>=23.0.0",
            "psutil>=5.9.0",
            "rich>=13.0.0",
            "python-dotenv>=1.0.0",
            # Training dependencies
            "matplotlib>=3.7.0",
            "tqdm>=4.65.0",
            "datasets>=2.14.0",
            "peft>=0.7.0",
            "tensorboard>=2.14.0"
        ]
        
        try:
            print_info("Installing server and training dependencies...")
            secure_pip_install(
                self.pip_path,
                server_deps,
                timeout=600
            )
            return True
            
        except SecureSubprocessError as e:
            self._log_install("WARNING", f"Some server deps failed: {e}")
            # Don't fail installation for server deps
            return True
    
    def handle_quantization_dependencies(
        self,
        base_deps: List[str],
        optional_deps: List[str],
        dtype_preference: str,
        quantization: str
    ) -> Tuple[List[str], List[str]]:
        """Handle quantization-specific dependencies.
        
        Args:
            base_deps: Base dependencies list (will be modified)
            optional_deps: Optional dependencies list (will be modified)
            dtype_preference: Data type preference
            quantization: Quantization method
            
        Returns:
            Tuple of (updated_base_deps, updated_optional_deps)
        """
        needs_quantization = (
            dtype_preference in ['int8', 'int4'] or
            quantization in ['8bit', '4bit']
        )
        
        if not needs_quantization:
            return base_deps, optional_deps
        
        # Check if bitsandbytes is already in base_deps
        bitsandbytes_in_base = any('bitsandbytes' in dep for dep in base_deps)
        
        if not bitsandbytes_in_base:
            # Check if bitsandbytes is in optional deps
            bitsandbytes_found = False
            for dep in optional_deps[:]:  # Copy to modify during iteration
                if 'bitsandbytes' in dep:
                    # Move from optional to required
                    base_deps.append(dep)
                    optional_deps.remove(dep)
                    bitsandbytes_found = True
                    self._log_install("INFO",
                        f"Moved {dep} from optional to required dependencies for quantization")
                    break
            
            # If not found anywhere, add it
            if not bitsandbytes_found:
                from core.quantization_config import QuantizationConfig
                bitsandbytes_version = QuantizationConfig.get_bitsandbytes_version()
                base_deps.append(bitsandbytes_version)
                self._log_install("INFO",
                    f"Added {bitsandbytes_version} for quantization support")
        else:
            self._log_install("INFO",
                "bitsandbytes already in base dependencies for quantization")
        
        return base_deps, optional_deps
    
    def _separate_cuda_packages(self, deps: List[str]) -> Tuple[List[str], List[str]]:
        """Separate CUDA-dependent packages from regular PyPI packages.
        
        Args:
            deps: List of dependencies
            
        Returns:
            Tuple of (cuda_packages, pypi_packages)
        """
        # Only packages that are actually in PyTorch index
        cuda_deps = ["xformers", "triton"]
        
        cuda_packages = []
        pypi_packages = []
        
        for dep in deps:
            # Check if this is a CUDA-dependent package
            is_cuda_dep = any(cuda_dep in dep.lower() for cuda_dep in cuda_deps)
            if is_cuda_dep:
                cuda_packages.append(dep)
            else:
                pypi_packages.append(dep)
        
        return cuda_packages, pypi_packages
    
    def _install_pypi_packages(self, packages: List[str]) -> bool:
        """Install regular PyPI packages.
        
        Args:
            packages: List of package specifications
            
        Returns:
            True if successful
        """
        if not packages:
            return True
        
        print_info(f"Installing PyPI packages: {', '.join(packages)}")
        secure_pip_install(
            self.pip_path,
            packages,
            timeout=600
        )
        return True
    
    def _install_cuda_packages(self, packages: List[str]) -> bool:
        """Install CUDA-dependent packages.
        
        Args:
            packages: List of package specifications
            
        Returns:
            True if successful
        """
        if not packages:
            return True
        
        print_info(f"Installing CUDA packages: {', '.join(packages)}")
        torch_info = self._get_torch_info()
        
        secure_pip_install(
            self.pip_path,
            packages,
            index_url=torch_info.get("index_url"),
            timeout=600
        )
        return True
    
    def _get_torch_info(self) -> Dict[str, Any]:
        """Get PyTorch installation information.
        
        Returns:
            Dict with torch version, cuda version, and index URL
        """
        try:
            result = secure_run(
                [str(self.pip_path), "show", "torch"],
                capture_output=True
            )
            
            # Parse version info
            version = None
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
                    break
            
            if not version:
                return {}
            
            # Determine CUDA version from torch version
            cuda_version = None
            index_url = None
            
            if '+cu' in version:
                cuda_part = version.split('+cu')[1]
                cuda_version = cuda_part[:3]  # e.g., "118", "121"
                index_url = f"https://download.pytorch.org/whl/cu{cuda_version}"
            elif '+rocm' in version:
                rocm_part = version.split('+rocm')[1]
                index_url = f"https://download.pytorch.org/whl/rocm{rocm_part}"
            
            return {
                "version": version,
                "cuda_version": cuda_version,
                "index_url": index_url
            }
            
        except Exception:
            return {}
    
    def _get_compatible_flash_attn_version(self, torch_info: Dict[str, Any]) -> str:
        """Get compatible flash-attn version for current PyTorch.
        
        Args:
            torch_info: PyTorch information
            
        Returns:
            Compatible flash-attn version string
        """
        # Default to latest known compatible version
        return "flash-attn==2.7.2.post1"
    
    def _install_xformers_with_cuda(self) -> None:
        """Try to install xformers with CUDA support."""
        try:
            torch_info = self._get_torch_info()
            if torch_info.get("cuda_version"):
                secure_pip_install(
                    self.pip_path,
                    ["xformers"],
                    index_url=torch_info["index_url"],
                    timeout=600
                )
                self._log_install("INFO", "Successfully installed xformers with CUDA")
        except Exception:
            self._log_install("INFO", "Could not install xformers with CUDA")