"""Model installer for LLM Installer.

This module implements the logic to download and install models from HuggingFace
into isolated environments with all dependencies.
"""

import sys
import json
import shutil
import signal
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import time

from huggingface_hub import (
    snapshot_download, HfApi
)
from huggingface_hub.utils import HfHubHTTPError, GatedRepoError

from core.config import get_config
from core.utils import (
    print_error, print_success, print_info, print_warning,
    console, safe_model_name
)
from core.checker import ModelChecker
from handlers.registry import get_handler_registry
from rich.progress import (
    Progress, BarColumn, DownloadColumn,
    TransferSpeedColumn, TimeRemainingColumn,
    SpinnerColumn, TextColumn
)


class ModelInstaller:
    """Install models from HuggingFace with isolated environments."""

    def __init__(self):
        self.config = get_config()
        self.api = HfApi(token=self.config.huggingface_token)
        self.logger = logging.getLogger(__name__)
        self.checker = ModelChecker()
        self.handler_registry = get_handler_registry()
        self._interrupted = False

        # Set up signal handler for interruption
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C interruption."""
        self._interrupted = True
        print_warning("\nInstallation interrupted. Cleaning up...")

    def install_model(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        quantization: str = "none",
        force: bool = False
    ) -> bool:
        """Install a model from HuggingFace.

        Args:
            model_id: HuggingFace model identifier.
            device: Device preference (auto/cuda/cpu/mps).
            dtype: Data type preference (auto/float16/float32/int8/int4).
            quantization: Quantization method (none/8bit/4bit).
            force: Force reinstall if already exists.

        Returns:
            True if installation successful, False otherwise.
        """
        print_info(f"Starting installation of {model_id}")

        # Step 1: Check or run model compatibility check
        check_result = self._get_or_run_check(model_id)
        if not check_result:
            return False

        model_info, requirements = check_result

        # Step 2: Verify compatibility
        if not self._verify_compatibility(requirements):
            return False

        # Step 3: Set up installation directory
        model_dir = self._setup_model_directory(model_id, force)
        if not model_dir:
            return False

        # Initialize installation log
        install_log_path = model_dir / "install.log"
        self._init_install_log(install_log_path, model_id)

        try:
            # Step 4: Check disk space
            if not self._check_disk_space(requirements):
                self._log_install(install_log_path, "ERROR", "Insufficient disk space")
                return False

            # Step 5: Download model
            print_info("Downloading model files...")
            if not self._download_model(model_id, model_dir, install_log_path):
                return False

            # Step 6: Create virtual environment
            print_info("Creating virtual environment...")
            if not self._create_venv(model_dir, install_log_path):
                return False

            # Step 7: Install dependencies
            print_info("Installing dependencies...")
            if not self._install_dependencies(
                model_dir, requirements, install_log_path
            ):
                return False

            # Step 8: Copy universal scripts
            print_info("Setting up scripts...")
            if not self._copy_scripts(model_dir, install_log_path):
                return False

            # Step 9: Save model info
            if not self._save_model_info(
                model_dir, model_info, requirements, install_log_path
            ):
                return False

            # Step 10: Test installation
            print_info("Testing installation...")
            if not self._test_installation(model_dir, model_info, install_log_path):
                print_warning("Installation test failed, but continuing...")

            # Step 11: Finalize installation
            self._finalize_installation(model_dir, model_id)

            return True

        except Exception as e:
            self._log_install(install_log_path, "ERROR", f"Installation failed: {e}")
            print_error(f"Installation failed: {e}")

            # Clean up on failure
            if model_dir.exists():
                print_info("Cleaning up failed installation...")
                shutil.rmtree(model_dir)

            return False

    def _get_or_run_check(
        self, model_id: str
    ) -> Optional[Tuple[Dict[str, Any], Any]]:
        """Get cached check result or run new check.

        Args:
            model_id: Model identifier.

        Returns:
            Tuple of (model_info, requirements) or None if check fails.
        """
        # Check for cached result
        cache_file = self.config.logs_dir / "checks" / f"{safe_model_name(model_id)}.json"

        if cache_file.exists():
            # Check if cache is fresh (< 24 hours)
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 24 * 60 * 60:  # 24 hours
                print_info("Using cached compatibility check result")
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)

                    # Convert requirements dict back to object
                    from dataclasses import dataclass, field

                    @dataclass
                    class ModelRequirements:
                        model_type: str = ""
                        model_family: str = ""
                        architecture_type: str = ""
                        primary_library: str = ""
                        base_dependencies: list = field(default_factory=list)
                        special_dependencies: list = field(default_factory=list)
                        optional_dependencies: list = field(default_factory=list)
                        disk_space_gb: float = 0.0
                        memory_requirements: dict = field(default_factory=dict)
                        capabilities: dict = field(default_factory=dict)
                        special_config: dict = field(default_factory=dict)
                        compatibility_result: dict = field(default_factory=dict)

                    req_data = cached_data.get('requirements', {})
                    requirements = ModelRequirements(**req_data)

                    return cached_data.get('model_info'), requirements

        # Run new check
        print_info("Running compatibility check...")
        success, requirements = self.checker.check_model(model_id)

        if not success:
            print_error("Model compatibility check failed")
            return None

        # Get model info from checker
        model_info = getattr(self.checker, '_last_model_info', {})

        return model_info, requirements

    def _verify_compatibility(self, requirements: Any) -> bool:
        """Verify model is compatible with system.

        Args:
            requirements: Model requirements from checker.

        Returns:
            True if compatible, False otherwise.
        """
        # Check if model can run on this system
        compatibility = getattr(requirements, 'compatibility_result', None)
        
        # Handle both dict (from cache) and object forms
        if compatibility is None:
            print_error("No compatibility information available")
            return False
            
        # Check can_run - handle both dict and object
        can_run = compatibility.get('can_run', False) if isinstance(compatibility, dict) else getattr(compatibility, 'can_run', False)
        if not can_run:
            print_error("Model is not compatible with your system")
            print_info("See the compatibility report above for details")
            return False

        # Show warnings if any - handle both dict and object
        notes = compatibility.get('notes', []) if isinstance(compatibility, dict) else getattr(compatibility, 'notes', [])
        if notes:
            for note in notes:
                print_warning(f"Note: {note}")

        return True

    def _setup_model_directory(
        self, model_id: str, force: bool
    ) -> Optional[Path]:
        """Set up the model installation directory.

        Args:
            model_id: Model identifier.
            force: Whether to force reinstall.

        Returns:
            Path to model directory or None if setup fails.
        """
        model_name = safe_model_name(model_id)
        model_dir = self.config.models_dir / model_name

        if model_dir.exists():
            if not force:
                print_error(f"Model already installed at: {model_dir}")
                print_info("Use --force to reinstall")
                return None
            else:
                print_warning(f"Removing existing installation at: {model_dir}")
                shutil.rmtree(model_dir)

        # Create directory structure
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model").mkdir(exist_ok=True)
            (model_dir / "logs").mkdir(exist_ok=True)

            return model_dir

        except Exception as e:
            print_error(f"Failed to create model directory: {e}")
            return None

    def _init_install_log(self, log_path: Path, model_id: str):
        """Initialize installation log file.

        Args:
            log_path: Path to log file.
            model_id: Model identifier.
        """
        with open(log_path, 'w') as f:
            f.write(f"Installation Log for {model_id}\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write(f"Installer version: {self.config.version}\n")
            f.write("=" * 80 + "\n\n")

    def _log_install(self, log_path: Path, level: str, message: str):
        """Log installation message.

        Args:
            log_path: Path to log file.
            level: Log level (INFO/WARNING/ERROR).
            message: Log message.
        """
        timestamp = datetime.now().isoformat()
        with open(log_path, 'a') as f:
            f.write(f"[{timestamp}] {level}: {message}\n")

    def _check_disk_space(self, requirements: Any) -> bool:
        """Check if there's enough disk space.

        Args:
            requirements: Model requirements.

        Returns:
            True if enough space, False otherwise.
        """
        import psutil

        # Get required space
        required_gb = requirements.disk_space_gb + 3  # Add 3GB for venv

        # Get available space
        disk_usage = psutil.disk_usage(str(self.config.models_dir))
        available_gb = disk_usage.free / (1024**3)

        if available_gb < required_gb:
            print_error(
                f"Insufficient disk space. Required: {required_gb:.1f} GB, "
                f"Available: {available_gb:.1f} GB"
            )
            return False

        return True

    def _download_model(
        self, model_id: str, model_dir: Path, log_path: Path
    ) -> bool:
        """Download model files from HuggingFace.

        Args:
            model_id: Model identifier.
            model_dir: Model installation directory.
            log_path: Path to installation log.

        Returns:
            True if download successful, False otherwise.
        """
        local_dir = model_dir / "model"
        cache_dir = self.config.cache_dir

        self._log_install(log_path, "INFO", f"Starting download to {local_dir}")

        try:
            # Set up progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True
            ) as progress:
                # Download with progress callback
                def progress_callback(current, total):
                    if self._interrupted:
                        raise KeyboardInterrupt()

                    if hasattr(progress_callback, 'task_id'):
                        progress.update(
                            progress_callback.task_id,
                            completed=current,
                            total=total
                        )

                # Create progress task
                task_id = progress.add_task(
                    f"Downloading {model_id}",
                    total=None
                )
                progress_callback.task_id = task_id

                # Download model
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(local_dir),
                    cache_dir=str(cache_dir),
                    max_workers=self.config.max_download_workers,
                    token=self.config.huggingface_token
                )

                progress.update(task_id, description=f"Downloaded {model_id}")

            self._log_install(log_path, "INFO", "Download completed successfully")
            print_success("Model downloaded successfully")
            return True

        except KeyboardInterrupt:
            self._log_install(log_path, "WARNING", "Download interrupted by user")
            return False

        except (HfHubHTTPError, GatedRepoError) as e:
            self._log_install(log_path, "ERROR", f"HuggingFace error: {e}")
            print_error(f"Failed to download model: {e}")
            return False

        except Exception as e:
            self._log_install(log_path, "ERROR", f"Download error: {e}")
            print_error(f"Download failed: {e}")
            return False

    def _create_venv(self, model_dir: Path, log_path: Path) -> bool:
        """Create virtual environment for the model.

        Args:
            model_dir: Model installation directory.
            log_path: Path to installation log.

        Returns:
            True if venv created successfully, False otherwise.
        """
        venv_path = model_dir / ".venv"

        self._log_install(log_path, "INFO", f"Creating virtual environment at {venv_path}")

        try:
            # Create venv
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                check=True,
                capture_output=True,
                text=True
            )

            # Upgrade pip, setuptools, wheel
            pip_path = venv_path / "bin" / "pip"
            if not pip_path.exists():  # Windows
                pip_path = venv_path / "Scripts" / "pip.exe"

            subprocess.run(
                [str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"],
                check=True,
                capture_output=True,
                text=True
            )

            self._log_install(log_path, "INFO", "Virtual environment created successfully")
            return True

        except subprocess.CalledProcessError as e:
            self._log_install(log_path, "ERROR", f"Failed to create venv: {e}")
            print_error(f"Failed to create virtual environment: {e}")
            return False

        except Exception as e:
            self._log_install(log_path, "ERROR", f"Venv creation error: {e}")
            print_error(f"Virtual environment creation failed: {e}")
            return False

    def _install_dependencies(
        self,
        model_dir: Path,
        requirements: Any,
        log_path: Path
    ) -> bool:
        """Install model dependencies in virtual environment.

        Args:
            model_dir: Model installation directory.
            requirements: Model requirements from checker.
            log_path: Path to installation log.

        Returns:
            True if dependencies installed successfully, False otherwise.
        """
        venv_path = model_dir / ".venv"
        pip_path = venv_path / "bin" / "pip"
        if not pip_path.exists():  # Windows
            pip_path = venv_path / "Scripts" / "pip.exe"

        # Get dependencies from requirements
        base_deps = requirements.base_dependencies
        special_deps = requirements.special_dependencies

        self._log_install(log_path, "INFO", f"Installing dependencies: {base_deps}")

        # Install PyTorch first if needed
        if any('torch' in dep for dep in base_deps):
            if not self._install_pytorch(pip_path, log_path):
                return False
            # Remove torch from base_deps to avoid reinstalling
            base_deps = [dep for dep in base_deps if 'torch' not in dep]

        # Install base dependencies
        if base_deps:
            try:
                print_info(f"Installing base dependencies: {', '.join(base_deps)}")
                subprocess.run(
                    [str(pip_path), "install"] + base_deps,
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                self._log_install(log_path, "ERROR", f"Failed to install base deps: {e}")
                print_error(f"Failed to install dependencies: {e}")
                return False

        # Install special dependencies with error handling
        for dep in special_deps:
            if not self._install_special_dependency(dep, pip_path, model_dir, log_path):
                # Special dependencies might fail but shouldn't block installation
                print_warning(f"Failed to install {dep}, continuing...")
        
        # Install optional dependencies
        optional_deps = getattr(requirements, 'optional_dependencies', [])
        if optional_deps:
            print_info(f"Installing optional dependencies: {', '.join(optional_deps)}")
            for dep in optional_deps:
                try:
                    subprocess.run(
                        [str(pip_path), "install", dep],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    self._log_install(log_path, "INFO", f"Installed optional dependency: {dep}")
                except subprocess.CalledProcessError:
                    # Optional dependencies can fail silently
                    self._log_install(log_path, "INFO", f"Skipped optional dependency: {dep}")
                    
                    # Try special handling for xformers
                    if dep == "xformers":
                        self._install_xformers_with_cuda(pip_path, log_path)

        # Install server dependencies for universal scripts
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
            "python-dotenv>=1.0.0"
        ]

        try:
            print_info("Installing server dependencies...")
            subprocess.run(
                [str(pip_path), "install"] + server_deps,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            self._log_install(log_path, "WARNING", f"Some server deps failed: {e}")
            # Continue anyway, core functionality might still work

        self._log_install(log_path, "INFO", "Dependencies installation completed")
        return True

    def _install_pytorch(self, pip_path: Path, log_path: Path) -> bool:
        """Install PyTorch with appropriate CUDA support.

        Args:
            pip_path: Path to pip executable.
            log_path: Path to installation log.

        Returns:
            True if PyTorch installed successfully, False otherwise.
        """
        import platform

        # Detect CUDA availability
        cuda_available = False
        cuda_version = None

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                cuda_available = True
                # Try to get CUDA version
                result = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and "release" in result.stdout:
                    # Extract version (e.g., "release 11.8")
                    import re
                    match = re.search(r"release (\d+\.\d+)", result.stdout)
                    if match:
                        cuda_version = match.group(1)
        except Exception:
            pass

        # Determine PyTorch index URL
        system = platform.system().lower()

        if cuda_available and cuda_version:
            # Map CUDA version to PyTorch index
            if cuda_version.startswith("12"):
                index_url = "https://download.pytorch.org/whl/cu121"
                torch_cmd = ["torch", "torchvision", "torchaudio"]
            elif cuda_version.startswith("11.8"):
                index_url = "https://download.pytorch.org/whl/cu118"
                torch_cmd = ["torch", "torchvision", "torchaudio"]
            else:
                # Default CUDA
                index_url = "https://download.pytorch.org/whl/cu118"
                torch_cmd = ["torch", "torchvision", "torchaudio"]
        elif system == "darwin":  # macOS
            # MPS support
            torch_cmd = ["torch", "torchvision", "torchaudio"]
            index_url = None
        else:
            # CPU only
            index_url = "https://download.pytorch.org/whl/cpu"
            torch_cmd = ["torch", "torchvision", "torchaudio"]

        try:
            print_info(f"Installing PyTorch for {system} "
                      f"{'with CUDA ' + cuda_version if cuda_available else 'CPU only'}")

            cmd = [str(pip_path), "install"] + torch_cmd
            if index_url:
                cmd.extend(["--index-url", index_url])

            subprocess.run(cmd, check=True, capture_output=True, text=True)

            self._log_install(log_path, "INFO", "PyTorch installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            self._log_install(log_path, "ERROR", f"Failed to install PyTorch: {e}")
            print_error(f"Failed to install PyTorch: {e}")
            return False

    def _install_special_dependency(
        self,
        dep: str,
        pip_path: Path,
        model_dir: Path,
        log_path: Path
    ) -> bool:
        """Install special dependency with custom handling.

        Args:
            dep: Dependency name.
            pip_path: Path to pip executable.
            model_dir: Model directory.
            log_path: Path to installation log.

        Returns:
            True if installed successfully, False otherwise.
        """
        special_instructions = {
            "mamba-ssm": """
This model requires mamba-ssm which needs additional setup:

1. Ensure CUDA toolkit is installed:
   nvcc --version

2. Install build dependencies:
   - Ubuntu/Debian: sudo apt-get install build-essential
   - macOS: xcode-select --install

3. Try manual installation:
   cd {model_dir}
   source .venv/bin/activate
   pip install mamba-ssm --no-cache-dir
""",
            "flash-attn": """
This model uses Flash Attention for better performance.
Flash Attention requires:
- CUDA 11.6 or higher
- PyTorch with CUDA support
- C++ compiler

To install manually:
   cd {model_dir}
   source .venv/bin/activate
   pip install flash-attn --no-build-isolation
""",
            "causal-conv1d": """
This dependency requires CUDA and compilation.
Make sure you have nvcc installed:
   nvcc --version
""",
            "xformers": """
xformers provides memory efficient attention for better performance.
If installation fails, the model will still work with standard attention.

To install manually:
   cd {model_dir}
   source .venv/bin/activate
   pip install xformers --index-url https://download.pytorch.org/whl/cu118
"""
        }

        try:
            print_info(f"Installing {dep}...")
            subprocess.run(
                [str(pip_path), "install", dep],
                check=True,
                capture_output=True,
                text=True
            )
            self._log_install(log_path, "INFO", f"Installed {dep} successfully")
            return True

        except subprocess.CalledProcessError as e:
            self._log_install(log_path, "WARNING", f"Failed to install {dep}: {e}")

            # Show special instructions if available
            if dep in special_instructions:
                print_warning(f"\nFailed to install {dep}")
                print_info(special_instructions[dep].format(model_dir=model_dir))

            return False
    
    def _install_xformers_with_cuda(self, pip_path: Path, log_path: Path) -> bool:
        """Try to install xformers with appropriate CUDA version.
        
        Args:
            pip_path: Path to pip executable.
            log_path: Path to installation log.
            
        Returns:
            True if installed successfully, False otherwise.
        """
        try:
            # First check installed torch version
            result = subprocess.run(
                [str(pip_path), "show", "torch"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False
                
            # Extract torch version
            torch_version = None
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    torch_version = line.split(':')[1].strip()
                    break
            
            if not torch_version:
                return False
            
            # Check CUDA version from torch
            cuda_suffix = None
            if '+cu' in torch_version:
                # Extract CUDA version from torch version (e.g., "2.5.1+cu121" -> "cu121")
                cuda_suffix = torch_version.split('+')[1]
            else:
                # Try to detect from system
                result = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'release' in line:
                            cuda_version = line.split('release')[1].split(',')[0].strip()
                            if cuda_version.startswith("12.1"):
                                cuda_suffix = "cu121"
                            elif cuda_version.startswith("11.8"):
                                cuda_suffix = "cu118"
                            break
            
            if cuda_suffix:
                # Map CUDA suffix to index URL
                index_urls = {
                    "cu121": "https://download.pytorch.org/whl/cu121",
                    "cu118": "https://download.pytorch.org/whl/cu118",
                    "cu117": "https://download.pytorch.org/whl/cu117",
                }
                index_url = index_urls.get(cuda_suffix, "https://download.pytorch.org/whl/cu118")
                
                print_info(f"Installing xformers compatible with torch {torch_version}...")
                
                # Try to install compatible xformers
                subprocess.run(
                    [str(pip_path), "install", "xformers", "--index-url", index_url],
                    check=True,
                    capture_output=True,
                    text=True
                )
                self._log_install(log_path, "INFO", f"Installed xformers for {cuda_suffix}")
                return True
                
        except Exception as e:
            # Silent fail for optional dependency
            self._log_install(log_path, "INFO", f"Could not install xformers: {str(e)}")
            pass
        
        return False

    def _copy_scripts(self, model_dir: Path, log_path: Path) -> bool:
        """Copy universal scripts and required libraries to model directory.

        Args:
            model_dir: Model installation directory.
            log_path: Path to installation log.

        Returns:
            True if scripts copied successfully, False otherwise.
        """
        scripts_dir = Path(__file__).parent.parent / "scripts"
        installer_root = Path(__file__).parent.parent

        if not scripts_dir.exists():
            self._log_install(log_path, "WARNING", "Scripts directory not found")
            print_warning("Scripts directory not found, skipping...")
            return True

        scripts_to_copy = [
            ("start.sh", True),
            ("train.sh", True),
            ("serve_api.py", False),
            ("serve_ui.html", False),
            ("model_loader.py", False)
        ]

        try:
            # Copy scripts
            for script_name, make_executable in scripts_to_copy:
                src = scripts_dir / script_name
                dst = model_dir / script_name

                if src.exists():
                    shutil.copy2(src, dst)

                    if make_executable:
                        # Make executable
                        dst.chmod(dst.stat().st_mode | 0o755)

                    self._log_install(log_path, "INFO", f"Copied {script_name}")
                else:
                    self._log_install(log_path, "WARNING", f"Script {script_name} not found")

            # Copy handlers directory
            handlers_src = installer_root / "handlers"
            handlers_dst = model_dir / "handlers"
            if handlers_src.exists():
                print_info("Copying handlers library...")
                shutil.copytree(handlers_src, handlers_dst, dirs_exist_ok=True)
                self._log_install(log_path, "INFO", "Copied handlers library")

            # Copy detectors directory
            detectors_src = installer_root / "detectors"
            detectors_dst = model_dir / "detectors"
            if detectors_src.exists():
                print_info("Copying detectors library...")
                shutil.copytree(detectors_src, detectors_dst, dirs_exist_ok=True)
                self._log_install(log_path, "INFO", "Copied detectors library")

            # Copy core utilities (only needed files)
            core_utils_src = installer_root / "core" / "utils.py"
            core_dst = model_dir / "core"
            core_dst.mkdir(exist_ok=True)
            
            if core_utils_src.exists():
                shutil.copy2(core_utils_src, core_dst / "utils.py")
                # Create __init__.py
                (core_dst / "__init__.py").touch()
                self._log_install(log_path, "INFO", "Copied core utilities")

            return True

        except Exception as e:
            self._log_install(log_path, "ERROR", f"Failed to copy scripts: {e}")
            print_error(f"Failed to copy scripts: {e}")
            return False

    def _save_model_info(
        self,
        model_dir: Path,
        model_info: Dict[str, Any],
        requirements: Any,
        log_path: Path
    ) -> bool:
        """Save model information to model_info.json.

        Args:
            model_dir: Model installation directory.
            model_info: Model information from checker.
            requirements: Model requirements.
            log_path: Path to installation log.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            # Prepare model info
            info_to_save = {
                **model_info,
                "install_date": datetime.now().isoformat(),
                "installer_version": self.config.version,
                "model_path": "./model",
                "requirements": {
                    "base_dependencies": requirements.base_dependencies,
                    "special_dependencies": requirements.special_dependencies,
                    "optional_dependencies": requirements.optional_dependencies,
                    "disk_space_gb": requirements.disk_space_gb,
                    "memory_requirements": requirements.memory_requirements
                },
                "capabilities": requirements.capabilities,
                "special_config": requirements.special_config
            }

            # Save to file
            info_path = model_dir / "model_info.json"
            with open(info_path, 'w') as f:
                json.dump(info_to_save, f, indent=2)

            self._log_install(log_path, "INFO", "Saved model_info.json")
            return True

        except Exception as e:
            self._log_install(log_path, "ERROR", f"Failed to save model info: {e}")
            print_error(f"Failed to save model information: {e}")
            return False

    def _test_installation(
        self,
        model_dir: Path,
        model_info: Dict[str, Any],
        log_path: Path
    ) -> bool:
        """Test the model installation.

        Args:
            model_dir: Model installation directory.
            model_info: Model information.
            log_path: Path to installation log.

        Returns:
            True if test passed, False otherwise.
        """
        venv_python = model_dir / ".venv" / "bin" / "python"
        if not venv_python.exists():  # Windows
            venv_python = model_dir / ".venv" / "Scripts" / "python.exe"

        # Create test script
        test_script = model_dir / "test_load.py"
        test_content = '''
import sys
import json
import os

# Add installer to path
installer_path = os.path.expanduser("~/LLM/installer")
if os.path.exists(installer_path):
    sys.path.insert(0, installer_path)

try:
    # Load model info
    with open("model_info.json", "r") as f:
        model_info = json.load(f)

    print("Model info loaded successfully")

    # Test imports based on model type
    primary_lib = model_info.get("primary_library", "transformers")

    if primary_lib == "transformers":
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    elif primary_lib == "diffusers":
        import diffusers
        print(f"Diffusers version: {diffusers.__version__}")

    # Test PyTorch
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("\\nBasic imports successful!")

except Exception as e:
    print(f"Test failed: {e}")
    sys.exit(1)
'''

        try:
            # Write test script
            with open(test_script, 'w') as f:
                f.write(test_content)

            # Run test
            result = subprocess.run(
                [str(venv_python), str(test_script)],
                capture_output=True,
                text=True,
                cwd=model_dir
            )

            if result.returncode == 0:
                self._log_install(log_path, "INFO", "Installation test passed")
                self._log_install(log_path, "INFO", f"Test output: {result.stdout}")
                print_success("Installation test passed")
                return True
            else:
                self._log_install(log_path, "WARNING", f"Installation test failed: {result.stderr}")
                print_warning(f"Installation test failed: {result.stderr}")
                return False

        except Exception as e:
            self._log_install(log_path, "WARNING", f"Failed to run installation test: {e}")
            print_warning(f"Failed to run installation test: {e}")
            return False

        finally:
            # Clean up test script
            if test_script.exists():
                test_script.unlink()

    def _finalize_installation(self, model_dir: Path, model_id: str):
        """Finalize the installation.

        Args:
            model_dir: Model installation directory.
            model_id: Model identifier.
        """
        # Create completion marker
        completion_marker = model_dir / ".install_complete"
        with open(completion_marker, 'w') as f:
            f.write(f"Installation completed at: {datetime.now().isoformat()}\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"Installer version: {self.config.version}\n")

        # Calculate sizes
        model_size = self._calculate_directory_size(model_dir / "model")
        venv_size = self._calculate_directory_size(model_dir / ".venv")

        # Show success message
        print_success("\n✓ Model installed successfully!")

        console.print(f"\nLocation: {model_dir}")
        console.print(f"Model size: {model_size / (1024**3):.1f} GB")
        console.print(f"Virtual environment: {venv_size / (1024**3):.1f} GB")

        console.print("\nTo start the model:")
        console.print(f"  cd {model_dir}")
        console.print("  ./start.sh")

        console.print("\nTo train/fine-tune:")
        console.print("  ./train.sh --data your_data.json")

        console.print(f"\nLogs saved to: install.log")

    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of a directory.

        Args:
            directory: Directory path.

        Returns:
            Total size in bytes.
        """
        total_size = 0
        for item in directory.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size
    
    def update_scripts(self, model_dir: str) -> bool:
        """Update scripts and libraries in an existing model installation.
        
        Args:
            model_dir: Path to model directory to update.
            
        Returns:
            True if update successful, False otherwise.
        """
        model_path = Path(model_dir).resolve()
        
        # Check if directory exists
        if not model_path.exists():
            print_error(f"Model directory not found: {model_path}")
            return False
            
        # Check if it's a valid model installation
        if not (model_path / ".install_complete").exists():
            print_error(f"Not a valid model installation: {model_path}")
            return False
            
        print_info(f"Updating scripts in: {model_path}")
        
        # Create a temporary log file
        log_path = model_path / "logs" / "update.log"
        log_path.parent.mkdir(exist_ok=True)
        self._init_install_log(log_path, f"Update for {model_path.name}")
        
        # Use the existing copy scripts method
        if self._copy_scripts(model_path, log_path):
            print_success("Scripts and libraries updated successfully!")
            self._log_install(log_path, "INFO", "Update completed successfully")
            return True
        else:
            print_error("Failed to update scripts")
            return False

