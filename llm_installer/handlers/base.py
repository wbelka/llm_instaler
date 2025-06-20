"""
Base handler class for model-specific installation logic
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """Base class for model handlers"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name"""
        pass
    
    @abstractmethod
    def can_handle(self, model_info: Dict[str, Any]) -> bool:
        """Check if this handler can process the model"""
        pass
    
    @abstractmethod
    def get_dependencies(self, model_info: Dict[str, Any]) -> List[str]:
        """Get list of required dependencies for the model"""
        pass
    
    def get_additional_files(self, model_info: Dict[str, Any]) -> List[str]:
        """Get list of additional files to download (optional)"""
        return []
    
    def post_install_setup(self, install_path: str, model_info: Dict[str, Any]) -> None:
        """Perform post-installation setup (optional)"""
        pass
    
    def get_environment_vars(self, model_info: Dict[str, Any]) -> Dict[str, str]:
        """Get environment variables needed for the model (optional)"""
        return {}
    
    def get_optimization_options(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization options for the model (optional)"""
        return {}