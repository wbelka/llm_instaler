"""Registry for model detectors.

This module manages the registration and ordering of model detectors,
allowing the system to determine the appropriate handler for any model.
"""

from typing import List, Optional, Dict, Any
import importlib
import inspect
from pathlib import Path

from detectors.base import BaseDetector


class DetectorRegistry:
    """Registry for managing model detectors."""

    def __init__(self):
        """Initialize the detector registry."""
        self._detectors: List[BaseDetector] = []
        self._loaded = False

    def register(self, detector: BaseDetector) -> None:
        """Register a detector instance.

        Args:
            detector: Detector instance to register.
        """
        self._detectors.append(detector)
        # Sort by priority (highest first)
        self._detectors.sort(key=lambda d: d.priority, reverse=True)

    def _auto_discover_detectors(self) -> None:
        """Automatically discover and register all detectors in the package."""
        if self._loaded:
            return

        detectors_dir = Path(__file__).parent

        # Find all Python files in the detectors directory
        for py_file in detectors_dir.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name in ["base.py", "registry.py"]:
                continue

            module_name = f"detectors.{py_file.stem}"

            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Find all detector classes in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, BaseDetector) and
                            obj != BaseDetector):
                        # Create instance and register
                        try:
                            detector_instance = obj()
                            self.register(detector_instance)
                        except Exception as e:
                            # Log error but continue
                            print(f"Failed to instantiate {name}: {e}")

            except ImportError as e:
                # Log error but continue
                print(f"Failed to import {module_name}: {e}")

        self._loaded = True

    def find_detector(self, model_info: Dict[str, Any]) -> Optional[BaseDetector]:
        """Find the appropriate detector for a model.

        Args:
            model_info: Model metadata dictionary.

        Returns:
            Matching detector instance or None.
        """
        # Ensure detectors are loaded
        self._auto_discover_detectors()

        # Try each detector in priority order
        for detector in self._detectors:
            try:
                if detector.matches(model_info):
                    return detector
            except Exception as e:
                # Log error but continue with next detector
                print(f"Error in {detector.name}: {e}")
                continue

        return None

    def analyze_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a model using the appropriate detector.

        Args:
            model_info: Model metadata dictionary.

        Returns:
            Extended model information with analysis results.

        Raises:
            ValueError: If no suitable detector is found.
        """
        detector = self.find_detector(model_info)

        if detector is None:
            raise ValueError(
                f"No detector found for model: {model_info.get('model_id', 'unknown')}"
            )

        # Run analysis
        analysis = detector.analyze(model_info)

        # Add detector information
        analysis['detector_name'] = detector.name
        analysis['handler_class'] = detector.get_handler_class()

        # Merge with original info
        result = model_info.copy()
        result.update(analysis)

        return result

    def get_registered_detectors(self) -> List[str]:
        """Get list of registered detector names.

        Returns:
            List of detector names in priority order.
        """
        self._auto_discover_detectors()
        return [d.name for d in self._detectors]

    def get_detectors(self) -> List[BaseDetector]:
        """Get list of all registered detectors.

        Returns:
            List of detector instances in priority order.
        """
        self._auto_discover_detectors()
        return self._detectors

    def clear(self) -> None:
        """Clear all registered detectors.

        Mainly used for testing.
        """
        self._detectors.clear()
        self._loaded = False


# Global registry instance
_registry = DetectorRegistry()


def get_registry() -> DetectorRegistry:
    """Get the global detector registry.

    Returns:
        Global DetectorRegistry instance.
    """
    return _registry


def analyze_model(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to analyze a model.

    Args:
        model_info: Model metadata dictionary.

    Returns:
        Extended model information with analysis results.
    """
    return _registry.analyze_model(model_info)


def find_detector(model_info: Dict[str, Any]) -> Optional[BaseDetector]:
    """Convenience function to find a detector.

    Args:
        model_info: Model metadata dictionary.

    Returns:
        Matching detector instance or None.
    """
    return _registry.find_detector(model_info)


def get_detector_registry() -> DetectorRegistry:
    """Get the global detector registry.

    Returns:
        Global DetectorRegistry instance.
    """
    return _registry
