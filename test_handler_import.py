#!/usr/bin/env python3
"""Test if handlers can be imported in the model directory."""

import sys
from pathlib import Path


def test_handler_import(model_path: Path):
    """Test importing handlers from model directory."""
    
    # Add model path to Python path
    sys.path.insert(0, str(model_path))
    
    print(f"Testing handler imports in: {model_path}")
    print(f"Python path includes: {model_path}")
    
    # Test registry import
    try:
        from handlers.registry import get_handler_registry
        print("✓ Successfully imported handler registry")
        
        registry = get_handler_registry()
        print(f"✓ Registry created, has {len(registry.list_handlers())} handlers")
        
        # List all handlers
        print("\nRegistered handlers:")
        for key, handler_class in registry.list_handlers().items():
            print(f"  - {key}: {handler_class.__name__}")
        
    except Exception as e:
        print(f"✗ Failed to import handler registry: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test multimodal handler import
    try:
        from handlers.multimodal import MultimodalHandler
        print("\n✓ Successfully imported MultimodalHandler")
        
        # Test instantiation with dummy model info
        model_info = {
            "model_id": "test/model",
            "model_type": "multi_modality",
            "model_family": "multimodal"
        }
        handler = MultimodalHandler(model_info)
        print("✓ Successfully created MultimodalHandler instance")
        
    except Exception as e:
        print(f"\n✗ Failed to import/create MultimodalHandler: {e}")
        import traceback
        traceback.print_exc()
    
    # Test if handler can be found for multi_modality
    try:
        test_model_info = {
            "model_type": "multi_modality",
            "model_family": "multimodal"
        }
        handler_class = registry.get_handler_for_model(test_model_info)
        if handler_class:
            print(f"\n✓ Registry found handler for multi_modality: {handler_class.__name__}")
        else:
            print("\n✗ Registry did not find handler for multi_modality")
    except Exception as e:
        print(f"\n✗ Error getting handler from registry: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_handler_import.py <model_path>")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    test_handler_import(model_path)


if __name__ == "__main__":
    main()