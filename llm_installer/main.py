#!/usr/bin/env python3
"""
LLM Installer v2 - Main entry point
Universal installer for HuggingFace models
"""

import argparse
import sys
import logging
from pathlib import Path

from .checker import ModelChecker
from .utils import get_models_dir

__version__ = "2.0.0"


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def check_command(args: argparse.Namespace) -> int:
    """Check model compatibility without downloading"""
    try:
        checker = ModelChecker()
        profile = checker.check_model(args.model, save_profile=True)

        if profile:
            checker.print_compatibility_report(profile)
            return 0
        else:
            logging.error(f"Failed to check model: {args.model}")
            return 1

    except Exception as e:
        logging.error(f"Error checking model: {e}")
        if args.debug:
            logging.exception("Full traceback:")
        return 1


def install_command(args: argparse.Namespace) -> int:
    """Install model with all dependencies"""
    logging.info(f"Installing model: {args.model}")

    if args.quantization:
        logging.info(f"Using quantization: {args.quantization}")

    if args.path:
        logging.info(f"Installation path: {args.path}")

    # TODO: Implement model installation logic
    logging.info("Model installation functionality not yet implemented")

    return 0


def list_command(args: argparse.Namespace) -> int:
    """List installed models"""
    try:
        models_dir = get_models_dir()

        if not models_dir.exists():
            print("No models directory found. No models installed yet.")
            return 0

        # Find all model directories
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        if not model_dirs:
            print("No models installed yet.")
            return 0

        print(f"\nInstalled models in {models_dir}:")
        print("=" * 60)

        for model_dir in sorted(model_dirs):
            # Check if it has model files
            has_model = (model_dir / "model").exists()
            has_venv = (model_dir / ".venv").exists()

            status = "✓" if has_model and has_venv else "⚠"
            print(f"{status} {model_dir.name}")

            # Try to read model info
            model_info_file = model_dir / "model_info.json"
            if model_info_file.exists():
                try:
                    import json
                    with open(model_info_file) as f:
                        info = json.load(f)
                    print(f"  Type: {info.get('model_type', 'unknown')}")
                    print(f"  Library: {info.get('library', 'unknown')}")
                except Exception:
                    pass

        print("=" * 60)
        print(f"\nTotal: {len(model_dirs)} models")
        return 0

    except Exception as e:
        logging.error(f"Error listing models: {e}")
        return 1


def remove_command(args: argparse.Namespace) -> int:
    """Remove installed model"""
    logging.info(f"Removing model: {args.model}")

    # TODO: Implement removal logic
    logging.info("Model removal functionality not yet implemented")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        prog='llm-installer',
        description='Universal installer for HuggingFace models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llm-installer check meta-llama/Llama-3-8B
  llm-installer install meta-llama/Llama-3-8B --quantization 4bit
  llm-installer list
  llm-installer remove meta-llama/Llama-3-8B
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )

    # Check command
    check_parser = subparsers.add_parser(
        'check',
        help='Check model compatibility without downloading'
    )
    check_parser.add_argument(
        'model',
        help='HuggingFace model ID (e.g., meta-llama/Llama-3-8B)'
    )

    # Install command
    install_parser = subparsers.add_parser(
        'install',
        help='Install model with all dependencies'
    )
    install_parser.add_argument(
        'model',
        help='HuggingFace model ID (e.g., meta-llama/Llama-3-8B)'
    )
    install_parser.add_argument(
        '--quantization',
        choices=['4bit', '8bit', 'fp16', 'fp32'],
        help='Quantization level for model loading'
    )
    install_parser.add_argument(
        '--path',
        type=Path,
        help='Custom installation path (default: ~/LLM/models/)'
    )
    install_parser.add_argument(
        '--force',
        action='store_true',
        help='Force reinstall if model already exists'
    )

    # List command
    subparsers.add_parser(
        'list',
        help='List installed models'
    )

    # Remove command
    remove_parser = subparsers.add_parser(
        'remove',
        help='Remove installed model'
    )
    remove_parser.add_argument(
        'model',
        help='Model name to remove'
    )
    remove_parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )

    return parser


def main() -> int:
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    setup_logging(args.debug)

    # Dispatch to appropriate command handler
    command_handlers = {
        'check': check_command,
        'install': install_command,
        'list': list_command,
        'remove': remove_command,
    }

    handler = command_handlers.get(args.command)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            logging.info("Operation cancelled by user")
            return 130
        except Exception as e:
            logging.error(f"Error: {e}")
            if args.debug:
                logging.exception("Full traceback:")
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
