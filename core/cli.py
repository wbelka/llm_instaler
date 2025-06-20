"""Command-line interface for LLM Installer.

This module implements the CLI commands using Click framework.
"""

import sys
import click
from pathlib import Path
from typing import Optional

from core.config import get_config, ConfigError
from core.utils import (
    setup_logging, check_system_requirements, print_system_info,
    print_error, print_success, print_warning, print_info,
    console, get_models_dir, safe_model_name
)

# Version
VERSION = "2.0.0"


@click.group()
@click.version_option(version=VERSION, prog_name="LLM Installer")
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx: click.Context, debug: bool):
    """LLM Installer v2 - Automated installation and management of LLM models.

    This tool helps you check compatibility, install, and manage Large Language Models
    from HuggingFace with automatic dependency management and environment isolation.
    """
    # Set up context
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug

    try:
        # Initialize configuration
        config = get_config()
        ctx.obj['config'] = config

        # Set up logging
        logger = setup_logging(debug_mode=debug)
        ctx.obj['logger'] = logger

        if debug:
            logger.debug("Debug mode enabled")
            logger.debug(f"Configuration loaded from: {config.config_path}")

    except ConfigError as e:
        print_error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Initialization error: {e}")
        if debug:
            raise
        sys.exit(1)


@cli.command()
@click.argument('model_id')
@click.pass_context
def check(ctx: click.Context, model_id: str):
    """Check model compatibility without downloading.

    MODEL_ID: HuggingFace model identifier (e.g., meta-llama/Llama-3-8B)

    This command analyzes the model's requirements and checks if your system
    can run it, without downloading any model weights.
    """
    logger = ctx.obj['logger']
    logger.info(f"Checking model: {model_id}")

    # For now, show a placeholder message
    print_info(f"Checking compatibility for model: {model_id}")
    console.print("\n[yellow]Note:[/yellow] Check functionality will be implemented in Step 2")

    # Show system info as example
    print_info("Analyzing system capabilities...")
    system_info = check_system_requirements()
    print_system_info(system_info)


@cli.command()
@click.argument('model_id')
@click.option('--device', default='auto', help='Device to use (auto/cuda/cpu/mps)')
@click.option('--dtype', default='auto', help='Data type (auto/float16/float32/int8/int4)')
@click.option('--quantization', type=click.Choice(['none', '8bit', '4bit']), default='none',
              help='Quantization method')
@click.option('--force', '-f', is_flag=True, help='Force reinstall if already exists')
@click.pass_context
def install(ctx: click.Context, model_id: str, device: str, dtype: str,
            quantization: str, force: bool):
    """Install a model from HuggingFace.

    MODEL_ID: HuggingFace model identifier (e.g., meta-llama/Llama-3-8B)

    This command downloads the model, sets up an isolated environment,
    installs all dependencies, and prepares the model for use.
    """
    logger = ctx.obj['logger']
    logger.info(f"Installing model: {model_id}")

    # Check if model already exists
    models_dir = get_models_dir()
    model_dir = models_dir / safe_model_name(model_id)

    if model_dir.exists() and not force:
        print_error(f"Model already installed at: {model_dir}")
        print_info("Use --force to reinstall")
        return

    print_info(f"Installing model: {model_id}")
    console.print("\n[yellow]Note:[/yellow] Install functionality will be implemented in Step 3")


@cli.command(name='list')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed information')
@click.pass_context
def list_models(ctx: click.Context, detailed: bool):
    """List installed models.

    Shows all models that have been installed using this tool,
    along with their status and basic information.
    """
    logger = ctx.obj['logger']
    logger.info("Listing installed models")

    models_dir = get_models_dir()

    if not models_dir.exists():
        print_info("No models directory found")
        return

    # Find all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not model_dirs:
        print_info("No models installed yet")
        return

    console.print(f"\n[bold]Installed Models[/bold] ({len(model_dirs)} total)\n")

    for model_dir in sorted(model_dirs):
        # Check if installation is complete
        if (model_dir / '.install_complete').exists():
            status = "[green] Ready[/green]"
        else:
            status = "[yellow]  Incomplete[/yellow]"

        console.print(f"{status} {model_dir.name}")

        if detailed:
            # Show more information if available
            model_info_file = model_dir / 'model_info.json'
            if model_info_file.exists():
                # Would load and display model info here
                console.print(f"   Path: {model_dir}")


@cli.command()
@click.option('--show-paths', is_flag=True, help='Show expanded paths')
@click.pass_context
def config(ctx: click.Context, show_paths: bool):
    """Show current configuration.

    Displays the active configuration settings, including paths,
    tokens, and system settings.
    """
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    logger.info("Showing configuration")

    console.print("\n[bold]LLM Installer Configuration[/bold]\n")

    # Create configuration table
    from rich.table import Table

    table = Table(show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Add configuration values
    config_dict = config.to_dict()

    # Group settings
    groups = {
        'Paths': ['models_dir', 'cache_dir', 'logs_dir'],
        'Installation': ['default_device', 'max_download_workers', 'resume_downloads'],
        'System': ['min_disk_space_gb', 'warn_disk_space_gb'],
        'Logging': ['log_level', 'log_rotation'],
        'Authentication': ['huggingface_token']
    }

    for group_name, keys in groups.items():
        table.add_row(f"[bold]{group_name}[/bold]", "")
        for key in keys:
            value = config_dict.get(key, 'Not set')

            # Hide token value for security
            if key == 'huggingface_token':
                if value:
                    value = '***' + value[-4:] if len(value) > 4 else '****'
                else:
                    value = '[dim]Not set[/dim]'

            # Format paths
            elif key.endswith('_dir') and not show_paths:
                # Show relative paths unless --show-paths is used
                value = value.replace(str(Path.home()), '~')

            table.add_row(f"  {key}", str(value))

    console.print(table)

    # Show config file location
    console.print(f"\n[dim]Configuration file: {config.config_path}[/dim]")


@cli.command()
@click.pass_context
def doctor(ctx: click.Context):
    """Run system diagnostics.

    Checks your system configuration, dependencies, and readiness
    for installing and running LLM models.
    """
    logger = ctx.obj['logger']
    logger.info("Running system diagnostics")

    console.print("\n[bold]System Diagnostics[/bold]\n")

    # Check system requirements
    print_info("Checking system requirements...")
    system_info = check_system_requirements()

    # Validate system
    issues = []
    warnings = []

    # Check Python version
    python_version = tuple(map(int, system_info['python_version'].split('.')[:2]))
    if python_version < (3, 8):
        issues.append(f"Python {system_info['python_version']} is too old (need >= 3.8)")

    # Check memory
    if system_info['available_memory_gb'] < 8:
        warnings.append(f"Low memory: {system_info['available_memory_gb']:.1f} GB available")

    # Check disk space
    config = ctx.obj['config']
    home_free = system_info['disk_space_gb']['home']['free']
    if home_free < config.min_disk_space_gb:
        issues.append(f"Insufficient disk space: {home_free:.1f} GB (need >= {config.min_disk_space_gb} GB)")
    elif home_free < config.warn_disk_space_gb:
        warnings.append(f"Low disk space: {home_free:.1f} GB")

    # Display results
    print_system_info(system_info)

    console.print("\n[bold]Diagnostic Results[/bold]\n")

    if not issues and not warnings:
        print_success("All checks passed! Your system is ready.")
    else:
        if issues:
            console.print("[red]Issues found:[/red]")
            for issue in issues:
                console.print(f"  • {issue}")

        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")

    # Check configuration
    console.print("\n[bold]Configuration Status[/bold]\n")

    config_checks = {
        'Configuration file': config.config_path.exists(),
        'Models directory': config.models_dir.exists(),
        'Cache directory': config.cache_dir.exists(),
        'Logs directory': config.logs_dir.exists(),
        'HuggingFace token': bool(config.huggingface_token)
    }

    for check, status in config_checks.items():
        if status:
            console.print(f"[green][/green] {check}")
        else:
            console.print(f"[yellow] [/yellow] {check}")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
