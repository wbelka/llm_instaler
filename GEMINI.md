## Gemini Development Guide

This document outlines the key aspects of the LLM Installer project and provides guidance for Gemini on how to interact with and extend the system.

## Core Concepts

The LLM Installer is a tool designed to simplify the installation, management, and use of Large Language Models (LLMs) from sources like HuggingFace. It automates dependency management, environment setup, and provides a unified interface for various model types.

### Key Principles:
- **Automation:** Minimize manual configuration and setup.
- **Isolation:** Each model runs in its own isolated environment.
- **Extensibility:** New model architectures and types can be added with minimal effort.
- **User-Friendliness:** Provide a simple and intuitive command-line interface.

## Project Structure

The project is organized into several key directories:

- `core/`: Contains the main application logic, including the installer, checker, and configuration management.
- `handlers/`: Implements the logic for specific model types (e.g., transformers, diffusion models).
- `detectors/`: Contains the logic for identifying model types based on their configuration and file structure.
- `scripts/`: Provides universal scripts that are copied into each installed model's directory for easy execution.
- `tests/`: Contains unit and integration tests for the project.

## Development Workflow

When adding support for a new model or modifying existing functionality, please follow these steps:

1.  **Understand the Request:** Carefully analyze the user's request to determine the scope of the changes.
2.  **Explore the Codebase:** Use the available tools to read relevant files and understand the existing code, conventions, and architecture.
3.  **Formulate a Plan:** Create a clear and concise plan for implementing the changes.
4.  **Implement the Changes:** Use the `replace` or `write_file` tools to modify the code.
5.  **Test the Changes:** If possible, run existing tests or create new ones to verify the changes.
6.  **Commit the Changes:** Once the changes are verified, commit them to the repository with a clear and descriptive commit message.

## Important Files

- `GEMINI.md`: This file. Your primary source of information about the project and your role in it.
- `README.md`: Provides a general overview of the project.
- `DETECTION_ALGORITHM.md`: Explains how the system identifies model types.
- `HANDLER_DEVELOPMENT_GUIDE.md`: A detailed guide to creating new model handlers.
- `QUANTIZATION_SUPPORT.md`: Describes how quantization is implemented and supported.

## How to Approach Common Tasks

### Adding a New Model Handler

1.  **Create a new handler file** in the `handlers/` directory (e.g., `my_handler.py`).
2.  **Implement the `BaseHandler` interface**, including methods for dependency management, model loading, and inference.
3.  **Create a new detector** in the `detectors/` directory to identify the new model type.
4.  **Register the new handler and detector** in the corresponding registry files.
5.  **Add a new test case** in the `tests/` directory to verify the new handler.

### Modifying an Existing Handler

1.  **Locate the handler file** in the `handlers/` directory.
2.  **Read the file** to understand its current implementation.
3.  **Make the necessary changes** using the `replace` tool.
4.  **Run the corresponding tests** to ensure that the changes have not introduced any regressions.

### Fixing a Bug

1.  **Identify the source of the bug** by reading the relevant code and logs.
2.  **Formulate a plan** to fix the bug.
3.  **Implement the fix** using the `replace` tool.
4.  **Add a new test case** that reproduces the bug and verifies the fix.

## Available Tools

You have access to a variety of tools to help you with your tasks. These include:

- `list_directory`: List the contents of a directory.
- `read_file`: Read the contents of a file.
- `search_file_content`: Search for a pattern in the contents of files.
- `glob`: Find files matching a specific pattern.
- `replace`: Replace text within a file.
- `write_file`: Write content to a file.
- `run_shell_command`: Execute a shell command.

Please use these tools to interact with the file system and the codebase.

## Final Notes

This project is under active development, and your contributions are valuable. Please adhere to the existing coding style and conventions, and do not hesitate to ask for clarification if you are unsure about anything.