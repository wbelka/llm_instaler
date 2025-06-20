#!/bin/bash

# LLM Installer Setup Script
# This script performs initial setup and system checks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Header
echo "======================================"
echo "    LLM Installer v2 Setup"
echo "======================================"
echo ""

# Check operating system
print_info "Checking operating system..."
OS=$(uname -s)
if [[ "$OS" == "MINGW"* ]] || [[ "$OS" == "CYGWIN"* ]] || [[ "$OS" == "MSYS"* ]]; then
    print_error "Windows is not supported. Please use Linux or macOS."
    exit 1
elif [[ "$OS" == "Darwin" ]]; then
    print_success "macOS detected"
elif [[ "$OS" == "Linux" ]]; then
    print_success "Linux detected"
else
    print_warning "Unknown OS: $OS. Proceeding with caution..."
fi

# Check Python version
print_info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher is required. Found: Python $PYTHON_VERSION"
    exit 1
fi
print_success "Python $PYTHON_VERSION found"

# Check disk space
print_info "Checking available disk space..."
if [[ "$OS" == "Darwin" ]]; then
    # macOS
    AVAILABLE_GB=$(df -g ~ | awk 'NR==2 {print $4}')
else
    # Linux
    AVAILABLE_GB=$(df -BG ~ | awk 'NR==2 {print $4}' | sed 's/G//')
fi

if [ "$AVAILABLE_GB" -lt 50 ]; then
    print_error "Insufficient disk space. At least 50GB required, found: ${AVAILABLE_GB}GB"
    exit 1
elif [ "$AVAILABLE_GB" -lt 100 ]; then
    print_warning "Low disk space: ${AVAILABLE_GB}GB available. Recommended: 100GB+"
else
    print_success "${AVAILABLE_GB}GB available disk space"
fi

# Check CUDA/nvidia-smi (optional)
print_info "Checking for CUDA/GPU support..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    print_success "CUDA $CUDA_VERSION detected with GPU: $GPU_INFO"
elif [[ "$OS" == "Darwin" ]] && [[ $(uname -m) == "arm64" ]]; then
    print_success "Apple Silicon detected - Metal Performance Shaders available"
else
    print_warning "No CUDA support detected. Models will run on CPU (slower)"
fi

# Create directories
print_info "Creating directory structure..."
mkdir -p ~/LLM/models
mkdir -p ~/LLM/cache
mkdir -p ~/LLM/logs
mkdir -p ~/LLM/logs/checks
print_success "Directories created"

# Create virtual environment
print_info "Creating Python virtual environment..."
if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv .venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_info "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_success "Pip upgraded"

# Install dependencies
print_info "Installing Python dependencies..."
if pip install -r requirements.txt; then
    print_success "Dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Copy config file
print_info "Setting up configuration..."
if [ ! -f "config.yaml" ]; then
    if [ -f "config.yaml.example" ]; then
        cp config.yaml.example config.yaml
        print_success "Configuration file created from example"
        print_info "Please edit config.yaml to customize settings"
    else
        print_error "config.yaml.example not found!"
        exit 1
    fi
else
    print_info "config.yaml already exists, skipping"
fi

# Add to shell configuration
print_info "Setting up shell alias..."
SHELL_CONFIG=""
if [ -f ~/.zshrc ]; then
    SHELL_CONFIG=~/.zshrc
elif [ -f ~/.bashrc ]; then
    SHELL_CONFIG=~/.bashrc
fi

if [ ! -z "$SHELL_CONFIG" ]; then
    INSTALLER_PATH=$(pwd)
    ALIAS_LINE="alias llm-installer='$INSTALLER_PATH/llm-installer'"
    
    if ! grep -q "llm-installer" "$SHELL_CONFIG"; then
        echo "" >> "$SHELL_CONFIG"
        echo "# LLM Installer alias" >> "$SHELL_CONFIG"
        echo "$ALIAS_LINE" >> "$SHELL_CONFIG"
        print_success "Added llm-installer alias to $SHELL_CONFIG"
        print_info "Run 'source $SHELL_CONFIG' or restart your terminal to use the alias"
    else
        print_info "Alias already exists in $SHELL_CONFIG"
    fi
fi

# Create marker file
touch .setup_complete

# Summary
echo ""
echo "======================================"
echo "    Setup Complete!"
echo "======================================"
echo ""
print_success "LLM Installer has been set up successfully!"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml if needed"
echo "2. Set your HuggingFace token: export HF_TOKEN='your-token'"
echo "3. Run: ./llm-installer --help"
echo ""
echo "Installation directory: $(pwd)"
echo "Models will be installed to: ~/LLM/models"
echo ""