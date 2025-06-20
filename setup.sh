#!/bin/bash
# Setup script for LLM Installer v2
# Checks system requirements and installs dependencies

set -e

echo "==================================="
echo "LLM Installer v2 Setup"
echo "==================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    DISTRO=$(lsb_release -si 2>/dev/null || echo "Unknown")
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "❌ Error: Unsupported operating system: $OSTYPE"
    echo "This installer only supports Linux and macOS."
    exit 1
fi

echo "Detected OS: $OS"
[[ "$OS" == "linux" ]] && echo "Distribution: $DISTRO"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install system packages
install_system_packages() {
    local packages=("$@")
    
    if [[ "$OS" == "linux" ]]; then
        if command_exists apt-get; then
            echo "Installing with apt-get..."
            sudo apt-get update
            sudo apt-get install -y "${packages[@]}"
        elif command_exists yum; then
            echo "Installing with yum..."
            sudo yum install -y "${packages[@]}"
        elif command_exists dnf; then
            echo "Installing with dnf..."
            sudo dnf install -y "${packages[@]}"
        else
            echo "❌ Error: No supported package manager found"
            return 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command_exists brew; then
            echo "Installing with Homebrew..."
            brew install "${packages[@]}"
        else
            echo "❌ Error: Homebrew not found. Please install it first:"
            echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            return 1
        fi
    fi
}

# Check Python
echo "Checking Python..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "✓ Python $PYTHON_VERSION found"
    
    # Check if version is 3.8+
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
    
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
        echo "❌ Error: Python 3.8 or higher is required (found $PYTHON_VERSION)"
        exit 1
    fi
else
    echo "❌ Python 3 not found"
    echo "Attempting to install Python 3..."
    
    if [[ "$OS" == "linux" ]]; then
        install_system_packages python3 python3-pip python3-venv
    elif [[ "$OS" == "macos" ]]; then
        install_system_packages python3
    fi
    
    # Verify installation
    if ! command_exists python3; then
        echo "❌ Error: Failed to install Python 3"
        exit 1
    fi
fi

# Check venv module
echo ""
echo "Checking Python venv module..."
if python3 -c "import venv" 2>/dev/null; then
    echo "✓ venv module found"
else
    echo "❌ venv module not found"
    echo "Attempting to install python3-venv..."
    
    if [[ "$OS" == "linux" ]]; then
        install_system_packages python3-venv
    else
        echo "❌ Error: venv module is missing. Please reinstall Python 3."
        exit 1
    fi
fi

# Check Git
echo ""
echo "Checking Git..."
if command_exists git; then
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    echo "✓ Git $GIT_VERSION found"
else
    echo "❌ Git not found"
    echo "Attempting to install Git..."
    
    install_system_packages git
    
    # Verify installation
    if ! command_exists git; then
        echo "❌ Error: Failed to install Git"
        exit 1
    fi
fi

# Check for GPU support (optional)
echo ""
echo "Checking GPU support (optional)..."
if command_exists nvidia-smi; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
elif [[ "$OS" == "macos" ]] && system_profiler SPDisplaysDataType | grep -q "Metal"; then
    echo "✓ Apple Metal GPU detected"
else
    echo "⚠ No GPU detected. Models will run on CPU (slower)"
fi

# Create virtual environment if it doesn't exist
echo ""
echo "Setting up virtual environment..."
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
    echo "✓ Python dependencies installed"
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Install the package in development mode
echo ""
echo "Installing llm-installer in development mode..."
pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
if command_exists llm-installer; then
    echo "✓ llm-installer command is available"
    llm-installer --version
else
    echo "❌ Error: llm-installer command not found"
    exit 1
fi

# Create default directories
echo ""
echo "Creating default directories..."
mkdir -p ~/LLM/models
mkdir -p ~/.config/llm-installer
echo "✓ Directories created"

# Success message
echo ""
echo "==================================="
echo "✓ Setup completed successfully!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Save your HuggingFace token (for gated models):"
echo "   llm-installer save-token YOUR_HF_TOKEN"
echo ""
echo "2. Check a model:"
echo "   llm-installer check meta-llama/Llama-3-8B"
echo ""
echo "3. Install a model:"
echo "   llm-installer install meta-llama/Llama-3-8B"
echo ""
echo "To activate the virtual environment in the future:"
echo "   source .venv/bin/activate"
echo ""