#!/bin/bash

# LLM Installer v2 - Setup Script
# This script checks system requirements and sets up the development environment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            echo "ubuntu"
        elif command_exists yum; then
            echo "centos"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unsupported"
    fi
}

# Function to install system dependencies
install_system_deps() {
    local os_type=$1
    
    case $os_type in
        ubuntu)
            print_info "Installing system dependencies for Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv git curl
            ;;
        macos)
            print_info "Installing system dependencies for macOS..."
            if ! command_exists brew; then
                print_error "Homebrew is not installed. Please install it first:"
                print_error "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
            brew install python3 git
            ;;
        *)
            print_warning "Automatic installation not supported for your OS."
            print_warning "Please manually install: python3, pip3, python3-venv, git"
            ;;
    esac
}

# Main setup process
main() {
    print_info "LLM Installer v2 - Setup Script"
    print_info "================================"
    
    # Detect OS
    OS_TYPE=$(detect_os)
    print_info "Detected OS: $OS_TYPE"
    
    if [ "$OS_TYPE" == "unsupported" ]; then
        print_error "Your operating system is not supported."
        print_error "This installer only works on Linux and macOS."
        exit 1
    fi
    
    # Check for required commands
    print_info "Checking system requirements..."
    
    MISSING_DEPS=()
    
    if ! command_exists python3; then
        MISSING_DEPS+=("python3")
    fi
    
    if ! command_exists pip3 && ! python3 -m pip --version >/dev/null 2>&1; then
        MISSING_DEPS+=("pip3")
    fi
    
    if ! command_exists git; then
        MISSING_DEPS+=("git")
    fi
    
    # Check if python3-venv is available
    if ! python3 -m venv --help >/dev/null 2>&1; then
        MISSING_DEPS+=("python3-venv")
    fi
    
    # Install missing dependencies
    if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
        print_warning "Missing dependencies: ${MISSING_DEPS[*]}"
        read -p "Do you want to install them automatically? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_system_deps "$OS_TYPE"
        else
            print_error "Please install the missing dependencies manually and run this script again."
            exit 1
        fi
    else
        print_success "All system requirements are satisfied!"
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.8"
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_error "Python $REQUIRED_VERSION or higher is required. You have Python $PYTHON_VERSION"
        exit 1
    fi
    print_success "Python version $PYTHON_VERSION is compatible!"
    
    # Create virtual environment
    print_info "Setting up virtual environment..."
    
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists. Skipping creation."
    else
        python3 -m venv .venv
        print_success "Virtual environment created!"
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source .venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    print_info "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed from requirements.txt!"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Install package in development mode
    print_info "Installing llm-installer in development mode..."
    pip install -e .
    
    # Verify installation
    print_info "Verifying installation..."
    if llm-installer --version >/dev/null 2>&1; then
        print_success "LLM Installer is ready to use!"
        echo
        print_info "Activation command: source .venv/bin/activate"
        print_info "Usage: llm-installer --help"
    else
        print_error "Installation verification failed!"
        exit 1
    fi
    
    # Create default directories
    print_info "Creating default directories..."
    mkdir -p ~/LLM/models
    mkdir -p ~/LLM/cache
    print_success "Default directories created at ~/LLM/"
    
    # Final success message
    echo
    print_success "Setup completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment: source .venv/bin/activate"
    echo "2. Check a model: llm-installer check meta-llama/Llama-3-8B"
    echo "3. Install a model: llm-installer install meta-llama/Llama-3-8B"
    echo
    
    # Create a development config file if it doesn't exist
    if [ ! -f "config.yaml" ]; then
        cat > config.yaml << EOF
# LLM Installer Configuration
debug: false
cache_dir: ~/LLM/cache
models_dir: ~/LLM/models
log_dir: ./logs
EOF
        print_success "Created default config.yaml"
    fi
}

# Run main function
main "$@"