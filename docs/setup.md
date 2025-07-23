# Developer Setup Guide

This guide walks you through setting up a development environment for InfiniCore, building the project, and running tests.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup](#quick-setup)
3. [Manual Setup](#manual-setup)
4. [Hardware-Specific Setup](#hardware-specific-setup)
5. [Building and Testing](#building-and-testing)
6. [IDE Configuration](#ide-configuration)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools
- **C/C++ Compiler**: GCC 7+ or Clang 6+ (MSVC 2019+ on Windows)
- **Build System**: [XMake](https://xmake.io/) 2.8.9+ (recommended on Windows)
- **Python**: 3.7+ (for tests and code generation)
- **Git**: For source code management

### Hardware Requirements
- **Minimum**: x86_64 CPU with 4GB RAM
- **Recommended**: 8GB+ RAM, SSD storage
- **Optional**: GPU/NPU hardware for accelerated backends

### Operating Systems
- **Linux**: Ubuntu 18.04+, CentOS 7+, or equivalent
- **Windows**: Windows 10+ (with Visual Studio 2019+)
- **macOS**: 10.15+ (limited support, CPU backend only)

## Quick Setup

### Automated Installation
Use the provided installation script for fastest setup:

```bash
# Clone the repository
git clone https://github.com/hootandy321/InfiniCore.git
cd InfiniCore

# Run automated installation (CPU backend only)
python scripts/install.py

# For GPU support (NVIDIA example)
python scripts/install.py --nv-gpu=y --cuda=$CUDA_HOME

# For multiple backends
python scripts/install.py --nv-gpu=y --ascend-npu=y --cambricon-mlu=y
```

### Verification
After installation, verify the setup:

```bash
# Set environment variables (follow the output instructions)
export INFINI_ROOT=$HOME/.infini
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH

# Run a simple test
python test/infiniop/add.py --cpu
```

## Manual Setup

For more control over the build process, follow these manual steps:

### 1. Install XMake
```bash
# Linux/macOS
curl -fsSL https://xmake.io/shget.text | bash

# Windows (PowerShell as Administrator)
Invoke-Expression (Invoke-WebRequest 'https://xmake.io/psget.text' -UseBasicParsing).Content

# Or install via package manager
# Ubuntu: apt install xmake
# Homebrew: brew install xmake
```

### 2. Configure Build Options
View available configuration options:
```bash
xmake f --help
```

Configure for your target platform:
```bash
# CPU-only build (default)
xmake f -cv

# View current configuration
xmake f -v
```

### 3. Build and Install
```bash
# Build all targets
xmake build

# Install to default location ($HOME/.infini)
xmake install

# Or specify custom installation directory
xmake install -o /path/to/install
```

### 4. Set Environment Variables
Add to your shell profile (`.bashrc`, `.zshrc`, etc.):
```bash
export INFINI_ROOT=$HOME/.infini  # or your custom path
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH
export CPATH=$INFINI_ROOT/include:$CPATH
```

## Hardware-Specific Setup

### NVIDIA GPU
```bash
# Ensure CUDA is installed and CUDA_HOME is set
export CUDA_HOME=/usr/local/cuda  # or your CUDA installation path

# Configure and build
xmake f --nv-gpu=y --cuda=$CUDA_HOME -cv
xmake build && xmake install

# Test NVIDIA backend
python test/infiniop/gemm.py --nvidia
```

**Requirements**:
- CUDA Toolkit 11.0+
- cuDNN 8.0+
- Compatible GPU with compute capability 6.0+

### Huawei Ascend NPU
```bash
# Ensure Ascend toolkit is installed
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Configure and build
xmake f --ascend-npu=y -cv
xmake build && xmake install

# Test Ascend backend
python test/infiniop/gemm.py --ascend
```

**Requirements**:
- Ascend Toolkit 6.0+
- Compatible Ascend hardware (910, 310P, etc.)

### Cambricon MLU
```bash
# Configure and build
xmake f --cambricon-mlu=y -cv
xmake build && xmake install

# Test Cambricon backend
python test/infiniop/gemm.py --cambricon
```

**Requirements**:
- Cambricon Neuware SDK
- Compatible MLU hardware

### Other Platforms
Similar patterns apply for other supported platforms:
- **MetaX GPU**: `--metax-gpu=y`
- **Moore Threads GPU**: `--moore-gpu=y`
- **Iluvatar GPU**: `--iluvatar-gpu=y`
- **Kunlun XPU**: `--kunlun-xpu=y`
- **Sugon DCU**: `--sugon-dcu=y`

## Building and Testing

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `--omp=[y\|n]` | Enable OpenMP | y |
| `--cpu=[y\|n]` | Build CPU backend | y |
| `--nv-gpu=[y\|n]` | Build NVIDIA GPU backend | n |
| `--ascend-npu=[y\|n]` | Build Ascend NPU backend | n |
| `--cambricon-mlu=[y\|n]` | Build Cambricon MLU backend | n |
| `--ccl=[y\|n]` | Build InfiniCCL communication library | n |
| `--ninetoothed=[y\|n]` | Build NineToothed (Triton) operators | n |

### Build Targets
```bash
# Build specific targets
xmake build infinirt      # Runtime library only
xmake build infiniop      # Operator library
xmake build infiniccl     # Communication library
xmake build all           # All libraries

# Build tests
xmake build infiniop-test      # Operator test framework
xmake build infiniccl-test     # Communication test
xmake build utils-test         # Utility tests
```

### Running Tests

#### Python Operator Tests
```bash
# Run single operator test
python test/infiniop/gemm.py --cpu
python test/infiniop/conv.py --nvidia

# Run all tests for a backend
python scripts/python_test.py --cpu
python scripts/python_test.py --nvidia
```

#### GGUF Test Framework
```bash
# Build test framework
xmake build infiniop-test

# Generate test cases
cd test/infiniop-test
python -m test_generate.testcases.gemm

# Run tests
infiniop-test gemm.gguf --cpu --warmup 20 --run 1000
```

#### Communication Tests
```bash
# Build and run communication tests
xmake build infiniccl-test
infiniccl-test --nvidia  # Uses all visible GPUs
```

### Performance Profiling
```bash
# Enable profiling in Python tests
python test/infiniop/gemm.py --nvidia --profile

# Use external profilers
nsys profile --trace=cuda python test/infiniop/gemm.py --nvidia
```

## IDE Configuration

### Visual Studio Code

#### Required Extensions
- **C/C++** (Microsoft)
- **clangd** (LLVM)
- **XMake** (xmake-io)

#### Configuration
Create `.vscode/settings.json`:
```json
{
    "clangd.arguments": [
        "--compile-commands-dir=.vscode"
    ],
    "xmake.additionalConfigArguments": [
        "--nv-gpu=y",
        "--cpu=y"
    ],
    "C_Cpp.intelliSenseEngine": "Disabled",
    "files.associations": {
        "*.h": "c",
        "*.cuh": "cuda-cpp"
    }
}
```

#### Generating Compile Commands
Save `xmake.lua` to automatically generate `.vscode/compile_commands.json` for IntelliSense.

### Other IDEs
- **CLion**: Import as XMake project
- **Qt Creator**: Use compile_commands.json
- **Vim/Neovim**: Use clangd LSP with compile_commands.json

## Code Formatting

InfiniCore uses `clang-format-16` for C/C++ and `black` for Python:

```bash
# Check formatting
python scripts/format.py --check

# Apply formatting
python scripts/format.py

# Format specific files
python scripts/format.py --path src/infiniop/ops/gemm/

# Format changes since commit
python scripts/format.py --ref HEAD~1
```

## NineToothed Integration

NineToothed is a Triton-based DSL for high-performance operator development:

### Prerequisites
```bash
# Install NineToothed and ntops
git clone https://github.com/InfiniTensor/ntops.git
cd ntops
pip install -e .  # This also installs ninetoothed
```

### Building with NineToothed
```bash
# Generate operator source files
PYTHONPATH=${PYTHONPATH}:src python scripts/build_ntops.py

# Build with NineToothed support
python scripts/install.py --ninetoothed=y --nv-gpu=y
```

## Environment Management

### Conda Environment
```bash
# Create dedicated environment
conda create -n infinicore python=3.9
conda activate infinicore

# Install Python dependencies
pip install torch numpy pytest black
```

### Docker Setup
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    git \
    curl

# Install XMake
RUN curl -fsSL https://xmake.io/shget.text | bash

# Clone and build InfiniCore
COPY . /opt/InfiniCore
WORKDIR /opt/InfiniCore
RUN python scripts/install.py --nv-gpu=y
```

## Troubleshooting

### Common Issues

#### XMake Not Found
```bash
# Add XMake to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### CUDA Not Found
```bash
# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### Library Loading Errors
```bash
# Check library paths
ldd $INFINI_ROOT/lib/libinfiniop.so

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH
```

#### Python Import Errors
```bash
# Ensure PYTHONPATH includes test directory
export PYTHONPATH=$PWD/test:$PYTHONPATH

# Check if libraries are accessible
python -c "from test.infiniop.libinfiniop import LIBINFINIOP; print('OK')"
```

### Debug Builds
```bash
# Build in debug mode for more verbose output
xmake f -m debug -cv
xmake build
```

### Getting Help
- Check [Troubleshooting Guide](troubleshooting.md)
- Search existing [GitHub Issues](https://github.com/hootandy321/InfiniCore/issues)
- Create a new issue with detailed error information

## Next Steps

Now that you have InfiniCore set up:
1. **[Learn the Architecture](architecture.md)** - Understand the system design
2. **[Develop Operators](operators.md)** - Create your first custom operator
3. **[Integrate Models](models.md)** - Adapt large language models
4. **[Optimize Performance](performance.md)** - Tune for maximum efficiency