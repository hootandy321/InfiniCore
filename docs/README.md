# InfiniCore Documentation

Welcome to the comprehensive documentation for InfiniCore - a cross-platform unified programming toolkit for AI/ML operations across different hardware platforms.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Documentation Structure](#documentation-structure)
4. [Getting Help](#getting-help)

## Project Overview

InfiniCore provides unified C language interfaces for different chip platforms, enabling developers to write hardware-agnostic AI/ML code that runs efficiently across:

- **CPU** platforms
- **NVIDIA GPUs** 
- **Moore Threads GPUs**
- **Iluvatar GPUs** (天数智芯)
- **MetaX GPUs** (沐曦)
- **Sugon DCU** (曙光)
- **Huawei Ascend NPUs** (华为昇腾)
- **Cambricon MLUs** (寒武纪)
- **Kunlun XPUs** (昆仑芯)

### Key Features

- **Unified API**: Single C interface for all supported hardware
- **High Performance**: Optimized implementations for each platform
- **Extensible**: Easy to add new operators and hardware backends
- **Comprehensive**: Covers computing, runtime, and communication operations
- **Production Ready**: Used in real-world AI/ML deployments

## Quick Start

1. **Setup**: Follow the [Developer Setup Guide](setup.md)
2. **Build**: Learn how to build and install InfiniCore
3. **First Operator**: Create your first custom operator using our [Operator Development Guide](operators.md)
4. **Model Integration**: Adapt large models like Qwen3 with our [Model Adaptation Guide](models.md)
5. **Optimization**: Improve performance using our [Performance Guide](performance.md)

## Documentation Structure

### Core Documentation
- **[Architecture](architecture.md)** - System design and module overview
- **[Setup Guide](setup.md)** - Development environment setup and building
- **[Operator Development](operators.md)** - How to implement new operators
- **[Model Adaptation](models.md)** - Integrating large language models
- **[Performance Optimization](performance.md)** - Tuning for maximum performance

### Reference Materials
- **[API Reference](api/)** - Complete API documentation
- **[Examples](examples/)** - Sample code and tutorials
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

### Advanced Topics
- **Hardware-Specific Guides** - Platform-specific optimization tips
- **Communication Library (InfiniCCL)** - Multi-GPU/distributed computing
- **NineToothed Integration** - Using Triton-based operator implementations

## Getting Help

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/hootandy321/InfiniCore/issues)
- **Discussions**: Join community discussions
- **Documentation**: This documentation is continuously updated

---

## Contributing to Documentation

We welcome contributions to improve this documentation! Please:

1. Follow the existing structure and style
2. Test any code examples
3. Submit pull requests with clear descriptions
4. Check for spelling and grammar

For more details, see the main [DEV.md](../DEV.md) file.