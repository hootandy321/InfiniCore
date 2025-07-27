# InfiniCore Architecture

This document provides a comprehensive overview of InfiniCore's architecture, design principles, and module organization.

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [System Architecture](#system-architecture)
3. [Module Breakdown](#module-breakdown)
4. [Hardware Abstraction](#hardware-abstraction)
5. [Memory Management](#memory-management)
6. [Error Handling](#error-handling)

## Design Philosophy

InfiniCore follows several key design principles:

### Hardware Abstraction
- **Unified Interface**: Single C API across all hardware platforms
- **Runtime Dispatch**: Dynamic selection of optimal implementation based on available hardware
- **Zero-Copy Operations**: Minimize memory transfers between host and device

### Performance First
- **Platform-Specific Optimizations**: Hand-tuned kernels for each hardware type
- **Asynchronous Execution**: Non-blocking operations with stream/queue support
- **Memory Pool Management**: Efficient memory allocation and reuse

### Extensibility
- **Plugin Architecture**: Easy addition of new hardware backends
- **Operator Framework**: Standardized way to implement new operations
- **Compile-Time Configuration**: Enable only needed backends to reduce binary size

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (User Code, Python Bindings, Model Implementations)        │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    InfiniCore C API                         │
│  (infiniop.h, infinirt.h, infiniccl.h, infinicore.h)       │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────┬──────────────────┬─────────────────────────┐
│   InfiniOP      │    InfiniRT      │      InfiniCCL          │
│ (Operators)     │  (Runtime)       │  (Communication)        │
├─────────────────┼──────────────────┼─────────────────────────┤
│ • GEMM          │ • Device Mgmt    │ • AllReduce             │
│ • Convolution   │ • Memory Alloc   │ • AllGather             │
│ • Attention     │ • Stream Mgmt    │ • Broadcast             │
│ • RMSNorm       │ • Error Handling │ • P2P Communication     │
│ • RoPE          │ • Type System    │ • Multi-GPU Coordination│
│ • Elementwise   │ • Sync Primitives│                         │
└─────────────────┴──────────────────┴─────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                  InfiniUtils                                │
│     (Common utilities, logging, data structures)            │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────┬──────────┬──────────┬──────────┬──────────────┐
│    CPU      │ NVIDIA   │ Ascend   │Cambricon │    ...       │
│            │  GPU     │   NPU    │   MLU    │              │
├─────────────┼──────────┼──────────┼──────────┼──────────────┤
│ • OpenMP    │ • CUDA   │ • CANN   │ • BANG   │ • Platform   │
│ • Intel MKL │ • cuBLAS │ • ACL    │ • CNNL   │   Specific   │
│ • OpenBLAS  │ • cuDNN  │ • AscendCL│ • CNRT   │   Libraries  │
│ • Eigen     │ • NCCL   │ • HCCL   │ • CNCL   │              │
└─────────────┴──────────┴──────────┴──────────┴──────────────┘
```

## Module Breakdown

### InfiniUtils
**Purpose**: Foundation utilities used across all modules

**Components**:
- **Logging System**: Unified logging with different verbosity levels
- **Data Structures**: Common containers, hash maps, arrays
- **Type System**: Data type definitions and conversions
- **Platform Utilities**: OS-specific functionality abstraction

**Key Files**:
- `src/utils/` - Implementation
- No public headers (internal use only)

### InfiniRT (Runtime)
**Purpose**: Device management, memory allocation, and execution context

**Components**:
- **Device Discovery**: Automatic detection of available hardware
- **Context Management**: Device contexts, streams, and synchronization
- **Memory Management**: Unified memory allocation interface
- **Error Handling**: Status codes and error reporting

**Key Files**:
- `include/infinirt.h` - Public API
- `src/infinirt/` - Implementation
- `src/infinirt/devices/` - Device-specific runtime code

**Core Types**:
```c
typedef struct InfinirtHandle *infinirtHandle_t;
typedef struct InfinirtContext *infinirtContext_t;
typedef struct InfinirtStream *infinirtStream_t;
```

### InfiniOP (Operators)
**Purpose**: Mathematical operations and neural network primitives

**Components**:
- **Basic Operations**: Element-wise operations (add, mul, relu, etc.)
- **Linear Algebra**: GEMM, convolution, matrix operations
- **Neural Network**: Attention, normalization, activation functions
- **Memory Operations**: Copy, reshape, transpose

**Key Files**:
- `include/infiniop.h` - Main public API
- `include/infiniop/ops/` - Individual operator headers
- `src/infiniop/ops/` - Operator implementations
- `src/infiniop/devices/` - Device-specific operator code

**Operator Structure**:
```c
// Each operator follows this pattern:
typedef struct InfiniopDescriptor *infiniopXXXDescriptor_t;

infiniStatus_t infiniopCreateXXXDescriptor(...);
infiniStatus_t infiniopGetXXXWorkspaceSize(...);
infiniStatus_t infiniopXXX(...);
infiniStatus_t infiniopDestroyXXXDescriptor(...);
```

### InfiniCCL (Communication)
**Purpose**: Multi-device and distributed computing primitives

**Components**:
- **Collective Operations**: AllReduce, AllGather, Broadcast, etc.
- **Point-to-Point**: Direct device-to-device communication
- **Topology Management**: Understanding device interconnects
- **Synchronization**: Cross-device synchronization primitives

**Key Files**:
- `include/infiniccl.h` - Public API
- `src/infiniccl/` - Implementation

## Hardware Abstraction

### Device Types
InfiniCore defines a unified device enumeration:

```c
typedef enum {
    INFINI_DEVICE_CPU = 0,
    INFINI_DEVICE_NVIDIA = 1,
    INFINI_DEVICE_CAMBRICON = 2,
    INFINI_DEVICE_ASCEND = 3,
    INFINI_DEVICE_METAX = 4,
    INFINI_DEVICE_MOORE = 5,
    INFINI_DEVICE_ILUVATAR = 6,
    INFINI_DEVICE_KUNLUN = 7,
    INFINI_DEVICE_SUGON = 8,
} infiniDevice_t;
```

### Runtime Dispatch
Each operator implementation uses compile-time conditionals and runtime dispatch:

```c
switch (handle->device) {
#ifdef ENABLE_CPU_API
    case INFINI_DEVICE_CPU:
        return cpu_implementation();
#endif
#ifdef ENABLE_NVIDIA_API
    case INFINI_DEVICE_NVIDIA:
        return nvidia_implementation();
#endif
    // ... other devices
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
```

### Platform-Specific Optimizations
- **CPU**: OpenMP parallelization, SIMD instructions, optimized BLAS
- **NVIDIA**: CUDA kernels, cuBLAS, cuDNN, Tensor Cores
- **Ascend**: CANN framework, ACL operators, NPU-specific optimizations
- **Others**: Platform-specific libraries and optimization techniques

## Memory Management

### Unified Memory Interface
InfiniCore provides a unified memory allocation interface that abstracts platform differences:

```c
// Runtime memory management
infiniStatus_t infinirtMalloc(void **ptr, size_t size, infinirtContext_t ctx);
infiniStatus_t infinirtFree(void *ptr, infinirtContext_t ctx);
infiniStatus_t infinirtMemcpy(void *dst, const void *src, size_t size, 
                             infinirtMemcpyKind_t kind, infinirtStream_t stream);
```

### Memory Types
- **Host Memory**: CPU-accessible memory
- **Device Memory**: GPU/NPU-accessible memory  
- **Unified Memory**: Accessible from both host and device (when supported)
- **Pinned Memory**: Page-locked host memory for faster transfers

### Memory Optimization Strategies
- **Memory Pools**: Pre-allocated memory pools to reduce allocation overhead
- **Workspace Management**: Temporary memory for operations
- **Zero-Copy**: Direct device-to-device transfers when possible

## Error Handling

### Status Codes
All InfiniCore functions return `infiniStatus_t` for consistent error handling:

```c
typedef enum {
    INFINI_STATUS_SUCCESS = 0,
    INFINI_STATUS_INTERNAL_ERROR = 1,
    INFINI_STATUS_NOT_IMPLEMENTED = 2,
    INFINI_STATUS_BAD_PARAM = 3,
    INFINI_STATUS_NULL_POINTER = 4,
    INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED = 5,
    // ... more status codes
} infiniStatus_t;
```

### Error Handling Best Practices
1. **Always Check Return Values**: Every API call should be checked
2. **Graceful Degradation**: Fall back to alternative implementations when possible
3. **Resource Cleanup**: Use RAII-style cleanup for C++ components
4. **Detailed Error Messages**: Provide context for debugging

### Debugging Support
- **Verbose Logging**: Enable with `DEBUG_MODE` compile flag
- **Parameter Validation**: Extensive input validation in debug builds
- **Memory Leak Detection**: Built-in tracking for memory allocations

## Thread Safety

InfiniCore is designed with thread safety in mind:
- **Context Isolation**: Each thread should use separate contexts
- **Handle Sharing**: Handles can be shared across threads with proper synchronization
- **Stream Independence**: Operations on different streams are thread-safe

## Next Steps

- **[Setup Guide](setup.md)**: Learn how to build and configure InfiniCore
- **[Operator Development](operators.md)**: Implement your first custom operator
- **[API Reference](api/)**: Detailed API documentation