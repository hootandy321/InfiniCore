# Operator Development Guide

This guide teaches you how to implement custom operators in InfiniCore, from simple element-wise operations to complex neural network primitives.

## Table of Contents

1. [Overview](#overview)
2. [Operator Anatomy](#operator-anatomy)
3. [Quick Start: Add Operator](#quick-start-add-operator)
4. [Step-by-Step Tutorial](#step-by-step-tutorial)
5. [Advanced Topics](#advanced-topics)
6. [Testing and Validation](#testing-and-validation)
7. [Performance Optimization](#performance-optimization)
8. [Common Patterns](#common-patterns)

## Overview

InfiniCore operators follow a standardized pattern that provides:
- **Hardware Abstraction**: Single implementation works across all supported devices
- **Type Safety**: Strong type checking and validation
- **Memory Management**: Automatic workspace handling
- **Error Handling**: Consistent error reporting
- **Performance**: Optimized kernels for each platform

### Operator Categories

1. **Element-wise Operations**: add, mul, relu, etc.
2. **Linear Algebra**: gemm, conv, matmul
3. **Neural Network Primitives**: attention, normalization, activation
4. **Memory Operations**: copy, reshape, transpose
5. **Communication**: collective operations (via InfiniCCL)

## Operator Anatomy

Every InfiniCore operator consists of these components:

```
src/infiniop/ops/[operator_name]/
├── operator.cc              # Main C API implementation
├── [device1]/               # Device-specific implementations
│   ├── [op]_[device1].h    # Device header
│   └── [op]_[device1].cc   # Device implementation
├── [device2]/
│   ├── [op]_[device2].cuh  # CUDA header (.cuh for GPU)
│   └── [op]_[device2].cu   # CUDA implementation
└── ...

include/infiniop/ops/
└── [operator_name].h        # Public API header
```

### Core API Pattern

Every operator follows this C API pattern:

```c
// 1. Create descriptor (validates inputs, prepares execution)
infiniStatus_t infiniopCreate[Op]Descriptor(
    infiniopHandle_t handle,
    infiniop[Op]Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input1_desc,
    infiniopTensorDescriptor_t input2_desc,
    /* additional parameters */);

// 2. Query workspace requirements
infiniStatus_t infiniopGet[Op]WorkspaceSize(
    infiniop[Op]Descriptor_t desc, 
    size_t *size);

// 3. Execute operation
infiniStatus_t infiniop[Op](
    infiniop[Op]Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input1,
    const void *input2,
    /* operation-specific parameters */
    void *stream);

// 4. Cleanup
infiniStatus_t infiniopDestroy[Op]Descriptor(infiniop[Op]Descriptor_t desc);
```

## Quick Start: Add Operator

Let's examine the `add` operator as a simple example:

### Public API Header
```c
// include/infiniop/ops/add.h
#ifndef __INFINIOP_ADD_API_H__
#define __INFINIOP_ADD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAddDescriptor_t;

__C __export infiniStatus_t infiniopCreateAddDescriptor(
    infiniopHandle_t handle,
    infiniopAddDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c,     // output
    infiniopTensorDescriptor_t a,     // input 1
    infiniopTensorDescriptor_t b);    // input 2

__C __export infiniStatus_t infiniopGetAddWorkspaceSize(
    infiniopAddDescriptor_t desc, 
    size_t *size);

__C __export infiniStatus_t infiniopAdd(
    infiniopAddDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,            // output
    const void *a,      // input 1
    const void *b,      // input 2
    void *stream);

__C __export infiniStatus_t infiniopDestroyAddDescriptor(
    infiniopAddDescriptor_t desc);

#endif
```

### Main Implementation
```c
// src/infiniop/ops/add/operator.cc
#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/add.h"

// Include device-specific headers
#ifdef ENABLE_CPU_API
#include "cpu/add_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/add_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateAddDescriptor(
    infiniopHandle_t handle,
    infiniopAddDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

    // Dispatch to device-specific implementation
    switch (handle->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return op::add::cpu::Descriptor::create(
                handle,
                reinterpret_cast<op::add::cpu::Descriptor **>(desc_ptr),
                c_desc, {a_desc, b_desc});
#endif
#ifdef ENABLE_NVIDIA_API
        case INFINI_DEVICE_NVIDIA:
            return op::add::nvidia::Descriptor::create(
                handle,
                reinterpret_cast<op::add::nvidia::Descriptor **>(desc_ptr),
                c_desc, {a_desc, b_desc});
#endif
        default:
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

// Similar pattern for other API functions...
```

## Step-by-Step Tutorial

Let's create a new operator called `square` that computes element-wise square.

### Step 1: Define Public API

Create `include/infiniop/ops/square.h`:

```c
#ifndef __INFINIOP_SQUARE_API_H__
#define __INFINIOP_SQUARE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSquareDescriptor_t;

__C __export infiniStatus_t infiniopCreateSquareDescriptor(
    infiniopHandle_t handle,
    infiniopSquareDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc);

__C __export infiniStatus_t infiniopGetSquareWorkspaceSize(
    infiniopSquareDescriptor_t desc, 
    size_t *size);

__C __export infiniStatus_t infiniopSquare(
    infiniopSquareDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__C __export infiniStatus_t infiniopDestroySquareDescriptor(
    infiniopSquareDescriptor_t desc);

#endif
```

### Step 2: Create Directory Structure

```bash
mkdir -p src/infiniop/ops/square/cpu
mkdir -p src/infiniop/ops/square/nvidia
```

### Step 3: Implement CPU Version

Create `src/infiniop/ops/square/cpu/square_cpu.h`:

```c
#ifndef __SQUARE_CPU_H__
#define __SQUARE_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

// Use elementwise framework for simple operations
ELEMENTWISE_DESCRIPTOR(square, cpu)

namespace op::square::cpu {

struct SquareOp {
    static constexpr size_t num_inputs = 1;
    
    template <typename T>
    T operator()(const T &input) const {
        return input * input;
    }
};

} // namespace op::square::cpu

#endif
```

Create `src/infiniop/ops/square/cpu/square_cpu.cc`:

```c
#include "square_cpu.h"

namespace op::square::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();
    
    // Validate inputs
    const auto &input_desc = input_desc_vec.at(0);
    const auto &output_shape = out_desc->shape();
    const auto &input_shape = input_desc->shape();
    
    // Check supported data types
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    
    // Check shapes match
    CHECK_SAME_SHAPE(output_shape, input_shape);
    
    // Create descriptor using elementwise framework
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<SquareOp, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<SquareOp, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<SquareOp, double>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::square::cpu
```

### Step 4: Implement Main Operator

Create `src/infiniop/ops/square/operator.cc`:

```c
#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/square.h"

#ifdef ENABLE_CPU_API
#include "cpu/square_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/square_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateSquareDescriptor(
    infiniopHandle_t handle,
    infiniopSquareDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc) {

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return op::square::cpu::Descriptor::create(
                handle,
                reinterpret_cast<op::square::cpu::Descriptor **>(desc_ptr),
                output_desc, {input_desc});
#endif
#ifdef ENABLE_NVIDIA_API
        case INFINI_DEVICE_NVIDIA:
            return op::square::nvidia::Descriptor::create(
                handle,
                reinterpret_cast<op::square::nvidia::Descriptor **>(desc_ptr),
                output_desc, {input_desc});
#endif
        default:
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopGetSquareWorkspaceSize(
    infiniopSquareDescriptor_t desc, 
    size_t *size) {
    return static_cast<InfiniopOperatorDescriptor *>(desc)->getWorkspaceSize(size);
}

__C infiniStatus_t infiniopSquare(
    infiniopSquareDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {
    
    return static_cast<InfiniopOperatorDescriptor *>(desc)->calculate(
        workspace, workspace_size, output, {input}, stream);
}

__C infiniStatus_t infiniopDestroySquareDescriptor(infiniopSquareDescriptor_t desc) {
    delete static_cast<InfiniopOperatorDescriptor *>(desc);
    return INFINI_STATUS_SUCCESS;
}
```

### Step 5: Register in Build System

Edit `src/infiniop/ops/*/operator.cc` files to ensure they're built. The build system automatically includes all `operator.cc` files.

### Step 6: Add to Public Header

Edit `include/infiniop.h` to include your new operator:

```c
#include "infiniop/ops/square.h"
```

### Step 7: Create Test

Create `test/infiniop/square.py`:

```python
import torch
from libinfiniop import (
    LIBINFINIOP, TestTensor, get_test_devices, check_error,
    test_operator, get_args, TestWorkspace, InfiniDtype
)

# Test cases: (shape, input_stride, output_stride)
_TEST_CASES = [
    ((10, 20), None, None),
    ((5, 4, 3), None, None),
    ((100,), (2,), (2,)),  # Test with custom strides
]

def test_square_op(device, dtype, shape, input_stride, output_stride):
    """Test square operation"""
    
    # Create tensors
    input_tensor = TestTensor(shape, dtype, device, stride=input_stride)
    output_tensor = TestTensor(shape, dtype, device, stride=output_stride)
    workspace = TestWorkspace()
    
    # Fill with test data
    input_data = torch.randn(shape, dtype=torch.float32)
    input_tensor.from_torch(input_data)
    
    # Create descriptor
    desc = LIBINFINIOP.infiniopCreateSquareDescriptor(
        device.handle,
        output_tensor.descriptor,
        input_tensor.descriptor
    )
    
    # Get workspace size
    workspace_size = LIBINFINIOP.infiniopGetSquareWorkspaceSize(desc)
    workspace.malloc(workspace_size)
    
    # Execute operation
    check_error(LIBINFINIOP.infiniopSquare(
        desc,
        workspace.ptr, workspace.size,
        output_tensor.ctypes_ptr,
        input_tensor.ctypes_ptr,
        device.stream
    ))
    
    # Verify result
    expected = input_data * input_data
    actual = output_tensor.to_torch()
    
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    
    # Cleanup
    LIBINFINIOP.infiniopDestroySquareDescriptor(desc)

def main():
    args = get_args()
    devices = get_test_devices(args)
    
    for device in devices:
        for dtype in [InfiniDtype.F32, InfiniDtype.F16]:
            for shape, input_stride, output_stride in _TEST_CASES:
                print(f"Testing square: device={device.name}, dtype={dtype.name}, shape={shape}")
                test_square_op(device, dtype, shape, input_stride, output_stride)
    
    print("All tests passed!")

if __name__ == "__main__":
    main()
```

## Advanced Topics

### Complex Operators

For non-element-wise operations (like GEMM, convolution), you'll need custom implementations:

```c
// Example: Custom GEMM-like operator
namespace op::mygemm::cpu {

class Descriptor : public InfiniopOperatorDescriptor {
private:
    // Store operation parameters
    int m_, n_, k_;
    bool transpose_a_, transpose_b_;
    float alpha_, beta_;
    
public:
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        bool transpose_a, bool transpose_b,
        float alpha, float beta);
    
    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *c,
        std::vector<const void *> inputs,
        void *stream) const override;
        
    size_t getWorkspaceSize() const override { return 0; }
};

} // namespace op::mygemm::cpu
```

### GPU Implementation

For NVIDIA GPUs, create CUDA kernels:

```cuda
// src/infiniop/ops/square/nvidia/square_nvidia.cu
#include "square_nvidia.cuh"

namespace op::square::nvidia {

template<typename T>
__global__ void square_kernel(T* output, const T* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * input[idx];
    }
}

template<typename T>
infiniStatus_t launch_square_kernel(
    T* output, const T* input, int n, cudaStream_t stream) {
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    square_kernel<<<grid_size, block_size, 0, stream>>>(output, input, n);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

// Explicit instantiations
template infiniStatus_t launch_square_kernel<float>(float*, const float*, int, cudaStream_t);
template infiniStatus_t launch_square_kernel<half>(half*, const half*, int, cudaStream_t);

} // namespace op::square::nvidia
```

### Memory Management

For operators requiring temporary memory:

```c
size_t Descriptor::getWorkspaceSize() const override {
    // Calculate required workspace size
    size_t temp_size = _output_elements * sizeof(float);
    size_t alignment = 256;  // GPU memory alignment
    return (temp_size + alignment - 1) & ~(alignment - 1);
}

infiniStatus_t Descriptor::calculate(...) const {
    // Use workspace for temporary allocations
    float *temp_buffer = static_cast<float*>(workspace);
    // ... use temp_buffer for intermediate calculations
}
```

## Testing and Validation

### Unit Tests
Every operator should have comprehensive tests:

```python
# Test different data types
for dtype in [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.F64, InfiniDtype.BF16]:
    test_operator(dtype, ...)

# Test different shapes
test_cases = [
    (1,),           # 1D
    (10, 20),       # 2D  
    (5, 4, 3),      # 3D
    (2, 3, 4, 5),   # 4D
]

# Test custom strides (broadcasting, memory layout)
stride_cases = [
    ((10, 20), None, None),                    # Contiguous
    ((10, 20), (40, 1), (40, 1)),            # Custom stride
    ((10, 20), (0, 1), None),                 # Broadcasting input
]

# Test edge cases
edge_cases = [
    (1,),           # Single element
    (0,),           # Empty tensor (if supported)
    (1000000,),     # Large tensor
]
```

### GGUF Test Framework

For integration tests, create GGUF test cases:

```python
# test/infiniop-test/test_generate/testcases/square.py
import numpy as np
from ..gguf_utils import GGUFTestCaseBuilder

class SquareTestCase(GGUFTestCaseBuilder):
    def __init__(self):
        super().__init__("square")
    
    def generate_random_case(self, shape, dtype):
        # Generate random input
        input_data = np.random.randn(*shape).astype(dtype)
        
        # Compute expected output
        output_data = input_data ** 2
        
        # Build test case
        case_id = self.add_test_case()
        self.add_tensor(f"test.{case_id}.input", input_data)
        self.add_tensor(f"test.{case_id}.output", output_data)
        self.add_tensor(f"test.{case_id}.expected", output_data.astype(np.float64))

def main():
    builder = SquareTestCase()
    
    # Generate various test cases
    builder.generate_random_case((100, 100), np.float32)
    builder.generate_random_case((50, 50, 4), np.float32)
    
    builder.save("square.gguf")

if __name__ == "__main__":
    main()
```

## Performance Optimization

### CPU Optimizations
- **Vectorization**: Use SIMD instructions
- **Threading**: Leverage OpenMP for parallelization
- **Memory Access**: Optimize for cache locality

```c
// Example: OpenMP parallelization
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    output[i] = input[i] * input[i];
}
```

### GPU Optimizations
- **Occupancy**: Maximize thread utilization
- **Memory Coalescing**: Ensure efficient memory access patterns
- **Shared Memory**: Use for data reuse
- **Tensor Cores**: Leverage for mixed-precision operations

```cuda
// Example: Optimized CUDA kernel
template<typename T, int BLOCK_SIZE>
__global__ void optimized_kernel(T* output, const T* input, int n) {
    __shared__ T shared_data[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Coalesced memory access
    if (idx < n) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();
    
    // Compute and write back
    if (idx < n) {
        output[idx] = shared_data[tid] * shared_data[tid];
    }
}
```

## Common Patterns

### 1. Element-wise Operations
Use the elementwise framework for operations like add, mul, relu:
- Inherit from elementwise base classes
- Define operation functor
- Automatic broadcasting and stride handling

### 2. Reduction Operations  
For operations like sum, max, mean:
- Use reduction framework
- Handle different reduction dimensions
- Manage workspace for intermediate results

### 3. Linear Algebra
For GEMM, convolution operations:
- Leverage platform BLAS libraries (cuBLAS, oneDNN, etc.)
- Handle different layouts and strides
- Optimize for specific shapes and data types

### 4. Memory-bound Operations
For copy, transpose, reshape:
- Focus on memory bandwidth optimization
- Minimize kernel launch overhead
- Use async memcpy when possible

## Best Practices

1. **Error Handling**: Always validate inputs and return appropriate status codes
2. **Type Safety**: Use templates for type-generic implementations
3. **Documentation**: Document operator semantics, supported types, and limitations
4. **Testing**: Comprehensive test coverage including edge cases
5. **Performance**: Profile and optimize critical code paths
6. **Consistency**: Follow naming conventions and API patterns

## Next Steps

- **[Model Adaptation Guide](models.md)**: Learn how to integrate operators into large language models
- **[Performance Guide](performance.md)**: Advanced optimization techniques
- **[API Reference](api/)**: Complete API documentation
- **[Examples](examples/)**: More complex operator implementations