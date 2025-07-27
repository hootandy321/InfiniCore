# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when working with InfiniCore.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Build and Installation Problems](#build-and-installation-problems)
3. [Runtime Errors](#runtime-errors)
4. [Performance Issues](#performance-issues)
5. [Memory Problems](#memory-problems)
6. [Hardware-Specific Issues](#hardware-specific-issues)
7. [Debugging Tools](#debugging-tools)
8. [Getting Help](#getting-help)

## Common Issues

### Build and Installation Problems

#### XMake Not Found
```bash
# Error: xmake: command not found

# Solution 1: Install XMake
curl -fsSL https://xmake.io/shget.text | bash

# Solution 2: Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Solution 3: Use package manager
# Ubuntu
sudo apt install xmake
# macOS
brew install xmake
```

#### CUDA Compilation Errors
```bash
# Error: CUDA headers not found

# Solution: Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
nvcc --version
nvidia-smi

# Configure with correct CUDA path
xmake f --nv-gpu=y --cuda=$CUDA_HOME -cv
```

#### Library Linking Errors
```bash
# Error: cannot find -linfiniop

# Solution 1: Check installation
ls $INFINI_ROOT/lib/

# Solution 2: Set library path
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH

# Solution 3: Reinstall
xmake clean
xmake build && xmake install
```

#### Python Import Errors
```python
# Error: ImportError: cannot import name 'LIBINFINIOP'

# Solution 1: Check PYTHONPATH
import sys
sys.path.append('/path/to/InfiniCore/test')

# Solution 2: Install in development mode
export PYTHONPATH=$PWD/test:$PYTHONPATH

# Solution 3: Check library compilation
python -c "import ctypes; ctypes.CDLL('$INFINI_ROOT/lib/libinfiniop.so')"
```

### Runtime Errors

#### Status Code Reference
```c
// Common status codes and solutions
switch (status) {
    case INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED:
        // Device not compiled or not available
        // Check build configuration and hardware
        break;
        
    case INFINI_STATUS_BAD_TENSOR_DTYPE:
        // Unsupported data type for operation
        // Check operator documentation for supported types
        break;
        
    case INFINI_STATUS_BAD_TENSOR_SHAPE:
        // Incompatible tensor shapes
        // Verify input/output dimensions match operator requirements
        break;
        
    case INFINI_STATUS_INSUFFICIENT_WORKSPACE:
        // Workspace too small
        // Query workspace size and allocate sufficient memory
        break;
        
    case INFINI_STATUS_DEVICE_NOT_INITIALIZED:
        // Device context not properly initialized
        // Check handle creation and device availability
        break;
}
```

#### Device Initialization Issues
```c
// Problem: Device not found or initialization fails
infiniopHandle_t handle;
infiniStatus_t status = infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

if (status != INFINI_STATUS_SUCCESS) {
    // Debugging steps:
    
    // 1. Check device availability
    int device_count;
    cudaGetDeviceCount(&device_count);  // For NVIDIA
    if (device_count == 0) {
        fprintf(stderr, "No NVIDIA devices found\n");
        // Check driver installation, hardware connection
    }
    
    // 2. Check device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s, Compute: %d.%d\n", prop.name, prop.major, prop.minor);
    
    // 3. Try different device ID
    for (int i = 0; i < device_count; i++) {
        status = infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, i);
        if (status == INFINI_STATUS_SUCCESS) {
            printf("Successfully initialized device %d\n", i);
            break;
        }
    }
}
```

#### Tensor Descriptor Errors
```c
// Problem: Invalid tensor shapes or strides
infiniopTensorDescriptor_t desc;
int shape[] = {2, 3, 4};
int stride[] = {12, 4, 1};  // Contiguous strides

// Debugging: Validate shapes and strides
bool validate_tensor_descriptor(int *shape, int *stride, int ndim) {
    if (ndim <= 0 || ndim > MAX_TENSOR_DIMS) {
        printf("Invalid number of dimensions: %d\n", ndim);
        return false;
    }
    
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            printf("Invalid shape[%d]: %d\n", i, shape[i]);
            return false;
        }
    }
    
    // Check stride consistency (for debugging)
    if (stride) {
        for (int i = 0; i < ndim - 1; i++) {
            if (stride[i] < stride[i + 1] * shape[i + 1]) {
                printf("Warning: Non-contiguous strides detected\n");
                break;
            }
        }
    }
    
    return true;
}
```

### Performance Issues

#### Slow Execution
```c
// Problem: Unexpectedly slow performance

// Debugging approach:
void debug_performance_issues(QwenModel *model) {
    // 1. Profile execution
    DetailedProfileInfo profile;
    profile_model_execution(model, /* inputs */, &profile);
    
    printf("Performance Analysis:\n");
    printf("  Total time: %.2f ms\n", profile.total_time);
    printf("  Compute utilization: %.1f%%\n", profile.compute_utilization * 100);
    printf("  Memory efficiency: %.1f%%\n", profile.memory_efficiency * 100);
    
    // 2. Check common issues
    if (profile.compute_utilization < 50.0) {
        printf("Low compute utilization - possible causes:\n");
        printf("  - Small batch size\n");
        printf("  - Memory bandwidth bottleneck\n");
        printf("  - Suboptimal kernel launch parameters\n");
        printf("  - Synchronization overhead\n");
    }
    
    if (profile.memory_efficiency < 60.0) {
        printf("Low memory efficiency - possible causes:\n");
        printf("  - Poor memory access patterns\n");
        printf("  - Cache misses\n");
        printf("  - Non-coalesced memory access\n");
    }
    
    // 3. Check workspace allocation
    size_t workspace_size;
    infiniopGetGemmWorkspaceSize(model->q_proj_desc[0], &workspace_size);
    printf("Workspace size: %zu MB\n", workspace_size / (1024 * 1024));
    
    // 4. Verify optimal batch size
    int optimal_batch = calculate_optimal_batch_size(
        /* max_batch */ 32, /* available_memory */ get_available_memory(),
        /* per_sample_memory */ calculate_per_sample_memory(model),
        /* compute_intensity */ 100);
    printf("Recommended batch size: %d\n", optimal_batch);
}
```

#### Memory Allocation Failures
```c
// Problem: Out of memory errors

void debug_memory_issues(infiniopHandle_t handle) {
    // 1. Check available memory
    size_t free_memory, total_memory;
    switch (handle->device) {
        case INFINI_DEVICE_NVIDIA:
            cudaMemGetInfo(&free_memory, &total_memory);
            break;
        case INFINI_DEVICE_ASCEND:
            // Ascend-specific memory query
            aclrtGetMemInfo(ACL_HBM_MEM, &free_memory, &total_memory);
            break;
        // Add other platforms
    }
    
    printf("Memory status:\n");
    printf("  Total: %zu MB\n", total_memory / (1024 * 1024));
    printf("  Free: %zu MB\n", free_memory / (1024 * 1024));
    printf("  Used: %zu MB (%.1f%%)\n", 
           (total_memory - free_memory) / (1024 * 1024),
           100.0 * (total_memory - free_memory) / total_memory);
    
    // 2. Check memory fragmentation
    if (free_memory > 0 && free_memory < required_memory) {
        printf("Possible memory fragmentation\n");
        printf("Solutions:\n");
        printf("  - Use memory pools\n");
        printf("  - Reduce batch size\n");
        printf("  - Enable gradient checkpointing\n");
        printf("  - Use lower precision (FP16/INT8)\n");
    }
    
    // 3. Memory leak detection
    static size_t previous_free_memory = 0;
    if (previous_free_memory > 0 && free_memory < previous_free_memory - 100*1024*1024) {
        printf("Warning: Potential memory leak detected\n");
        printf("Previous free: %zu MB, Current free: %zu MB\n",
               previous_free_memory / (1024 * 1024),
               free_memory / (1024 * 1024));
    }
    previous_free_memory = free_memory;
}
```

### Hardware-Specific Issues

#### NVIDIA GPU Issues
```bash
# Problem: CUDA errors

# Check NVIDIA driver and CUDA installation
nvidia-smi
nvcc --version

# Common CUDA errors and solutions:
# 1. CUDA_ERROR_OUT_OF_MEMORY
#    - Reduce batch size
#    - Use gradient checkpointing
#    - Enable mixed precision

# 2. CUDA_ERROR_INVALID_DEVICE
#    - Check device ID
#    - Verify GPU is not in exclusive mode

# 3. CUDA_ERROR_LAUNCH_FAILED
#    - Check kernel launch parameters
#    - Verify shared memory usage
#    - Check for stack overflow in device code

# Debug CUDA kernels
export CUDA_LAUNCH_BLOCKING=1  # Synchronous kernel launches
```

#### Ascend NPU Issues
```bash
# Problem: Ascend initialization fails

# Check Ascend installation
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# Set environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Check device status
npu-smi info

# Common Ascend issues:
# 1. Device not found
#    - Check driver installation
#    - Verify device permissions

# 2. Memory allocation fails
#    - Check device memory limit
#    - Use HBM memory type for large allocations

# Debug Ascend operations
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=1
```

#### Cambricon MLU Issues
```bash
# Problem: Cambricon device errors

# Check Cambricon installation
cnmon info

# Set environment
source /opt/cambricon/cnrt/set_env.sh

# Common MLU issues:
# 1. Driver version mismatch
#    - Update driver to match toolkit version

# 2. Device busy
#    - Check for other processes using MLU
#    - Reset device if necessary

# Debug MLU operations
export CNRT_LOG_LEVEL=4
```

### Debugging Tools

#### Memory Debugging
```c
// Enable debug allocations
#ifdef DEBUG_MODE
typedef struct {
    void *ptr;
    size_t size;
    const char *file;
    int line;
    double timestamp;
} DebugAllocation;

static DebugAllocation debug_allocations[MAX_DEBUG_ALLOCS];
static int num_debug_allocs = 0;

void* debug_malloc(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    
    if (num_debug_allocs < MAX_DEBUG_ALLOCS) {
        debug_allocations[num_debug_allocs] = (DebugAllocation){
            .ptr = ptr,
            .size = size,
            .file = file,
            .line = line,
            .timestamp = get_time()
        };
        num_debug_allocs++;
    }
    
    printf("ALLOC: %p (%zu bytes) at %s:%d\n", ptr, size, file, line);
    return ptr;
}

void debug_free(void *ptr, const char *file, int line) {
    // Find and remove from debug list
    for (int i = 0; i < num_debug_allocs; i++) {
        if (debug_allocations[i].ptr == ptr) {
            printf("FREE: %p (%zu bytes) at %s:%d (lifetime: %.2f ms)\n",
                   ptr, debug_allocations[i].size, file, line,
                   get_time() - debug_allocations[i].timestamp);
            
            // Remove from list
            memmove(&debug_allocations[i], &debug_allocations[i+1],
                    (num_debug_allocs - i - 1) * sizeof(DebugAllocation));
            num_debug_allocs--;
            break;
        }
    }
    
    free(ptr);
}

// Macros for debug allocation
#define DEBUG_MALLOC(size) debug_malloc(size, __FILE__, __LINE__)
#define DEBUG_FREE(ptr) debug_free(ptr, __FILE__, __LINE__)

void print_memory_leaks() {
    if (num_debug_allocs > 0) {
        printf("Memory leaks detected:\n");
        for (int i = 0; i < num_debug_allocs; i++) {
            printf("  %p (%zu bytes) allocated at %s:%d\n",
                   debug_allocations[i].ptr,
                   debug_allocations[i].size,
                   debug_allocations[i].file,
                   debug_allocations[i].line);
        }
    }
}
#endif
```

#### Tensor Validation
```c
// Validate tensor contents for debugging
bool validate_tensor_values(const void *tensor, size_t num_elements,
                           infiniDtype_t dtype, const char *name) {
    bool has_nan = false, has_inf = false;
    double min_val = INFINITY, max_val = -INFINITY;
    
    // Copy to host for validation
    void *host_data = malloc(num_elements * get_dtype_size(dtype));
    infinirtMemcpy(host_data, tensor, num_elements * get_dtype_size(dtype),
                   INFINIRT_MEMCPY_DEVICE_TO_HOST, NULL);
    
    switch (dtype) {
        case INFINI_DTYPE_F32: {
            float *data = (float*)host_data;
            for (size_t i = 0; i < num_elements; i++) {
                if (isnan(data[i])) has_nan = true;
                if (isinf(data[i])) has_inf = true;
                min_val = fmin(min_val, data[i]);
                max_val = fmax(max_val, data[i]);
            }
            break;
        }
        case INFINI_DTYPE_F16: {
            // Handle FP16 validation
            break;
        }
        // Add other types
    }
    
    free(host_data);
    
    if (has_nan || has_inf) {
        printf("Tensor validation failed for %s:\n", name);
        if (has_nan) printf("  Contains NaN values\n");
        if (has_inf) printf("  Contains Inf values\n");
        return false;
    }
    
    printf("Tensor %s: min=%.6f, max=%.6f\n", name, min_val, max_val);
    return true;
}

// Add validation calls after critical operations
#ifdef DEBUG_MODE
#define VALIDATE_TENSOR(tensor, num_elements, dtype, name) \
    validate_tensor_values(tensor, num_elements, dtype, name)
#else
#define VALIDATE_TENSOR(tensor, num_elements, dtype, name) true
#endif
```

#### Performance Debugging
```c
// Detailed timing for each operation
typedef struct {
    const char *operation_name;
    double start_time;
    double total_time;
    int call_count;
} OperationTimer;

static OperationTimer operation_timers[MAX_OPERATIONS];
static int num_timers = 0;

void start_operation_timer(const char *name) {
    for (int i = 0; i < num_timers; i++) {
        if (strcmp(operation_timers[i].operation_name, name) == 0) {
            operation_timers[i].start_time = get_high_precision_time();
            return;
        }
    }
    
    // Add new timer
    if (num_timers < MAX_OPERATIONS) {
        operation_timers[num_timers] = (OperationTimer){
            .operation_name = name,
            .start_time = get_high_precision_time(),
            .total_time = 0.0,
            .call_count = 0
        };
        num_timers++;
    }
}

void end_operation_timer(const char *name) {
    double end_time = get_high_precision_time();
    
    for (int i = 0; i < num_timers; i++) {
        if (strcmp(operation_timers[i].operation_name, name) == 0) {
            operation_timers[i].total_time += end_time - operation_timers[i].start_time;
            operation_timers[i].call_count++;
            return;
        }
    }
}

void print_operation_timings() {
    printf("Operation Timings:\n");
    for (int i = 0; i < num_timers; i++) {
        printf("  %s: %.2f ms total, %.2f ms avg (%d calls)\n",
               operation_timers[i].operation_name,
               operation_timers[i].total_time,
               operation_timers[i].total_time / operation_timers[i].call_count,
               operation_timers[i].call_count);
    }
}

// Macros for easy timing
#define START_TIMER(name) start_operation_timer(name)
#define END_TIMER(name) end_operation_timer(name)
```

### Environment Debugging
```bash
# Debug script to check environment
#!/bin/bash

echo "=== InfiniCore Environment Debug ==="

# 1. Check basic environment
echo "Environment Variables:"
echo "  INFINI_ROOT: $INFINI_ROOT"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  CUDA_HOME: $CUDA_HOME"

# 2. Check library files
echo -e "\nLibrary Files:"
if [ -d "$INFINI_ROOT/lib" ]; then
    ls -la $INFINI_ROOT/lib/
else
    echo "  ERROR: $INFINI_ROOT/lib not found"
fi

# 3. Check headers
echo -e "\nHeader Files:"
if [ -d "$INFINI_ROOT/include" ]; then
    ls -la $INFINI_ROOT/include/
else
    echo "  ERROR: $INFINI_ROOT/include not found"
fi

# 4. Check device availability
echo -e "\nDevice Status:"
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPUs:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

if command -v npu-smi &> /dev/null; then
    echo "Ascend NPUs:"
    npu-smi info
fi

# 5. Test library loading
echo -e "\nLibrary Loading Test:"
python3 -c "
try:
    import ctypes
    lib = ctypes.CDLL('$INFINI_ROOT/lib/libinfiniop.so')
    print('  libinfiniop.so: OK')
except Exception as e:
    print(f'  libinfiniop.so: ERROR - {e}')

try:
    import sys
    sys.path.append('$PWD/test')
    from test.infiniop.libinfiniop import LIBINFINIOP
    print('  Python bindings: OK')
except Exception as e:
    print(f'  Python bindings: ERROR - {e}')
"

echo -e "\n=== Debug Complete ==="
```

### Getting Help

#### Before Asking for Help
1. **Check this troubleshooting guide** for common solutions
2. **Search existing issues** on GitHub
3. **Gather debug information**:
   - InfiniCore version/commit
   - Hardware platform and drivers
   - Operating system and compiler
   - Complete error messages and stack traces
   - Minimal reproducing example

#### Information to Include
```bash
# System information script
#!/bin/bash

echo "=== InfiniCore Bug Report ==="
echo "Date: $(date)"
echo "InfiniCore commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "OS: $(uname -a)"
echo "Compiler: $(gcc --version | head -1)"

if [ -f "xmake.lua" ]; then
    echo "Build configuration:"
    xmake f -v
fi

echo "Hardware:"
lscpu | grep "Model name"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -L
fi

echo "Libraries:"
ldd $INFINI_ROOT/lib/libinfiniop.so 2>/dev/null || echo "Library not found"

echo "Environment:"
env | grep -E "(INFINI|CUDA|PATH)" | sort
```

#### Creating Minimal Examples
```c
// Template for bug reports
#include <infiniop.h>
#include <stdio.h>

int main() {
    // Initialize
    infiniopHandle_t handle;
    infiniStatus_t status = infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);
    if (status != INFINI_STATUS_SUCCESS) {
        printf("Failed to create handle: %d\n", status);
        return 1;
    }
    
    // Create minimal test case that reproduces the issue
    // ...
    
    // Cleanup
    infiniopDestroyHandle(handle);
    return 0;
}
```

#### Where to Get Help
1. **GitHub Issues**: https://github.com/hootandy321/InfiniCore/issues
2. **Documentation**: Check all documentation sections
3. **Community Forums**: If available
4. **Direct Contact**: For urgent issues or commercial support

#### Contributing Fixes
If you find and fix an issue:
1. **Create a test case** that reproduces the problem
2. **Implement the fix** with minimal changes
3. **Verify the fix** doesn't break existing functionality
4. **Submit a pull request** with clear description

This troubleshooting guide should help you resolve most common issues with InfiniCore. Remember to always check the latest documentation and search for existing solutions before reporting new issues.