# Performance Optimization Guide

This guide provides comprehensive strategies for maximizing performance when using InfiniCore across different hardware platforms.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [General Optimization Principles](#general-optimization-principles)
3. [CPU Optimizations](#cpu-optimizations)
4. [GPU Optimizations](#gpu-optimizations)
5. [Hardware-Specific Techniques](#hardware-specific-techniques)
6. [Memory Optimization](#memory-optimization)
7. [Operator-Level Optimizations](#operator-level-optimizations)
8. [Model-Level Optimizations](#model-level-optimizations)
9. [Profiling and Analysis](#profiling-and-analysis)
10. [Best Practices](#best-practices)

## Performance Overview

InfiniCore performance depends on several factors:
- **Hardware Utilization**: Maximizing compute unit usage
- **Memory Efficiency**: Minimizing bandwidth bottlenecks
- **Algorithm Selection**: Choosing optimal implementations
- **Data Layout**: Optimizing memory access patterns
- **Precision**: Balancing accuracy vs. speed
- **Parallelization**: Efficient use of available resources

### Performance Hierarchy
```
Application Level
├── Model Architecture Optimization
├── Operator Fusion and Scheduling
└── Memory Management Strategy

Operator Level  
├── Algorithm Selection
├── Kernel Optimization
└── Data Layout Optimization

Hardware Level
├── Platform-Specific Tuning
├── Resource Utilization
└── Memory Hierarchy Optimization
```

## General Optimization Principles

### 1. Understand Your Workload
Before optimizing, analyze your workload characteristics:

```c
// Workload analysis structure
typedef struct {
    // Compute characteristics
    float compute_intensity;     // FLOPs per byte
    bool is_memory_bound;       // vs compute bound
    bool has_regular_patterns;  // vs irregular access
    
    // Data characteristics
    size_t working_set_size;    // Total data size
    float data_reuse_factor;    // How often data is reused
    bool requires_precision;    // FP32 vs FP16/INT8
    
    // Execution characteristics
    bool can_pipeline;          // Overlapping opportunities
    bool can_batch;            // Batching opportunities
    int parallelism_degree;    // Available parallelism
} WorkloadProfile;

void analyze_workload(WorkloadProfile *profile, const char *model_name) {
    // Profile your specific model to understand characteristics
    // This guides optimization strategy
}
```

### 2. Hardware-Aware Development
Design with target hardware in mind:

```c
// Hardware capability query
typedef struct {
    int compute_units;          // CUDA cores, Ascend cores, etc.
    size_t memory_bandwidth;    // GB/s
    size_t cache_sizes[4];      // L1, L2, L3, etc.
    int tensor_cores;           // Special units (if available)
    bool supports_mixed_precision;
    bool supports_async_copy;
} HardwareProfile;

infiniStatus_t query_hardware_capabilities(infiniopHandle_t handle,
                                          HardwareProfile *profile) {
    switch (handle->device) {
        case INFINI_DEVICE_NVIDIA:
            return query_nvidia_capabilities(handle, profile);
        case INFINI_DEVICE_ASCEND:
            return query_ascend_capabilities(handle, profile);
        // ... other devices
    }
}
```

### 3. Performance Measurement
Establish baseline and track improvements:

```c
// Performance metrics
typedef struct {
    double execution_time;      // Total time (ms)
    double compute_time;        // Pure compute time
    double memory_time;         // Memory transfer time
    double throughput;          // Operations/second
    float memory_utilization;   // % of peak bandwidth used
    float compute_utilization;  // % of peak compute used
    size_t memory_footprint;    // Peak memory usage
} PerformanceMetrics;

infiniStatus_t benchmark_operation(infiniopOperatorDescriptor_t desc,
                                  void *inputs[], void *outputs[],
                                  int num_iterations,
                                  PerformanceMetrics *metrics) {
    // Accurate timing with warmup
    // Multiple iterations for statistical significance
    // Separate timing for different phases
}
```

## CPU Optimizations

### 1. Vectorization and SIMD
Leverage SIMD instructions for parallel processing:

```c
// Example: Vectorized element-wise operations
#include <immintrin.h>  // For AVX/AVX2

void vectorized_add_f32(const float *a, const float *b, float *c, size_t n) {
    size_t simd_end = n - (n % 8);
    
    // Process 8 elements at once with AVX
    for (size_t i = 0; i < simd_end; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_store_ps(&c[i], vc);
    }
    
    // Handle remaining elements
    for (size_t i = simd_end; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Auto-vectorization hints
#pragma omp simd aligned(a,b,c:32)
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}
```

### 2. Multi-threading with OpenMP
Efficient parallel execution:

```c
// Thread-level parallelism
void parallel_matrix_multiply(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// Nested parallelism for large workloads
void nested_parallel_operation(float *data, int batch_size, int channels, int height, int width) {
    #pragma omp parallel for num_threads(4)  // Outer level: batches
    for (int b = 0; b < batch_size; b++) {
        #pragma omp parallel for num_threads(2) // Inner level: channels
        for (int c = 0; c < channels; c++) {
            // Process batch b, channel c
            process_channel(&data[b*channels*height*width + c*height*width], height, width);
        }
    }
}
```

### 3. Cache Optimization
Optimize for CPU cache hierarchy:

```c
// Cache-friendly matrix multiplication (blocking)
void blocked_matrix_multiply(const float *A, const float *B, float *C,
                           int M, int N, int K, int block_size) {
    for (int i = 0; i < M; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < K; k += block_size) {
                // Process block
                int max_i = min(i + block_size, M);
                int max_j = min(j + block_size, N);
                int max_k = min(k + block_size, K);
                
                for (int bi = i; bi < max_i; bi++) {
                    for (int bj = j; bj < max_j; bj++) {
                        float sum = 0.0f;
                        for (int bk = k; bk < max_k; bk++) {
                            sum += A[bi*K + bk] * B[bk*N + bj];
                        }
                        C[bi*N + bj] += sum;
                    }
                }
            }
        }
    }
}

// Memory prefetching
void prefetch_optimized_loop(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        // Prefetch data for next iteration
        if (i + 64 < n) {
            __builtin_prefetch(&input[i + 64], 0, 3);  // Read, high locality
            __builtin_prefetch(&output[i + 64], 1, 3); // Write, high locality
        }
        
        // Process current element
        output[i] = compute_function(input[i]);
    }
}
```

### 4. Optimized BLAS Libraries
Use high-performance BLAS implementations:

```c
// Intel MKL integration
#ifdef USE_MKL
#include <mkl.h>

void optimized_gemm_cpu(const float *A, const float *B, float *C,
                       int M, int N, int K, float alpha, float beta) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}
#endif

// OpenBLAS integration
#ifdef USE_OPENBLAS
#include <cblas.h>

void optimized_gemm_cpu_openblas(const float *A, const float *B, float *C,
                                int M, int N, int K, float alpha, float beta) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}
#endif
```

## GPU Optimizations

### 1. CUDA Kernel Optimization
Write efficient CUDA kernels:

```cuda
// Optimized element-wise kernel
template<typename T, int BLOCK_SIZE>
__global__ void optimized_elementwise_kernel(T *output, const T *input1, const T *input2, int n) {
    // Shared memory for data reuse
    __shared__ T shared_data1[BLOCK_SIZE];
    __shared__ T shared_data2[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread
    for (int i = idx; i < n; i += stride) {
        // Coalesced memory access
        if (tid < BLOCK_SIZE && i + tid < n) {
            shared_data1[tid] = input1[i + tid];
            shared_data2[tid] = input2[i + tid];
        }
        __syncthreads();
        
        // Compute
        if (i < n) {
            T result = shared_data1[tid] + shared_data2[tid];
            output[i] = result;
        }
        __syncthreads();
    }
}

// Launch configuration optimization
void launch_optimized_kernel(float *output, const float *input1, const float *input2, int n, cudaStream_t stream) {
    // Calculate optimal block and grid sizes
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int block_size = 256;  // Usually optimal for most operations
    int max_blocks = prop.multiProcessorCount * 8;  // Occupancy heuristic
    int grid_size = min(max_blocks, (n + block_size - 1) / block_size);
    
    optimized_elementwise_kernel<float, 256><<<grid_size, block_size, 0, stream>>>(
        output, input1, input2, n);
}
```

### 2. Memory Coalescing
Ensure efficient memory access patterns:

```cuda
// Bad: Strided access
__global__ void bad_transpose(float *output, const float *input, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // Non-coalesced access
        output[col * rows + row] = input[row * cols + col];
    }
}

// Good: Coalesced access with shared memory
__global__ void good_transpose(float *output, const float *input, int rows, int cols) {
    __shared__ float tile[32][32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Coalesced read into shared memory
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
    }
    __syncthreads();
    
    // Transpose indices for output
    row = blockIdx.x * blockDim.x + threadIdx.y;
    col = blockIdx.y * blockDim.y + threadIdx.x;
    
    // Coalesced write from shared memory
    if (row < cols && col < rows) {
        output[row * rows + col] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 3. Tensor Core Utilization
Leverage Tensor Cores for mixed precision:

```cuda
// Tensor Core GEMM using WMMA API
#include <mma.h>
using namespace nvcuda;

__global__ void tensor_core_gemm(half *a, half *b, half *c, int M, int N, int K) {
    // Declare fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    
    // Calculate warp and lane positions
    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Perform matrix multiplication
    for (int k = 0; k < K; k += 16) {
        // Load matrix fragments
        wmma::load_matrix_sync(a_frag, a + warp_m * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, b + k * N + warp_n * 16, N);
        
        // Perform matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result
    wmma::store_matrix_sync(c + warp_m * 16 * N + warp_n * 16, c_frag, N, wmma::mem_row_major);
}
```

### 4. Asynchronous Execution
Overlap computation with memory transfers:

```cuda
// Asynchronous execution pattern
void async_pipeline_execution(float *host_data, float *device_data, 
                             int batch_size, int data_size, cudaStream_t *streams) {
    const int num_streams = 4;
    const int chunk_size = batch_size / num_streams;
    
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size * data_size;
        
        // Async memory copy H2D
        cudaMemcpyAsync(device_data + offset, host_data + offset,
                       chunk_size * data_size * sizeof(float),
                       cudaMemcpyHostToDevice, streams[i]);
        
        // Launch kernel on this chunk
        process_chunk<<<grid_size, block_size, 0, streams[i]>>>(
            device_data + offset, chunk_size);
        
        // Async memory copy D2H for result
        cudaMemcpyAsync(host_data + offset, device_data + offset,
                       chunk_size * data_size * sizeof(float),
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
}
```

## Hardware-Specific Techniques

### NVIDIA GPU Optimizations
```cuda
// Optimizations specific to NVIDIA architectures

// 1. Occupancy optimization
__global__ void __launch_bounds__(256, 4)  // 256 threads, 4 blocks per SM
occupancy_optimized_kernel(float *data, int n) {
    // Kernel implementation
}

// 2. Persistent threads for irregular workloads
__global__ void persistent_kernel(float *data, int *work_queue, int queue_size) {
    __shared__ int shared_work_index;
    
    while (true) {
        // Cooperatively get work
        if (threadIdx.x == 0) {
            shared_work_index = atomicAdd(&global_work_index, 1);
        }
        __syncthreads();
        
        if (shared_work_index >= queue_size) break;
        
        // Process work item
        process_work_item(data, work_queue[shared_work_index]);
    }
}

// 3. Warp-level primitives
__global__ void warp_optimized_reduction(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    
    float val = (tid < n) ? input[tid] : 0.0f;
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // Write result (first thread in warp)
    if (lane == 0) {
        atomicAdd(output, val);
    }
}
```

### Huawei Ascend NPU Optimizations
```c
// Ascend-specific optimizations using ACL

// 1. Tile-based processing for large tensors
acl_status_t ascend_tiled_operation(aclrtContext context,
                                   float *input, float *output,
                                   int batch, int height, int width, int channels) {
    // Split large tensors into tiles that fit in NPU memory
    const int tile_size = 1024 * 1024;  // Adjust based on NPU memory
    int tiles_h = (height + tile_size - 1) / tile_size;
    int tiles_w = (width + tile_size - 1) / tile_size;
    
    for (int th = 0; th < tiles_h; th++) {
        for (int tw = 0; tw < tiles_w; tw++) {
            // Process tile
            int tile_h_start = th * tile_size;
            int tile_w_start = tw * tile_size;
            int tile_h_size = min(tile_size, height - tile_h_start);
            int tile_w_size = min(tile_size, width - tile_w_start);
            
            // Execute operation on tile
            process_tile_on_npu(context, input, output, 
                               tile_h_start, tile_w_start, 
                               tile_h_size, tile_w_size, channels);
        }
    }
    
    return ACL_SUCCESS;
}

// 2. Operator fusion for Ascend
acl_status_t create_fused_ascend_op(aclnnHandle_t handle,
                                   aclTensor *input1, aclTensor *input2, aclTensor *output) {
    // Create fused operation combining multiple primitives
    // This reduces memory traffic and improves performance
    
    aclnnHandle_t fused_handle;
    aclnnCreateConv2d(&fused_handle, /* conv params */);
    aclnnAddTensorOp(fused_handle, input1, input2);  // Add after conv
    aclnnActivationOp(fused_handle, ACL_ACTIVATION_RELU);  // ReLU activation
    
    return aclnnExecute(fused_handle, output);
}
```

### Cambricon MLU Optimizations
```c
// Cambricon-specific optimizations using BANG

// 1. BANG kernel for element-wise operations
__mlu_global__ void bang_elementwise_add(float *output, float *input1, float *input2, int n) {
    // Use BANG vector instructions
    int task_id = taskId;
    int core_num = taskDim;
    int elements_per_core = n / core_num;
    int start = task_id * elements_per_core;
    int end = (task_id == core_num - 1) ? n : start + elements_per_core;
    
    // BANG vectorized addition
    __bang_add(output + start, input1 + start, input2 + start, end - start);
}

// 2. Memory optimization for MLU
cnrt_status_t optimize_mlu_memory_layout(void **optimized_data, void *original_data,
                                        int batch, int channels, int height, int width) {
    // Convert to MLU-optimized memory layout (NHWC to NCHW or similar)
    size_t size = batch * channels * height * width * sizeof(float);
    cnrtMalloc(optimized_data, size);
    
    // Transpose data to optimal layout
    cnrtMemcpy2D(*optimized_data, width * sizeof(float),
                 original_data, height * width * sizeof(float),
                 width * sizeof(float), height, CNRT_MEM_TRANS_DIR_HOST2DEV);
    
    return CNRT_RET_SUCCESS;
}
```

## Memory Optimization

### 1. Memory Pool Management
Efficient memory allocation and reuse:

```c
// Advanced memory pool with size classes
typedef struct {
    void **free_blocks;      // Available blocks per size class
    int *free_counts;        // Number of free blocks per class
    size_t *size_classes;    // Predefined sizes
    int num_classes;
    size_t total_allocated;
    size_t peak_usage;
    infiniopHandle_t handle;
} AdvancedMemoryPool;

infiniStatus_t memory_pool_get_optimized(AdvancedMemoryPool *pool, 
                                        void **ptr, size_t size) {
    // Find best-fit size class
    int class_idx = find_size_class(pool, size);
    
    if (pool->free_counts[class_idx] > 0) {
        // Reuse existing block
        pool->free_counts[class_idx]--;
        *ptr = pool->free_blocks[class_idx * MAX_BLOCKS + pool->free_counts[class_idx]];
        return INFINI_STATUS_SUCCESS;
    }
    
    // Allocate new block
    size_t actual_size = pool->size_classes[class_idx];
    infinirtMalloc(ptr, actual_size, pool->handle->context);
    pool->total_allocated += actual_size;
    pool->peak_usage = max(pool->peak_usage, pool->total_allocated);
    
    return INFINI_STATUS_SUCCESS;
}

// Memory defragmentation
infiniStatus_t memory_pool_defragment(AdvancedMemoryPool *pool) {
    // Compact fragmented memory blocks
    // This is platform-specific and may not be available on all devices
}
```

### 2. Workspace Optimization
Optimize temporary memory usage:

```c
// Workspace size calculation with sharing
typedef struct {
    size_t sizes[MAX_LAYERS];     // Workspace needed per layer
    bool can_share[MAX_LAYERS];   // Whether layer can share workspace
    size_t shared_size;           // Size of shared workspace
    size_t total_size;            // Total workspace needed
} WorkspaceAnalysis;

infiniStatus_t analyze_workspace_requirements(QwenModel *model, 
                                             WorkspaceAnalysis *analysis) {
    analysis->shared_size = 0;
    analysis->total_size = 0;
    
    for (int i = 0; i < model->config.num_hidden_layers; i++) {
        // Calculate workspace for each layer
        size_t attention_workspace, mlp_workspace, norm_workspace;
        
        infiniopGetAttentionWorkspaceSize(model->attention_desc[i], &attention_workspace);
        infiniopGetGemmWorkspaceSize(model->gate_proj_desc[i], &mlp_workspace);
        infiniopGetRMSNormWorkspaceSize(model->input_norm_desc[i], &norm_workspace);
        
        analysis->sizes[i] = attention_workspace + mlp_workspace + norm_workspace;
        
        // Layers can share workspace if they don't execute concurrently
        analysis->can_share[i] = true;  // Sequential execution
        analysis->shared_size = max(analysis->shared_size, analysis->sizes[i]);
    }
    
    // Total size is shared workspace + any permanent allocations
    analysis->total_size = analysis->shared_size;
    return INFINI_STATUS_SUCCESS;
}
```

### 3. Memory Access Optimization
Optimize data layout and access patterns:

```c
// Data layout transformation for better cache utilization
infiniStatus_t optimize_tensor_layout(void *output, const void *input,
                                     int *original_shape, int *optimal_shape,
                                     int ndim, infiniDtype_t dtype,
                                     infiniopHandle_t handle) {
    // Transform from generic layout to hardware-optimal layout
    // For example: NCHW -> NHWC for some platforms, or blocked layouts
    
    switch (handle->device) {
        case INFINI_DEVICE_NVIDIA:
            // NVIDIA often prefers NCHW for convolutions
            return transform_to_nchw(output, input, original_shape, ndim, dtype, handle);
            
        case INFINI_DEVICE_ASCEND:
            // Ascend may prefer different layouts
            return transform_to_ascend_optimal(output, input, original_shape, ndim, dtype, handle);
            
        default:
            // Generic layout transformation
            break;
    }
}

// Memory prefetching strategies
void prefetch_next_layer_weights(QwenModel *model, int current_layer, void *stream) {
    if (current_layer + 1 < model->config.num_hidden_layers) {
        // Asynchronously prefetch weights for next layer
        async_prefetch_weights(model->q_proj_weight[current_layer + 1],
                              model->k_proj_weight[current_layer + 1],
                              model->v_proj_weight[current_layer + 1],
                              stream);
    }
}
```

## Operator-Level Optimizations

### 1. Kernel Fusion
Combine multiple operations to reduce memory traffic:

```c
// Fused operations
infiniStatus_t fused_linear_relu(infiniopGemmDescriptor_t gemm_desc,
                                infiniopReLUDescriptor_t relu_desc,
                                void *workspace, size_t workspace_size,
                                void *output, const void *input, const void *weight,
                                float alpha, float beta, void *stream) {
    
    // Check if platform supports fused kernel
    if (supports_fused_linear_relu(gemm_desc->handle)) {
        return platform_fused_linear_relu(gemm_desc, relu_desc, workspace, workspace_size,
                                         output, input, weight, alpha, beta, stream);
    }
    
    // Fallback: separate operations
    void *temp_output = workspace;
    
    infiniopGemm(gemm_desc, workspace + temp_output_size, workspace_size - temp_output_size,
                 temp_output, input, weight, alpha, beta, stream);
    
    infiniopReLU(relu_desc, workspace + temp_output_size, workspace_size - temp_output_size,
                 output, temp_output, stream);
    
    return INFINI_STATUS_SUCCESS;
}

// Multi-operation fusion
infiniStatus_t fused_attention_block(QwenModel *model, int layer_idx,
                                    void *output, const void *input,
                                    void *workspace, size_t workspace_size,
                                    void *stream) {
    // Fuse: LayerNorm + QKV projection + Attention + Output projection
    // This can significantly reduce memory bandwidth requirements
    
    switch (model->handle->device) {
        case INFINI_DEVICE_NVIDIA:
            return nvidia_fused_attention_block(model, layer_idx, output, input,
                                               workspace, workspace_size, stream);
        case INFINI_DEVICE_ASCEND:
            return ascend_fused_attention_block(model, layer_idx, output, input,
                                              workspace, workspace_size, stream);
        default:
            // Fallback to individual operations
            return standard_attention_block(model, layer_idx, output, input,
                                          workspace, workspace_size, stream);
    }
}
```

### 2. Algorithm Selection
Choose optimal algorithms based on problem size:

```c
// Dynamic algorithm selection
typedef enum {
    GEMM_ALGORITHM_CUBLAS,
    GEMM_ALGORITHM_CUTLASS,
    GEMM_ALGORITHM_TENSOR_CORES,
    GEMM_ALGORITHM_CUSTOM_KERNEL
} GemmAlgorithm;

GemmAlgorithm select_optimal_gemm_algorithm(int m, int n, int k, 
                                           infiniDtype_t dtype,
                                           infiniopHandle_t handle) {
    // Algorithm selection based on problem characteristics
    if (handle->device == INFINI_DEVICE_NVIDIA) {
        // Use Tensor Cores for suitable shapes and types
        if (dtype == INFINI_DTYPE_F16 && m >= 16 && n >= 16 && k >= 16 &&
            m % 16 == 0 && n % 16 == 0 && k % 16 == 0) {
            return GEMM_ALGORITHM_TENSOR_CORES;
        }
        
        // Use cuBLAS for large matrices
        if (m * n * k > 1000000) {
            return GEMM_ALGORITHM_CUBLAS;
        }
        
        // Use custom kernel for small matrices
        return GEMM_ALGORITHM_CUSTOM_KERNEL;
    }
    
    // Platform-specific selection for other devices
    return select_platform_optimal_algorithm(m, n, k, dtype, handle);
}

// Adaptive batch size
int calculate_optimal_batch_size(int max_batch_size, size_t available_memory,
                                size_t per_sample_memory, int compute_intensity) {
    // Memory constraint
    int memory_limited_batch = available_memory / per_sample_memory;
    
    // Compute efficiency constraint (avoid too small batches)
    int min_efficient_batch = compute_intensity > 100 ? 4 : 1;
    
    // Hardware utilization constraint
    int hw_optimal_batch = get_hardware_optimal_batch_size();
    
    return min(max_batch_size, max(min_efficient_batch, 
                                  min(memory_limited_batch, hw_optimal_batch)));
}
```

### 3. Precision Optimization
Balance accuracy and performance:

```c
// Mixed precision execution
typedef struct {
    infiniDtype_t input_dtype;      // Input precision
    infiniDtype_t compute_dtype;    // Computation precision
    infiniDtype_t output_dtype;     // Output precision
    bool use_loss_scaling;          // For training
    float loss_scale_factor;        // Scale factor
} MixedPrecisionConfig;

infiniStatus_t mixed_precision_gemm(infiniopGemmDescriptor_t desc,
                                   MixedPrecisionConfig *mp_config,
                                   void *workspace, size_t workspace_size,
                                   void *output, const void *input, const void *weight,
                                   float alpha, float beta, void *stream) {
    
    // Convert inputs to computation precision if needed
    void *converted_input = input;
    void *converted_weight = weight;
    
    if (mp_config->input_dtype != mp_config->compute_dtype) {
        converted_input = workspace;
        convert_precision(converted_input, input, /* size */, 
                         mp_config->input_dtype, mp_config->compute_dtype, stream);
    }
    
    // Perform computation in compute precision
    void *compute_output = (mp_config->output_dtype == mp_config->compute_dtype) ? 
                          output : (char*)workspace + converted_input_size;
    
    infiniopGemm(desc, workspace + temp_size, workspace_size - temp_size,
                 compute_output, converted_input, converted_weight, alpha, beta, stream);
    
    // Convert output to desired precision
    if (mp_config->output_dtype != mp_config->compute_dtype) {
        convert_precision(output, compute_output, /* size */,
                         mp_config->compute_dtype, mp_config->output_dtype, stream);
    }
    
    return INFINI_STATUS_SUCCESS;
}
```

## Model-Level Optimizations

### 1. Layer Scheduling
Optimize execution order and parallelism:

```c
// Pipeline parallelism for transformer layers
typedef struct {
    int pipeline_stages;        // Number of pipeline stages
    int micro_batch_size;      // Size of each micro-batch
    QwenModel **stage_models;  // Model partition for each stage
    void **stage_buffers;      // Intermediate buffers
    infinirtStream_t *streams; // Streams for each stage
} PipelineConfig;

infiniStatus_t pipeline_forward_pass(PipelineConfig *config,
                                    int *input_ids, int batch_size, int seq_len,
                                    float *output_logits) {
    
    int micro_batches = batch_size / config->micro_batch_size;
    
    for (int mb = 0; mb < micro_batches; mb++) {
        // Start pipeline for this micro-batch
        for (int stage = 0; stage < config->pipeline_stages; stage++) {
            // Calculate which micro-batch this stage should process
            int active_mb = mb - stage;
            if (active_mb >= 0) {
                // Process micro-batch on this stage
                int mb_start = active_mb * config->micro_batch_size;
                int mb_end = min(mb_start + config->micro_batch_size, batch_size);
                
                process_pipeline_stage(config->stage_models[stage],
                                      input_ids + mb_start,
                                      mb_end - mb_start, seq_len,
                                      config->stage_buffers[stage],
                                      config->streams[stage]);
            }
        }
    }
    
    // Wait for all stages to complete
    for (int stage = 0; stage < config->pipeline_stages; stage++) {
        infinirtStreamSynchronize(config->streams[stage]);
    }
    
    return INFINI_STATUS_SUCCESS;
}

// Dynamic layer skipping (for inference)
infiniStatus_t adaptive_layer_execution(QwenModel *model,
                                       void *hidden_states,
                                       float confidence_threshold,
                                       int *layers_executed) {
    *layers_executed = 0;
    
    for (int layer = 0; layer < model->config.num_hidden_layers; layer++) {
        // Execute layer
        qwen_decoder_layer_forward(model, layer, hidden_states, 
                                  NULL, NULL, NULL, NULL);
        (*layers_executed)++;
        
        // Check confidence (simplified - would need actual confidence metric)
        float confidence = calculate_output_confidence(hidden_states);
        
        // Early exit if confident enough
        if (confidence > confidence_threshold && layer >= MIN_LAYERS) {
            break;
        }
    }
    
    return INFINI_STATUS_SUCCESS;
}
```

### 2. KV Cache Optimization
Efficient key-value cache management:

```c
// Optimized KV cache with compression
typedef struct {
    void *compressed_k_cache;   // Compressed key cache
    void *compressed_v_cache;   // Compressed value cache
    void *full_k_cache;        // Full precision cache (recent tokens)
    void *full_v_cache;        // Full precision cache (recent tokens)
    
    int compression_ratio;      // How much to compress old entries
    int full_cache_size;       // Number of recent tokens in full precision
    int total_cache_size;      // Total cache capacity
    int current_pos;           // Current position
} OptimizedKVCache;

infiniStatus_t kv_cache_compress_old_entries(OptimizedKVCache *cache) {
    // Compress old cache entries to save memory
    if (cache->current_pos > cache->full_cache_size) {
        int compress_start = cache->current_pos - cache->full_cache_size;
        
        // Move recent entries to full cache
        infinirtMemcpy(cache->full_k_cache, 
                       (char*)cache->full_k_cache + compress_start * sizeof(float),
                       cache->full_cache_size * sizeof(float),
                       INFINIRT_MEMCPY_DEVICE_TO_DEVICE, NULL);
        
        // Compress older entries
        compress_cache_entries(cache->compressed_k_cache,
                              cache->full_k_cache,
                              compress_start,
                              cache->compression_ratio);
    }
    
    return INFINI_STATUS_SUCCESS;
}

// Page-based KV cache management
typedef struct {
    void **cache_pages;        // Array of cache pages
    int *page_usage;          // Usage count for each page
    int page_size;            // Tokens per page
    int num_pages;            // Total number of pages
    int active_pages;         // Currently active pages
} PagedKVCache;

infiniStatus_t paged_kv_cache_allocate_sequence(PagedKVCache *cache,
                                               int sequence_length,
                                               int **allocated_pages,
                                               int *num_allocated) {
    int pages_needed = (sequence_length + cache->page_size - 1) / cache->page_size;
    *allocated_pages = malloc(pages_needed * sizeof(int));
    *num_allocated = 0;
    
    // Find available pages using LRU or similar policy
    for (int i = 0; i < cache->num_pages && *num_allocated < pages_needed; i++) {
        if (cache->page_usage[i] == 0) {
            (*allocated_pages)[*num_allocated] = i;
            cache->page_usage[i] = 1;
            (*num_allocated)++;
        }
    }
    
    return (*num_allocated == pages_needed) ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INSUFFICIENT_WORKSPACE;
}
```

## Profiling and Analysis

### 1. Performance Profiling
Comprehensive performance measurement:

```c
// Detailed profiling structure
typedef struct {
    // Timing information
    double total_time;
    double compute_time;
    double memory_time;
    double synchronization_time;
    
    // Memory usage
    size_t peak_memory_usage;
    size_t average_memory_usage;
    float memory_efficiency;    // % of peak bandwidth used
    
    // Compute utilization
    float compute_utilization;  // % of peak compute used
    float cache_hit_rate;      // Cache efficiency
    
    // Operation breakdown
    double embedding_time;
    double attention_time;
    double mlp_time;
    double normalization_time;
    
    // Per-layer timing
    double *layer_times;
    int num_layers;
} DetailedProfileInfo;

infiniStatus_t profile_model_execution(QwenModel *model,
                                      int *input_ids, int batch_size, int seq_len,
                                      DetailedProfileInfo *profile) {
    
    profile->layer_times = malloc(model->config.num_hidden_layers * sizeof(double));
    
    auto total_start = get_high_precision_time();
    
    // Profile embedding
    auto embed_start = get_high_precision_time();
    // ... embedding operations ...
    profile->embedding_time = get_high_precision_time() - embed_start;
    
    // Profile each layer
    for (int layer = 0; layer < model->config.num_hidden_layers; layer++) {
        auto layer_start = get_high_precision_time();
        
        // Profile attention
        auto attn_start = get_high_precision_time();
        // ... attention operations ...
        profile->attention_time += get_high_precision_time() - attn_start;
        
        // Profile MLP
        auto mlp_start = get_high_precision_time();
        // ... MLP operations ...
        profile->mlp_time += get_high_precision_time() - mlp_start;
        
        profile->layer_times[layer] = get_high_precision_time() - layer_start;
    }
    
    profile->total_time = get_high_precision_time() - total_start;
    
    // Get memory statistics
    get_memory_statistics(model->handle, &profile->peak_memory_usage,
                         &profile->memory_efficiency);
    
    // Get compute utilization
    get_compute_statistics(model->handle, &profile->compute_utilization);
    
    return INFINI_STATUS_SUCCESS;
}
```

### 2. Bottleneck Analysis
Identify performance bottlenecks:

```c
// Bottleneck detection
typedef enum {
    BOTTLENECK_COMPUTE,     // Compute bound
    BOTTLENECK_MEMORY,      // Memory bandwidth bound
    BOTTLENECK_CACHE,       // Cache miss bound
    BOTTLENECK_SYNC,        // Synchronization bound
    BOTTLENECK_UNKNOWN
} BottleneckType;

BottleneckType analyze_bottleneck(DetailedProfileInfo *profile) {
    // Heuristic bottleneck detection
    if (profile->compute_utilization < 0.6) {
        if (profile->memory_efficiency > 0.8) {
            return BOTTLENECK_SYNC;  // High memory usage but low compute
        } else {
            return BOTTLENECK_MEMORY;  // Low memory and compute usage
        }
    } else if (profile->cache_hit_rate < 0.7) {
        return BOTTLENECK_CACHE;
    } else {
        return BOTTLENECK_COMPUTE;
    }
}

// Performance recommendations
void generate_optimization_recommendations(DetailedProfileInfo *profile,
                                         BottleneckType bottleneck) {
    printf("Performance Analysis Results:\n");
    printf("  Total Time: %.2f ms\n", profile->total_time);
    printf("  Compute Utilization: %.1f%%\n", profile->compute_utilization * 100);
    printf("  Memory Efficiency: %.1f%%\n", profile->memory_efficiency * 100);
    
    switch (bottleneck) {
        case BOTTLENECK_COMPUTE:
            printf("Recommendations:\n");
            printf("  - Consider operator fusion to reduce kernel launch overhead\n");
            printf("  - Use mixed precision to increase compute throughput\n");
            printf("  - Optimize algorithm selection for better compute efficiency\n");
            break;
            
        case BOTTLENECK_MEMORY:
            printf("Recommendations:\n");
            printf("  - Increase batch size to improve memory efficiency\n");
            printf("  - Use data layout optimization for better cache utilization\n");
            printf("  - Consider compression techniques for weights/activations\n");
            break;
            
        case BOTTLENECK_CACHE:
            printf("Recommendations:\n");
            printf("  - Optimize data access patterns\n");
            printf("  - Use blocking/tiling techniques\n");
            printf("  - Consider prefetching strategies\n");
            break;
            
        case BOTTLENECK_SYNC:
            printf("Recommendations:\n");
            printf("  - Use asynchronous execution where possible\n");
            printf("  - Optimize stream usage and dependencies\n");
            printf("  - Consider pipeline parallelism\n");
            break;
    }
}
```

## Best Practices

### 1. Development Workflow
```c
// Performance-oriented development workflow
typedef struct {
    // Baseline measurements
    DetailedProfileInfo baseline_profile;
    
    // Optimization targets
    double target_latency;       // Target inference time
    size_t target_memory;        // Target memory usage
    float target_accuracy;       // Minimum accuracy requirement
    
    // Current measurements
    DetailedProfileInfo current_profile;
    float current_accuracy;
    
    // Optimization history
    OptimizationStep *optimization_history;
    int num_optimizations;
} OptimizationTracker;

// Systematic optimization approach
infiniStatus_t systematic_optimization(QwenModel *model, OptimizationTracker *tracker) {
    // 1. Establish baseline
    profile_model_execution(model, /* test inputs */, &tracker->baseline_profile);
    
    // 2. Apply optimizations in order of impact
    optimization_steps[] = {
        optimize_memory_layout,
        apply_operator_fusion,
        tune_precision_settings,
        optimize_batch_processing,
        apply_advanced_techniques
    };
    
    for (int i = 0; i < sizeof(optimization_steps)/sizeof(optimization_steps[0]); i++) {
        // Apply optimization
        optimization_steps[i](model);
        
        // Measure performance
        profile_model_execution(model, /* test inputs */, &tracker->current_profile);
        
        // Validate accuracy
        float accuracy = validate_model_accuracy(model);
        
        // Check if optimization is beneficial
        if (tracker->current_profile.total_time < tracker->baseline_profile.total_time &&
            accuracy >= tracker->target_accuracy) {
            // Keep optimization
            tracker->baseline_profile = tracker->current_profile;
            printf("Optimization %d: Improved performance by %.1f%%\n", i,
                   (1.0 - tracker->current_profile.total_time / tracker->baseline_profile.total_time) * 100);
        } else {
            // Revert optimization
            revert_optimization(model, i);
            printf("Optimization %d: Reverted due to performance/accuracy regression\n", i);
        }
    }
    
    return INFINI_STATUS_SUCCESS;
}
```

### 2. Production Deployment
```c
// Production-ready optimization settings
typedef struct {
    // Performance settings
    int optimal_batch_size;
    MixedPrecisionConfig precision_config;
    MemoryPoolConfig memory_config;
    
    // Reliability settings
    bool enable_checkpointing;
    int checkpoint_frequency;
    bool enable_fallback_algorithms;
    
    // Monitoring settings
    bool enable_performance_monitoring;
    int monitoring_frequency;
    
    // Adaptive settings
    bool enable_dynamic_batching;
    bool enable_adaptive_precision;
    bool enable_load_balancing;
} ProductionConfig;

infiniStatus_t deploy_optimized_model(QwenModel *model, ProductionConfig *config) {
    // Apply production optimizations
    set_optimal_batch_size(model, config->optimal_batch_size);
    configure_mixed_precision(model, &config->precision_config);
    setup_memory_pools(model, &config->memory_config);
    
    // Setup monitoring
    if (config->enable_performance_monitoring) {
        setup_performance_monitoring(model, config->monitoring_frequency);
    }
    
    // Setup adaptive features
    if (config->enable_dynamic_batching) {
        enable_dynamic_batching(model);
    }
    
    return INFINI_STATUS_SUCCESS;
}
```

This comprehensive performance optimization guide provides strategies for maximizing InfiniCore performance across different hardware platforms. The key is to understand your specific workload characteristics and apply optimizations systematically while maintaining accuracy and reliability.

## Next Steps

- **[API Reference](api/)**: Detailed documentation of optimization-related APIs
- **[Examples](examples/)**: Performance optimization examples for specific models
- **[Troubleshooting](troubleshooting.md)**: Common performance issues and solutions