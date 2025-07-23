# InfiniCore Examples

This directory contains practical examples demonstrating how to use InfiniCore for various tasks.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Operator Examples](#operator-examples)
3. [Model Examples](#model-examples)
4. [Performance Examples](#performance-examples)

## Basic Examples

### Simple Matrix Addition
```c
// examples/basic/matrix_add.c
#include <infiniop.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize
    infiniopHandle_t handle;
    infiniStatus_t status = infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);
    if (status != INFINI_STATUS_SUCCESS) {
        printf("Failed to create handle\n");
        return 1;
    }
    
    // Define matrix dimensions
    const int M = 4, N = 4;
    const size_t size = M * N * sizeof(float);
    
    // Create tensor descriptors
    infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
    int shape[] = {M, N};
    
    infiniopCreateTensorDescriptor(&a_desc, INFINI_DTYPE_F32, 2, shape, NULL);
    infiniopCreateTensorDescriptor(&b_desc, INFINI_DTYPE_F32, 2, shape, NULL);
    infiniopCreateTensorDescriptor(&c_desc, INFINI_DTYPE_F32, 2, shape, NULL);
    
    // Allocate memory
    float *a = (float*)malloc(size);
    float *b = (float*)malloc(size);
    float *c = (float*)malloc(size);
    
    // Initialize data
    for (int i = 0; i < M * N; i++) {
        a[i] = i * 0.1f;
        b[i] = i * 0.2f;
    }
    
    // Create add descriptor
    infiniopAddDescriptor_t add_desc;
    infiniopCreateAddDescriptor(handle, &add_desc, c_desc, a_desc, b_desc);
    
    // Get workspace size
    size_t workspace_size;
    infiniopGetAddWorkspaceSize(add_desc, &workspace_size);
    void *workspace = malloc(workspace_size);
    
    // Perform addition
    infiniopAdd(add_desc, workspace, workspace_size, c, a, b, NULL);
    
    // Print results
    printf("Matrix A + B = C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", c[i * N + j]);
        }
        printf("\n");
    }
    
    // Cleanup
    free(workspace);
    free(a);
    free(b);
    free(c);
    infiniopDestroyAddDescriptor(add_desc);
    infiniopDestroyTensorDescriptor(a_desc);
    infiniopDestroyTensorDescriptor(b_desc);
    infiniopDestroyTensorDescriptor(c_desc);
    infiniopDestroyHandle(handle);
    
    return 0;
}
```

### Matrix Multiplication
```c
// examples/basic/matrix_multiply.c
#include <infiniop.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);
    
    // Matrix dimensions: A(M,K) * B(K,N) = C(M,N)
    const int M = 3, K = 4, N = 2;
    
    // Create tensor descriptors
    infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
    int a_shape[] = {M, K};
    int b_shape[] = {K, N};
    int c_shape[] = {M, N};
    
    infiniopCreateTensorDescriptor(&a_desc, INFINI_DTYPE_F32, 2, a_shape, NULL);
    infiniopCreateTensorDescriptor(&b_desc, INFINI_DTYPE_F32, 2, b_shape, NULL);
    infiniopCreateTensorDescriptor(&c_desc, INFINI_DTYPE_F32, 2, c_shape, NULL);
    
    // Allocate and initialize data
    float *a = (float*)malloc(M * K * sizeof(float));
    float *b = (float*)malloc(K * N * sizeof(float));
    float *c = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) a[i] = i + 1;
    for (int i = 0; i < K * N; i++) b[i] = (i + 1) * 0.1f;
    
    // Create GEMM descriptor
    infiniopGemmDescriptor_t gemm_desc;
    infiniopCreateGemmDescriptor(handle, &gemm_desc, c_desc, a_desc, b_desc);
    
    // Get workspace and perform multiplication
    size_t workspace_size;
    infiniopGetGemmWorkspaceSize(gemm_desc, &workspace_size);
    void *workspace = malloc(workspace_size);
    
    infiniopGemm(gemm_desc, workspace, workspace_size, c, a, b, 1.0f, 0.0f, NULL);
    
    // Print result
    printf("Result C = A * B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", c[i * N + j]);
        }
        printf("\n");
    }
    
    // Cleanup
    free(workspace);
    free(a); free(b); free(c);
    infiniopDestroyGemmDescriptor(gemm_desc);
    infiniopDestroyTensorDescriptor(a_desc);
    infiniopDestroyTensorDescriptor(b_desc);
    infiniopDestroyTensorDescriptor(c_desc);
    infiniopDestroyHandle(handle);
    
    return 0;
}
```

## Operator Examples

### Custom ReLU Implementation
```c
// examples/operators/custom_relu.c
#include <infiniop.h>

// This example shows how to implement a custom ReLU operator
// following InfiniCore patterns

// CPU implementation
static infiniStatus_t relu_cpu_forward(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = fmaxf(0.0f, input[i]);
    }
    return INFINI_STATUS_SUCCESS;
}

// Wrapper function that follows InfiniCore operator pattern
infiniStatus_t custom_relu_forward(infiniopHandle_t handle,
                                  const void *input, void *output,
                                  int num_elements, infiniDtype_t dtype,
                                  void *stream) {
    switch (handle->device) {
        case INFINI_DEVICE_CPU:
            if (dtype == INFINI_DTYPE_F32) {
                return relu_cpu_forward((const float*)input, (float*)output, num_elements);
            }
            break;
        // Add other devices as needed
        default:
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

int main() {
    // Test the custom ReLU implementation
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);
    
    const int n = 8;
    float input[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    float output[n];
    
    custom_relu_forward(handle, input, output, n, INFINI_DTYPE_F32, NULL);
    
    printf("ReLU Results:\n");
    for (int i = 0; i < n; i++) {
        printf("ReLU(%.1f) = %.1f\n", input[i], output[i]);
    }
    
    infiniopDestroyHandle(handle);
    return 0;
}
```

### Layer Normalization Example
```c
// examples/operators/layer_norm_example.c
#include <infiniop.h>
#include <math.h>

// Simple layer normalization implementation
void layer_norm_cpu(const float *input, float *output, const float *weight, 
                   const float *bias, int batch_size, int hidden_size, float epsilon) {
    for (int b = 0; b < batch_size; b++) {
        const float *x = input + b * hidden_size;
        float *y = output + b * hidden_size;
        
        // Calculate mean
        float mean = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            mean += x[i];
        }
        mean /= hidden_size;
        
        // Calculate variance
        float variance = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float diff = x[i] - mean;
            variance += diff * diff;
        }
        variance /= hidden_size;
        
        // Normalize
        float inv_std = 1.0f / sqrtf(variance + epsilon);
        for (int i = 0; i < hidden_size; i++) {
            y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

int main() {
    const int batch_size = 2;
    const int hidden_size = 4;
    const float epsilon = 1e-5f;
    
    // Test data
    float input[] = {
        1.0f, 2.0f, 3.0f, 4.0f,    // batch 0
        0.5f, 1.5f, 2.5f, 3.5f     // batch 1
    };
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float bias[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float output[batch_size * hidden_size];
    
    layer_norm_cpu(input, output, weight, bias, batch_size, hidden_size, epsilon);
    
    printf("Layer Normalization Results:\n");
    for (int b = 0; b < batch_size; b++) {
        printf("Batch %d: ", b);
        for (int i = 0; i < hidden_size; i++) {
            printf("%.3f ", output[b * hidden_size + i]);
        }
        printf("\n");
    }
    
    return 0;
}
```

## Model Examples

### Simple Transformer Layer
```c
// examples/models/transformer_layer.c
#include <infiniop.h>

typedef struct {
    infiniopHandle_t handle;
    
    // Attention weights
    float *q_weight, *k_weight, *v_weight, *o_weight;
    
    // MLP weights  
    float *gate_weight, *up_weight, *down_weight;
    
    // Normalization weights
    float *input_norm_weight, *post_attn_norm_weight;
    
    // Descriptors
    infiniopGemmDescriptor_t q_proj, k_proj, v_proj, o_proj;
    infiniopGemmDescriptor_t gate_proj, up_proj, down_proj;
    infiniopRMSNormDescriptor_t input_norm, post_attn_norm;
    
    // Configuration
    int hidden_size;
    int intermediate_size;
    int num_heads;
    float norm_epsilon;
} SimpleTransformerLayer;

infiniStatus_t transformer_layer_forward(SimpleTransformerLayer *layer,
                                        const float *input, float *output,
                                        void *workspace, size_t workspace_size) {
    const int seq_len = 1;  // Single token for simplicity
    const int hidden_size = layer->hidden_size;
    
    // Use workspace for intermediate results
    float *norm_output = (float*)workspace;
    float *attn_output = norm_output + hidden_size;
    float *residual = attn_output + hidden_size;
    float *mlp_gate = residual + hidden_size;
    float *mlp_up = mlp_gate + layer->intermediate_size;
    float *mlp_output = mlp_up + layer->intermediate_size;
    
    // Store input for residual connection
    memcpy(residual, input, hidden_size * sizeof(float));
    
    // 1. Input layer normalization
    infiniopRMSNorm(layer->input_norm, workspace, workspace_size,
                   norm_output, input, layer->input_norm_weight, NULL);
    
    // 2. Self-attention (simplified - just using as linear layers)
    // In a real implementation, you'd need proper attention computation
    float *q_out = mlp_output;  // Reuse space
    infiniopGemm(layer->q_proj, workspace, workspace_size,
                q_out, norm_output, layer->q_weight, 1.0f, 0.0f, NULL);
    
    // For simplicity, output = q_projection * o_weight
    infiniopGemm(layer->o_proj, workspace, workspace_size,
                attn_output, q_out, layer->o_weight, 1.0f, 0.0f, NULL);
    
    // 3. Residual connection after attention
    for (int i = 0; i < hidden_size; i++) {
        attn_output[i] += residual[i];
    }
    
    // Store for next residual
    memcpy(residual, attn_output, hidden_size * sizeof(float));
    
    // 4. Post-attention normalization
    infiniopRMSNorm(layer->post_attn_norm, workspace, workspace_size,
                   norm_output, attn_output, layer->post_attn_norm_weight, NULL);
    
    // 5. MLP
    infiniopGemm(layer->gate_proj, workspace, workspace_size,
                mlp_gate, norm_output, layer->gate_weight, 1.0f, 0.0f, NULL);
    infiniopGemm(layer->up_proj, workspace, workspace_size,
                mlp_up, norm_output, layer->up_weight, 1.0f, 0.0f, NULL);
    
    // SwiGLU: gate * silu(up)
    for (int i = 0; i < layer->intermediate_size; i++) {
        float up_val = mlp_up[i];
        float silu_val = up_val / (1.0f + expf(-up_val));  // SiLU activation
        mlp_gate[i] = mlp_gate[i] * silu_val;
    }
    
    infiniopGemm(layer->down_proj, workspace, workspace_size,
                mlp_output, mlp_gate, layer->down_weight, 1.0f, 0.0f, NULL);
    
    // 6. Final residual connection
    for (int i = 0; i < hidden_size; i++) {
        output[i] = mlp_output[i] + residual[i];
    }
    
    return INFINI_STATUS_SUCCESS;
}

int main() {
    // This is a conceptual example - real implementation would need
    // proper weight initialization and more complete attention mechanism
    printf("Transformer layer example (conceptual)\n");
    printf("See docs/models.md for complete implementation\n");
    return 0;
}
```

## Performance Examples

### Batch Processing Example
```c
// examples/performance/batch_processing.c
#include <infiniop.h>
#include <time.h>

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void benchmark_batch_sizes(infiniopHandle_t handle) {
    printf("Benchmarking different batch sizes for GEMM:\n");
    printf("Batch Size | Time (ms) | Throughput (GFLOPS)\n");
    printf("-----------|-----------|-------------------\n");
    
    const int M = 1024, K = 1024, N = 1024;
    int batch_sizes[] = {1, 4, 8, 16, 32, 64};
    int num_batch_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
    
    for (int b = 0; b < num_batch_sizes; b++) {
        int batch_size = batch_sizes[b];
        
        // Create descriptors for batched GEMM
        infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
        int a_shape[] = {batch_size, M, K};
        int b_shape[] = {batch_size, K, N};  
        int c_shape[] = {batch_size, M, N};
        
        infiniopCreateTensorDescriptor(&a_desc, INFINI_DTYPE_F32, 3, a_shape, NULL);
        infiniopCreateTensorDescriptor(&b_desc, INFINI_DTYPE_F32, 3, b_shape, NULL);
        infiniopCreateTensorDescriptor(&c_desc, INFINI_DTYPE_F32, 3, c_shape, NULL);
        
        infiniopGemmDescriptor_t gemm_desc;
        infiniopCreateGemmDescriptor(handle, &gemm_desc, c_desc, a_desc, b_desc);
        
        // Allocate memory
        size_t a_size = batch_size * M * K * sizeof(float);
        size_t b_size = batch_size * K * N * sizeof(float);
        size_t c_size = batch_size * M * N * sizeof(float);
        
        float *a = (float*)malloc(a_size);
        float *b = (float*)malloc(b_size);
        float *c = (float*)malloc(c_size);
        
        // Initialize with random data
        for (int i = 0; i < batch_size * M * K; i++) a[i] = rand() / (float)RAND_MAX;
        for (int i = 0; i < batch_size * K * N; i++) b[i] = rand() / (float)RAND_MAX;
        
        // Get workspace
        size_t workspace_size;
        infiniopGetGemmWorkspaceSize(gemm_desc, &workspace_size);
        void *workspace = malloc(workspace_size);
        
        // Benchmark
        const int num_runs = 10;
        double start_time = get_time();
        
        for (int run = 0; run < num_runs; run++) {
            infiniopGemm(gemm_desc, workspace, workspace_size, c, a, b, 1.0f, 0.0f, NULL);
        }
        
        double end_time = get_time();
        double avg_time = (end_time - start_time) / num_runs * 1000;  // ms
        
        // Calculate throughput
        double flops = 2.0 * batch_size * M * N * K;  // FMA operations
        double gflops = (flops / (avg_time / 1000)) / 1e9;
        
        printf("%10d | %9.2f | %17.2f\n", batch_size, avg_time, gflops);
        
        // Cleanup
        free(workspace);
        free(a); free(b); free(c);
        infiniopDestroyGemmDescriptor(gemm_desc);
        infiniopDestroyTensorDescriptor(a_desc);
        infiniopDestroyTensorDescriptor(b_desc);
        infiniopDestroyTensorDescriptor(c_desc);
    }
}

int main() {
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);
    
    benchmark_batch_sizes(handle);
    
    infiniopDestroyHandle(handle);
    return 0;
}
```

## Building Examples

### Makefile
```makefile
# examples/Makefile
CC = gcc
CFLAGS = -Wall -O2 -std=c99
INCLUDES = -I$(INFINI_ROOT)/include
LIBS = -L$(INFINI_ROOT)/lib -linfiniop -linfinicore -lm

# Source files
BASIC_SOURCES = basic/matrix_add.c basic/matrix_multiply.c
OPERATOR_SOURCES = operators/custom_relu.c operators/layer_norm_example.c
MODEL_SOURCES = models/transformer_layer.c
PERF_SOURCES = performance/batch_processing.c

# Targets
BASIC_TARGETS = $(BASIC_SOURCES:.c=)
OPERATOR_TARGETS = $(OPERATOR_SOURCES:.c=)
MODEL_TARGETS = $(MODEL_SOURCES:.c=)
PERF_TARGETS = $(PERF_SOURCES:.c=)

ALL_TARGETS = $(BASIC_TARGETS) $(OPERATOR_TARGETS) $(MODEL_TARGETS) $(PERF_TARGETS)

all: $(ALL_TARGETS)

%: %.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

clean:
	rm -f $(ALL_TARGETS)

test: all
	@echo "Running basic examples..."
	./basic/matrix_add
	./basic/matrix_multiply
	@echo "Running operator examples..."
	./operators/custom_relu
	./operators/layer_norm_example

.PHONY: all clean test
```

### CMake Alternative
```cmake
# examples/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(InfiniCoreExamples)

set(CMAKE_C_STANDARD 99)

# Find InfiniCore
find_path(INFINICORE_INCLUDE_DIR infiniop.h PATHS ${INFINI_ROOT}/include)
find_library(INFINICORE_LIB infiniop PATHS ${INFINI_ROOT}/lib)

include_directories(${INFINICORE_INCLUDE_DIR})

# Basic examples
add_executable(matrix_add basic/matrix_add.c)
target_link_libraries(matrix_add ${INFINICORE_LIB} m)

add_executable(matrix_multiply basic/matrix_multiply.c)
target_link_libraries(matrix_multiply ${INFINICORE_LIB} m)

# Operator examples
add_executable(custom_relu operators/custom_relu.c)
target_link_libraries(custom_relu ${INFINICORE_LIB} m)

# Performance examples
add_executable(batch_processing performance/batch_processing.c)
target_link_libraries(batch_processing ${INFINICORE_LIB} m)
```

These examples provide a solid foundation for learning InfiniCore usage patterns. Start with the basic examples and gradually move to more complex scenarios as you become familiar with the API.