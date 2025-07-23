# Large Model Adaptation Guide

This guide shows you how to adapt large language models (like Qwen3, LLaMA, GPT, etc.) to use InfiniCore for high-performance inference across different hardware platforms.

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture Analysis](#model-architecture-analysis)
3. [Qwen3 Adaptation Example](#qwen3-adaptation-example)
4. [Common LLM Components](#common-llm-components)
5. [Model Loading and Weight Management](#model-loading-and-weight-management)
6. [Inference Pipeline](#inference-pipeline)
7. [Memory Optimization](#memory-optimization)
8. [Performance Tuning](#performance-tuning)
9. [Multi-GPU Deployment](#multi-gpu-deployment)
10. [Troubleshooting](#troubleshooting)

## Overview

Large language models consist of several key components that can be efficiently implemented using InfiniCore operators:

- **Embedding Layers**: Token and positional embeddings
- **Transformer Blocks**: Self-attention and feed-forward networks
- **Normalization**: LayerNorm, RMSNorm
- **Attention Mechanisms**: Multi-head attention with various optimizations
- **Activation Functions**: SwiGLU, GELU, ReLU variants
- **Output Projection**: Final linear layer and sampling

### Benefits of InfiniCore for LLMs

1. **Hardware Portability**: Same code runs on NVIDIA, Ascend, Cambricon, etc.
2. **Optimized Kernels**: Hand-tuned implementations for each platform
3. **Memory Efficiency**: Advanced memory management and optimization
4. **Scalability**: Multi-GPU and distributed inference support
5. **Precision Flexibility**: FP32, FP16, BF16, and mixed precision

## Model Architecture Analysis

Before adapting a model, analyze its architecture to identify required operators:

### Example: Qwen3 Architecture
```python
# Typical Qwen3 components:
class QwenMLP:
    # gate_proj: Linear(hidden_size, intermediate_size)
    # up_proj: Linear(hidden_size, intermediate_size) 
    # down_proj: Linear(intermediate_size, hidden_size)
    # activation: SwiGLU

class QwenAttention:
    # q_proj, k_proj, v_proj: Linear layers
    # o_proj: Output projection
    # rotary_emb: RoPE positional encoding
    # attention: Scaled dot-product with causal masking

class QwenDecoderLayer:
    # input_layernorm: RMSNorm
    # self_attn: QwenAttention
    # post_attention_layernorm: RMSNorm
    # mlp: QwenMLP

class QwenModel:
    # embed_tokens: Embedding layer
    # layers: Stack of QwenDecoderLayer
    # norm: Final RMSNorm
    # lm_head: Output projection (often tied with embeddings)
```

### Required InfiniCore Operators
Based on this analysis, you'll need:
- `infiniopGemm` - Matrix multiplication for linear layers
- `infiniopRMSNorm` - Layer normalization  
- `infiniopAttention` - Multi-head attention
- `infiniopRoPE` - Rotary positional encoding
- `infiniopSwiGLU` - SwiGLU activation
- `infiniopAdd` - Residual connections
- Element-wise operations for activations

## Qwen3 Adaptation Example

### Step 1: Model Configuration

Create a configuration structure:

```c
// qwen_config.h
typedef struct {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int max_position_embeddings;
    float rms_norm_eps;
    float rope_theta;
    bool tie_word_embeddings;
} QwenConfig;

typedef struct {
    infiniopHandle_t handle;
    QwenConfig config;
    
    // Model weights
    void *embed_tokens_weight;      // [vocab_size, hidden_size]
    void *lm_head_weight;          // [vocab_size, hidden_size] or shared
    void *norm_weight;             // [hidden_size]
    
    // Layer weights (arrays of size num_hidden_layers)
    void **input_layernorm_weight;     // [hidden_size]
    void **post_attention_layernorm_weight; // [hidden_size]
    void **q_proj_weight;              // [hidden_size, num_heads * head_dim]
    void **k_proj_weight;              // [hidden_size, num_kv_heads * head_dim]
    void **v_proj_weight;              // [hidden_size, num_kv_heads * head_dim]
    void **o_proj_weight;              // [num_heads * head_dim, hidden_size]
    void **gate_proj_weight;           // [hidden_size, intermediate_size]
    void **up_proj_weight;             // [hidden_size, intermediate_size]
    void **down_proj_weight;           // [intermediate_size, hidden_size]
    
    // Operator descriptors (pre-created for efficiency)
    infiniopGemmDescriptor_t *q_proj_desc;
    infiniopGemmDescriptor_t *k_proj_desc;
    infiniopGemmDescriptor_t *v_proj_desc;
    infiniopGemmDescriptor_t *o_proj_desc;
    infiniopGemmDescriptor_t *gate_proj_desc;
    infiniopGemmDescriptor_t *up_proj_desc;
    infiniopGemmDescriptor_t *down_proj_desc;
    infiniopRMSNormDescriptor_t *input_norm_desc;
    infiniopRMSNormDescriptor_t *post_attn_norm_desc;
    infiniopAttentionDescriptor_t *attention_desc;
    infiniopSwiGLUDescriptor_t *swiglu_desc;
    infiniopRoPEDescriptor_t *rope_desc;
    
    // Workspace management
    void *workspace;
    size_t workspace_size;
} QwenModel;
```

### Step 2: Model Initialization

```c
// qwen_model.c
#include "qwen_config.h"
#include <infiniop.h>

infiniStatus_t qwen_model_create(QwenModel **model_ptr, 
                                infiniopHandle_t handle,
                                const QwenConfig *config) {
    QwenModel *model = malloc(sizeof(QwenModel));
    model->handle = handle;
    model->config = *config;
    
    // Allocate weight storage
    const int hidden_size = config->hidden_size;
    const int intermediate_size = config->intermediate_size;
    const int vocab_size = config->vocab_size;
    const int num_layers = config->num_hidden_layers;
    
    // Allocate device memory for weights (example for one layer)
    infinirtMalloc(&model->embed_tokens_weight, 
                   vocab_size * hidden_size * sizeof(float), 
                   handle->context);
    
    // Allocate arrays for per-layer weights
    model->q_proj_weight = malloc(num_layers * sizeof(void*));
    model->q_proj_desc = malloc(num_layers * sizeof(infiniopGemmDescriptor_t));
    
    for (int i = 0; i < num_layers; i++) {
        // Allocate weight memory
        infinirtMalloc(&model->q_proj_weight[i],
                       hidden_size * hidden_size * sizeof(float),
                       handle->context);
        
        // Create tensor descriptors
        infiniopTensorDescriptor_t q_weight_desc, q_input_desc, q_output_desc;
        
        infiniopCreateTensorDescriptor(&q_weight_desc, INFINI_DTYPE_F32, 2, 
                                       (int[]){hidden_size, hidden_size}, NULL);
        infiniopCreateTensorDescriptor(&q_input_desc, INFINI_DTYPE_F32, 3,
                                       (int[]){1, 1, hidden_size}, NULL);  // [batch, seq, hidden]
        infiniopCreateTensorDescriptor(&q_output_desc, INFINI_DTYPE_F32, 3,
                                       (int[]){1, 1, hidden_size}, NULL);
        
        // Create GEMM descriptor for Q projection
        infiniopCreateGemmDescriptor(handle, &model->q_proj_desc[i],
                                     q_output_desc, q_input_desc, q_weight_desc);
        
        // Clean up temp descriptors
        infiniopDestroyTensorDescriptor(q_weight_desc);
        infiniopDestroyTensorDescriptor(q_input_desc);
        infiniopDestroyTensorDescriptor(q_output_desc);
    }
    
    // Calculate total workspace size needed
    size_t total_workspace = 0;
    for (int i = 0; i < num_layers; i++) {
        size_t layer_workspace;
        infiniopGetGemmWorkspaceSize(model->q_proj_desc[i], &layer_workspace);
        total_workspace = max(total_workspace, layer_workspace);
    }
    
    // Add workspace for attention, normalization, etc.
    total_workspace += hidden_size * 1024 * sizeof(float);  // Attention workspace
    
    model->workspace_size = total_workspace;
    infinirtMalloc(&model->workspace, total_workspace, handle->context);
    
    *model_ptr = model;
    return INFINI_STATUS_SUCCESS;
}
```

### Step 3: Layer Implementation

```c
// Transformer decoder layer implementation
infiniStatus_t qwen_decoder_layer_forward(QwenModel *model, int layer_idx,
                                         void *hidden_states,      // [batch, seq_len, hidden_size]
                                         void *attention_mask,     // [batch, seq_len, seq_len]
                                         void *position_ids,       // [batch, seq_len]
                                         void *past_key_value,     // KV cache
                                         void *stream) {
    
    const int hidden_size = model->config.hidden_size;
    const int intermediate_size = model->config.intermediate_size;
    
    // Allocate temporary buffers from workspace
    void *residual = model->workspace;
    void *norm_output = (char*)model->workspace + hidden_size * sizeof(float);
    void *attn_output = (char*)norm_output + hidden_size * sizeof(float);
    void *gate_output = (char*)attn_output + hidden_size * sizeof(float);
    void *up_output = (char*)gate_output + intermediate_size * sizeof(float);
    void *mlp_output = (char*)up_output + intermediate_size * sizeof(float);
    
    // Store residual connection
    infinirtMemcpy(residual, hidden_states, hidden_size * sizeof(float),
                   INFINIRT_MEMCPY_DEVICE_TO_DEVICE, stream);
    
    // 1. Input LayerNorm
    infiniopRMSNorm(model->input_norm_desc[layer_idx],
                    model->workspace, model->workspace_size,
                    norm_output, hidden_states,
                    model->input_layernorm_weight[layer_idx],
                    stream);
    
    // 2. Self-Attention
    // Note: This is simplified - real attention needs Q, K, V projections
    infiniopAttention(model->attention_desc[layer_idx],
                      model->workspace, model->workspace_size,
                      attn_output, norm_output, norm_output, norm_output,
                      past_key_value, past_key_value, stream);
    
    // 3. Residual connection after attention
    infiniopAdd(/* add descriptor */, 
                model->workspace, model->workspace_size,
                hidden_states, residual, attn_output, stream);
    
    // Store new residual
    infinirtMemcpy(residual, hidden_states, hidden_size * sizeof(float),
                   INFINIRT_MEMCPY_DEVICE_TO_DEVICE, stream);
    
    // 4. Post-attention LayerNorm
    infiniopRMSNorm(model->post_attn_norm_desc[layer_idx],
                    model->workspace, model->workspace_size,
                    norm_output, hidden_states,
                    model->post_attention_layernorm_weight[layer_idx],
                    stream);
    
    // 5. MLP - Gate and Up projections
    infiniopGemm(model->gate_proj_desc[layer_idx],
                 model->workspace, model->workspace_size,
                 gate_output, norm_output, model->gate_proj_weight[layer_idx],
                 1.0f, 0.0f, stream);
                 
    infiniopGemm(model->up_proj_desc[layer_idx],
                 model->workspace, model->workspace_size,
                 up_output, norm_output, model->up_proj_weight[layer_idx],
                 1.0f, 0.0f, stream);
    
    // 6. SwiGLU activation
    infiniopSwiGLU(model->swiglu_desc[layer_idx],
                   model->workspace, model->workspace_size,
                   mlp_output, gate_output, up_output, stream);
    
    // 7. Down projection
    infiniopGemm(model->down_proj_desc[layer_idx],
                 model->workspace, model->workspace_size,
                 norm_output, mlp_output, model->down_proj_weight[layer_idx],
                 1.0f, 0.0f, stream);
    
    // 8. Final residual connection
    infiniopAdd(/* add descriptor */,
                model->workspace, model->workspace_size,
                hidden_states, residual, norm_output, stream);
    
    return INFINI_STATUS_SUCCESS;
}
```

### Step 4: Full Model Forward Pass

```c
infiniStatus_t qwen_model_forward(QwenModel *model,
                                 int *input_ids,           // [batch_size, seq_len]
                                 int batch_size,
                                 int seq_len,
                                 float *logits,            // [batch_size, seq_len, vocab_size]
                                 void *stream) {
    
    const int hidden_size = model->config.hidden_size;
    const int vocab_size = model->config.vocab_size;
    const int num_layers = model->config.num_hidden_layers;
    
    // Allocate activation memory
    void *hidden_states, *position_ids;
    infinirtMalloc(&hidden_states, batch_size * seq_len * hidden_size * sizeof(float),
                   model->handle->context);
    infinirtMalloc(&position_ids, batch_size * seq_len * sizeof(int),
                   model->handle->context);
    
    // 1. Token embeddings (simplified - would use lookup table)
    // For each token: hidden_states[i] = embed_tokens_weight[input_ids[i]]
    // This could be implemented as a custom embedding operator
    
    // 2. Position IDs (0, 1, 2, ... seq_len-1)
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            ((int*)position_ids)[b * seq_len + s] = s;
        }
    }
    
    // 3. Apply RoPE to embeddings
    infiniopRoPE(model->rope_desc,
                 model->workspace, model->workspace_size,
                 hidden_states, hidden_states, position_ids,
                 /* sin_table */ NULL, /* cos_table */ NULL,
                 stream);
    
    // 4. Pass through transformer layers
    for (int layer = 0; layer < num_layers; layer++) {
        qwen_decoder_layer_forward(model, layer, hidden_states,
                                  NULL, position_ids, NULL, stream);
    }
    
    // 5. Final layer normalization
    infiniopRMSNorm(model->norm_desc,
                    model->workspace, model->workspace_size,
                    hidden_states, hidden_states, model->norm_weight,
                    stream);
    
    // 6. Language modeling head (output projection)
    infiniopGemm(model->lm_head_desc,
                 model->workspace, model->workspace_size,
                 logits, hidden_states, model->lm_head_weight,
                 1.0f, 0.0f, stream);
    
    // Cleanup temporary memory
    infinirtFree(hidden_states, model->handle->context);
    infinirtFree(position_ids, model->handle->context);
    
    return INFINI_STATUS_SUCCESS;
}
```

## Common LLM Components

### Embedding Layers
```c
// Token embedding lookup
infiniStatus_t token_embedding_forward(void *output,        // [batch, seq, hidden]
                                      const int *input_ids, // [batch, seq]
                                      const void *weight,   // [vocab_size, hidden]
                                      int batch_size, int seq_len, int hidden_size,
                                      infiniopHandle_t handle, void *stream) {
    // Implement as gather operation or custom kernel
    // output[b][s] = weight[input_ids[b][s]]
}

// Positional embeddings (if not using RoPE)
infiniStatus_t positional_embedding_forward(void *output,     // [batch, seq, hidden]
                                           const int *pos_ids, // [batch, seq]
                                           const void *weight,  // [max_pos, hidden]
                                           int batch_size, int seq_len, int hidden_size,
                                           infiniopHandle_t handle, void *stream) {
    // Similar to token embedding
}
```

### KV Cache Management
```c
typedef struct {
    void *k_cache;  // [num_layers, batch, num_heads, max_seq, head_dim]
    void *v_cache;  // [num_layers, batch, num_heads, max_seq, head_dim]
    int *cache_lengths;  // [batch] - current sequence length for each batch
    int max_seq_len;
    int current_pos;
} KVCache;

infiniStatus_t kv_cache_update(KVCache *cache, int layer_idx,
                              const void *new_k, const void *new_v,
                              int batch_size, int seq_len,
                              infiniopHandle_t handle, void *stream) {
    // Copy new keys/values to appropriate position in cache
    // Update cache_lengths
}
```

### Attention Patterns
```c
// Different attention mechanisms
typedef enum {
    ATTENTION_FULL,           // Full attention (for prefill)
    ATTENTION_CAUSAL,         // Causal masking
    ATTENTION_SLIDING_WINDOW, // Sliding window attention
    ATTENTION_SPARSE,         // Sparse attention patterns
} AttentionType;

infiniStatus_t create_attention_mask(void *mask,           // [batch, seq, seq]
                                    AttentionType type,
                                    int batch_size, int seq_len,
                                    int window_size,       // for sliding window
                                    infiniopHandle_t handle, void *stream);
```

## Model Loading and Weight Management

### Weight Loading from Checkpoints
```c
// Load model weights from file (HuggingFace format, GGUF, etc.)
infiniStatus_t qwen_load_weights(QwenModel *model, const char *model_path) {
    // 1. Parse model configuration
    QwenConfig config;
    load_config_from_path(model_path, &config);
    
    // 2. Load and transfer weights to device
    for (int layer = 0; layer < config.num_hidden_layers; layer++) {
        // Load weights from disk
        float *q_proj_host = malloc(config.hidden_size * config.hidden_size * sizeof(float));
        load_tensor_from_file(model_path, layer, "q_proj.weight", q_proj_host);
        
        // Transfer to device
        infinirtMemcpy(model->q_proj_weight[layer], q_proj_host,
                       config.hidden_size * config.hidden_size * sizeof(float),
                       INFINIRT_MEMCPY_HOST_TO_DEVICE, NULL);
        
        free(q_proj_host);
    }
    
    return INFINI_STATUS_SUCCESS;
}
```

### Weight Quantization
```c
// Support for different quantization schemes
typedef enum {
    QUANT_NONE,     // FP32/FP16
    QUANT_INT8,     // 8-bit quantization
    QUANT_INT4,     // 4-bit quantization
    QUANT_GPTQ,     // GPTQ quantization
    QUANT_AWQ,      // AWQ quantization
} QuantizationType;

infiniStatus_t load_quantized_weights(QwenModel *model,
                                     const char *model_path,
                                     QuantizationType quant_type) {
    // Load quantized weights and dequantization parameters
    // Create appropriate descriptors for quantized operations
}
```

## Inference Pipeline

### Text Generation
```c
typedef struct {
    QwenModel *model;
    int *input_ids;      // Current input sequence
    int seq_len;         // Current sequence length
    int max_new_tokens;  // Maximum tokens to generate
    float temperature;   // Sampling temperature
    int top_k;          // Top-k sampling
    float top_p;        // Top-p (nucleus) sampling
    KVCache *kv_cache;  // Key-value cache
} GenerationConfig;

infiniStatus_t qwen_generate_text(GenerationConfig *config,
                                 int *output_tokens,  // Generated tokens
                                 int *output_length,  // Number of generated tokens
                                 void *stream) {
    
    while (config->seq_len < config->max_new_tokens) {
        // 1. Forward pass to get logits
        float *logits = malloc(config->model->config.vocab_size * sizeof(float));
        qwen_model_forward(config->model, config->input_ids + config->seq_len - 1,
                          1, 1, logits, stream);  // Generate next token
        
        // 2. Apply temperature scaling
        if (config->temperature != 1.0f) {
            for (int i = 0; i < config->model->config.vocab_size; i++) {
                logits[i] /= config->temperature;
            }
        }
        
        // 3. Sample next token
        int next_token = sample_token(logits, config->model->config.vocab_size,
                                     config->top_k, config->top_p);
        
        // 4. Add to sequence
        config->input_ids[config->seq_len] = next_token;
        output_tokens[config->seq_len - config->max_new_tokens] = next_token;
        config->seq_len++;
        
        // 5. Check for EOS token
        if (next_token == EOS_TOKEN_ID) break;
        
        free(logits);
    }
    
    *output_length = config->seq_len - (config->max_new_tokens - config->seq_len);
    return INFINI_STATUS_SUCCESS;
}
```

### Batched Inference
```c
// Batch multiple sequences for efficiency
typedef struct {
    int **input_ids;      // [batch_size][seq_len]
    int *seq_lengths;     // [batch_size]
    int batch_size;
    int max_seq_len;
} BatchedInput;

infiniStatus_t qwen_batched_forward(QwenModel *model,
                                   BatchedInput *input,
                                   float **output_logits,  // [batch_size][vocab_size]
                                   void *stream) {
    // Handle variable-length sequences in batch
    // Use padding and attention masks
}
```

## Memory Optimization

### Memory Pool Management
```c
// Pre-allocate memory pools for different sizes
typedef struct {
    void **pools;          // Array of memory pools
    size_t *pool_sizes;    // Size of each pool
    bool *pool_in_use;     // Availability flags
    int num_pools;
    infiniopHandle_t handle;
} MemoryPool;

infiniStatus_t memory_pool_create(MemoryPool **pool_ptr,
                                 infiniopHandle_t handle,
                                 size_t *sizes, int num_sizes) {
    MemoryPool *pool = malloc(sizeof(MemoryPool));
    pool->pools = malloc(num_sizes * sizeof(void*));
    pool->pool_sizes = malloc(num_sizes * sizeof(size_t));
    pool->pool_in_use = calloc(num_sizes, sizeof(bool));
    pool->num_pools = num_sizes;
    pool->handle = handle;
    
    // Pre-allocate all pools
    for (int i = 0; i < num_sizes; i++) {
        infinirtMalloc(&pool->pools[i], sizes[i], handle->context);
        pool->pool_sizes[i] = sizes[i];
    }
    
    *pool_ptr = pool;
    return INFINI_STATUS_SUCCESS;
}

void* memory_pool_get(MemoryPool *pool, size_t size) {
    // Find smallest available pool that fits the request
    for (int i = 0; i < pool->num_pools; i++) {
        if (!pool->pool_in_use[i] && pool->pool_sizes[i] >= size) {
            pool->pool_in_use[i] = true;
            return pool->pools[i];
        }
    }
    return NULL;  // No suitable pool available
}
```

### Gradient Checkpointing (for fine-tuning)
```c
// Store only essential activations, recompute others during backward pass
typedef struct {
    void **checkpoints;    // Stored activations
    int *checkpoint_layers; // Which layers to checkpoint
    int num_checkpoints;
} CheckpointManager;

infiniStatus_t forward_with_checkpointing(QwenModel *model,
                                         CheckpointManager *ckpt_mgr,
                                         /* ... other params ... */) {
    // Forward pass with selective activation storage
}
```

## Performance Tuning

### Operator Fusion
```c
// Fuse multiple operations into single kernels
infiniStatus_t fused_attention_mlp(QwenModel *model, int layer_idx,
                                  void *hidden_states, void *stream) {
    // Combine attention + MLP in single kernel launch
    // Reduces memory bandwidth requirements
}

// Fuse normalization with other operations
infiniStatus_t fused_rmsnorm_linear(infiniopRMSNormDescriptor_t norm_desc,
                                   infiniopGemmDescriptor_t gemm_desc,
                                   /* ... params ... */) {
    // RMSNorm + Linear in single operation
}
```

### Mixed Precision
```c
// Use different precisions for different parts
typedef struct {
    infiniDtype_t weight_dtype;     // FP16 for weights
    infiniDtype_t activation_dtype; // FP16 for activations
    infiniDtype_t output_dtype;     // FP32 for final output
} PrecisionConfig;

infiniStatus_t create_mixed_precision_descriptors(QwenModel *model,
                                                 PrecisionConfig *config) {
    // Create descriptors with appropriate data types
    // Handle automatic casting where needed
}
```

### Asynchronous Execution
```c
// Overlap computation with memory transfers
infiniStatus_t async_layer_forward(QwenModel *model, int layer_idx,
                                  infinirtStream_t compute_stream,
                                  infinirtStream_t copy_stream) {
    // Start next layer's weight transfer while computing current layer
    if (layer_idx + 1 < model->config.num_hidden_layers) {
        // Async copy next layer weights
        infinirtMemcpyAsync(/* next layer weights */, copy_stream);
    }
    
    // Compute current layer
    qwen_decoder_layer_forward(model, layer_idx, /* ... */, compute_stream);
    
    // Synchronize streams as needed
    infinirtStreamSynchronize(compute_stream);
}
```

## Multi-GPU Deployment

### Model Parallelism
```c
// Split model across multiple GPUs
typedef struct {
    int num_gpus;
    QwenModel **models;        // One model per GPU
    int *layer_assignment;     // Which GPU handles which layers
    infiniopHandle_t *handles; // Handles for each GPU
} MultiGPUModel;

infiniStatus_t multi_gpu_forward(MultiGPUModel *mgpu_model,
                                int *input_ids, int batch_size, int seq_len,
                                float *output_logits) {
    // Distribute layers across GPUs
    // Handle inter-GPU communication for activations
    
    for (int gpu = 0; gpu < mgpu_model->num_gpus; gpu++) {
        // Process assigned layers on each GPU
        int start_layer = mgpu_model->layer_assignment[gpu * 2];
        int end_layer = mgpu_model->layer_assignment[gpu * 2 + 1];
        
        for (int layer = start_layer; layer <= end_layer; layer++) {
            qwen_decoder_layer_forward(mgpu_model->models[gpu], layer,
                                      /* ... params ... */);
        }
        
        // Transfer activations to next GPU if needed
        if (gpu + 1 < mgpu_model->num_gpus) {
            // P2P transfer or through host memory
        }
    }
}
```

### Data Parallelism
```c
// Process different batches on different GPUs
infiniStatus_t data_parallel_forward(MultiGPUModel *mgpu_model,
                                    BatchedInput *input,
                                    float **output_logits) {
    int batch_per_gpu = input->batch_size / mgpu_model->num_gpus;
    
    // Distribute batches across GPUs
    for (int gpu = 0; gpu < mgpu_model->num_gpus; gpu++) {
        int start_batch = gpu * batch_per_gpu;
        int end_batch = (gpu + 1) * batch_per_gpu;
        
        // Process batch subset on this GPU
        BatchedInput gpu_input = {
            .input_ids = input->input_ids + start_batch,
            .seq_lengths = input->seq_lengths + start_batch,
            .batch_size = end_batch - start_batch,
            .max_seq_len = input->max_seq_len
        };
        
        qwen_batched_forward(mgpu_model->models[gpu], &gpu_input,
                            output_logits + start_batch,
                            mgpu_model->handles[gpu]->stream);
    }
    
    // Synchronize all GPUs
    for (int gpu = 0; gpu < mgpu_model->num_gpus; gpu++) {
        infinirtStreamSynchronize(mgpu_model->handles[gpu]->stream);
    }
}
```

## Troubleshooting

### Common Issues

#### Memory Allocation Failures
```c
// Check memory requirements before allocation
size_t available_memory, total_memory;
infinirtGetMemInfo(&available_memory, &total_memory, handle->context);

if (required_memory > available_memory) {
    // Try memory optimizations:
    // 1. Reduce batch size
    // 2. Use gradient checkpointing
    // 3. Enable memory pooling
    // 4. Use lower precision
}
```

#### Performance Bottlenecks
```c
// Profile execution times
typedef struct {
    double embedding_time;
    double attention_time;
    double mlp_time;
    double normalization_time;
    double total_time;
} ProfileInfo;

void profile_model_forward(QwenModel *model, ProfileInfo *profile) {
    // Add timing around each component
    auto start = get_time();
    
    // ... embedding operations ...
    profile->embedding_time = get_time() - start;
    
    start = get_time();
    // ... attention operations ...
    profile->attention_time = get_time() - start;
    
    // etc.
}
```

#### Numerical Issues
```c
// Check for NaN/Inf values
bool check_tensor_validity(const void *tensor, size_t num_elements,
                          infiniDtype_t dtype) {
    // Device-side kernel to check for invalid values
    // Return false if any NaN or Inf found
}

// Add validation after critical operations
if (!check_tensor_validity(hidden_states, batch_size * seq_len * hidden_size, INFINI_DTYPE_F32)) {
    return INFINI_STATUS_INTERNAL_ERROR;
}
```

### Debugging Tools
```c
// Tensor dumping for debugging
infiniStatus_t dump_tensor_to_file(const void *tensor, const char *filename,
                                  size_t num_elements, infiniDtype_t dtype,
                                  infiniopHandle_t handle) {
    // Copy tensor to host and save to file
    // Useful for comparing with reference implementations
}

// Layer-by-layer validation
infiniStatus_t validate_layer_output(QwenModel *model, int layer_idx,
                                    const void *expected_output,
                                    const void *actual_output,
                                    float tolerance) {
    // Compare actual vs expected outputs
    // Return status indicating validation result
}
```

## Next Steps

- **[Performance Optimization Guide](performance.md)**: Advanced optimization techniques
- **[API Reference](api/)**: Complete operator documentation
- **[Examples](examples/)**: Complete model implementations
- **[Troubleshooting](troubleshooting.md)**: Common issues and solutions

This guide provides a comprehensive foundation for adapting large language models to InfiniCore. Start with a simple model like GPT-2 to learn the patterns, then apply the same techniques to more complex models like Qwen3, LLaMA, or your custom architectures.