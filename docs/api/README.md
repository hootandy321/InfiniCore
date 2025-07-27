# API Reference

This section provides detailed API documentation for InfiniCore libraries.

## Table of Contents

1. [Core APIs](#core-apis)
2. [InfiniRT (Runtime)](#infinirt-runtime)
3. [InfiniOP (Operators)](#infiniop-operators)
4. [InfiniCCL (Communication)](#infiniccl-communication)
5. [Data Types](#data-types)
6. [Error Codes](#error-codes)

## Core APIs

### Common Types and Constants

#### Status Codes
```c
typedef enum {
    INFINI_STATUS_SUCCESS = 0,                      // Operation completed successfully
    INFINI_STATUS_INTERNAL_ERROR = 1,               // Internal implementation error
    INFINI_STATUS_NOT_IMPLEMENTED = 2,              // Feature not implemented
    INFINI_STATUS_BAD_PARAM = 3,                    // Invalid parameter
    INFINI_STATUS_NULL_POINTER = 4,                 // Null pointer error
    INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED = 5,    // Device type not supported
    INFINI_STATUS_DEVICE_NOT_FOUND = 6,             // Device not found
    INFINI_STATUS_DEVICE_NOT_INITIALIZED = 7,       // Device not initialized
    INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED = 8, // Architecture not supported
    INFINI_STATUS_BAD_TENSOR_DTYPE = 10,            // Invalid tensor data type
    INFINI_STATUS_BAD_TENSOR_SHAPE = 11,            // Invalid tensor shape
    INFINI_STATUS_BAD_TENSOR_STRIDES = 12,          // Invalid tensor strides
    INFINI_STATUS_INSUFFICIENT_WORKSPACE = 13,      // Insufficient workspace memory
} infiniStatus_t;
```

#### Device Types
```c
typedef enum {
    INFINI_DEVICE_CPU = 0,        // CPU device
    INFINI_DEVICE_NVIDIA = 1,     // NVIDIA GPU
    INFINI_DEVICE_CAMBRICON = 2,  // Cambricon MLU
    INFINI_DEVICE_ASCEND = 3,     // Huawei Ascend NPU
    INFINI_DEVICE_METAX = 4,      // MetaX GPU
    INFINI_DEVICE_MOORE = 5,      // Moore Threads GPU
    INFINI_DEVICE_ILUVATAR = 6,   // Iluvatar GPU
    INFINI_DEVICE_KUNLUN = 7,     // Kunlun XPU
    INFINI_DEVICE_SUGON = 8,      // Sugon DCU
    INFINI_DEVICE_TYPE_COUNT
} infiniDevice_t;
```

#### Data Types
```c
typedef enum {
    INFINI_DTYPE_INVALID = 0,  // Invalid type
    INFINI_DTYPE_BYTE = 1,     // Byte (8-bit)
    INFINI_DTYPE_BOOL = 2,     // Boolean
    INFINI_DTYPE_I8 = 3,       // Signed 8-bit integer
    INFINI_DTYPE_I16 = 4,      // Signed 16-bit integer
    INFINI_DTYPE_I32 = 5,      // Signed 32-bit integer
    INFINI_DTYPE_I64 = 6,      // Signed 64-bit integer
    INFINI_DTYPE_U8 = 7,       // Unsigned 8-bit integer
    INFINI_DTYPE_U16 = 8,      // Unsigned 16-bit integer
    INFINI_DTYPE_U32 = 9,      // Unsigned 32-bit integer
    INFINI_DTYPE_U64 = 10,     // Unsigned 64-bit integer
    INFINI_DTYPE_F8 = 11,      // 8-bit float
    INFINI_DTYPE_F16 = 12,     // 16-bit float (half precision)
    INFINI_DTYPE_F32 = 13,     // 32-bit float (single precision)
    INFINI_DTYPE_F64 = 14,     // 64-bit float (double precision)
    INFINI_DTYPE_C16 = 15,     // 16-bit complex
    INFINI_DTYPE_C32 = 16,     // 32-bit complex
    INFINI_DTYPE_C64 = 17,     // 64-bit complex
    INFINI_DTYPE_C128 = 18,    // 128-bit complex
    INFINI_DTYPE_BF16 = 19,    // 16-bit bfloat
} infiniDtype_t;
```

## InfiniRT (Runtime)

### Handle Management

#### infinirtCreateHandle
Creates a runtime handle for device operations.

```c
infiniStatus_t infinirtCreateHandle(infinirtHandle_t *handle,
                                   infiniDevice_t device,
                                   int device_id);
```

**Parameters:**
- `handle` - Pointer to store the created handle
- `device` - Device type (CPU, NVIDIA, etc.)
- `device_id` - Device ID (0 for first device)

**Returns:** Status code indicating success or failure

**Example:**
```c
infinirtHandle_t handle;
infiniStatus_t status = infinirtCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);
if (status != INFINI_STATUS_SUCCESS) {
    // Handle error
}
```

#### infinirtDestroyHandle
Destroys a runtime handle and releases resources.

```c
infiniStatus_t infinirtDestroyHandle(infinirtHandle_t handle);
```

### Memory Management

#### infinirtMalloc
Allocates device memory.

```c
infiniStatus_t infinirtMalloc(void **ptr, size_t size, infinirtContext_t ctx);
```

**Parameters:**
- `ptr` - Pointer to store allocated memory address
- `size` - Size in bytes to allocate
- `ctx` - Device context

#### infinirtFree
Frees device memory.

```c
infiniStatus_t infinirtFree(void *ptr, infinirtContext_t ctx);
```

#### infinirtMemcpy
Copies memory between host and device.

```c
infiniStatus_t infinirtMemcpy(void *dst, const void *src, size_t size,
                             infinirtMemcpyKind_t kind, infinirtStream_t stream);
```

**Memory Copy Types:**
```c
typedef enum {
    INFINIRT_MEMCPY_HOST_TO_HOST,
    INFINIRT_MEMCPY_HOST_TO_DEVICE,
    INFINIRT_MEMCPY_DEVICE_TO_HOST,
    INFINIRT_MEMCPY_DEVICE_TO_DEVICE
} infinirtMemcpyKind_t;
```

### Stream Management

#### infinirtStreamCreate
Creates an execution stream.

```c
infiniStatus_t infinirtStreamCreate(infinirtStream_t *stream, infinirtContext_t ctx);
```

#### infinirtStreamSynchronize
Synchronizes a stream (waits for completion).

```c
infiniStatus_t infinirtStreamSynchronize(infinirtStream_t stream);
```

## InfiniOP (Operators)

### Handle Management

#### infiniopCreateHandle
Creates an operator handle.

```c
infiniStatus_t infiniopCreateHandle(infiniopHandle_t *handle,
                                   infiniDevice_t device,
                                   int device_id);
```

### Tensor Descriptors

#### infiniopCreateTensorDescriptor
Creates a tensor descriptor.

```c
infiniStatus_t infiniopCreateTensorDescriptor(infiniopTensorDescriptor_t *desc,
                                             infiniDtype_t dtype,
                                             int ndim,
                                             const int *shape,
                                             const int *stride);
```

**Parameters:**
- `desc` - Pointer to store tensor descriptor
- `dtype` - Data type
- `ndim` - Number of dimensions
- `shape` - Array of dimension sizes
- `stride` - Array of strides (NULL for contiguous)

### Basic Operations

#### Addition (infiniopAdd)

##### infiniopCreateAddDescriptor
```c
infiniStatus_t infiniopCreateAddDescriptor(infiniopHandle_t handle,
                                          infiniopAddDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t c_desc,
                                          infiniopTensorDescriptor_t a_desc,
                                          infiniopTensorDescriptor_t b_desc);
```

Creates descriptor for element-wise addition: `c = a + b`

##### infiniopAdd
```c
infiniStatus_t infiniopAdd(infiniopAddDescriptor_t desc,
                          void *workspace, size_t workspace_size,
                          void *c, const void *a, const void *b,
                          void *stream);
```

Performs element-wise addition operation.

#### Matrix Multiplication (infiniopGemm)

##### infiniopCreateGemmDescriptor
```c
infiniStatus_t infiniopCreateGemmDescriptor(infiniopHandle_t handle,
                                           infiniopGemmDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t c_desc,
                                           infiniopTensorDescriptor_t a_desc,
                                           infiniopTensorDescriptor_t b_desc);
```

##### infiniopGemm
```c
infiniStatus_t infiniopGemm(infiniopGemmDescriptor_t desc,
                           void *workspace, size_t workspace_size,
                           void *c, const void *a, const void *b,
                           float alpha, float beta, void *stream);
```

Performs general matrix multiplication: `c = alpha * a * b + beta * c`

### Advanced Operations

#### Multi-Head Attention

##### infiniopCreateAttentionDescriptor
```c
infiniStatus_t infiniopCreateAttentionDescriptor(infiniopHandle_t handle,
                                                infiniopAttentionDescriptor_t *desc_ptr,
                                                infiniopTensorDescriptor_t out_desc,
                                                infiniopTensorDescriptor_t q_desc,
                                                infiniopTensorDescriptor_t k_desc,
                                                infiniopTensorDescriptor_t v_desc,
                                                infiniopTensorDescriptor_t k_cache_desc,
                                                infiniopTensorDescriptor_t v_cache_desc,
                                                size_t pos);
```

##### infiniopAttention
```c
infiniStatus_t infiniopAttention(infiniopAttentionDescriptor_t desc,
                                void *workspace, size_t workspace_size,
                                void *out, const void *q, const void *k, const void *v,
                                void *k_cache, void *v_cache, void *stream);
```

#### RMS Normalization

##### infiniopCreateRMSNormDescriptor
```c
infiniStatus_t infiniopCreateRMSNormDescriptor(infiniopHandle_t handle,
                                              infiniopRMSNormDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t y_desc,
                                              infiniopTensorDescriptor_t x_desc,
                                              infiniopTensorDescriptor_t w_desc,
                                              float epsilon);
```

##### infiniopRMSNorm
```c
infiniStatus_t infiniopRMSNorm(infiniopRMSNormDescriptor_t desc,
                              void *workspace, size_t workspace_size,
                              void *y, const void *x, const void *w,
                              void *stream);
```

#### Rotary Position Embedding (RoPE)

##### infiniopCreateRoPEDescriptor
```c
infiniStatus_t infiniopCreateRoPEDescriptor(infiniopHandle_t handle,
                                           infiniopRoPEDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t x,
                                           infiniopTensorDescriptor_t pos_ids,
                                           infiniopTensorDescriptor_t sin_table,
                                           infiniopTensorDescriptor_t cos_table);
```

##### infiniopRoPE
```c
infiniStatus_t infiniopRoPE(infiniopRoPEDescriptor_t desc,
                           void *workspace, size_t workspace_size,
                           void *y, const void *x, void const *pos_ids,
                           void const *sin_table, void const *cos_table,
                           void *stream);
```

### Common Patterns

#### Workspace Size Query
All operators follow this pattern for workspace management:

```c
infiniStatus_t infiniopGet[Operation]WorkspaceSize([Operation]Descriptor_t desc,
                                                   size_t *size);
```

#### Descriptor Cleanup
All descriptors should be destroyed after use:

```c
infiniStatus_t infiniopDestroy[Operation]Descriptor([Operation]Descriptor_t desc);
```

## InfiniCCL (Communication)

### Collective Operations

#### All-Reduce
```c
infiniStatus_t infinicclAllReduce(const void *sendbuf, void *recvbuf,
                                 size_t count, infinicclDataType_t datatype,
                                 infinicclRedOp_t op, infinicclComm_t comm,
                                 infinicclStream_t stream);
```

#### All-Gather
```c
infiniStatus_t infinicclAllGather(const void *sendbuf, void *recvbuf,
                                 size_t sendcount, infinicclDataType_t datatype,
                                 infinicclComm_t comm, infinicclStream_t stream);
```

#### Broadcast
```c
infiniStatus_t infinicclBroadcast(void *buffer, size_t count,
                                 infinicclDataType_t datatype, int root,
                                 infinicclComm_t comm, infinicclStream_t stream);
```

## Data Types

### Size Information
```c
size_t get_dtype_size(infiniDtype_t dtype) {
    switch (dtype) {
        case INFINI_DTYPE_I8:
        case INFINI_DTYPE_U8:
        case INFINI_DTYPE_BYTE:
        case INFINI_DTYPE_BOOL:
            return 1;
        case INFINI_DTYPE_I16:
        case INFINI_DTYPE_U16:
        case INFINI_DTYPE_F16:
        case INFINI_DTYPE_BF16:
            return 2;
        case INFINI_DTYPE_I32:
        case INFINI_DTYPE_U32:
        case INFINI_DTYPE_F32:
            return 4;
        case INFINI_DTYPE_I64:
        case INFINI_DTYPE_U64:
        case INFINI_DTYPE_F64:
            return 8;
        default:
            return 0;
    }
}
```

## Error Codes

### Status Check Macro
```c
#define CHECK_INFINI_STATUS(status) \
    do { \
        if ((status) != INFINI_STATUS_SUCCESS) { \
            fprintf(stderr, "InfiniCore error at %s:%d: %d\n", \
                    __FILE__, __LINE__, (status)); \
            return (status); \
        } \
    } while(0)
```

### Error Handling Example
```c
infiniStatus_t safe_operation_example() {
    infiniopHandle_t handle;
    infiniopTensorDescriptor_t desc;
    
    // Create handle with error checking
    infiniStatus_t status = infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);
    CHECK_INFINI_STATUS(status);
    
    // Create tensor descriptor
    int shape[] = {2, 3, 4};
    status = infiniopCreateTensorDescriptor(&desc, INFINI_DTYPE_F32, 3, shape, NULL);
    CHECK_INFINI_STATUS(status);
    
    // Cleanup
    infiniopDestroyTensorDescriptor(desc);
    infiniopDestroyHandle(handle);
    
    return INFINI_STATUS_SUCCESS;
}
```

This API reference provides the essential interface documentation for InfiniCore. For complete operator-specific details, see the individual operator documentation files and header files in the `include/` directory.