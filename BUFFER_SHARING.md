# Backward Buffer Sharing Feature

## Overview

The `init_attention` function now supports an optional "predecessor" parameter to enable backward buffer sharing between attention layers. This feature allows multiple attention instances to share the same backward pass buffers, reducing memory usage in scenarios where gradients need to be accumulated across multiple layers.

## Function Signature

### CPU Version
```c
Attention* init_attention(int seq_len, int d_model, int batch_size, bool is_causal, Attention* predecessor);
```

### GPU Version  
```c
Attention* init_attention(int seq_len, int d_model, int batch_size, bool is_causal, cublasLtHandle_t cublaslt_handle, Attention* predecessor);
```

## Parameters

- `seq_len`: Sequence length
- `d_model`: Model dimension  
- `batch_size`: Batch size
- `is_causal`: Whether to use causal attention
- `cublaslt_handle`: cuBLAS handle (GPU version only)
- `predecessor`: Optional pointer to an existing Attention instance whose backward buffers should be shared. Pass `NULL` to allocate new buffers.

## Behavior

### When predecessor is NULL:
- New backward buffers are allocated
- The instance owns its backward buffers (`owns_grad_buffers = true`)
- Buffers are freed when `free_attention()` is called

### When predecessor is provided:
- Backward buffers are shared from the predecessor
- The instance does not own backward buffers (`owns_grad_buffers = false`)
- Backward buffers are NOT freed when `free_attention()` is called
- Forward buffers are still allocated separately (not shared)

## Shared Buffers

The following backward pass buffers are shared when a predecessor is provided:

### CPU Version:
- `grad_output`
- `grad_attn_output` 
- `grad_weights`
- `grad_scores`
- `grad_Q`
- `grad_K`
- `grad_V`

### GPU Version:
- `d_grad_output`
- `d_grad_attn_output`
- `d_grad_weights` 
- `d_grad_scores`
- `d_grad_Q`
- `d_grad_K`
- `d_grad_V`

## Example Usage

```c
// Create first attention layer (owns its buffers)
Attention* attn1 = init_attention(seq_len, d_model, batch_size, false, NULL);

// Create second attention layer sharing backward buffers
Attention* attn2 = init_attention(seq_len, d_model, batch_size, false, attn1);

// attn1 and attn2 now share backward buffers
// Forward buffers remain separate

// Free in any order - memory management is handled correctly
free_attention(attn2);  // Backward buffers not freed (not owned)
free_attention(attn1);  // Backward buffers freed (owned)
```

## Memory Management

The implementation uses the `owns_grad_buffers` flag to track ownership:
- Only the original owner frees the shared backward buffers
- No double-free errors can occur
- Instances can be freed in any order

## Backward Compatibility

All existing code continues to work unchanged by passing `NULL` for the predecessor parameter.