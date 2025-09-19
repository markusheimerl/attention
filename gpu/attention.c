#include "attention.h"

// CUDA kernel for reshaping [batch, seq, d_model] -> [batch, num_heads, seq, d_head]
__global__ static void reshape_for_heads_kernel(float* output, float* input, 
                                               int batch_size, int seq_len, int num_heads, int d_head) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;
    int d = threadIdx.x;
    
    if (b >= batch_size || h >= num_heads || s >= seq_len || d >= d_head) return;
    
    int input_idx = b * seq_len * num_heads * d_head + s * num_heads * d_head + h * d_head + d;
    int output_idx = b * num_heads * seq_len * d_head + h * seq_len * d_head + s * d_head + d;
    
    output[output_idx] = input[input_idx];
}

// CUDA kernel for reshaping [batch, num_heads, seq, d_head] -> [batch, seq, d_model]
__global__ static void reshape_from_heads_kernel(float* output, float* input, 
                                                int batch_size, int seq_len, int num_heads, int d_head) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;
    int d = threadIdx.x;
    
    if (b >= batch_size || h >= num_heads || s >= seq_len || d >= d_head) return;
    
    int input_idx = b * num_heads * seq_len * d_head + h * seq_len * d_head + s * d_head + d;
    int output_idx = b * seq_len * num_heads * d_head + s * num_heads * d_head + h * d_head + d;
    
    output[output_idx] = input[input_idx];
}

// Initialize the attention layer
Attention* init_attention(int seq_len, int d_model, int num_heads, int batch_size, bool is_causal, cublasLtHandle_t cublaslt_handle) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->seq_len = seq_len;
    attn->d_model = d_model;
    attn->num_heads = num_heads;
    attn->d_head = d_model / num_heads;
    attn->batch_size = batch_size;
    attn->scale = 1.0f / sqrtf(attn->d_head);
    attn->is_causal = is_causal;
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    // Initialize cuBLASLt
    attn->cublaslt_handle = cublaslt_handle;
    
    int weight_size = d_model * d_model;
    int seq_batch_size = batch_size * seq_len * d_model;
    int head_seq_size = batch_size * num_heads * seq_len * attn->d_head;
    int head_attn_size = batch_size * num_heads * seq_len * seq_len;
    
    // Allocate host memory for weight initialization
    float* h_W_q = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o = (float*)malloc(weight_size * sizeof(float));
    
    // Initialize weights on host
    float scale_W = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < weight_size; i++) {
        h_W_q[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
        h_W_k[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
        h_W_v[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
        h_W_o[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&attn->d_W_q, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o, weight_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&attn->d_W_q_grad, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_grad, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_grad, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_grad, weight_size * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&attn->d_W_q_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_q_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_v, weight_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&attn->d_Q, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_K, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_V, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_Q_reshaped, head_seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_K_reshaped, head_seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_V_reshaped, head_seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_scores, head_attn_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_weights, head_attn_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_output, head_seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_concat_output, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_output, seq_batch_size * sizeof(float)));
    
    // Allocate device memory for backward pass buffers
    CHECK_CUDA(cudaMalloc(&attn->d_grad_output, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_concat_output, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_attn_output, head_seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_weights, head_attn_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_scores, head_attn_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_Q_reshaped, head_seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_K_reshaped, head_seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_V_reshaped, head_seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_Q, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_K, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_V, seq_batch_size * sizeof(float)));
    
    // Allocate single device float for loss computation
    CHECK_CUDA(cudaMalloc(&attn->d_loss_result, sizeof(float)));
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(attn->d_W_q, h_W_q, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k, h_W_k, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v, h_W_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o, h_W_o, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(attn->d_W_q_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_q_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_v, 0, weight_size * sizeof(float)));
    
    // Create cuBLASLt matrix multiplication descriptors
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&attn->matmul_NN_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&attn->matmul_NT_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&attn->matmul_TN_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    
    // Set transpose operations
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(attn->matmul_NT_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(attn->matmul_NT_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    
    transA = CUBLAS_OP_T;
    transB = CUBLAS_OP_N;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(attn->matmul_TN_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(attn->matmul_TN_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    
    // Row-major layout order
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    
    // Create matrix layout descriptors
    // 1. Weight matrices [d_model x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->weight_layout, CUDA_R_32F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // 2. Sequence data [seq_len x d_model] - batched
    int64_t seq_batch_stride = seq_len * d_model;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->seq_batch_layout, CUDA_R_32F, seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_batch_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_batch_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_batch_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &seq_batch_stride, sizeof(seq_batch_stride)));
    
    // 3. Head sequence data [seq_len x d_head] - batched per head
    int total_head_batches = batch_size * num_heads;
    int64_t head_seq_stride = seq_len * attn->d_head;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->head_seq_layout, CUDA_R_32F, seq_len, attn->d_head, attn->d_head));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_seq_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_seq_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &total_head_batches, sizeof(total_head_batches)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_seq_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &head_seq_stride, sizeof(head_seq_stride)));
    
    // 4. Head attention matrices [seq_len x seq_len] - batched per head
    int64_t head_attn_stride = seq_len * seq_len;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->head_attn_layout, CUDA_R_32F, seq_len, seq_len, seq_len));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_attn_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_attn_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &total_head_batches, sizeof(total_head_batches)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_attn_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &head_attn_stride, sizeof(head_attn_stride)));
    
    // 5. Weight matrices [d_model x d_model] - broadcasted
    int64_t zero_stride = 0;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->weight_broadcast_layout, CUDA_R_32F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weight_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weight_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weight_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &zero_stride, sizeof(zero_stride)));
    
    // 6. Sequence data [batch_size*seq_len x d_model] - flattened
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->flattened_seq_layout, CUDA_R_32F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->flattened_seq_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Free host memory
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    
    return attn;
}

// Free attention memory
void free_attention(Attention* attn) {
    // Destroy cuBLASLt descriptors
    cublasLtMatmulDescDestroy(attn->matmul_NN_desc);
    cublasLtMatmulDescDestroy(attn->matmul_NT_desc);
    cublasLtMatmulDescDestroy(attn->matmul_TN_desc);
    
    // Destroy layouts
    cublasLtMatrixLayoutDestroy(attn->weight_layout);
    cublasLtMatrixLayoutDestroy(attn->seq_batch_layout);
    cublasLtMatrixLayoutDestroy(attn->head_seq_layout);
    cublasLtMatrixLayoutDestroy(attn->head_attn_layout);
    cublasLtMatrixLayoutDestroy(attn->weight_broadcast_layout);
    cublasLtMatrixLayoutDestroy(attn->flattened_seq_layout);
    
    // Free device memory
    cudaFree(attn->d_W_q); cudaFree(attn->d_W_k); cudaFree(attn->d_W_v); cudaFree(attn->d_W_o);
    cudaFree(attn->d_W_q_grad); cudaFree(attn->d_W_k_grad); cudaFree(attn->d_W_v_grad); cudaFree(attn->d_W_o_grad);
    cudaFree(attn->d_W_q_m); cudaFree(attn->d_W_q_v); cudaFree(attn->d_W_k_m); cudaFree(attn->d_W_k_v);
    cudaFree(attn->d_W_v_m); cudaFree(attn->d_W_v_v); cudaFree(attn->d_W_o_m); cudaFree(attn->d_W_o_v);
    cudaFree(attn->d_Q); cudaFree(attn->d_K); cudaFree(attn->d_V);
    cudaFree(attn->d_Q_reshaped); cudaFree(attn->d_K_reshaped); cudaFree(attn->d_V_reshaped);
    cudaFree(attn->d_scores); cudaFree(attn->d_attn_weights);
    cudaFree(attn->d_attn_output); cudaFree(attn->d_concat_output); cudaFree(attn->d_output);
    cudaFree(attn->d_grad_output); cudaFree(attn->d_grad_concat_output); cudaFree(attn->d_grad_attn_output);
    cudaFree(attn->d_grad_weights); cudaFree(attn->d_grad_scores);
    cudaFree(attn->d_grad_Q_reshaped); cudaFree(attn->d_grad_K_reshaped); cudaFree(attn->d_grad_V_reshaped);
    cudaFree(attn->d_grad_Q); cudaFree(attn->d_grad_K); cudaFree(attn->d_grad_V);
    
    // Free loss computation buffer
    cudaFree(attn->d_loss_result);
    
    free(attn);
}

// CUDA kernel for softmax forward pass (multi-head)
__global__ static void softmax_forward_kernel_attention(float* weights, float* scores, int batch_size, int num_heads, int seq_len) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    float* scores_bh = &scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len];
    float* weights_bh = &weights[b * num_heads * seq_len * seq_len + h * seq_len * seq_len];
    
    // Find max for numerical stability
    float max_val = -1e30f;
    for (int j = 0; j < seq_len; j++) {
        float val = scores_bh[i * seq_len + j];
        if (val > max_val) max_val = val;
    }
    
    // A_ij = exp(S_ij)/∑_k exp(S_ik)
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        float exp_val = expf(scores_bh[i * seq_len + j] - max_val);
        weights_bh[i * seq_len + j] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize
    for (int j = 0; j < seq_len; j++) {
        weights_bh[i * seq_len + j] /= sum_exp;
    }
}

// CUDA kernel for causal softmax forward pass (multi-head)
__global__ static void softmax_causal_forward_kernel_attention(float* weights, float* scores, int batch_size, int num_heads, int seq_len) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    float* scores_bh = &scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len];
    float* weights_bh = &weights[b * num_heads * seq_len * seq_len + h * seq_len * seq_len];
    
    // Find max for numerical stability (only consider positions <= i)
    float max_val = -1e30f;
    for (int j = 0; j <= i; j++) {
        float val = scores_bh[i * seq_len + j];
        if (val > max_val) max_val = val;
    }
    
    // A_ij = exp(S_ij)/∑_k exp(S_ik) for j <= i, 0 for j > i
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        if (j <= i) {
            float exp_val = expf(scores_bh[i * seq_len + j] - max_val);
            weights_bh[i * seq_len + j] = exp_val;
            sum_exp += exp_val;
        } else {
            weights_bh[i * seq_len + j] = 0.0f;
        }
    }
    
    // Normalize only the valid positions
    for (int j = 0; j <= i; j++) {
        weights_bh[i * seq_len + j] /= sum_exp;
    }
}

// CUDA kernel for softmax backward pass (multi-head)
__global__ static void softmax_backward_kernel_attention(float* grad_scores, float* grad_weights, float* weights, int batch_size, int num_heads, int seq_len) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    float* grad_weights_bh = &grad_weights[b * num_heads * seq_len * seq_len + h * seq_len * seq_len];
    float* weights_bh = &weights[b * num_heads * seq_len * seq_len + h * seq_len * seq_len];
    float* grad_scores_bh = &grad_scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len];
    
    // Compute ∑_j ∂L/∂A⊙A
    float sum_term = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        int idx = i * seq_len + k;
        sum_term += grad_weights_bh[idx] * weights_bh[idx];
    }
    
    // ∂L/∂S = A⊙(∂L/∂A - ∑_j ∂L/∂A⊙A)
    for (int j = 0; j < seq_len; j++) {
        int idx = i * seq_len + j;
        grad_scores_bh[idx] = weights_bh[idx] * (grad_weights_bh[idx] - sum_term);
    }
}

// CUDA kernel for causal softmax backward pass (multi-head)
__global__ static void softmax_causal_backward_kernel_attention(float* grad_scores, float* grad_weights, float* weights, int batch_size, int num_heads, int seq_len) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    float* grad_weights_bh = &grad_weights[b * num_heads * seq_len * seq_len + h * seq_len * seq_len];
    float* weights_bh = &weights[b * num_heads * seq_len * seq_len + h * seq_len * seq_len];
    float* grad_scores_bh = &grad_scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len];
    
    // Compute ∑_j ∂L/∂A⊙A (only for j <= i)
    float sum_term = 0.0f;
    for (int k = 0; k <= i; k++) {
        int idx = i * seq_len + k;
        sum_term += grad_weights_bh[idx] * weights_bh[idx];
    }
    
    // ∂L/∂S = A⊙(∂L/∂A - ∑_j ∂L/∂A⊙A) for j <= i, 0 for j > i
    for (int j = 0; j < seq_len; j++) {
        int idx = i * seq_len + j;
        if (j <= i) {
            grad_scores_bh[idx] = weights_bh[idx] * (grad_weights_bh[idx] - sum_term);
        } else {
            grad_scores_bh[idx] = 0.0f;
        }
    }
}

// Forward pass
void forward_pass_attention(Attention* attn, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 1: Compute Q, K, V using strided batched operations with broadcasted weights
    // Q = XW_q, K = XW_k, V = XW_v
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NN_desc,
                                  &alpha,
                                  d_X, attn->seq_batch_layout,
                                  attn->d_W_q, attn->weight_broadcast_layout,
                                  &beta,
                                  attn->d_Q, attn->seq_batch_layout,
                                  attn->d_Q, attn->seq_batch_layout,
                                  NULL, NULL, 0, 0));
    
    // K = XW_k
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NN_desc,
                                  &alpha,
                                  d_X, attn->seq_batch_layout,
                                  attn->d_W_k, attn->weight_broadcast_layout,
                                  &beta,
                                  attn->d_K, attn->seq_batch_layout,
                                  attn->d_K, attn->seq_batch_layout,
                                  NULL, NULL, 0, 0));
    
    // V = XW_v
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NN_desc,
                                  &alpha,
                                  d_X, attn->seq_batch_layout,
                                  attn->d_W_v, attn->weight_broadcast_layout,
                                  &beta,
                                  attn->d_V, attn->seq_batch_layout,
                                  attn->d_V, attn->seq_batch_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 2: Reshape Q, K, V for multi-head attention
    // [batch, seq, d_model] -> [batch, num_heads, seq, d_head]
    dim3 reshape_grid(attn->batch_size, attn->num_heads, attn->seq_len);
    int reshape_threads = attn->d_head;
    
    reshape_for_heads_kernel<<<reshape_grid, reshape_threads>>>(
        attn->d_Q_reshaped, attn->d_Q, attn->batch_size, attn->seq_len, attn->num_heads, attn->d_head);
    reshape_for_heads_kernel<<<reshape_grid, reshape_threads>>>(
        attn->d_K_reshaped, attn->d_K, attn->batch_size, attn->seq_len, attn->num_heads, attn->d_head);
    reshape_for_heads_kernel<<<reshape_grid, reshape_threads>>>(
        attn->d_V_reshaped, attn->d_V, attn->batch_size, attn->seq_len, attn->num_heads, attn->d_head);
    
    // Step 3: Compute attention scores for all heads
    // S = QKᵀ/√d_head (batched across all heads)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NT_desc,
                                  &attn->scale,
                                  attn->d_Q_reshaped, attn->head_seq_layout,
                                  attn->d_K_reshaped, attn->head_seq_layout,
                                  &beta,
                                  attn->d_scores, attn->head_attn_layout,
                                  attn->d_scores, attn->head_attn_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 4: Apply softmax
    dim3 softmax_grid(attn->batch_size, attn->num_heads, attn->seq_len);
    if (attn->is_causal) {
        softmax_causal_forward_kernel_attention<<<softmax_grid, 1>>>(
            attn->d_attn_weights, attn->d_scores, attn->batch_size, attn->num_heads, attn->seq_len);
    } else {
        softmax_forward_kernel_attention<<<softmax_grid, 1>>>(
            attn->d_attn_weights, attn->d_scores, attn->batch_size, attn->num_heads, attn->seq_len);
    }
    
    // Step 5: Compute attention output for all heads
    // Z = AV (batched across all heads)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NN_desc,
                                  &alpha,
                                  attn->d_attn_weights, attn->head_attn_layout,
                                  attn->d_V_reshaped, attn->head_seq_layout,
                                  &beta,
                                  attn->d_attn_output, attn->head_seq_layout,
                                  attn->d_attn_output, attn->head_seq_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 6: Concatenate heads
    // [batch, num_heads, seq, d_head] -> [batch, seq, d_model]
    reshape_from_heads_kernel<<<reshape_grid, reshape_threads>>>(
        attn->d_concat_output, attn->d_attn_output, attn->batch_size, attn->seq_len, attn->num_heads, attn->d_head);
    
    // Step 7: Apply output projection
    // Y = ConcatW_o
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NN_desc,
                                  &alpha,
                                  attn->d_concat_output, attn->seq_batch_layout,
                                  attn->d_W_o, attn->weight_broadcast_layout,
                                  &beta,
                                  attn->d_output, attn->seq_batch_layout,
                                  attn->d_output, attn->seq_batch_layout,
                                  NULL, NULL, 0, 0));
}

// CUDA kernel for computing loss and gradient
__global__ static void compute_loss_and_gradient_kernel_attention(float* grad_output, float* predictions, float* targets, float* loss_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_output[idx] = predictions[idx] - targets[idx];
        atomicAdd(loss_result, grad_output[idx] * grad_output[idx]);
    }
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* d_y) {
    // ∂L/∂Y = Y - Y_true
    int total_elements = attn->batch_size * attn->seq_len * attn->d_model;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Reset loss accumulator to zero
    CHECK_CUDA(cudaMemset(attn->d_loss_result, 0, sizeof(float)));
    
    // Compute gradient and accumulate loss
    compute_loss_and_gradient_kernel_attention<<<num_blocks, block_size>>>(
        attn->d_grad_output, attn->d_output, d_y, attn->d_loss_result, total_elements
    );
    
    // Copy result back to host
    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, attn->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / total_elements;
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    int weight_size = attn->d_model * attn->d_model;
    
    CHECK_CUDA(cudaMemset(attn->d_W_q_grad, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_grad, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_grad, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_grad, 0, weight_size * sizeof(float)));
}

// Backward pass
void backward_pass_attention(Attention* attn, float* d_X, float* d_grad_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 7 (backward): Gradient through output projection
    // ∂L/∂W_o = Concat^T(∂L/∂Y)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &alpha,
                                  attn->d_concat_output, attn->flattened_seq_layout,
                                  attn->d_grad_output, attn->flattened_seq_layout,
                                  &beta,
                                  attn->d_W_o_grad, attn->weight_layout,
                                  attn->d_W_o_grad, attn->weight_layout,
                                  NULL, NULL, 0, 0));
    
    // ∂L/∂Concat = (∂L/∂Y)W_o^T
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NT_desc,
                                  &alpha,
                                  attn->d_grad_output, attn->seq_batch_layout,
                                  attn->d_W_o, attn->weight_broadcast_layout,
                                  &beta,
                                  attn->d_grad_concat_output, attn->seq_batch_layout,
                                  attn->d_grad_concat_output, attn->seq_batch_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 6 (backward): Gradient through head concatenation
    // [batch, seq, d_model] -> [batch, num_heads, seq, d_head]
    dim3 reshape_grid(attn->batch_size, attn->num_heads, attn->seq_len);
    int reshape_threads = attn->d_head;
    
    reshape_for_heads_kernel<<<reshape_grid, reshape_threads>>>(
        attn->d_grad_attn_output, attn->d_grad_concat_output, attn->batch_size, attn->seq_len, attn->num_heads, attn->d_head);
    
    // Step 5 (backward): Gradient through attention output computation
    // ∂L/∂A = (∂L/∂Z)V^T (batched across all heads)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NT_desc,
                                  &alpha,
                                  attn->d_grad_attn_output, attn->head_seq_layout,
                                  attn->d_V_reshaped, attn->head_seq_layout,
                                  &beta,
                                  attn->d_grad_weights, attn->head_attn_layout,
                                  attn->d_grad_weights, attn->head_attn_layout,
                                  NULL, NULL, 0, 0));
    
    // ∂L/∂V = A^T(∂L/∂Z) (batched across all heads)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &alpha,
                                  attn->d_attn_weights, attn->head_attn_layout,
                                  attn->d_grad_attn_output, attn->head_seq_layout,
                                  &beta,
                                  attn->d_grad_V_reshaped, attn->head_seq_layout,
                                  attn->d_grad_V_reshaped, attn->head_seq_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 4 (backward): Gradient through softmax
    dim3 softmax_grid(attn->batch_size, attn->num_heads, attn->seq_len);
    if (attn->is_causal) {
        softmax_causal_backward_kernel_attention<<<softmax_grid, 1>>>(
            attn->d_grad_scores, attn->d_grad_weights, attn->d_attn_weights, attn->batch_size, attn->num_heads, attn->seq_len);
    } else {
        softmax_backward_kernel_attention<<<softmax_grid, 1>>>(
            attn->d_grad_scores, attn->d_grad_weights, attn->d_attn_weights, attn->batch_size, attn->num_heads, attn->seq_len);
    }
    
    // Step 3 (backward): Gradient through attention scores
    // ∂L/∂Q = (∂L/∂S)(K/√d_head) (batched across all heads)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NN_desc,
                                  &attn->scale,
                                  attn->d_grad_scores, attn->head_attn_layout,
                                  attn->d_K_reshaped, attn->head_seq_layout,
                                  &beta,
                                  attn->d_grad_Q_reshaped, attn->head_seq_layout,
                                  attn->d_grad_Q_reshaped, attn->head_seq_layout,
                                  NULL, NULL, 0, 0));
    
    // ∂L/∂K = (∂L/∂S)^T(Q/√d_head) (batched across all heads)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &attn->scale,
                                  attn->d_grad_scores, attn->head_attn_layout,
                                  attn->d_Q_reshaped, attn->head_seq_layout,
                                  &beta,
                                  attn->d_grad_K_reshaped, attn->head_seq_layout,
                                  attn->d_grad_K_reshaped, attn->head_seq_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 2 (backward): Reshape gradients back
    // [batch, num_heads, seq, d_head] -> [batch, seq, d_model]
    reshape_from_heads_kernel<<<reshape_grid, reshape_threads>>>(
        attn->d_grad_Q, attn->d_grad_Q_reshaped, attn->batch_size, attn->seq_len, attn->num_heads, attn->d_head);
    reshape_from_heads_kernel<<<reshape_grid, reshape_threads>>>(
        attn->d_grad_K, attn->d_grad_K_reshaped, attn->batch_size, attn->seq_len, attn->num_heads, attn->d_head);
    reshape_from_heads_kernel<<<reshape_grid, reshape_threads>>>(
        attn->d_grad_V, attn->d_grad_V_reshaped, attn->batch_size, attn->seq_len, attn->num_heads, attn->d_head);
    
    // Step 1 (backward): Gradient through linear projections
    // ∂L/∂W_q = X^T(∂L/∂Q)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &alpha,
                                  d_X, attn->flattened_seq_layout,
                                  attn->d_grad_Q, attn->flattened_seq_layout,
                                  &beta,
                                  attn->d_W_q_grad, attn->weight_layout,
                                  attn->d_W_q_grad, attn->weight_layout,
                                  NULL, NULL, 0, 0));
    
    // ∂L/∂W_k = X^T(∂L/∂K)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &alpha,
                                  d_X, attn->flattened_seq_layout,
                                  attn->d_grad_K, attn->flattened_seq_layout,
                                  &beta,
                                  attn->d_W_k_grad, attn->weight_layout,
                                  attn->d_W_k_grad, attn->weight_layout,
                                  NULL, NULL, 0, 0));
    
    // ∂L/∂W_v = X^T(∂L/∂V)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &alpha,
                                  d_X, attn->flattened_seq_layout,
                                  attn->d_grad_V, attn->flattened_seq_layout,
                                  &beta,
                                  attn->d_W_v_grad, attn->weight_layout,
                                  attn->d_W_v_grad, attn->weight_layout,
                                  NULL, NULL, 0, 0));
    
    // ∂L/∂X = (∂L/∂Q)W_q^T + (∂L/∂K)W_k^T + (∂L/∂V)W_v^T
    if (d_grad_X != NULL) {
        // ∂L/∂X = (∂L/∂Q)W_q^T
        CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                      attn->matmul_NT_desc,
                                      &alpha,
                                      attn->d_grad_Q, attn->seq_batch_layout,
                                      attn->d_W_q, attn->weight_broadcast_layout,
                                      &beta,
                                      d_grad_X, attn->seq_batch_layout,
                                      d_grad_X, attn->seq_batch_layout,
                                      NULL, NULL, 0, 0));
        
        // ∂L/∂X += (∂L/∂K)W_k^T
        CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                      attn->matmul_NT_desc,
                                      &alpha,
                                      attn->d_grad_K, attn->seq_batch_layout,
                                      attn->d_W_k, attn->weight_broadcast_layout,
                                      &alpha,
                                      d_grad_X, attn->seq_batch_layout,
                                      d_grad_X, attn->seq_batch_layout,
                                      NULL, NULL, 0, 0));
        
        // ∂L/∂X += (∂L/∂V)W_v^T
        CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                      attn->matmul_NT_desc,
                                      &alpha,
                                      attn->d_grad_V, attn->seq_batch_layout,
                                      attn->d_W_v, attn->weight_broadcast_layout,
                                      &alpha,
                                      d_grad_X, attn->seq_batch_layout,
                                      d_grad_X, attn->seq_batch_layout,
                                      NULL, NULL, 0, 0));
    }
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel_attention(float* weight, float* grad, float* m, float* v,
                                             float beta1, float beta2, float epsilon, float learning_rate,
                                             float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int weight_size = attn->d_model * attn->d_model;
    int block_size = 256;
    int num_blocks = (weight_size + block_size - 1) / block_size;
    
    // Update all weight matrices (W_q, W_k, W_v, W_o)
    float* weights[] = {attn->d_W_q, attn->d_W_k, attn->d_W_v, attn->d_W_o};
    float* grads[] = {attn->d_W_q_grad, attn->d_W_k_grad, attn->d_W_v_grad, attn->d_W_o_grad};
    float* m_arrays[] = {attn->d_W_q_m, attn->d_W_k_m, attn->d_W_v_m, attn->d_W_o_m};
    float* v_arrays[] = {attn->d_W_q_v, attn->d_W_k_v, attn->d_W_v_v, attn->d_W_o_v};
    
    for (int w = 0; w < 4; w++) {
        adamw_update_kernel_attention<<<num_blocks, block_size>>>(
            weights[w], grads[w], m_arrays[w], v_arrays[w],
            attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
            alpha_t, weight_size, attn->batch_size
        );
    }
}

// Save attention weights to binary file
void save_attention(Attention* attn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&attn->seq_len, sizeof(int), 1, file);
    fwrite(&attn->d_model, sizeof(int), 1, file);
    fwrite(&attn->num_heads, sizeof(int), 1, file);
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    fwrite(&attn->is_causal, sizeof(bool), 1, file);
    
    int weight_size = attn->d_model * attn->d_model;
    
    // Allocate temporary host memory for weights
    float* h_W_q = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o = (float*)malloc(weight_size * sizeof(float));
    
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(h_W_q, attn->d_W_q, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k, attn->d_W_k, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v, attn->d_W_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o, attn->d_W_o, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_W_q, sizeof(float), weight_size, file);
    fwrite(h_W_k, sizeof(float), weight_size, file);
    fwrite(h_W_v, sizeof(float), weight_size, file);
    fwrite(h_W_o, sizeof(float), weight_size, file);
    
    // Save Adam state
    fwrite(&attn->t, sizeof(int), 1, file);
    
    float* h_W_q_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_q_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_v = (float*)malloc(weight_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_W_q_m, attn->d_W_q_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_q_v, attn->d_W_q_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k_m, attn->d_W_k_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k_v, attn->d_W_k_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v_m, attn->d_W_v_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v_v, attn->d_W_v_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o_m, attn->d_W_o_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o_v, attn->d_W_o_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_W_q_m, sizeof(float), weight_size, file);
    fwrite(h_W_q_v, sizeof(float), weight_size, file);
    fwrite(h_W_k_m, sizeof(float), weight_size, file);
    fwrite(h_W_k_v, sizeof(float), weight_size, file);
    fwrite(h_W_v_m, sizeof(float), weight_size, file);
    fwrite(h_W_v_v, sizeof(float), weight_size, file);
    fwrite(h_W_o_m, sizeof(float), weight_size, file);
    fwrite(h_W_o_v, sizeof(float), weight_size, file);
    
    // Free temporary host memory
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    free(h_W_q_m); free(h_W_q_v); free(h_W_k_m); free(h_W_k_v);
    free(h_W_v_m); free(h_W_v_v); free(h_W_o_m); free(h_W_o_v);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load attention weights from binary file
Attention* load_attention(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int seq_len, d_model, num_heads, stored_batch_size;
    bool is_causal;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&num_heads, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    Attention* attn = init_attention(seq_len, d_model, num_heads, batch_size, is_causal, cublaslt_handle);
    
    int weight_size = d_model * d_model;
    
    // Load weights
    float* h_W_q = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o = (float*)malloc(weight_size * sizeof(float));
    
    fread(h_W_q, sizeof(float), weight_size, file);
    fread(h_W_k, sizeof(float), weight_size, file);
    fread(h_W_v, sizeof(float), weight_size, file);
    fread(h_W_o, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(attn->d_W_q, h_W_q, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k, h_W_k, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v, h_W_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o, h_W_o, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    fread(&attn->t, sizeof(int), 1, file);
    
    float* h_W_q_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_q_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_v = (float*)malloc(weight_size * sizeof(float));
    
    fread(h_W_q_m, sizeof(float), weight_size, file);
    fread(h_W_q_v, sizeof(float), weight_size, file);
    fread(h_W_k_m, sizeof(float), weight_size, file);
    fread(h_W_k_v, sizeof(float), weight_size, file);
    fread(h_W_v_m, sizeof(float), weight_size, file);
    fread(h_W_v_v, sizeof(float), weight_size, file);
    fread(h_W_o_m, sizeof(float), weight_size, file);
    fread(h_W_o_v, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(attn->d_W_q_m, h_W_q_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_q_v, h_W_q_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k_m, h_W_k_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k_v, h_W_k_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v_m, h_W_v_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v_v, h_W_v_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o_m, h_W_o_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o_v, h_W_o_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    free(h_W_q_m); free(h_W_q_v); free(h_W_k_m); free(h_W_k_v);
    free(h_W_v_m); free(h_W_v_v); free(h_W_o_m); free(h_W_o_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return attn;
}