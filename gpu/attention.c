#include "attention.h"

// Initialize the attention layer
Attention* init_attention(int seq_len, int d_model, int batch_size, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->seq_len = seq_len;
    attn->d_model = d_model;
    attn->batch_size = batch_size;
    attn->scale = 1.0f / sqrtf(d_model);
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    // Initialize cuBLAS and cuBLASLt
    attn->cublas_handle = cublas_handle;
    attn->cublaslt_handle = cublaslt_handle;
    
    int weight_size = d_model * d_model;
    int seq_batch_size = batch_size * seq_len * d_model;
    int attn_matrix_size = batch_size * seq_len * seq_len;
    
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
    CHECK_CUDA(cudaMalloc(&attn->d_scores, attn_matrix_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_weights, attn_matrix_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_output, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_output, seq_batch_size * sizeof(float)));
    
    // Allocate device memory for backward pass buffers
    CHECK_CUDA(cudaMalloc(&attn->d_grad_output, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_attn_output, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_weights, attn_matrix_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_scores, attn_matrix_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_Q, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_K, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_V, seq_batch_size * sizeof(float)));
    
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
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&attn->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
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
    
    // Create matrix layout descriptors for single batch operations (used for weight gradients)
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->W_layout, CUDA_R_32F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->seq_layout, CUDA_R_32F, seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->W_grad_layout, CUDA_R_32F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_grad_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Create batched matrix layout descriptors
    int64_t batch_stride_seq = seq_len * d_model;
    int64_t batch_stride_attn = seq_len * seq_len;
    int64_t zero_stride = 0;  // For broadcasting weights
    
    // Input/output sequence layouts (batched)
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->Q_layout, CUDA_R_32F, seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->Q_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->Q_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->Q_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_seq, sizeof(batch_stride_seq)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->K_layout, CUDA_R_32F, seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->K_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->K_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->K_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_seq, sizeof(batch_stride_seq)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->V_layout, CUDA_R_32F, seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->V_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->V_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->V_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_seq, sizeof(batch_stride_seq)));
    
    // Attention matrix layouts (batched)
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->scores_layout, CUDA_R_32F, seq_len, seq_len, seq_len));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->scores_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->scores_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->scores_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_attn, sizeof(batch_stride_attn)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->weights_layout, CUDA_R_32F, seq_len, seq_len, seq_len));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weights_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weights_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weights_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_attn, sizeof(batch_stride_attn)));
        
    // Weight matrix layouts for broadcasting (batch_count=batch_size, stride=0)
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->W_q_broadcast_layout, CUDA_R_32F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_q_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_q_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_q_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &zero_stride, sizeof(zero_stride)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->W_k_broadcast_layout, CUDA_R_32F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_k_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_k_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_k_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &zero_stride, sizeof(zero_stride)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->W_v_broadcast_layout, CUDA_R_32F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_v_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_v_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_v_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &zero_stride, sizeof(zero_stride)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->W_o_broadcast_layout, CUDA_R_32F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_o_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_o_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->W_o_broadcast_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &zero_stride, sizeof(zero_stride)));
    
    // Special layouts for weight gradient computation (input batched, output single)
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->X_for_wgrad_layout, CUDA_R_32F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->X_for_wgrad_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->grad_QKV_for_wgrad_layout, CUDA_R_32F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->grad_QKV_for_wgrad_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->attn_out_for_wgrad_layout, CUDA_R_32F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->attn_out_for_wgrad_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->grad_out_for_wgrad_layout, CUDA_R_32F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->grad_out_for_wgrad_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Free host memory
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    
    return attn;
}

// Free attention memory
void free_attention(Attention* attn) {
    // Destroy cuBLASLt descriptors
    cublasLtMatmulDescDestroy(attn->matmul_desc);
    cublasLtMatmulDescDestroy(attn->matmul_NT_desc);
    cublasLtMatmulDescDestroy(attn->matmul_TN_desc);
    
    // Destroy layouts
    cublasLtMatrixLayoutDestroy(attn->W_layout);
    cublasLtMatrixLayoutDestroy(attn->seq_layout);
    cublasLtMatrixLayoutDestroy(attn->W_grad_layout);
    cublasLtMatrixLayoutDestroy(attn->Q_layout);
    cublasLtMatrixLayoutDestroy(attn->K_layout);
    cublasLtMatrixLayoutDestroy(attn->V_layout);
    cublasLtMatrixLayoutDestroy(attn->scores_layout);
    cublasLtMatrixLayoutDestroy(attn->weights_layout);
    cublasLtMatrixLayoutDestroy(attn->W_q_broadcast_layout);
    cublasLtMatrixLayoutDestroy(attn->W_k_broadcast_layout);
    cublasLtMatrixLayoutDestroy(attn->W_v_broadcast_layout);
    cublasLtMatrixLayoutDestroy(attn->W_o_broadcast_layout);
    cublasLtMatrixLayoutDestroy(attn->X_for_wgrad_layout);
    cublasLtMatrixLayoutDestroy(attn->grad_QKV_for_wgrad_layout);
    cublasLtMatrixLayoutDestroy(attn->attn_out_for_wgrad_layout);
    cublasLtMatrixLayoutDestroy(attn->grad_out_for_wgrad_layout);
    
    // Free device memory
    cudaFree(attn->d_W_q); cudaFree(attn->d_W_k); cudaFree(attn->d_W_v); cudaFree(attn->d_W_o);
    cudaFree(attn->d_W_q_grad); cudaFree(attn->d_W_k_grad); cudaFree(attn->d_W_v_grad); cudaFree(attn->d_W_o_grad);
    cudaFree(attn->d_W_q_m); cudaFree(attn->d_W_q_v); cudaFree(attn->d_W_k_m); cudaFree(attn->d_W_k_v);
    cudaFree(attn->d_W_v_m); cudaFree(attn->d_W_v_v); cudaFree(attn->d_W_o_m); cudaFree(attn->d_W_o_v);
    cudaFree(attn->d_Q); cudaFree(attn->d_K); cudaFree(attn->d_V);
    cudaFree(attn->d_scores); cudaFree(attn->d_attn_weights);
    cudaFree(attn->d_attn_output); cudaFree(attn->d_output);
    cudaFree(attn->d_grad_output); cudaFree(attn->d_grad_attn_output); cudaFree(attn->d_grad_weights);
    cudaFree(attn->d_grad_scores); cudaFree(attn->d_grad_Q); cudaFree(attn->d_grad_K); cudaFree(attn->d_grad_V);
    free(attn);
}

// CUDA kernel for softmax forward pass
__global__ void softmax_forward_kernel_attention(float* weights, float* scores, int batch_size, int seq_len) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    
    if (b >= batch_size || i >= seq_len) return;
    
    float* scores_b = &scores[b * seq_len * seq_len];
    float* weights_b = &weights[b * seq_len * seq_len];
    
    // Find max for numerical stability
    float max_val = -1e30f;
    for (int j = 0; j < seq_len; j++) {
        float val = scores_b[i * seq_len + j];
        if (val > max_val) max_val = val;
    }
    
    // Compute softmax
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        float exp_val = expf(scores_b[i * seq_len + j] - max_val);
        weights_b[i * seq_len + j] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize
    for (int j = 0; j < seq_len; j++) {
        weights_b[i * seq_len + j] /= sum_exp;
    }
}

// CUDA kernel for softmax backward pass
__global__ void softmax_backward_kernel_attention(float* grad_scores, float* grad_weights, float* weights, int batch_size, int seq_len) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    
    if (b >= batch_size || i >= seq_len) return;
    
    float* grad_weights_b = &grad_weights[b * seq_len * seq_len];
    float* weights_b = &weights[b * seq_len * seq_len];
    float* grad_scores_b = &grad_scores[b * seq_len * seq_len];
    
    // Compute sum term for softmax gradient
    float sum_term = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        int idx = i * seq_len + k;
        sum_term += grad_weights_b[idx] * weights_b[idx];
    }
    
    // Apply softmax gradient formula
    for (int j = 0; j < seq_len; j++) {
        int idx = i * seq_len + j;
        grad_scores_b[idx] = weights_b[idx] * (grad_weights_b[idx] - sum_term);
    }
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_attention(float* weight, float* grad, float* m, float* v,
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

// Forward pass
void forward_pass_attention(Attention* attn, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 1: Compute Q, K, V using strided batched operations with broadcasting weights
    // Q = X * W_q (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_desc,
                                  &alpha,
                                  d_X, attn->Q_layout,
                                  attn->d_W_q, attn->W_q_broadcast_layout,
                                  &beta,
                                  attn->d_Q, attn->Q_layout,
                                  attn->d_Q, attn->Q_layout,
                                  NULL, NULL, 0, 0));
    
    // K = X * W_k (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_desc,
                                  &alpha,
                                  d_X, attn->K_layout,
                                  attn->d_W_k, attn->W_k_broadcast_layout,
                                  &beta,
                                  attn->d_K, attn->K_layout,
                                  attn->d_K, attn->K_layout,
                                  NULL, NULL, 0, 0));
    
    // V = X * W_v (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_desc,
                                  &alpha,
                                  d_X, attn->V_layout,
                                  attn->d_W_v, attn->W_v_broadcast_layout,
                                  &beta,
                                  attn->d_V, attn->V_layout,
                                  attn->d_V, attn->V_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 2: Compute attention scores = Q * Kᵀ * scale (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NT_desc,
                                  &attn->scale,
                                  attn->d_Q, attn->Q_layout,
                                  attn->d_K, attn->K_layout,
                                  &beta,
                                  attn->d_scores, attn->scores_layout,
                                  attn->d_scores, attn->scores_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 3: Apply softmax row-wise
    dim3 grid(attn->batch_size, attn->seq_len);
    softmax_forward_kernel_attention<<<grid, 1>>>(attn->d_attn_weights, attn->d_scores, attn->batch_size, attn->seq_len);
    
    // Step 4: Compute attention output = weights * V (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_desc,
                                  &alpha,
                                  attn->d_attn_weights, attn->weights_layout,
                                  attn->d_V, attn->V_layout,
                                  &beta,
                                  attn->d_attn_output, attn->Q_layout,
                                  attn->d_attn_output, attn->Q_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 5: Apply output projection (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_desc,
                                  &alpha,
                                  attn->d_attn_output, attn->Q_layout,
                                  attn->d_W_o, attn->W_o_broadcast_layout,
                                  &beta,
                                  attn->d_output, attn->Q_layout,
                                  attn->d_output, attn->Q_layout,
                                  NULL, NULL, 0, 0));
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* d_y) {
    int total_elements = attn->batch_size * attn->seq_len * attn->d_model;
    
    // grad_output = output - y
    const float alpha = 1.0f;
    const float beta = -1.0f;
    CHECK_CUBLAS(cublasSgeam(attn->cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            total_elements, 1,
                            &alpha, attn->d_output, total_elements,
                            &beta, d_y, total_elements,
                            attn->d_grad_output, total_elements));
    
    // Calculate MSE loss
    float loss = 0.0f;
    CHECK_CUBLAS(cublasSdot(attn->cublas_handle, total_elements, 
                           attn->d_grad_output, 1, attn->d_grad_output, 1, &loss));
    return loss / total_elements;
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
    const float gamma = 1.0f / attn->batch_size; // For averaging gradients
    
    // Step 5 (backward): Gradient through output projection
    // grad_W_o = (1/batch_size) * sum_batches(attn_output_bᵀ * grad_output_b)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &gamma,  // Scale by 1/batch_size for averaging
                                  attn->d_attn_output, attn->attn_out_for_wgrad_layout,
                                  attn->d_grad_output, attn->grad_out_for_wgrad_layout,
                                  &beta,
                                  attn->d_W_o_grad, attn->W_grad_layout,
                                  attn->d_W_o_grad, attn->W_grad_layout,
                                  NULL, NULL, 0, 0));
    
    // grad_attn_output = grad_output * W_oᵀ (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NT_desc,
                                  &alpha,
                                  attn->d_grad_output, attn->Q_layout,
                                  attn->d_W_o, attn->W_o_broadcast_layout,
                                  &beta,
                                  attn->d_grad_attn_output, attn->Q_layout,
                                  attn->d_grad_attn_output, attn->Q_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 4 (backward): Gradient through attention output computation
    // grad_weights = grad_attn_output * Vᵀ (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_NT_desc,
                                  &alpha,
                                  attn->d_grad_attn_output, attn->Q_layout,
                                  attn->d_V, attn->V_layout,
                                  &beta,
                                  attn->d_grad_weights, attn->weights_layout,
                                  attn->d_grad_weights, attn->weights_layout,
                                  NULL, NULL, 0, 0));
    
    // grad_V = weightsᵀ * grad_attn_output (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &alpha,
                                  attn->d_attn_weights, attn->weights_layout,
                                  attn->d_grad_attn_output, attn->Q_layout,
                                  &beta,
                                  attn->d_grad_V, attn->V_layout,
                                  attn->d_grad_V, attn->V_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 3 (backward): Gradient through softmax
    dim3 grid(attn->batch_size, attn->seq_len);
    softmax_backward_kernel_attention<<<grid, 1>>>(attn->d_grad_scores, attn->d_grad_weights, attn->d_attn_weights, attn->batch_size, attn->seq_len);
    
    // Step 2 (backward): Gradient through attention scores
    // grad_Q = grad_scores * K * scale (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_desc,
                                  &attn->scale,
                                  attn->d_grad_scores, attn->scores_layout,
                                  attn->d_K, attn->K_layout,
                                  &beta,
                                  attn->d_grad_Q, attn->Q_layout,
                                  attn->d_grad_Q, attn->Q_layout,
                                  NULL, NULL, 0, 0));
    
    // grad_K = grad_scoresᵀ * Q * scale (batched)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &attn->scale,
                                  attn->d_grad_scores, attn->scores_layout,
                                  attn->d_Q, attn->Q_layout,
                                  &beta,
                                  attn->d_grad_K, attn->K_layout,
                                  attn->d_grad_K, attn->K_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 1 (backward): Gradient through linear projections
    // Weight gradients using batched operations with averaging
    // grad_W_q = (1/batch_size) * Xᵀ * grad_Q (reshape to handle batch dimension)
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &gamma,  // Scale by 1/batch_size for averaging
                                  d_X, attn->X_for_wgrad_layout,
                                  attn->d_grad_Q, attn->grad_QKV_for_wgrad_layout,
                                  &beta,
                                  attn->d_W_q_grad, attn->W_grad_layout,
                                  attn->d_W_q_grad, attn->W_grad_layout,
                                  NULL, NULL, 0, 0));
    
    // grad_W_k = (1/batch_size) * Xᵀ * grad_K
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &gamma,
                                  d_X, attn->X_for_wgrad_layout,
                                  attn->d_grad_K, attn->grad_QKV_for_wgrad_layout,
                                  &beta,
                                  attn->d_W_k_grad, attn->W_grad_layout,
                                  attn->d_W_k_grad, attn->W_grad_layout,
                                  NULL, NULL, 0, 0));
    
    // grad_W_v = (1/batch_size) * Xᵀ * grad_V
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                  attn->matmul_TN_desc,
                                  &gamma,
                                  d_X, attn->X_for_wgrad_layout,
                                  attn->d_grad_V, attn->grad_QKV_for_wgrad_layout,
                                  &beta,
                                  attn->d_W_v_grad, attn->W_grad_layout,
                                  attn->d_W_v_grad, attn->W_grad_layout,
                                  NULL, NULL, 0, 0));
    
    // grad_X = grad_Q * W_qᵀ + grad_K * W_kᵀ + grad_V * W_vᵀ (batched)
    if (d_grad_X != NULL) {
        CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                      attn->matmul_NT_desc,
                                      &alpha,
                                      attn->d_grad_Q, attn->Q_layout,
                                      attn->d_W_q, attn->W_q_broadcast_layout,
                                      &beta,
                                      d_grad_X, attn->Q_layout,
                                      d_grad_X, attn->Q_layout,
                                      NULL, NULL, 0, 0));
        
        CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                      attn->matmul_NT_desc,
                                      &alpha,
                                      attn->d_grad_K, attn->K_layout,
                                      attn->d_W_k, attn->W_k_broadcast_layout,
                                      &alpha,  // Note: using alpha (=1.0) to accumulate
                                      d_grad_X, attn->Q_layout,
                                      d_grad_X, attn->Q_layout,
                                      NULL, NULL, 0, 0));
        
        CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle,
                                      attn->matmul_NT_desc,
                                      &alpha,
                                      attn->d_grad_V, attn->V_layout,
                                      attn->d_W_v, attn->W_v_broadcast_layout,
                                      &alpha,  // Note: using alpha (=1.0) to accumulate
                                      d_grad_X, attn->Q_layout,
                                      d_grad_X, attn->Q_layout,
                                      NULL, NULL, 0, 0));
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
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    
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
Attention* load_attention(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int seq_len, d_model, stored_batch_size;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    Attention* attn = init_attention(seq_len, d_model, batch_size, cublas_handle, cublaslt_handle);
    
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