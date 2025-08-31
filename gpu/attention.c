#include "attention.h"

// Initialize the attention network
Attention* init_attention(int d_model, int seq_len, int batch_size, bool is_causal, cublasHandle_t cublas_handle) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->d_model = d_model;
    attn->seq_len = seq_len;
    attn->batch_size = batch_size;
    attn->is_causal = is_causal;
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    attn->cublas_handle = cublas_handle;
    
    int weight_size = d_model * d_model;
    int seq_size = batch_size * seq_len * d_model;
    int attn_size = batch_size * seq_len * seq_len;
    
    // Allocate host memory for weight initialization
    float* h_W_q = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o = (float*)malloc(weight_size * sizeof(float));
    
    // Initialize weights on host
    float scale = 1.0f / sqrtf(d_model);
    for (int i = 0; i < weight_size; i++) {
        h_W_q[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        h_W_k[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        h_W_v[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        h_W_o[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
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
    CHECK_CUDA(cudaMalloc(&attn->d_Q, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_K, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_V, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_scores, attn_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_weights, attn_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_output, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_layer_output, seq_size * sizeof(float)));
    
    // Allocate device memory for backward pass buffers
    CHECK_CUDA(cudaMalloc(&attn->d_grad_Q, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_K, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_V, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_scores, attn_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_weights, attn_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_attn_out, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_error_output, seq_size * sizeof(float)));
    
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
    
    // Free host memory
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    
    return attn;
}

// Free network memory
void free_attention(Attention* attn) {
    // Free device memory
    cudaFree(attn->d_W_q); cudaFree(attn->d_W_k); cudaFree(attn->d_W_v); cudaFree(attn->d_W_o);
    cudaFree(attn->d_W_q_grad); cudaFree(attn->d_W_k_grad); cudaFree(attn->d_W_v_grad); cudaFree(attn->d_W_o_grad);
    cudaFree(attn->d_W_q_m); cudaFree(attn->d_W_q_v); cudaFree(attn->d_W_k_m); cudaFree(attn->d_W_k_v);
    cudaFree(attn->d_W_v_m); cudaFree(attn->d_W_v_v); cudaFree(attn->d_W_o_m); cudaFree(attn->d_W_o_v);
    cudaFree(attn->d_Q); cudaFree(attn->d_K); cudaFree(attn->d_V);
    cudaFree(attn->d_attn_scores); cudaFree(attn->d_attn_weights);
    cudaFree(attn->d_attn_output); cudaFree(attn->d_layer_output);
    cudaFree(attn->d_grad_Q); cudaFree(attn->d_grad_K); cudaFree(attn->d_grad_V);
    cudaFree(attn->d_grad_scores); cudaFree(attn->d_grad_weights);
    cudaFree(attn->d_grad_attn_out); cudaFree(attn->d_error_output);
    free(attn);
}

// CUDA kernel for softmax forward pass
__global__ void softmax_forward_kernel_attention(float* weights, float* scores, int batch_size, int seq_len) {
    int batch = blockIdx.x;
    int row = threadIdx.x;
    
    if (batch < batch_size && row < seq_len) {
        int offset = batch * seq_len * seq_len + row * seq_len;
        float* row_scores = scores + offset;
        float* row_weights = weights + offset;
        
        // Find max for numerical stability
        float max_val = row_scores[0];
        for (int j = 1; j < seq_len; j++) {
            if (row_scores[j] > max_val) max_val = row_scores[j];
        }
        
        // Compute softmax
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            row_weights[j] = expf(row_scores[j] - max_val);
            sum += row_weights[j];
        }
        for (int j = 0; j < seq_len; j++) {
            row_weights[j] /= sum;
        }
    }
}

// CUDA kernel for softmax backward pass
__global__ void softmax_backward_kernel_attention(float* grad_scores, float* grad_weights, float* weights, int batch_size, int seq_len) {
    int batch = blockIdx.x;
    int row = threadIdx.x;
    
    if (batch < batch_size && row < seq_len) {
        int offset = batch * seq_len * seq_len + row * seq_len;
        float* weights_row = weights + offset;
        float* grad_weights_row = grad_weights + offset;
        float* grad_scores_row = grad_scores + offset;
        
        // Softmax gradient: grad_scores[i] = weights[i] * (grad_weights[i] - sum_j(grad_weights[j] * weights[j]))
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += grad_weights_row[j] * weights_row[j];
        }
        for (int j = 0; j < seq_len; j++) {
            grad_scores_row[j] = weights_row[j] * (grad_weights_row[j] - sum);
        }
    }
}

// CUDA kernel for causal softmax forward pass
__global__ void softmax_causal_forward_kernel_attention(float* weights, float* scores, int batch_size, int seq_len) {
    int batch = blockIdx.x;
    int row = threadIdx.x;
    
    if (batch < batch_size && row < seq_len) {
        int offset = batch * seq_len * seq_len + row * seq_len;
        float* row_scores = scores + offset;
        float* row_weights = weights + offset;
        
        // Find max for numerical stability (only consider positions <= row for causal mask)
        float max_val = row_scores[0];
        for (int j = 1; j <= row; j++) {
            if (row_scores[j] > max_val) max_val = row_scores[j];
        }
        
        // Compute softmax with causal masking
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            if (j <= row) {
                // Attend to current and previous positions
                row_weights[j] = expf(row_scores[j] - max_val);
                sum += row_weights[j];
            } else {
                // Mask future positions
                row_weights[j] = 0.0f;
            }
        }
        
        // Normalize
        for (int j = 0; j <= row; j++) {
            row_weights[j] /= sum;
        }
    }
}

// CUDA kernel for causal softmax backward pass
__global__ void softmax_causal_backward_kernel_attention(float* grad_scores, float* grad_weights, float* weights, int batch_size, int seq_len) {
    int batch = blockIdx.x;
    int row = threadIdx.x;
    
    if (batch < batch_size && row < seq_len) {
        int offset = batch * seq_len * seq_len + row * seq_len;
        float* weights_row = weights + offset;
        float* grad_weights_row = grad_weights + offset;
        float* grad_scores_row = grad_scores + offset;
        
        // Softmax gradient with causal masking
        float sum = 0.0f;
        for (int j = 0; j <= row; j++) {
            sum += grad_weights_row[j] * weights_row[j];
        }
        
        for (int j = 0; j < seq_len; j++) {
            if (j <= row) {
                // Gradient for non-masked positions
                grad_scores_row[j] = weights_row[j] * (grad_weights_row[j] - sum);
            } else {
                // Zero gradient for masked (future) positions
                grad_scores_row[j] = 0.0f;
            }
        }
    }
}

// Forward pass
void forward_pass_attention(Attention* attn, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int total_seq = attn->batch_size * attn->seq_len;
    
    // Q = XWq - Query transformation: maps inputs to query representations
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            attn->d_model, total_seq, attn->d_model,
                            &alpha, attn->d_W_q, attn->d_model,
                            d_X, attn->d_model,
                            &beta, attn->d_Q, attn->d_model));
    
    // K = XWk - Key transformation: produces keys for matching
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            attn->d_model, total_seq, attn->d_model,
                            &alpha, attn->d_W_k, attn->d_model,
                            d_X, attn->d_model,
                            &beta, attn->d_K, attn->d_model));
    
    // V = XWv - Value transformation: generates values to be aggregated
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            attn->d_model, total_seq, attn->d_model,
                            &alpha, attn->d_W_v, attn->d_model,
                            d_X, attn->d_model,
                            &beta, attn->d_V, attn->d_model));
    
    // S = QK^T / √d_model using batched GEMM - Scaled attention scores: measure compatibility between queries and keys
    float scale = 1.0f / sqrtf(attn->d_model);
    long long int strideA = (long long int)attn->seq_len * attn->d_model;  // Stride between Q matrices for each batch
    long long int strideB = (long long int)attn->seq_len * attn->d_model;  // Stride between K matrices for each batch
    long long int strideC = (long long int)attn->seq_len * attn->seq_len;  // Stride between score matrices for each batch
    
    CHECK_CUBLAS(cublasSgemmStridedBatched(attn->cublas_handle,
                                          CUBLAS_OP_T, CUBLAS_OP_N,
                                          attn->seq_len, attn->seq_len, attn->d_model,
                                          &scale,
                                          attn->d_K, attn->d_model, strideA,
                                          attn->d_Q, attn->d_model, strideB,
                                          &beta,
                                          attn->d_attn_scores, attn->seq_len, strideC,
                                          attn->batch_size));
    
    // A_ij = exp(S_ij) / Σ_k exp(S_ik) - Softmax normalization: produces attention weights
    dim3 grid(attn->batch_size);
    dim3 block(attn->seq_len);
    if (attn->is_causal) {
        softmax_causal_forward_kernel_attention<<<grid, block>>>(
            attn->d_attn_weights, attn->d_attn_scores, attn->batch_size, attn->seq_len
        );
    } else {
        softmax_forward_kernel_attention<<<grid, block>>>(
            attn->d_attn_weights, attn->d_attn_scores, attn->batch_size, attn->seq_len
        );
    }
    
    // Z = AV using batched GEMM - Weighted combination: attention output as weighted sum of values
    strideA = (long long int)attn->seq_len * attn->d_model;  // Stride between V matrices for each batch
    strideB = (long long int)attn->seq_len * attn->seq_len;  // Stride between attention weight matrices for each batch
    strideC = (long long int)attn->seq_len * attn->d_model;  // Stride between attention output matrices for each batch
    
    CHECK_CUBLAS(cublasSgemmStridedBatched(attn->cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          attn->d_model, attn->seq_len, attn->seq_len,
                                          &alpha,
                                          attn->d_V, attn->d_model, strideA,
                                          attn->d_attn_weights, attn->seq_len, strideB,
                                          &beta,
                                          attn->d_attn_output, attn->d_model, strideC,
                                          attn->batch_size));
    
    // Y = ZWo - Output projection: transforms attended values to final outputs
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            attn->d_model, total_seq, attn->d_model,
                            &alpha, attn->d_W_o, attn->d_model,
                            attn->d_attn_output, attn->d_model,
                            &beta, attn->d_layer_output, attn->d_model));
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* d_y) {
    // ∂L/∂Y = Y - Y_true
    int total_size = attn->batch_size * attn->seq_len * attn->d_model;
    float loss = 0.0f;

    const float alpha = 1.0f;
    const float beta = -1.0f;
    CHECK_CUBLAS(cublasSgeam(attn->cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            attn->d_model, attn->batch_size * attn->seq_len,
                            &alpha, attn->d_layer_output, attn->d_model,
                            &beta, d_y, attn->d_model,
                            attn->d_error_output, attn->d_model));
    CHECK_CUBLAS(cublasSdot(attn->cublas_handle, total_size, 
                           attn->d_error_output, 1, attn->d_error_output, 1, &loss));
    
    return loss / total_size;
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
    int total_seq = attn->batch_size * attn->seq_len;
    
    // ∂L/∂Wo = Z^T(∂L/∂Y) - Output projection weight gradient
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            attn->d_model, attn->d_model, total_seq,
                            &alpha, attn->d_error_output, attn->d_model,
                            attn->d_attn_output, attn->d_model,
                            &alpha, attn->d_W_o_grad, attn->d_model));
    
    // ∂L/∂Z = (∂L/∂Y)Wo^T - Gradient w.r.t. attention output
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            attn->d_model, total_seq, attn->d_model,
                            &alpha, attn->d_W_o, attn->d_model,
                            attn->d_error_output, attn->d_model,
                            &beta, attn->d_grad_attn_out, attn->d_model));
    
    // ∂L/∂V = A^T(∂L/∂Z) using batched GEMM - Gradient w.r.t. values
    long long int strideA = (long long int)attn->seq_len * attn->d_model;  // Stride between grad_attn_out matrices
    long long int strideB = (long long int)attn->seq_len * attn->seq_len;  // Stride between attention weight matrices
    long long int strideC = (long long int)attn->seq_len * attn->d_model;  // Stride between grad_V matrices
    
    CHECK_CUBLAS(cublasSgemmStridedBatched(attn->cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_T,
                                          attn->d_model, attn->seq_len, attn->seq_len,
                                          &alpha,
                                          attn->d_grad_attn_out, attn->d_model, strideA,
                                          attn->d_attn_weights, attn->seq_len, strideB,
                                          &beta,
                                          attn->d_grad_V, attn->d_model, strideC,
                                          attn->batch_size));
    
    // ∂L/∂A = (∂L/∂Z)V^T using batched GEMM - Gradient w.r.t. attention weights
    strideA = (long long int)attn->seq_len * attn->d_model;  // Stride between V matrices
    strideB = (long long int)attn->seq_len * attn->d_model;  // Stride between grad_attn_out matrices
    strideC = (long long int)attn->seq_len * attn->seq_len;  // Stride between grad_weights matrices
    
    CHECK_CUBLAS(cublasSgemmStridedBatched(attn->cublas_handle,
                                          CUBLAS_OP_T, CUBLAS_OP_N,
                                          attn->seq_len, attn->seq_len, attn->d_model,
                                          &alpha,
                                          attn->d_V, attn->d_model, strideA,
                                          attn->d_grad_attn_out, attn->d_model, strideB,
                                          &beta,
                                          attn->d_grad_weights, attn->seq_len, strideC,
                                          attn->batch_size));
    
    // ∂L/∂S_ij = A_ij(∂L/∂A_ij - Σ_k ∂L/∂A_ikA_ik) - Softmax backward pass
    dim3 grid(attn->batch_size);
    dim3 block(attn->seq_len);
    if (attn->is_causal) {
        softmax_causal_backward_kernel_attention<<<grid, block>>>(
            attn->d_grad_scores, attn->d_grad_weights, attn->d_attn_weights, attn->batch_size, attn->seq_len
        );
    } else {
        softmax_backward_kernel_attention<<<grid, block>>>(
            attn->d_grad_scores, attn->d_grad_weights, attn->d_attn_weights, attn->batch_size, attn->seq_len
        );
    }
    
    // ∂L/∂Q = (∂L/∂S)K / √d_model using batched GEMM - Gradient w.r.t. queries
    float scale = 1.0f / sqrtf(attn->d_model);
    strideA = (long long int)attn->seq_len * attn->d_model;  // Stride between K matrices
    strideB = (long long int)attn->seq_len * attn->seq_len;  // Stride between grad_scores matrices
    strideC = (long long int)attn->seq_len * attn->d_model;  // Stride between grad_Q matrices
    
    CHECK_CUBLAS(cublasSgemmStridedBatched(attn->cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          attn->d_model, attn->seq_len, attn->seq_len,
                                          &scale,
                                          attn->d_K, attn->d_model, strideA,
                                          attn->d_grad_scores, attn->seq_len, strideB,
                                          &beta,
                                          attn->d_grad_Q, attn->d_model, strideC,
                                          attn->batch_size));
    
    // ∂L/∂K = (∂L/∂S)^TQ / √d_model using batched GEMM - Gradient w.r.t. keys
    strideA = (long long int)attn->seq_len * attn->d_model;  // Stride between Q matrices
    strideB = (long long int)attn->seq_len * attn->seq_len;  // Stride between grad_scores matrices
    strideC = (long long int)attn->seq_len * attn->d_model;  // Stride between grad_K matrices
    
    CHECK_CUBLAS(cublasSgemmStridedBatched(attn->cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_T,
                                          attn->d_model, attn->seq_len, attn->seq_len,
                                          &scale,
                                          attn->d_Q, attn->d_model, strideA,
                                          attn->d_grad_scores, attn->seq_len, strideB,
                                          &beta,
                                          attn->d_grad_K, attn->d_model, strideC,
                                          attn->batch_size));
    
    // Accumulate weight gradients
    // ∂L/∂Wq = X^T(∂L/∂Q) - Query weight gradient
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            attn->d_model, attn->d_model, total_seq,
                            &alpha, attn->d_grad_Q, attn->d_model,
                            d_X, attn->d_model,
                            &alpha, attn->d_W_q_grad, attn->d_model));
    
    // ∂L/∂Wk = X^T(∂L/∂K) - Key weight gradient
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            attn->d_model, attn->d_model, total_seq,
                            &alpha, attn->d_grad_K, attn->d_model,
                            d_X, attn->d_model,
                            &alpha, attn->d_W_k_grad, attn->d_model));
    
    // ∂L/∂Wv = X^T(∂L/∂V) - Value weight gradient
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            attn->d_model, attn->d_model, total_seq,
                            &alpha, attn->d_grad_V, attn->d_model,
                            d_X, attn->d_model,
                            &alpha, attn->d_W_v_grad, attn->d_model));

    if (d_grad_X != NULL) {
        // ∂L/∂X = (∂L/∂Q)W_q^T
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                attn->d_model, total_seq, attn->d_model,
                                &alpha, attn->d_W_q, attn->d_model,
                                attn->d_grad_Q, attn->d_model,
                                &alpha, d_grad_X, attn->d_model));
        
        // ∂L/∂X += (∂L/∂K)W_k^T
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                attn->d_model, total_seq, attn->d_model,
                                &alpha, attn->d_W_k, attn->d_model,
                                attn->d_grad_K, attn->d_model,
                                &alpha, d_grad_X, attn->d_model));
        
        // ∂L/∂X += (∂L/∂V)W_v^T
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                attn->d_model, total_seq, attn->d_model,
                                &alpha, attn->d_W_v, attn->d_model,
                                attn->d_grad_V, attn->d_model,
                                &alpha, d_grad_X, attn->d_model));
    }
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_attention(float* weight, float* grad, float* m, float* v,
                                              float beta1, float beta2, float epsilon, float learning_rate,
                                              float weight_decay, float alpha_t, int size, int total_seq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / total_seq;
        
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
    int total_seq = attn->batch_size * attn->seq_len;
    int block_size = 256;
    int num_blocks = (weight_size + block_size - 1) / block_size;
    
    // Update W_q weights
    adamw_update_kernel_attention<<<num_blocks, block_size>>>(
        attn->d_W_q, attn->d_W_q_grad, attn->d_W_q_m, attn->d_W_q_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, total_seq
    );
    
    // Update W_k weights
    adamw_update_kernel_attention<<<num_blocks, block_size>>>(
        attn->d_W_k, attn->d_W_k_grad, attn->d_W_k_m, attn->d_W_k_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, total_seq
    );
    
    // Update W_v weights
    adamw_update_kernel_attention<<<num_blocks, block_size>>>(
        attn->d_W_v, attn->d_W_v_grad, attn->d_W_v_m, attn->d_W_v_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, total_seq
    );
    
    // Update W_o weights
    adamw_update_kernel_attention<<<num_blocks, block_size>>>(
        attn->d_W_o, attn->d_W_o_grad, attn->d_W_o_m, attn->d_W_o_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, total_seq
    );
}

// Save model weights to binary file
void save_attention(Attention* attn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&attn->d_model, sizeof(int), 1, file);
    fwrite(&attn->seq_len, sizeof(int), 1, file);
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    fwrite(&attn->is_causal, sizeof(bool), 1, file);
    
    // Save weights
    int weight_size = attn->d_model * attn->d_model;
    
    // Allocate temporary host memory
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
    free(h_W_q_m); free(h_W_q_v);
    free(h_W_k_m); free(h_W_k_v);
    free(h_W_v_m); free(h_W_v_v);
    free(h_W_o_m); free(h_W_o_v);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights from binary file
Attention* load_attention(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int d_model, seq_len, stored_batch_size;
    bool is_causal;
    fread(&d_model, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    Attention* attn = init_attention(d_model, seq_len, batch_size, is_causal, cublas_handle);
    
    // Load weights
    int weight_size = d_model * d_model;
    
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
    
    // Free temporary host memory
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    free(h_W_q_m); free(h_W_q_v);
    free(h_W_k_m); free(h_W_k_v);
    free(h_W_v_m); free(h_W_v_v);
    free(h_W_o_m); free(h_W_o_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return attn;
}