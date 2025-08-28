#include "attention.h"

// Row-major GEMM kernel: C = alpha * A * B + beta * C
__global__ void rowmajor_gemm_kernel(float* C, const float* A, const float* B, 
                                     int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Row-major GEMM with B transposed: C = alpha * A * B^T + beta * C
__global__ void rowmajor_gemm_transB_kernel(float* C, const float* A, const float* B,
                                            int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Row-major GEMM with A transposed: C = alpha * A^T * B + beta * C
__global__ void rowmajor_gemm_transA_kernel(float* C, const float* A, const float* B,
                                            int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[k * M + row] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Softmax kernel for attention weights
__global__ void softmax_kernel(float* weights, const float* scores, int batch_size, int seq_len) {
    int batch = blockIdx.x;
    int row = threadIdx.x;
    
    if (batch < batch_size && row < seq_len) {
        int offset = batch * seq_len * seq_len + row * seq_len;
        
        // Find max for numerical stability
        float max_val = scores[offset];
        for (int j = 1; j < seq_len; j++) {
            if (scores[offset + j] > max_val) {
                max_val = scores[offset + j];
            }
        }
        
        // Compute softmax
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            weights[offset + j] = expf(scores[offset + j] - max_val);
            sum += weights[offset + j];
        }
        
        for (int j = 0; j < seq_len; j++) {
            weights[offset + j] /= sum;
        }
    }
}

// Softmax backward kernel
__global__ void softmax_backward_kernel(float* d_scores, const float* d_weights, const float* weights, 
                                        int batch_size, int seq_len) {
    int batch = blockIdx.x;
    int row = threadIdx.x;
    
    if (batch < batch_size && row < seq_len) {
        int offset = batch * seq_len * seq_len + row * seq_len;
        
        // Compute sum for softmax gradient
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += d_weights[offset + j] * weights[offset + j];
        }
        
        for (int j = 0; j < seq_len; j++) {
            d_scores[offset + j] = weights[offset + j] * (d_weights[offset + j] - sum);
        }
    }
}

// AdamW update kernel
__global__ void adamw_update_kernel(float* weight, const float* grad, float* m, float* v,
                                   float beta1, float beta2, float epsilon, float learning_rate,
                                   float weight_decay, float alpha_t, int size, int batch_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_norm;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Simple kernel to compute error and squared error
__global__ void compute_error_kernel(float* error, const float* predictions, const float* targets, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - targets[idx];
    }
}

// Initialize the attention network
Attention* init_attention(int d_model, int seq_len, int batch_size) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->d_model = d_model;
    attn->seq_len = seq_len;
    attn->batch_size = batch_size;
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    int weight_size = d_model * d_model;
    int seq_size = batch_size * seq_len * d_model;
    int attn_size = batch_size * seq_len * seq_len;
    
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
    
    // Initialize weights on host and copy to device
    float* h_weights = (float*)malloc(weight_size * sizeof(float));
    float scale = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < weight_size; i++) {
        h_weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    CHECK_CUDA(cudaMemcpy(attn->d_W_q, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int i = 0; i < weight_size; i++) {
        h_weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    CHECK_CUDA(cudaMemcpy(attn->d_W_k, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int i = 0; i < weight_size; i++) {
        h_weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    CHECK_CUDA(cudaMemcpy(attn->d_W_v, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int i = 0; i < weight_size; i++) {
        h_weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    CHECK_CUDA(cudaMemcpy(attn->d_W_o, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_weights);
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(attn->d_W_q_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_q_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_v, 0, weight_size * sizeof(float)));
    
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

// Forward pass
void forward_pass_attention(Attention* attn, float* d_X) {
    int total_seq = attn->batch_size * attn->seq_len;
    
    dim3 blockDim(16, 16);
    dim3 gridDim_seq_model((attn->d_model + blockDim.x - 1) / blockDim.x, 
                           (total_seq + blockDim.y - 1) / blockDim.y);
    dim3 gridDim_seq_seq((attn->seq_len + blockDim.x - 1) / blockDim.x, 
                         (attn->seq_len + blockDim.y - 1) / blockDim.y);
    
    // Q = XW_q, K = XW_k, V = XW_v
    rowmajor_gemm_kernel<<<gridDim_seq_model, blockDim>>>(
        attn->d_Q, d_X, attn->d_W_q, total_seq, attn->d_model, attn->d_model, 1.0f, 0.0f);
    
    rowmajor_gemm_kernel<<<gridDim_seq_model, blockDim>>>(
        attn->d_K, d_X, attn->d_W_k, total_seq, attn->d_model, attn->d_model, 1.0f, 0.0f);
    
    rowmajor_gemm_kernel<<<gridDim_seq_model, blockDim>>>(
        attn->d_V, d_X, attn->d_W_v, total_seq, attn->d_model, attn->d_model, 1.0f, 0.0f);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Compute attention scores for each batch: QK^T / sqrt(d_model)
    float scale = 1.0f / sqrtf(attn->d_model);
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* Q_batch = attn->d_Q + batch * attn->seq_len * attn->d_model;
        float* K_batch = attn->d_K + batch * attn->seq_len * attn->d_model;
        float* scores_batch = attn->d_attn_scores + batch * attn->seq_len * attn->seq_len;
        
        rowmajor_gemm_transB_kernel<<<gridDim_seq_seq, blockDim>>>(
            scores_batch, Q_batch, K_batch, attn->seq_len, attn->seq_len, attn->d_model, scale, 0.0f);
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Apply softmax to get attention weights
    dim3 softmaxGrid(attn->batch_size);
    dim3 softmaxBlock(attn->seq_len);
    softmax_kernel<<<softmaxGrid, softmaxBlock>>>(
        attn->d_attn_weights, attn->d_attn_scores, attn->batch_size, attn->seq_len);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Compute weighted values for each batch: weights * V
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* weights_batch = attn->d_attn_weights + batch * attn->seq_len * attn->seq_len;
        float* V_batch = attn->d_V + batch * attn->seq_len * attn->d_model;
        float* output_batch = attn->d_attn_output + batch * attn->seq_len * attn->d_model;
        
        dim3 gridDim_seq_model_batch((attn->d_model + blockDim.x - 1) / blockDim.x, 
                                     (attn->seq_len + blockDim.y - 1) / blockDim.y);
        rowmajor_gemm_kernel<<<gridDim_seq_model_batch, blockDim>>>(
            output_batch, weights_batch, V_batch, attn->seq_len, attn->d_model, attn->seq_len, 1.0f, 0.0f);
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Apply output projection: layer_output = attn_output * W_o
    rowmajor_gemm_kernel<<<gridDim_seq_model, blockDim>>>(
        attn->d_layer_output, attn->d_attn_output, attn->d_W_o, 
        total_seq, attn->d_model, attn->d_model, 1.0f, 0.0f);
    
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* d_y) {
    int total_size = attn->batch_size * attn->seq_len * attn->d_model;
    
    // Compute error on GPU: error = predictions - targets
    dim3 blockDim(256);
    dim3 gridDim((total_size + blockDim.x - 1) / blockDim.x);
    
    compute_error_kernel<<<gridDim, blockDim>>>(
        attn->d_error_output, attn->d_layer_output, d_y, total_size);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy error back to CPU to compute loss (for simplicity)
    float* h_error = (float*)malloc(total_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_error, attn->d_error_output, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float loss = 0.0f;
    for (int i = 0; i < total_size; i++) {
        loss += h_error[i] * h_error[i];
    }
    
    free(h_error);
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
void backward_pass_attention(Attention* attn, float* d_X) {
    int total_seq = attn->batch_size * attn->seq_len;
    
    dim3 blockDim(16, 16);
    dim3 gridDim_model_model((attn->d_model + blockDim.x - 1) / blockDim.x, 
                             (attn->d_model + blockDim.y - 1) / blockDim.y);
    dim3 gridDim_seq_model((attn->d_model + blockDim.x - 1) / blockDim.x, 
                           (total_seq + blockDim.y - 1) / blockDim.y);
    dim3 gridDim_seq_seq((attn->seq_len + blockDim.x - 1) / blockDim.x, 
                         (attn->seq_len + blockDim.y - 1) / blockDim.y);
    
    // ∂L/∂W_o = attn_output^T * error_output
    rowmajor_gemm_transA_kernel<<<gridDim_model_model, blockDim>>>(
        attn->d_W_o_grad, attn->d_attn_output, attn->d_error_output, 
        attn->d_model, attn->d_model, total_seq, 1.0f, 1.0f);
    
    // ∂L/∂attn_output = error_output * W_o^T
    rowmajor_gemm_transB_kernel<<<gridDim_seq_model, blockDim>>>(
        attn->d_grad_attn_out, attn->d_error_output, attn->d_W_o, 
        total_seq, attn->d_model, attn->d_model, 1.0f, 0.0f);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Backpropagate through attention for each batch
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* d_attn_output_batch = attn->d_grad_attn_out + batch * attn->seq_len * attn->d_model;
        float* attn_weights_batch = attn->d_attn_weights + batch * attn->seq_len * attn->seq_len;
        float* V_batch = attn->d_V + batch * attn->seq_len * attn->d_model;
        float* d_weights_batch = attn->d_grad_weights + batch * attn->seq_len * attn->seq_len;
        float* dV_batch = attn->d_grad_V + batch * attn->seq_len * attn->d_model;
        
        // ∂L/∂V = weights^T * d_attn_output
        dim3 gridDim_seq_model_batch((attn->d_model + blockDim.x - 1) / blockDim.x, 
                                     (attn->seq_len + blockDim.y - 1) / blockDim.y);
        rowmajor_gemm_transA_kernel<<<gridDim_seq_model_batch, blockDim>>>(
            dV_batch, attn_weights_batch, d_attn_output_batch, 
            attn->seq_len, attn->d_model, attn->seq_len, 1.0f, 0.0f);
        
        // ∂L/∂weights = d_attn_output * V^T
        rowmajor_gemm_transB_kernel<<<gridDim_seq_seq, blockDim>>>(
            d_weights_batch, d_attn_output_batch, V_batch, 
            attn->seq_len, attn->seq_len, attn->d_model, 1.0f, 0.0f);
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Backpropagate through softmax
    dim3 softmaxGrid(attn->batch_size);
    dim3 softmaxBlock(attn->seq_len);
    softmax_backward_kernel<<<softmaxGrid, softmaxBlock>>>(
        attn->d_grad_scores, attn->d_grad_weights, attn->d_attn_weights, 
        attn->batch_size, attn->seq_len);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Backpropagate through attention scores for each batch
    float scale = 1.0f / sqrtf(attn->d_model);
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* d_scores_batch = attn->d_grad_scores + batch * attn->seq_len * attn->seq_len;
        float* Q_batch = attn->d_Q + batch * attn->seq_len * attn->d_model;
        float* K_batch = attn->d_K + batch * attn->seq_len * attn->d_model;
        float* dQ_batch = attn->d_grad_Q + batch * attn->seq_len * attn->d_model;
        float* dK_batch = attn->d_grad_K + batch * attn->seq_len * attn->d_model;
        
        // ∂L/∂Q = d_scores * K / sqrt(d_model)
        dim3 gridDim_seq_model_batch((attn->d_model + blockDim.x - 1) / blockDim.x, 
                                     (attn->seq_len + blockDim.y - 1) / blockDim.y);
        rowmajor_gemm_kernel<<<gridDim_seq_model_batch, blockDim>>>(
            dQ_batch, d_scores_batch, K_batch, 
            attn->seq_len, attn->d_model, attn->seq_len, scale, 0.0f);
        
        // ∂L/∂K = d_scores^T * Q / sqrt(d_model)
        rowmajor_gemm_transA_kernel<<<gridDim_seq_model_batch, blockDim>>>(
            dK_batch, d_scores_batch, Q_batch, 
            attn->seq_len, attn->d_model, attn->seq_len, scale, 0.0f);
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Accumulate weight gradients
    // ∂L/∂W_q = X^T * grad_Q
    rowmajor_gemm_transA_kernel<<<gridDim_model_model, blockDim>>>(
        attn->d_W_q_grad, d_X, attn->d_grad_Q, 
        attn->d_model, attn->d_model, total_seq, 1.0f, 1.0f);
    
    // ∂L/∂W_k = X^T * grad_K
    rowmajor_gemm_transA_kernel<<<gridDim_model_model, blockDim>>>(
        attn->d_W_k_grad, d_X, attn->d_grad_K, 
        attn->d_model, attn->d_model, total_seq, 1.0f, 1.0f);
    
    // ∂L/∂W_v = X^T * grad_V
    rowmajor_gemm_transA_kernel<<<gridDim_model_model, blockDim>>>(
        attn->d_W_v_grad, d_X, attn->d_grad_V, 
        attn->d_model, attn->d_model, total_seq, 1.0f, 1.0f);
    
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int weight_size = attn->d_model * attn->d_model;
    int total_seq = attn->batch_size * attn->seq_len;
    
    dim3 blockDim(256);
    dim3 gridDim((weight_size + blockDim.x - 1) / blockDim.x);
    
    // Update W_q weights
    adamw_update_kernel<<<gridDim, blockDim>>>(
        attn->d_W_q, attn->d_W_q_grad, attn->d_W_q_m, attn->d_W_q_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, total_seq);
    
    // Update W_k weights
    adamw_update_kernel<<<gridDim, blockDim>>>(
        attn->d_W_k, attn->d_W_k_grad, attn->d_W_k_m, attn->d_W_k_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, total_seq);
    
    // Update W_v weights
    adamw_update_kernel<<<gridDim, blockDim>>>(
        attn->d_W_v, attn->d_W_v_grad, attn->d_W_v_m, attn->d_W_v_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, total_seq);
    
    // Update W_o weights
    adamw_update_kernel<<<gridDim, blockDim>>>(
        attn->d_W_o, attn->d_W_o_grad, attn->d_W_o_m, attn->d_W_o_v,
        attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
        alpha_t, weight_size, total_seq);
    
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Save model weights and Adam state to binary file
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
    
    // Save weights
    int weight_size = attn->d_model * attn->d_model;
    float* h_weights = (float*)malloc(weight_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_q, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_k, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_o, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    
    // Save Adam state
    fwrite(&attn->t, sizeof(int), 1, file);
    
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_q_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_q_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_k_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_k_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_v_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_v_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_o_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(h_weights, attn->d_W_o_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_weights, sizeof(float), weight_size, file);

    free(h_weights);
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights and Adam state from binary file
Attention* load_attention(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int d_model, seq_len, stored_batch_size;
    fread(&d_model, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    Attention* attn = init_attention(d_model, seq_len, batch_size);
    
    // Load weights
    int weight_size = d_model * d_model;
    float* h_weights = (float*)malloc(weight_size * sizeof(float));
    
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_q, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_k, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_v, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_o, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    fread(&attn->t, sizeof(int), 1, file);
    
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_q_m, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_q_v, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_k_m, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_k_v, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_v_m, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_v_v, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_o_m, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    fread(h_weights, sizeof(float), weight_size, file);
    CHECK_CUDA(cudaMemcpy(attn->d_W_o_v, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_weights);
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return attn;
}