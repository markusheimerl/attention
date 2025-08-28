#include "attention.h"

// Initialize the attention network
Attention* init_attention(int d_model, int seq_len, int batch_size, cublasHandle_t cublas_handle) {
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

// Forward pass
void forward_pass_attention(Attention* attn, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int total_seq = attn->batch_size * attn->seq_len;
    
    // Q = XW_q, K = XW_k, V = XW_v
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            attn->d_model, total_seq, attn->d_model,
                            &alpha, attn->d_W_q, attn->d_model,
                            d_X, attn->d_model,
                            &beta, attn->d_Q, attn->d_model));
    
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            attn->d_model, total_seq, attn->d_model,
                            &alpha, attn->d_W_k, attn->d_model,
                            d_X, attn->d_model,
                            &beta, attn->d_K, attn->d_model));
    
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            attn->d_model, total_seq, attn->d_model,
                            &alpha, attn->d_W_v, attn->d_model,
                            d_X, attn->d_model,
                            &beta, attn->d_V, attn->d_model));
    
    // Compute attention scores and weights for each batch
    float scale = 1.0f / sqrtf(attn->d_model);
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* Q_batch = attn->d_Q + batch * attn->seq_len * attn->d_model;
        float* K_batch = attn->d_K + batch * attn->seq_len * attn->d_model;
        float* scores_batch = attn->d_attn_scores + batch * attn->seq_len * attn->seq_len;
        
        // Attention scores = QK^T / sqrt(d_model)
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                attn->seq_len, attn->seq_len, attn->d_model,
                                &scale, K_batch, attn->d_model,
                                Q_batch, attn->d_model,
                                &beta, scores_batch, attn->seq_len));
    }
    
    // Apply softmax to get attention weights
    dim3 grid(attn->batch_size);
    dim3 block(attn->seq_len);
    softmax_forward_kernel_attention<<<grid, block>>>(
        attn->d_attn_weights, attn->d_attn_scores, attn->batch_size, attn->seq_len
    );
    
    // Compute weighted values for each batch
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* weights_batch = attn->d_attn_weights + batch * attn->seq_len * attn->seq_len;
        float* V_batch = attn->d_V + batch * attn->seq_len * attn->d_model;
        float* output_batch = attn->d_attn_output + batch * attn->seq_len * attn->d_model;
        
        // Attention output = weights * V
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                attn->d_model, attn->seq_len, attn->seq_len,
                                &alpha, V_batch, attn->d_model,
                                weights_batch, attn->seq_len,
                                &beta, output_batch, attn->d_model));
    }
    
    // Apply output projection: layer_output = attn_output * W_o
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
void backward_pass_attention(Attention* attn, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int total_seq = attn->batch_size * attn->seq_len;
    
    // ∂L/∂W_o = attn_output^T * error_output
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            attn->d_model, attn->d_model, total_seq,
                            &alpha, attn->d_error_output, attn->d_model,
                            attn->d_attn_output, attn->d_model,
                            &alpha, attn->d_W_o_grad, attn->d_model));
    
    // ∂L/∂attn_output = error_output * W_o^T
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            attn->d_model, total_seq, attn->d_model,
                            &alpha, attn->d_W_o, attn->d_model,
                            attn->d_error_output, attn->d_model,
                            &beta, attn->d_grad_attn_out, attn->d_model));
    
    // Backpropagate through attention for each batch
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* d_attn_output_batch = attn->d_grad_attn_out + batch * attn->seq_len * attn->d_model;
        float* V_batch = attn->d_V + batch * attn->seq_len * attn->d_model;
        float* d_weights_batch = attn->d_grad_weights + batch * attn->seq_len * attn->seq_len;
        float* dV_batch = attn->d_grad_V + batch * attn->seq_len * attn->d_model;
        
        // ∂L/∂V = weights^T * d_attn_output
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                attn->d_model, attn->seq_len, attn->seq_len,
                                &alpha, d_attn_output_batch, attn->d_model,
                                attn->d_attn_weights + batch * attn->seq_len * attn->seq_len, attn->seq_len,
                                &beta, dV_batch, attn->d_model));
        
        // ∂L/∂weights = d_attn_output * V^T
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                attn->seq_len, attn->seq_len, attn->d_model,
                                &alpha, V_batch, attn->d_model,
                                d_attn_output_batch, attn->d_model,
                                &beta, d_weights_batch, attn->seq_len));
    }
    
    // Backpropagate through softmax
    dim3 grid(attn->batch_size);
    dim3 block(attn->seq_len);
    softmax_backward_kernel_attention<<<grid, block>>>(
        attn->d_grad_scores, attn->d_grad_weights, attn->d_attn_weights, attn->batch_size, attn->seq_len
    );
    
    // Backpropagate through attention scores
    float scale = 1.0f / sqrtf(attn->d_model);
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* d_scores_batch = attn->d_grad_scores + batch * attn->seq_len * attn->seq_len;
        float* Q_batch = attn->d_Q + batch * attn->seq_len * attn->d_model;
        float* K_batch = attn->d_K + batch * attn->seq_len * attn->d_model;
        float* dQ_batch = attn->d_grad_Q + batch * attn->seq_len * attn->d_model;
        float* dK_batch = attn->d_grad_K + batch * attn->seq_len * attn->d_model;
        
        // ∂L/∂Q = d_scores * K / sqrt(d_model)
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                attn->d_model, attn->seq_len, attn->seq_len,
                                &scale, K_batch, attn->d_model,
                                d_scores_batch, attn->seq_len,
                                &beta, dQ_batch, attn->d_model));
        
        // ∂L/∂K = d_scores^T * Q / sqrt(d_model)
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                attn->d_model, attn->seq_len, attn->seq_len,
                                &scale, Q_batch, attn->d_model,
                                d_scores_batch, attn->seq_len,
                                &beta, dK_batch, attn->d_model));
    }
    
    // Accumulate weight gradients
    // ∂L/∂W_q = X^T * grad_Q
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            attn->d_model, attn->d_model, total_seq,
                            &alpha, attn->d_grad_Q, attn->d_model,
                            d_X, attn->d_model,
                            &alpha, attn->d_W_q_grad, attn->d_model));
    
    // ∂L/∂W_k = X^T * grad_K
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            attn->d_model, attn->d_model, total_seq,
                            &alpha, attn->d_grad_K, attn->d_model,
                            d_X, attn->d_model,
                            &alpha, attn->d_W_k_grad, attn->d_model));
    
    // ∂L/∂W_v = X^T * grad_V
    CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            attn->d_model, attn->d_model, total_seq,
                            &alpha, attn->d_grad_V, attn->d_model,
                            d_X, attn->d_model,
                            &alpha, attn->d_W_v_grad, attn->d_model));
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
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
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
    fread(&d_model, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    Attention* attn = init_attention(d_model, seq_len, batch_size, cublas_handle);
    
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