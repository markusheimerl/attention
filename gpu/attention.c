#include "attention.h"

// Initialize the attention network
Attention* init_attention(int d_model, int seq_len, int num_layers, int batch_size, cublasHandle_t cublas_handle) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->d_model = d_model;
    attn->seq_len = seq_len;
    attn->num_layers = num_layers;
    attn->batch_size = batch_size;
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    attn->cublas_handle = cublas_handle;
    
    // Allocate arrays of device pointers
    attn->d_W_q = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_k = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_v = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_o = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_q_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_k_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_v_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_o_grad = (float**)malloc(num_layers * sizeof(float*));
    
    attn->d_W_q_m = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_q_v = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_k_m = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_k_v = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_v_m = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_v_v = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_o_m = (float**)malloc(num_layers * sizeof(float*));
    attn->d_W_o_v = (float**)malloc(num_layers * sizeof(float*));
    
    attn->d_Q = (float**)malloc(num_layers * sizeof(float*));
    attn->d_K = (float**)malloc(num_layers * sizeof(float*));
    attn->d_V = (float**)malloc(num_layers * sizeof(float*));
    attn->d_attn_scores = (float**)malloc(num_layers * sizeof(float*));
    attn->d_attn_weights = (float**)malloc(num_layers * sizeof(float*));
    attn->d_attn_output = (float**)malloc(num_layers * sizeof(float*));
    attn->d_layer_output = (float**)malloc(num_layers * sizeof(float*));
    
    attn->d_dQ = (float**)malloc(num_layers * sizeof(float*));
    attn->d_dK = (float**)malloc(num_layers * sizeof(float*));
    attn->d_dV = (float**)malloc(num_layers * sizeof(float*));
    attn->d_d_attn_scores = (float**)malloc(num_layers * sizeof(float*));
    attn->d_d_attn_weights = (float**)malloc(num_layers * sizeof(float*));
    attn->d_d_attn_output = (float**)malloc(num_layers * sizeof(float*));
    attn->d_d_layer_output = (float**)malloc(num_layers * sizeof(float*));
    
    for (int layer = 0; layer < num_layers; layer++) {
        int weight_size = d_model * d_model;
        int qkv_size = batch_size * seq_len * d_model;
        int attn_size = batch_size * seq_len * seq_len;
        
        // Allocate host memory for weight initialization
        float* W_q = (float*)malloc(weight_size * sizeof(float));
        float* W_k = (float*)malloc(weight_size * sizeof(float));
        float* W_v = (float*)malloc(weight_size * sizeof(float));
        float* W_o = (float*)malloc(weight_size * sizeof(float));
        
        // Initialize weights on host
        float scale = 1.0f / sqrtf(d_model);
        for (int i = 0; i < weight_size; i++) {
            W_q[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
            W_k[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
            W_v[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
            W_o[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        }
        
        // Allocate device memory for weights and gradients
        CHECK_CUDA(cudaMalloc(&attn->d_W_q[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_k[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_v[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_o[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_q_grad[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_k_grad[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_v_grad[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_o_grad[layer], weight_size * sizeof(float)));
        
        // Allocate device memory for Adam parameters
        CHECK_CUDA(cudaMalloc(&attn->d_W_q_m[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_q_v[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_k_m[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_k_v[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_v_m[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_v_v[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_o_m[layer], weight_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_W_o_v[layer], weight_size * sizeof(float)));
        
        // Allocate device memory for layer outputs and working buffers
        CHECK_CUDA(cudaMalloc(&attn->d_Q[layer], qkv_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_K[layer], qkv_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_V[layer], qkv_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_attn_scores[layer], attn_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_attn_weights[layer], attn_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_attn_output[layer], qkv_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_layer_output[layer], qkv_size * sizeof(float)));
        
        // Allocate device memory for gradient buffers
        CHECK_CUDA(cudaMalloc(&attn->d_dQ[layer], qkv_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_dK[layer], qkv_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_dV[layer], qkv_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_d_attn_scores[layer], attn_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_d_attn_weights[layer], attn_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_d_attn_output[layer], qkv_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&attn->d_d_layer_output[layer], qkv_size * sizeof(float)));
        
        // Copy weights to device
        CHECK_CUDA(cudaMemcpy(attn->d_W_q[layer], W_q, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_k[layer], W_k, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_v[layer], W_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_o[layer], W_o, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Initialize Adam parameters to zero
        CHECK_CUDA(cudaMemset(attn->d_W_q_m[layer], 0, weight_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_W_q_v[layer], 0, weight_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_W_k_m[layer], 0, weight_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_W_k_v[layer], 0, weight_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_W_v_m[layer], 0, weight_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_W_v_v[layer], 0, weight_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_W_o_m[layer], 0, weight_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_W_o_v[layer], 0, weight_size * sizeof(float)));
        
        // Free host memory
        free(W_q); free(W_k); free(W_v); free(W_o);
    }
    
    return attn;
}

// Free attention network memory
void free_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        // Free device memory
        cudaFree(attn->d_W_q[layer]); cudaFree(attn->d_W_k[layer]); 
        cudaFree(attn->d_W_v[layer]); cudaFree(attn->d_W_o[layer]);
        cudaFree(attn->d_W_q_grad[layer]); cudaFree(attn->d_W_k_grad[layer]); 
        cudaFree(attn->d_W_v_grad[layer]); cudaFree(attn->d_W_o_grad[layer]);
        cudaFree(attn->d_W_q_m[layer]); cudaFree(attn->d_W_q_v[layer]);
        cudaFree(attn->d_W_k_m[layer]); cudaFree(attn->d_W_k_v[layer]);
        cudaFree(attn->d_W_v_m[layer]); cudaFree(attn->d_W_v_v[layer]);
        cudaFree(attn->d_W_o_m[layer]); cudaFree(attn->d_W_o_v[layer]);
        cudaFree(attn->d_Q[layer]); cudaFree(attn->d_K[layer]); cudaFree(attn->d_V[layer]);
        cudaFree(attn->d_attn_scores[layer]); cudaFree(attn->d_attn_weights[layer]);
        cudaFree(attn->d_attn_output[layer]); cudaFree(attn->d_layer_output[layer]);
        cudaFree(attn->d_dQ[layer]); cudaFree(attn->d_dK[layer]); cudaFree(attn->d_dV[layer]);
        cudaFree(attn->d_d_attn_scores[layer]); cudaFree(attn->d_d_attn_weights[layer]);
        cudaFree(attn->d_d_attn_output[layer]); cudaFree(attn->d_d_layer_output[layer]);
    }
    
    free(attn->d_W_q); free(attn->d_W_k); free(attn->d_W_v); free(attn->d_W_o);
    free(attn->d_W_q_grad); free(attn->d_W_k_grad); free(attn->d_W_v_grad); free(attn->d_W_o_grad);
    free(attn->d_W_q_m); free(attn->d_W_q_v); free(attn->d_W_k_m); free(attn->d_W_k_v);
    free(attn->d_W_v_m); free(attn->d_W_v_v); free(attn->d_W_o_m); free(attn->d_W_o_v);
    free(attn->d_Q); free(attn->d_K); free(attn->d_V);
    free(attn->d_attn_scores); free(attn->d_attn_weights);
    free(attn->d_attn_output); free(attn->d_layer_output);
    free(attn->d_dQ); free(attn->d_dK); free(attn->d_dV);
    free(attn->d_d_attn_scores); free(attn->d_d_attn_weights);
    free(attn->d_d_attn_output); free(attn->d_d_layer_output);
    free(attn);
}

// CUDA kernel for softmax forward
__global__ void softmax_forward_kernel_attention(float* weights, float* scores, int batch_size, int seq_len) {
    int batch_idx = blockIdx.x;
    int row_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || row_idx >= seq_len) return;
    
    extern __shared__ float shared_data[];
    
    int offset = batch_idx * seq_len * seq_len + row_idx * seq_len;
    float* row_scores = scores + offset;
    float* row_weights = weights + offset;
    
    // Find maximum
    float local_max = -1e30f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, row_scores[i]);
    }
    shared_data[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_val = shared_data[0];
    __syncthreads();
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float exp_val = expf(row_scores[i] - max_val);
        row_weights[i] = exp_val;
        local_sum += exp_val;
    }
    shared_data[tid] = local_sum;
    __syncthreads();
    
    // Reduce to find global sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_val = shared_data[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < seq_len; i += blockDim.x) {
        row_weights[i] /= sum_val;
    }
}

// CUDA kernel for softmax backward
__global__ void softmax_backward_kernel_attention(float* d_scores, float* d_weights, float* weights, int batch_size, int seq_len) {
    int batch_idx = blockIdx.x;
    int row_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || row_idx >= seq_len) return;
    
    extern __shared__ float shared_sum[];
    
    int offset = batch_idx * seq_len * seq_len + row_idx * seq_len;
    float* row_weights = weights + offset;
    float* row_d_weights = d_weights + offset;
    float* row_d_scores = d_scores + offset;
    
    // Compute sum of d_weights[j] * weights[j]
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_sum += row_d_weights[i] * row_weights[i];
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce to find global sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_val = shared_sum[0];
    __syncthreads();
    
    // Compute softmax backward: d_scores[i] = weights[i] * (d_weights[i] - sum)
    for (int i = tid; i < seq_len; i += blockDim.x) {
        row_d_scores[i] = row_weights[i] * (row_d_weights[i] - sum_val);
    }
}

// Forward pass
void forward_pass_attention(Attention* attn, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    float* input = d_X;
    
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int total_seq = attn->batch_size * attn->seq_len;
        
        // Compute Q, K, V: Q = X * W_q, K = X * W_k, V = X * W_v
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                attn->d_model, total_seq, attn->d_model,
                                &alpha, attn->d_W_q[layer], attn->d_model,
                                input, attn->d_model,
                                &beta, attn->d_Q[layer], attn->d_model));
        
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                attn->d_model, total_seq, attn->d_model,
                                &alpha, attn->d_W_k[layer], attn->d_model,
                                input, attn->d_model,
                                &beta, attn->d_K[layer], attn->d_model));
        
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                attn->d_model, total_seq, attn->d_model,
                                &alpha, attn->d_W_v[layer], attn->d_model,
                                input, attn->d_model,
                                &beta, attn->d_V[layer], attn->d_model));
        
        // Compute attention scores for each sequence in the batch
        float scale = 1.0f / sqrtf(attn->d_model);
        for (int batch = 0; batch < attn->batch_size; batch++) {
            float* Q_batch = attn->d_Q[layer] + batch * attn->seq_len * attn->d_model;
            float* K_batch = attn->d_K[layer] + batch * attn->seq_len * attn->d_model;
            float* scores_batch = attn->d_attn_scores[layer] + batch * attn->seq_len * attn->seq_len;
            
            // scores = Q * K^T / sqrt(d_model)
            CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_T,
                                    attn->seq_len, attn->seq_len, attn->d_model,
                                    &scale, Q_batch, attn->seq_len,
                                    K_batch, attn->seq_len,
                                    &beta, scores_batch, attn->seq_len));
        }
        
        // Apply softmax to attention scores
        dim3 grid(attn->batch_size, attn->seq_len);
        int block_size = (256 < attn->seq_len) ? 256 : attn->seq_len;
        int shared_size = 2 * block_size * sizeof(float);
        softmax_forward_kernel_attention<<<grid, block_size, shared_size>>>(
            attn->d_attn_weights[layer], attn->d_attn_scores[layer], attn->batch_size, attn->seq_len);
        
        // Compute attention output for each sequence in the batch
        for (int batch = 0; batch < attn->batch_size; batch++) {
            float* weights_batch = attn->d_attn_weights[layer] + batch * attn->seq_len * attn->seq_len;
            float* V_batch = attn->d_V[layer] + batch * attn->seq_len * attn->d_model;
            float* output_batch = attn->d_attn_output[layer] + batch * attn->seq_len * attn->d_model;
            
            // attn_output = weights * V
            CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    attn->seq_len, attn->d_model, attn->seq_len,
                                    &alpha, weights_batch, attn->seq_len,
                                    V_batch, attn->seq_len,
                                    &beta, output_batch, attn->seq_len));
        }
        
        // Apply output projection: layer_output = attn_output * W_o
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                attn->d_model, total_seq, attn->d_model,
                                &alpha, attn->d_W_o[layer], attn->d_model,
                                attn->d_attn_output[layer], attn->d_model,
                                &beta, attn->d_layer_output[layer], attn->d_model));
        
        // Set input for next layer
        input = attn->d_layer_output[layer];
    }
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* d_y) {
    int last_layer = attn->num_layers - 1;
    int total_size = attn->batch_size * attn->seq_len * attn->d_model;
    float loss = 0.0f;

    const float alpha = 1.0f;
    const float beta = -1.0f;
    CHECK_CUBLAS(cublasSgeam(attn->cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            attn->d_model, attn->batch_size * attn->seq_len,
                            &alpha, attn->d_layer_output[last_layer], attn->d_model,
                            &beta, d_y, attn->d_model,
                            attn->d_d_layer_output[last_layer], attn->d_model));
    CHECK_CUBLAS(cublasSdot(attn->cublas_handle, total_size, 
                           attn->d_d_layer_output[last_layer], 1, 
                           attn->d_d_layer_output[last_layer], 1, &loss));
    
    return loss / total_size;
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int weight_size = attn->d_model * attn->d_model;
        
        CHECK_CUDA(cudaMemset(attn->d_W_q_grad[layer], 0, weight_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_W_k_grad[layer], 0, weight_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_W_v_grad[layer], 0, weight_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_W_o_grad[layer], 0, weight_size * sizeof(float)));
    }
}

// Backward pass
void backward_pass_attention(Attention* attn, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    for (int layer = attn->num_layers - 1; layer >= 0; layer--) {
        float* input = (layer == 0) ? d_X : attn->d_layer_output[layer - 1];
        int total_seq = attn->batch_size * attn->seq_len;
        
        // Gradient w.r.t. W_o: dW_o = attn_output^T * d_layer_output
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                attn->d_model, attn->d_model, total_seq,
                                &alpha, attn->d_attn_output[layer], attn->d_model,
                                attn->d_d_layer_output[layer], attn->d_model,
                                &alpha, attn->d_W_o_grad[layer], attn->d_model));
        
        // Gradient w.r.t. attention output: d_attn_output = d_layer_output * W_o^T
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                attn->d_model, total_seq, attn->d_model,
                                &alpha, attn->d_W_o[layer], attn->d_model,
                                attn->d_d_layer_output[layer], attn->d_model,
                                &beta, attn->d_d_attn_output[layer], attn->d_model));
        
        // Backpropagate through attention mechanism for each sequence in batch
        CHECK_CUDA(cudaMemset(attn->d_dQ[layer], 0, total_seq * attn->d_model * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_dK[layer], 0, total_seq * attn->d_model * sizeof(float)));
        CHECK_CUDA(cudaMemset(attn->d_dV[layer], 0, total_seq * attn->d_model * sizeof(float)));
        
        for (int batch = 0; batch < attn->batch_size; batch++) {
            float* d_attn_output_batch = attn->d_d_attn_output[layer] + batch * attn->seq_len * attn->d_model;
            float* attn_weights_batch = attn->d_attn_weights[layer] + batch * attn->seq_len * attn->seq_len;
            float* V_batch = attn->d_V[layer] + batch * attn->seq_len * attn->d_model;
            float* Q_batch = attn->d_Q[layer] + batch * attn->seq_len * attn->d_model;
            float* K_batch = attn->d_K[layer] + batch * attn->seq_len * attn->d_model;
            float* d_attn_weights_batch = attn->d_d_attn_weights[layer] + batch * attn->seq_len * attn->seq_len;
            float* dQ_batch = attn->d_dQ[layer] + batch * attn->seq_len * attn->d_model;
            float* dK_batch = attn->d_dK[layer] + batch * attn->seq_len * attn->d_model;
            float* dV_batch = attn->d_dV[layer] + batch * attn->seq_len * attn->d_model;
            
            // Gradient w.r.t. V: dV = attn_weights^T * d_attn_output
            CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    attn->seq_len, attn->d_model, attn->seq_len,
                                    &alpha, attn_weights_batch, attn->seq_len,
                                    d_attn_output_batch, attn->seq_len,
                                    &beta, dV_batch, attn->seq_len));
            
            // Gradient w.r.t. attention weights: d_attn_weights = d_attn_output * V^T
            CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_T,
                                    attn->seq_len, attn->seq_len, attn->d_model,
                                    &alpha, d_attn_output_batch, attn->seq_len,
                                    V_batch, attn->seq_len,
                                    &beta, d_attn_weights_batch, attn->seq_len));
            
            // Backpropagate through softmax
            float* d_attn_scores_batch = attn->d_d_attn_scores[layer] + batch * attn->seq_len * attn->seq_len;
            dim3 grid(1, attn->seq_len);
            int block_size = (256 < attn->seq_len) ? 256 : attn->seq_len;
            int shared_size = block_size * sizeof(float);
            softmax_backward_kernel_attention<<<grid, block_size, shared_size>>>(
                d_attn_scores_batch, d_attn_weights_batch, attn_weights_batch, 1, attn->seq_len);
            
            // Gradient w.r.t. Q: dQ = d_attn_scores * K / sqrt(d_model)
            float scale = 1.0f / sqrtf(attn->d_model);
            CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    attn->seq_len, attn->d_model, attn->seq_len,
                                    &scale, d_attn_scores_batch, attn->seq_len,
                                    K_batch, attn->seq_len,
                                    &beta, dQ_batch, attn->seq_len));
            
            // Gradient w.r.t. K: dK = d_attn_scores^T * Q / sqrt(d_model)
            CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    attn->seq_len, attn->d_model, attn->seq_len,
                                    &scale, d_attn_scores_batch, attn->seq_len,
                                    Q_batch, attn->seq_len,
                                    &beta, dK_batch, attn->seq_len));
        }
        
        // Gradient w.r.t. weight matrices
        // dW_q = input^T * dQ
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                attn->d_model, attn->d_model, total_seq,
                                &alpha, input, attn->d_model,
                                attn->d_dQ[layer], attn->d_model,
                                &alpha, attn->d_W_q_grad[layer], attn->d_model));
        
        // dW_k = input^T * dK
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                attn->d_model, attn->d_model, total_seq,
                                &alpha, input, attn->d_model,
                                attn->d_dK[layer], attn->d_model,
                                &alpha, attn->d_W_k_grad[layer], attn->d_model));
        
        // dW_v = input^T * dV
        CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                attn->d_model, attn->d_model, total_seq,
                                &alpha, input, attn->d_model,
                                attn->d_dV[layer], attn->d_model,
                                &alpha, attn->d_W_v_grad[layer], attn->d_model));
        
        // Propagate gradient to previous layer
        if (layer > 0) {
            // d_input = dQ * W_q^T + dK * W_k^T + dV * W_v^T
            CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    attn->d_model, total_seq, attn->d_model,
                                    &alpha, attn->d_W_q[layer], attn->d_model,
                                    attn->d_dQ[layer], attn->d_model,
                                    &beta, attn->d_d_layer_output[layer - 1], attn->d_model));
            
            CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    attn->d_model, total_seq, attn->d_model,
                                    &alpha, attn->d_W_k[layer], attn->d_model,
                                    attn->d_dK[layer], attn->d_model,
                                    &alpha, attn->d_d_layer_output[layer - 1], attn->d_model));
            
            CHECK_CUBLAS(cublasSgemm(attn->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    attn->d_model, total_seq, attn->d_model,
                                    &alpha, attn->d_W_v[layer], attn->d_model,
                                    attn->d_dV[layer], attn->d_model,
                                    &alpha, attn->d_d_layer_output[layer - 1], attn->d_model));
        }
    }
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_attention(float* weight, float* grad, float* m, float* v,
                                              float beta1, float beta2, float epsilon, float learning_rate,
                                              float weight_decay, float alpha_t, int size, int batch_size, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / (batch_size * seq_len);
        
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
    
    int block_size = 256;
    
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int weight_size = attn->d_model * attn->d_model;
        int num_blocks = (weight_size + block_size - 1) / block_size;
        
        // Update W_q weights
        adamw_update_kernel_attention<<<num_blocks, block_size>>>(
            attn->d_W_q[layer], attn->d_W_q_grad[layer], attn->d_W_q_m[layer], attn->d_W_q_v[layer],
            attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
            alpha_t, weight_size, attn->batch_size, attn->seq_len
        );
        
        // Update W_k weights
        adamw_update_kernel_attention<<<num_blocks, block_size>>>(
            attn->d_W_k[layer], attn->d_W_k_grad[layer], attn->d_W_k_m[layer], attn->d_W_k_v[layer],
            attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
            alpha_t, weight_size, attn->batch_size, attn->seq_len
        );
        
        // Update W_v weights
        adamw_update_kernel_attention<<<num_blocks, block_size>>>(
            attn->d_W_v[layer], attn->d_W_v_grad[layer], attn->d_W_v_m[layer], attn->d_W_v_v[layer],
            attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
            alpha_t, weight_size, attn->batch_size, attn->seq_len
        );
        
        // Update W_o weights
        adamw_update_kernel_attention<<<num_blocks, block_size>>>(
            attn->d_W_o[layer], attn->d_W_o_grad[layer], attn->d_W_o_m[layer], attn->d_W_o_v[layer],
            attn->beta1, attn->beta2, attn->epsilon, learning_rate, attn->weight_decay,
            alpha_t, weight_size, attn->batch_size, attn->seq_len
        );
    }
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
    fwrite(&attn->num_layers, sizeof(int), 1, file);
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    
    // Save weights for each layer
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int weight_size = attn->d_model * attn->d_model;
        
        // Allocate temporary host memory
        float* h_W_q = (float*)malloc(weight_size * sizeof(float));
        float* h_W_k = (float*)malloc(weight_size * sizeof(float));
        float* h_W_v = (float*)malloc(weight_size * sizeof(float));
        float* h_W_o = (float*)malloc(weight_size * sizeof(float));
        
        // Copy weights from device to host
        CHECK_CUDA(cudaMemcpy(h_W_q, attn->d_W_q[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_W_k, attn->d_W_k[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_W_v, attn->d_W_v[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_W_o, attn->d_W_o[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        fwrite(h_W_q, sizeof(float), weight_size, file);
        fwrite(h_W_k, sizeof(float), weight_size, file);
        fwrite(h_W_v, sizeof(float), weight_size, file);
        fwrite(h_W_o, sizeof(float), weight_size, file);
        
        free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    }
    
    // Save Adam state
    fwrite(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int weight_size = attn->d_model * attn->d_model;
        
        float* W_q_m = (float*)malloc(weight_size * sizeof(float));
        float* W_q_v = (float*)malloc(weight_size * sizeof(float));
        float* W_k_m = (float*)malloc(weight_size * sizeof(float));
        float* W_k_v = (float*)malloc(weight_size * sizeof(float));
        float* W_v_m = (float*)malloc(weight_size * sizeof(float));
        float* W_v_v = (float*)malloc(weight_size * sizeof(float));
        float* W_o_m = (float*)malloc(weight_size * sizeof(float));
        float* W_o_v = (float*)malloc(weight_size * sizeof(float));
        
        CHECK_CUDA(cudaMemcpy(W_q_m, attn->d_W_q_m[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W_q_v, attn->d_W_q_v[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W_k_m, attn->d_W_k_m[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W_k_v, attn->d_W_k_v[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W_v_m, attn->d_W_v_m[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W_v_v, attn->d_W_v_v[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W_o_m, attn->d_W_o_m[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W_o_v, attn->d_W_o_v[layer], weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        fwrite(W_q_m, sizeof(float), weight_size, file);
        fwrite(W_q_v, sizeof(float), weight_size, file);
        fwrite(W_k_m, sizeof(float), weight_size, file);
        fwrite(W_k_v, sizeof(float), weight_size, file);
        fwrite(W_v_m, sizeof(float), weight_size, file);
        fwrite(W_v_v, sizeof(float), weight_size, file);
        fwrite(W_o_m, sizeof(float), weight_size, file);
        fwrite(W_o_v, sizeof(float), weight_size, file);
        
        free(W_q_m); free(W_q_v);
        free(W_k_m); free(W_k_v);
        free(W_v_m); free(W_v_v);
        free(W_o_m); free(W_o_v);
    }

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
    int d_model, seq_len, num_layers, stored_batch_size;
    fread(&d_model, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    Attention* attn = init_attention(d_model, seq_len, num_layers, batch_size, cublas_handle);
    
    // Load weights for each layer
    for (int layer = 0; layer < num_layers; layer++) {
        int weight_size = d_model * d_model;
        
        float* h_W_q = (float*)malloc(weight_size * sizeof(float));
        float* h_W_k = (float*)malloc(weight_size * sizeof(float));
        float* h_W_v = (float*)malloc(weight_size * sizeof(float));
        float* h_W_o = (float*)malloc(weight_size * sizeof(float));
        
        fread(h_W_q, sizeof(float), weight_size, file);
        fread(h_W_k, sizeof(float), weight_size, file);
        fread(h_W_v, sizeof(float), weight_size, file);
        fread(h_W_o, sizeof(float), weight_size, file);
        
        CHECK_CUDA(cudaMemcpy(attn->d_W_q[layer], h_W_q, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_k[layer], h_W_k, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_v[layer], h_W_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_o[layer], h_W_o, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        
        free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    }
    
    // Load Adam state
    fread(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < num_layers; layer++) {
        int weight_size = d_model * d_model;
        
        float* W_q_m = (float*)malloc(weight_size * sizeof(float));
        float* W_q_v = (float*)malloc(weight_size * sizeof(float));
        float* W_k_m = (float*)malloc(weight_size * sizeof(float));
        float* W_k_v = (float*)malloc(weight_size * sizeof(float));
        float* W_v_m = (float*)malloc(weight_size * sizeof(float));
        float* W_v_v = (float*)malloc(weight_size * sizeof(float));
        float* W_o_m = (float*)malloc(weight_size * sizeof(float));
        float* W_o_v = (float*)malloc(weight_size * sizeof(float));
        
        fread(W_q_m, sizeof(float), weight_size, file);
        fread(W_q_v, sizeof(float), weight_size, file);
        fread(W_k_m, sizeof(float), weight_size, file);
        fread(W_k_v, sizeof(float), weight_size, file);
        fread(W_v_m, sizeof(float), weight_size, file);
        fread(W_v_v, sizeof(float), weight_size, file);
        fread(W_o_m, sizeof(float), weight_size, file);
        fread(W_o_v, sizeof(float), weight_size, file);
        
        CHECK_CUDA(cudaMemcpy(attn->d_W_q_m[layer], W_q_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_q_v[layer], W_q_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_k_m[layer], W_k_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_k_v[layer], W_k_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_v_m[layer], W_v_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_v_v[layer], W_v_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_o_m[layer], W_o_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_W_o_v[layer], W_o_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        
        free(W_q_m); free(W_q_v);
        free(W_k_m); free(W_k_v);
        free(W_v_m); free(W_v_v);
        free(W_o_m); free(W_o_v);
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return attn;
}