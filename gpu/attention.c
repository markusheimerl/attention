#include "attention.h"

// RoPE forward kernel - applies rotary position embeddings
__global__ static void rope_forward_kernel(float* Q, float* K, int batch_size, int seq_len, int num_heads, int head_dim) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d_pair = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d_pair >= head_dim / 2) return;
    
    int d_model = num_heads * head_dim;
    int d = d_pair * 2;
    
    // Compute rotation angle
    float theta = powf(10000.0f, -((float)d / (float)head_dim));
    float angle = (float)t * theta;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    
    // Apply to all heads
    for (int h = 0; h < num_heads; h++) {
        int idx = b * seq_len * d_model + t * d_model + h * head_dim + d;
        
        // Rotate Q
        float q0 = Q[idx];
        float q1 = Q[idx + 1];
        Q[idx] = q0 * cos_a - q1 * sin_a;
        Q[idx + 1] = q0 * sin_a + q1 * cos_a;
        
        // Rotate K
        float k0 = K[idx];
        float k1 = K[idx + 1];
        K[idx] = k0 * cos_a - k1 * sin_a;
        K[idx + 1] = k0 * sin_a + k1 * cos_a;
    }
}

// RoPE backward kernel - applies inverse rotation to gradients
__global__ static void rope_backward_kernel(float* grad_Q, float* grad_K, int batch_size, int seq_len, int num_heads, int head_dim) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d_pair = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d_pair >= head_dim / 2) return;
    
    int d_model = num_heads * head_dim;
    int d = d_pair * 2;
    
    // Compute rotation angle
    float theta = powf(10000.0f, -((float)d / (float)head_dim));
    float angle = (float)t * theta;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    
    // Apply inverse rotation to all heads
    for (int h = 0; h < num_heads; h++) {
        int idx = b * seq_len * d_model + t * d_model + h * head_dim + d;
        
        // Inverse rotate grad_Q
        float gq0 = grad_Q[idx];
        float gq1 = grad_Q[idx + 1];
        grad_Q[idx] = gq0 * cos_a + gq1 * sin_a;
        grad_Q[idx + 1] = -gq0 * sin_a + gq1 * cos_a;
        
        // Inverse rotate grad_K
        float gk0 = grad_K[idx];
        float gk1 = grad_K[idx + 1];
        grad_K[idx] = gk0 * cos_a + gk1 * sin_a;
        grad_K[idx + 1] = -gk0 * sin_a + gk1 * cos_a;
    }
}

// Reshape from [B, L, H, d] to [B, H, L, d]
__global__ static void reshape_to_heads_kernel(float* out, const float* in, int B, int L, int H, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * L * d;
    
    if (idx < total) {
        // Output index: [B, H, L, d]
        int d_idx = idx % d;
        int l_idx = (idx / d) % L;
        int h_idx = (idx / (d * L)) % H;
        int b_idx = idx / (d * L * H);
        
        // Input index: [B, L, H, d]
        int in_idx = b_idx * (L * H * d) + l_idx * (H * d) + h_idx * d + d_idx;
        out[idx] = in[in_idx];
    }
}

// Reshape from [B, H, L, d] to [B, L, H, d]
__global__ static void reshape_from_heads_kernel(float* out, const float* in, int B, int L, int H, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * L * H * d;
    
    if (idx < total) {
        // Output index: [B, L, H, d]
        int d_idx = idx % d;
        int h_idx = (idx / d) % H;
        int l_idx = (idx / (d * H)) % L;
        int b_idx = idx / (d * H * L);
        
        // Input index: [B, H, L, d]
        int in_idx = b_idx * (H * L * d) + h_idx * (L * d) + l_idx * d + d_idx;
        out[idx] = in[in_idx];
    }
}

// CUDA kernel for softmax forward pass
__global__ static void softmax_forward_kernel_attention(float* weights, float* scores, int batch_count, int seq_len) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    
    if (b >= batch_count || i >= seq_len) return;
    
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

// CUDA kernel for causal softmax forward pass
__global__ static void softmax_causal_forward_kernel_attention(float* weights, float* scores, int batch_count, int seq_len) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    
    if (b >= batch_count || i >= seq_len) return;
    
    float* scores_b = &scores[b * seq_len * seq_len];
    float* weights_b = &weights[b * seq_len * seq_len];
    
    // Find max for numerical stability (only consider positions <= i)
    float max_val = -1e30f;
    for (int j = 0; j <= i; j++) {
        float val = scores_b[i * seq_len + j];
        if (val > max_val) max_val = val;
    }
    
    // Compute softmax for j <= i, 0 for j > i
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        if (j <= i) {
            float exp_val = expf(scores_b[i * seq_len + j] - max_val);
            weights_b[i * seq_len + j] = exp_val;
            sum_exp += exp_val;
        } else {
            weights_b[i * seq_len + j] = 0.0f;
        }
    }
    
    // Normalize
    for (int j = 0; j <= i; j++) {
        weights_b[i * seq_len + j] /= sum_exp;
    }
}

// CUDA kernel for softmax backward pass
__global__ static void softmax_backward_kernel_attention(float* grad_scores, float* grad_weights, float* weights, int batch_count, int seq_len) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    
    if (b >= batch_count || i >= seq_len) return;
    
    float* grad_weights_b = &grad_weights[b * seq_len * seq_len];
    float* weights_b = &weights[b * seq_len * seq_len];
    float* grad_scores_b = &grad_scores[b * seq_len * seq_len];
    
    // Compute sum term
    float sum_term = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        int idx = i * seq_len + k;
        sum_term += grad_weights_b[idx] * weights_b[idx];
    }
    
    // Gradient through softmax
    for (int j = 0; j < seq_len; j++) {
        int idx = i * seq_len + j;
        grad_scores_b[idx] = weights_b[idx] * (grad_weights_b[idx] - sum_term);
    }
}

// CUDA kernel for causal softmax backward pass
__global__ static void softmax_causal_backward_kernel_attention(float* grad_scores, float* grad_weights, float* weights, int batch_count, int seq_len) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    
    if (b >= batch_count || i >= seq_len) return;
    
    float* grad_weights_b = &grad_weights[b * seq_len * seq_len];
    float* weights_b = &weights[b * seq_len * seq_len];
    float* grad_scores_b = &grad_scores[b * seq_len * seq_len];
    
    // Compute sum term (only for j <= i)
    float sum_term = 0.0f;
    for (int k = 0; k <= i; k++) {
        int idx = i * seq_len + k;
        sum_term += grad_weights_b[idx] * weights_b[idx];
    }
    
    // Gradient through softmax
    for (int j = 0; j < seq_len; j++) {
        int idx = i * seq_len + j;
        if (j <= i) {
            grad_scores_b[idx] = weights_b[idx] * (grad_weights_b[idx] - sum_term);
        } else {
            grad_scores_b[idx] = 0.0f;
        }
    }
}

// Initialize the attention layer
Attention* init_attention(int seq_len, int d_model, int num_heads, int batch_size, bool is_causal, cublasLtHandle_t cublaslt_handle) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->seq_len = seq_len;
    attn->d_model = d_model;
    attn->num_heads = num_heads;
    attn->head_dim = d_model / num_heads;
    attn->batch_size = batch_size;
    attn->scale = 1.0f / sqrtf(attn->head_dim);
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
    int attn_matrix_size = batch_size * num_heads * seq_len * seq_len;
    
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
    
    // Allocate device memory for forward pass buffers (OPTIMIZED - no duplicates)
    CHECK_CUDA(cudaMalloc(&attn->d_Q, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_K, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_V, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_scores, attn_matrix_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_output, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_output, seq_batch_size * sizeof(float)));
    
    // Allocate device memory for backward pass buffers (OPTIMIZED - no duplicates)
    CHECK_CUDA(cudaMalloc(&attn->d_grad_output, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_attn_output, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_scores, attn_matrix_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_Q, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_K, seq_batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_V, seq_batch_size * sizeof(float)));
    
    // Allocate ONE workspace buffer for reshape operations
    CHECK_CUDA(cudaMalloc(&attn->d_workspace, seq_batch_size * sizeof(float)));
    
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
    
    // Create cuBLASLt matrix multiplication descriptor
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&attn->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    
    // Row-major layout order
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    
    // Create matrix layout descriptors
    // 1. Weight matrices [d_model x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->weight_layout, CUDA_R_32F, d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // 2. Flattened sequence data [batch_size * seq_len x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->seq_flat_layout, CUDA_R_32F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_flat_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // 3. Head layout [seq_len x head_dim] - batched by batch_size * num_heads
    int head_batch_count = batch_size * num_heads;
    int64_t head_batch_stride = seq_len * attn->head_dim;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->head_layout, CUDA_R_32F, seq_len, attn->head_dim, attn->head_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &head_batch_count, sizeof(head_batch_count)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &head_batch_stride, sizeof(head_batch_stride)));
    
    // 4. Attention head layout [seq_len x seq_len] - batched by batch_size * num_heads
    int64_t attn_head_batch_stride = seq_len * seq_len;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->attn_head_layout, CUDA_R_32F, seq_len, seq_len, seq_len));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->attn_head_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->attn_head_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &head_batch_count, sizeof(head_batch_count)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->attn_head_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &attn_head_batch_stride, sizeof(attn_head_batch_stride)));
    
    // Free host memory
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    
    return attn;
}

// Free attention memory
void free_attention(Attention* attn) {
    // Destroy cuBLASLt descriptor
    cublasLtMatmulDescDestroy(attn->matmul_desc);
    
    // Destroy matrix layouts
    cublasLtMatrixLayoutDestroy(attn->weight_layout);
    cublasLtMatrixLayoutDestroy(attn->seq_flat_layout);
    cublasLtMatrixLayoutDestroy(attn->head_layout);
    cublasLtMatrixLayoutDestroy(attn->attn_head_layout);
    
    // Free device memory
    cudaFree(attn->d_W_q); cudaFree(attn->d_W_k); cudaFree(attn->d_W_v); cudaFree(attn->d_W_o);
    cudaFree(attn->d_W_q_grad); cudaFree(attn->d_W_k_grad); cudaFree(attn->d_W_v_grad); cudaFree(attn->d_W_o_grad);
    cudaFree(attn->d_W_q_m); cudaFree(attn->d_W_q_v); cudaFree(attn->d_W_k_m); cudaFree(attn->d_W_k_v);
    cudaFree(attn->d_W_v_m); cudaFree(attn->d_W_v_v); cudaFree(attn->d_W_o_m); cudaFree(attn->d_W_o_v);
    cudaFree(attn->d_Q); cudaFree(attn->d_K); cudaFree(attn->d_V);
    cudaFree(attn->d_scores);
    cudaFree(attn->d_attn_output); cudaFree(attn->d_output);
    cudaFree(attn->d_grad_output); cudaFree(attn->d_grad_attn_output);
    cudaFree(attn->d_grad_scores);
    cudaFree(attn->d_grad_Q); cudaFree(attn->d_grad_K); cudaFree(attn->d_grad_V);
    cudaFree(attn->d_workspace);
    
    // Free loss computation buffer
    cudaFree(attn->d_loss_result);
    
    free(attn);
}

// Forward pass
void forward_pass_attention(Attention* attn, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int total_elements = attn->batch_size * attn->seq_len * attn->d_model;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Step 1: Compute Q, K, V (layout: [B,L,D])
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_W_q, attn->weight_layout,
              &beta, attn->d_Q, attn->seq_flat_layout);
    
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_W_k, attn->weight_layout,
              &beta, attn->d_K, attn->seq_flat_layout);
    
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_W_v, attn->weight_layout,
              &beta, attn->d_V, attn->seq_flat_layout);
    
    // Step 2: Apply RoPE to Q and K (while still in [B,L,D] layout)
    dim3 rope_grid(attn->batch_size, attn->seq_len);
    int rope_threads = attn->head_dim / 2;
    rope_forward_kernel<<<rope_grid, rope_threads>>>(attn->d_Q, attn->d_K, attn->batch_size, attn->seq_len, attn->num_heads, attn->head_dim);
    
    // Step 3: Reshape Q, K, V from [B,L,D] to [B,H,L,d] using workspace
    reshape_to_heads_kernel<<<num_blocks, block_size>>>(
        attn->d_workspace, attn->d_Q, attn->batch_size, attn->seq_len, attn->num_heads, attn->head_dim);
    CHECK_CUDA(cudaMemcpy(attn->d_Q, attn->d_workspace, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    reshape_to_heads_kernel<<<num_blocks, block_size>>>(
        attn->d_workspace, attn->d_K, attn->batch_size, attn->seq_len, attn->num_heads, attn->head_dim);
    CHECK_CUDA(cudaMemcpy(attn->d_K, attn->d_workspace, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    reshape_to_heads_kernel<<<num_blocks, block_size>>>(
        attn->d_workspace, attn->d_V, attn->batch_size, attn->seq_len, attn->num_heads, attn->head_dim);
    CHECK_CUDA(cudaMemcpy(attn->d_V, attn->d_workspace, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 4: Compute attention scores (Q now has layout [B,H,L,d])
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &attn->scale,
              attn->d_Q, attn->head_layout,
              attn->d_K, attn->head_layout,
              &beta, attn->d_scores, attn->attn_head_layout);
    
    // Step 5: Apply softmax (in-place: scores become weights)
    int batch_count = attn->batch_size * attn->num_heads;
    dim3 grid(batch_count, attn->seq_len);
    if (attn->is_causal) {
        softmax_causal_forward_kernel_attention<<<grid, 1>>>(attn->d_scores, attn->d_scores, batch_count, attn->seq_len);
    } else {
        softmax_forward_kernel_attention<<<grid, 1>>>(attn->d_scores, attn->d_scores, batch_count, attn->seq_len);
    }
    
    // Step 6: Compute attention output (layout: [B,H,L,d])
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              attn->d_scores, attn->attn_head_layout,
              attn->d_V, attn->head_layout,
              &beta, attn->d_attn_output, attn->head_layout);
    
    // Step 7: Reshape attention output from [B,H,L,d] to [B,L,D]
    reshape_from_heads_kernel<<<num_blocks, block_size>>>(
        attn->d_workspace, attn->d_attn_output, attn->batch_size, attn->seq_len, attn->num_heads, attn->head_dim);
    CHECK_CUDA(cudaMemcpy(attn->d_attn_output, attn->d_workspace, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 8: Apply output projection
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              attn->d_attn_output, attn->seq_flat_layout,
              attn->d_W_o, attn->weight_layout,
              &beta, attn->d_output, attn->seq_flat_layout);
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
    
    int total_elements = attn->batch_size * attn->seq_len * attn->d_model;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Step 8 (backward): Gradient through output projection
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              attn->d_attn_output, attn->seq_flat_layout,
              attn->d_grad_output, attn->seq_flat_layout,
              &beta, attn->d_W_o_grad, attn->weight_layout);
    
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              attn->d_grad_output, attn->seq_flat_layout,
              attn->d_W_o, attn->weight_layout,
              &beta, attn->d_grad_attn_output, attn->seq_flat_layout);
    
    // Step 7 (backward): Reshape gradient from [B,L,D] to [B,H,L,d]
    reshape_to_heads_kernel<<<num_blocks, block_size>>>(
        attn->d_workspace, attn->d_grad_attn_output, attn->batch_size, attn->seq_len, attn->num_heads, attn->head_dim);
    CHECK_CUDA(cudaMemcpy(attn->d_grad_attn_output, attn->d_workspace, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 6 (backward): Gradient through attention output computation
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              attn->d_grad_attn_output, attn->head_layout,
              attn->d_V, attn->head_layout,
              &beta, attn->d_grad_scores, attn->attn_head_layout);
    
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              attn->d_scores, attn->attn_head_layout,
              attn->d_grad_attn_output, attn->head_layout,
              &beta, attn->d_grad_V, attn->head_layout);
    
    // Step 5 (backward): Gradient through softmax (in-place)
    int batch_count = attn->batch_size * attn->num_heads;
    dim3 grid(batch_count, attn->seq_len);
    if (attn->is_causal) {
        softmax_causal_backward_kernel_attention<<<grid, 1>>>(attn->d_grad_scores, attn->d_grad_scores, attn->d_scores, batch_count, attn->seq_len);
    } else {
        softmax_backward_kernel_attention<<<grid, 1>>>(attn->d_grad_scores, attn->d_grad_scores, attn->d_scores, batch_count, attn->seq_len);
    }
    
    // Step 4 (backward): Gradient through attention scores
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &attn->scale,
              attn->d_grad_scores, attn->attn_head_layout,
              attn->d_K, attn->head_layout,
              &beta, attn->d_grad_Q, attn->head_layout);
    
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &attn->scale,
              attn->d_grad_scores, attn->attn_head_layout,
              attn->d_Q, attn->head_layout,
              &beta, attn->d_grad_K, attn->head_layout);
    
    // Step 3 (backward): Reshape gradients from [B,H,L,d] to [B,L,D]
    reshape_from_heads_kernel<<<num_blocks, block_size>>>(
        attn->d_workspace, attn->d_grad_Q, attn->batch_size, attn->seq_len, attn->num_heads, attn->head_dim);
    CHECK_CUDA(cudaMemcpy(attn->d_grad_Q, attn->d_workspace, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    reshape_from_heads_kernel<<<num_blocks, block_size>>>(
        attn->d_workspace, attn->d_grad_K, attn->batch_size, attn->seq_len, attn->num_heads, attn->head_dim);
    CHECK_CUDA(cudaMemcpy(attn->d_grad_K, attn->d_workspace, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    reshape_from_heads_kernel<<<num_blocks, block_size>>>(
        attn->d_workspace, attn->d_grad_V, attn->batch_size, attn->seq_len, attn->num_heads, attn->head_dim);
    CHECK_CUDA(cudaMemcpy(attn->d_grad_V, attn->d_workspace, total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 2 (backward): Apply inverse RoPE to gradients (now in [B,L,D] layout)
    dim3 rope_grid(attn->batch_size, attn->seq_len);
    int rope_threads = attn->head_dim / 2;
    rope_backward_kernel<<<rope_grid, rope_threads>>>(attn->d_grad_Q, attn->d_grad_K, attn->batch_size, attn->seq_len, attn->num_heads, attn->head_dim);
    
    // Step 1 (backward): Gradient through linear projections
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_grad_Q, attn->seq_flat_layout,
              &beta, attn->d_W_q_grad, attn->weight_layout);
    
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_grad_K, attn->seq_flat_layout,
              &beta, attn->d_W_k_grad, attn->weight_layout);
    
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, attn->seq_flat_layout,
              attn->d_grad_V, attn->seq_flat_layout,
              &beta, attn->d_W_v_grad, attn->weight_layout);
    
    if (d_grad_X != NULL) {
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_Q, attn->seq_flat_layout,
                  attn->d_W_q, attn->weight_layout,
                  &beta, d_grad_X, attn->seq_flat_layout);
        
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_K, attn->seq_flat_layout,
                  attn->d_W_k, attn->weight_layout,
                  &alpha, d_grad_X, attn->seq_flat_layout);
        
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_V, attn->seq_flat_layout,
                  attn->d_W_v, attn->weight_layout,
                  &alpha, d_grad_X, attn->seq_flat_layout);
    }
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel_attention(float* weight, float* grad, float* m, float* v,
                                             float beta1, float beta2, float epsilon, float learning_rate,
                                             float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
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