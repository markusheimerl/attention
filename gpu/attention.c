#include "attention.h"

// Initialize the attention layer
Attention* init_attention(int seq_len, int d_model, int num_heads,
                          int batch_size, bool is_causal, bool use_rope,
                          cublasLtHandle_t cublaslt_handle) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    if (!attn) {
        fprintf(stderr, "Failed to allocate Attention struct\n");
        exit(EXIT_FAILURE);
    }

    if (num_heads <= 0 || d_model % num_heads != 0) {
        fprintf(stderr, "d_model (%d) must be divisible by num_heads (%d)\n",
                d_model, num_heads);
        exit(EXIT_FAILURE);
    }

    // Store dimensions
    attn->seq_len    = seq_len;
    attn->d_model    = d_model;
    attn->batch_size = batch_size;
    attn->num_heads  = num_heads;
    attn->head_dim   = d_model / num_heads;
    attn->scale      = 1.0f / sqrtf((float)attn->head_dim);
    attn->is_causal  = is_causal;
    attn->use_rope   = use_rope;

    // AdamW parameters
    attn->beta1        = 0.9f;
    attn->beta2        = 0.999f;
    attn->epsilon      = 1e-8f;
    attn->t            = 0;
    attn->weight_decay = 0.01f;

    // cuBLASLt handle
    attn->cublaslt_handle = cublaslt_handle;

    size_t weight_size      = (size_t)d_model * d_model;
    size_t seq_batch_size   = (size_t)batch_size * seq_len * d_model;
    size_t attn_matrix_size = (size_t)batch_size * num_heads * seq_len * seq_len;

    // Host buffers for weight init
    half* h_W_q = (half*)malloc(weight_size * sizeof(half));
    half* h_W_k = (half*)malloc(weight_size * sizeof(half));
    half* h_W_v = (half*)malloc(weight_size * sizeof(half));
    half* h_W_o = (half*)malloc(weight_size * sizeof(half));
    if (!h_W_q || !h_W_k || !h_W_v || !h_W_o) {
        fprintf(stderr, "Failed to allocate host weight buffers\n");
        exit(EXIT_FAILURE);
    }

    float scale_W = 1.0f / sqrtf((float)d_model);
    for (size_t i = 0; i < weight_size; i++) {
        float r_q = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
        float r_k = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
        float r_v = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
        float r_o = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
        h_W_q[i] = __float2half(r_q);
        h_W_k[i] = __float2half(r_k);
        h_W_v[i] = __float2half(r_v);
        h_W_o[i] = __float2half(r_o);
    }

    // Device weights and gradients
    CHECK_CUDA(cudaMalloc(&attn->d_W_q,      weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k,      weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v,      weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o,      weight_size * sizeof(half)));

    CHECK_CUDA(cudaMalloc(&attn->d_W_q_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_grad, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_grad, weight_size * sizeof(half)));

    // Adam buffers
    CHECK_CUDA(cudaMalloc(&attn->d_W_q_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_q_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_k_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_v_v, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_W_o_v, weight_size * sizeof(float)));

    // Forward buffers
    CHECK_CUDA(cudaMalloc(&attn->d_Q,            seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_K,            seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_V,            seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_scores,       attn_matrix_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_weights, attn_matrix_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_attn_output,  seq_batch_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&attn->d_output,       seq_batch_size * sizeof(half)));

    // Backward buffers (aliases where possible)
    attn->d_grad_output      = attn->d_output;
    attn->d_grad_attn_output = attn->d_attn_output;
    CHECK_CUDA(cudaMalloc(&attn->d_grad_weights, attn_matrix_size * sizeof(half)));
    attn->d_grad_scores = attn->d_scores;
    attn->d_grad_Q      = attn->d_attn_output;
    attn->d_grad_K      = attn->d_K;
    attn->d_grad_V      = attn->d_V;

    // Loss buffer
    CHECK_CUDA(cudaMalloc(&attn->d_loss_result, sizeof(float)));

    // Copy weights
    CHECK_CUDA(cudaMemcpy(attn->d_W_q, h_W_q, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k, h_W_k, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v, h_W_v, weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o, h_W_o, weight_size * sizeof(half), cudaMemcpyHostToDevice));

    // Zero Adam moments
    CHECK_CUDA(cudaMemset(attn->d_W_q_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_q_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_v, 0, weight_size * sizeof(float)));

    // cuBLASLt descriptors
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&attn->matmul_desc,
                                            CUBLAS_COMPUTE_32F_FAST_TF32,
                                            CUDA_R_32F));

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    // Weight layout [d_model x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->weight_layout,
                                              CUDA_R_16F,
                                              d_model, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->weight_layout,
                   CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    // Flattened sequence [batch_size * seq_len x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->seq_flat_layout,
                                              CUDA_R_16F,
                                              batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_flat_layout,
                   CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    // Original seq_batch_layout (kept to minimize changes, not used in MH math)
    int64_t seq_batch_stride = (int64_t)seq_len * d_model;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->seq_batch_layout,
                                              CUDA_R_16F,
                                              seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_batch_layout,
                   CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_batch_layout,
                   CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                   &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->seq_batch_layout,
                   CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                   &seq_batch_stride, sizeof(seq_batch_stride)));

    // Head sequence layout [seq_len x head_dim] batched over batch_size
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->head_seq_layout,
                                              CUDA_R_16F,
                                              seq_len, attn->head_dim, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_seq_layout,
                   CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_seq_layout,
                   CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                   &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->head_seq_layout,
                   CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                   &seq_batch_stride, sizeof(seq_batch_stride)));

    // Attention layout [seq_len x seq_len] batched over batch_size
    int64_t attn_batch_stride = (int64_t)seq_len * seq_len;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&attn->attn_batch_layout,
                                              CUDA_R_16F,
                                              seq_len, seq_len, seq_len));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->attn_batch_layout,
                   CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->attn_batch_layout,
                   CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                   &batch_size, sizeof(batch_size)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(attn->attn_batch_layout,
                   CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                   &attn_batch_stride, sizeof(attn_batch_stride)));

    // Free host weight buffers
    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);

    return attn;
}

// Free attention memory
void free_attention(Attention* attn) {
    cublasLtMatmulDescDestroy(attn->matmul_desc);

    cublasLtMatrixLayoutDestroy(attn->weight_layout);
    cublasLtMatrixLayoutDestroy(attn->seq_flat_layout);
    cublasLtMatrixLayoutDestroy(attn->seq_batch_layout);
    cublasLtMatrixLayoutDestroy(attn->attn_batch_layout);
    cublasLtMatrixLayoutDestroy(attn->head_seq_layout);

    cudaFree(attn->d_W_q); cudaFree(attn->d_W_k);
    cudaFree(attn->d_W_v); cudaFree(attn->d_W_o);
    cudaFree(attn->d_W_q_grad); cudaFree(attn->d_W_k_grad);
    cudaFree(attn->d_W_v_grad); cudaFree(attn->d_W_o_grad);
    cudaFree(attn->d_W_q_m); cudaFree(attn->d_W_q_v);
    cudaFree(attn->d_W_k_m); cudaFree(attn->d_W_k_v);
    cudaFree(attn->d_W_v_m); cudaFree(attn->d_W_v_v);
    cudaFree(attn->d_W_o_m); cudaFree(attn->d_W_o_v);
    cudaFree(attn->d_Q); cudaFree(attn->d_K); cudaFree(attn->d_V);
    cudaFree(attn->d_scores); cudaFree(attn->d_attn_weights);
    cudaFree(attn->d_attn_output); cudaFree(attn->d_output);
    cudaFree(attn->d_grad_weights);
    cudaFree(attn->d_loss_result);

    free(attn);
}

// CUDA kernel for softmax forward pass
__global__ static void softmax_forward_kernel_attention(half* weights,
                                                        half* scores,
                                                        int n_matrices,
                                                        int seq_len) {
    int m = blockIdx.x; // matrix index (over num_heads * batch_size)
    int i = blockIdx.y; // row

    if (m >= n_matrices || i >= seq_len) return;

    half* scores_m  = &scores[m * seq_len * seq_len];
    half* weights_m = &weights[m * seq_len * seq_len];

    float max_val = -1e30f;
    for (int j = 0; j < seq_len; j++) {
        float v = __half2float(scores_m[i * seq_len + j]);
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        float e = expf(__half2float(scores_m[i * seq_len + j]) - max_val);
        weights_m[i * seq_len + j] = __float2half(e);
        sum_exp += e;
    }

    for (int j = 0; j < seq_len; j++) {
        weights_m[i * seq_len + j] =
            __float2half(__half2float(weights_m[i * seq_len + j]) / sum_exp);
    }
}

// CUDA kernel for causal softmax forward pass
__global__ static void softmax_causal_forward_kernel_attention(half* weights,
                                                               half* scores,
                                                               int n_matrices,
                                                               int seq_len) {
    int m = blockIdx.x;
    int i = blockIdx.y;

    if (m >= n_matrices || i >= seq_len) return;

    half* scores_m  = &scores[m * seq_len * seq_len];
    half* weights_m = &weights[m * seq_len * seq_len];

    float max_val = -1e30f;
    for (int j = 0; j <= i; j++) {
        float v = __half2float(scores_m[i * seq_len + j]);
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        if (j <= i) {
            float e = expf(__half2float(scores_m[i * seq_len + j]) - max_val);
            weights_m[i * seq_len + j] = __float2half(e);
            sum_exp += e;
        } else {
            weights_m[i * seq_len + j] = __float2half(0.0f);
        }
    }

    for (int j = 0; j <= i; j++) {
        weights_m[i * seq_len + j] =
            __float2half(__half2float(weights_m[i * seq_len + j]) / sum_exp);
    }
}

// CUDA kernel for softmax backward pass
__global__ static void softmax_backward_kernel_attention(half* grad_scores,
                                                         half* grad_weights,
                                                         half* weights,
                                                         int n_matrices,
                                                         int seq_len) {
    int m = blockIdx.x;
    int i = blockIdx.y;

    if (m >= n_matrices || i >= seq_len) return;

    half* grad_weights_m = &grad_weights[m * seq_len * seq_len];
    half* weights_m      = &weights[m * seq_len * seq_len];
    half* grad_scores_m  = &grad_scores[m * seq_len * seq_len];

    float sum_term = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        int idx = i * seq_len + k;
        sum_term += __half2float(grad_weights_m[idx]) *
                    __half2float(weights_m[idx]);
    }

    for (int j = 0; j < seq_len; j++) {
        int idx = i * seq_len + j;
        float result = __half2float(weights_m[idx]) *
                       (__half2float(grad_weights_m[idx]) - sum_term);
        grad_scores_m[idx] = __float2half(result);
    }
}

// CUDA kernel for causal softmax backward pass
__global__ static void softmax_causal_backward_kernel_attention(half* grad_scores,
                                                                half* grad_weights,
                                                                half* weights,
                                                                int n_matrices,
                                                                int seq_len) {
    int m = blockIdx.x;
    int i = blockIdx.y;

    if (m >= n_matrices || i >= seq_len) return;

    half* grad_weights_m = &grad_weights[m * seq_len * seq_len];
    half* weights_m      = &weights[m * seq_len * seq_len];
    half* grad_scores_m  = &grad_scores[m * seq_len * seq_len];

    float sum_term = 0.0f;
    for (int k = 0; k <= i; k++) {
        int idx = i * seq_len + k;
        sum_term += __half2float(grad_weights_m[idx]) *
                    __half2float(weights_m[idx]);
    }

    for (int j = 0; j < seq_len; j++) {
        int idx = i * seq_len + j;
        if (j <= i) {
            float result = __half2float(weights_m[idx]) *
                           (__half2float(grad_weights_m[idx]) - sum_term);
            grad_scores_m[idx] = __float2half(result);
        } else {
            grad_scores_m[idx] = __float2half(0.0f);
        }
    }
}

// CUDA kernel for RoPE forward pass (full d_model, used before split into heads)
__global__ static void rope_forward_kernel_attention(half* Q, half* K,
                                                     int batch_size,
                                                     int seq_len,
                                                     int d_model) {
    int b      = blockIdx.x;
    int t      = blockIdx.y;
    int d_pair = threadIdx.x;

    if (b >= batch_size || t >= seq_len || d_pair >= d_model / 2) return;

    int d = d_pair * 2;

    float theta = powf(10000.0f, -((float)d / (float)d_model));
    float angle = (float)t * theta;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int idx = b * seq_len * d_model + t * d_model + d;

    float q0 = __half2float(Q[idx]);
    float q1 = __half2float(Q[idx + 1]);
    Q[idx]     = __float2half(q0 * cos_a - q1 * sin_a);
    Q[idx + 1] = __float2half(q0 * sin_a + q1 * cos_a);

    float k0 = __half2float(K[idx]);
    float k1 = __half2float(K[idx + 1]);
    K[idx]     = __float2half(k0 * cos_a - k1 * sin_a);
    K[idx + 1] = __float2half(k0 * sin_a + k1 * cos_a);
}

// CUDA kernel for RoPE backward pass
__global__ static void rope_backward_kernel_attention(half* grad_Q, half* grad_K,
                                                      int batch_size,
                                                      int seq_len,
                                                      int d_model) {
    int b      = blockIdx.x;
    int t      = blockIdx.y;
    int d_pair = threadIdx.x;

    if (b >= batch_size || t >= seq_len || d_pair >= d_model / 2) return;

    int d = d_pair * 2;

    float theta = powf(10000.0f, -((float)d / (float)d_model));
    float angle = (float)t * theta;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int idx = b * seq_len * d_model + t * d_model + d;

    float gq0 = __half2float(grad_Q[idx]);
    float gq1 = __half2float(grad_Q[idx + 1]);
    grad_Q[idx]     = __float2half(gq0 * cos_a + gq1 * sin_a);
    grad_Q[idx + 1] = __float2half(-gq0 * sin_a + gq1 * cos_a);

    float gk0 = __half2float(grad_K[idx]);
    float gk1 = __half2float(grad_K[idx + 1]);
    grad_K[idx]     = __float2half(gk0 * cos_a + gk1 * sin_a);
    grad_K[idx + 1] = __float2half(-gk0 * sin_a + gk1 * cos_a);
}

// Forward pass (multi-head)
void forward_pass_attention(Attention* attn, half* d_X) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Step 1: Compute Q, K, V for entire model dimension
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

    // Step 2: RoPE (over full d_model)
    if (attn->use_rope) {
        dim3 grid_rope(attn->batch_size, attn->seq_len);
        rope_forward_kernel_attention<<<grid_rope, attn->d_model / 2>>>(
            attn->d_Q, attn->d_K,
            attn->batch_size, attn->seq_len, attn->d_model
        );
    }

    // Step 3: Attention scores per head, batched over batch_size
    // d_scores layout: [num_heads * batch_size, seq_len, seq_len]
    for (int h = 0; h < attn->num_heads; ++h) {
        half* Q_h = attn->d_Q + h * attn->head_dim;
        half* K_h = attn->d_K + h * attn->head_dim;
        half* S_h = attn->d_scores + (size_t)h * attn->batch_size *
                                    attn->seq_len * attn->seq_len;

        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &attn->scale,
                  Q_h, attn->head_seq_layout,
                  K_h, attn->head_seq_layout,
                  &beta, S_h, attn->attn_batch_layout);
    }

    // Step 4: Softmax over all (head, batch) matrices
    int n_matrices = attn->num_heads * attn->batch_size;
    dim3 grid(n_matrices, attn->seq_len);
    if (attn->is_causal) {
        softmax_causal_forward_kernel_attention<<<grid, 1>>>(
            attn->d_attn_weights, attn->d_scores,
            n_matrices, attn->seq_len
        );
    } else {
        softmax_forward_kernel_attention<<<grid, 1>>>(
            attn->d_attn_weights, attn->d_scores,
            n_matrices, attn->seq_len
        );
    }

    // Step 5: Z_h = A_h V_h per head, batched over batch_size
    for (int h = 0; h < attn->num_heads; ++h) {
        half* A_h = attn->d_attn_weights + (size_t)h * attn->batch_size *
                                        attn->seq_len * attn->seq_len;
        half* V_h = attn->d_V + h * attn->head_dim;
        half* Z_h = attn->d_attn_output + h * attn->head_dim;

        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
                  A_h, attn->attn_batch_layout,
                  V_h, attn->head_seq_layout,
                  &beta, Z_h, attn->head_seq_layout);
    }

    // Step 6: Output projection Y = Z W_o (flattened)
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              attn->d_attn_output, attn->seq_flat_layout,
              attn->d_W_o,         attn->weight_layout,
              &beta, attn->d_output, attn->seq_flat_layout);
}

// CUDA kernel for computing loss and gradient (MSE)
__global__ static void compute_loss_and_gradient_kernel_attention(half* grad_output,
                                                                  half* predictions,
                                                                  half* targets,
                                                                  float* loss_result,
                                                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pred   = __half2float(predictions[idx]);
        float target = __half2float(targets[idx]);
        float diff   = pred - target;
        grad_output[idx] = __float2half(diff);
        atomicAdd(loss_result, diff * diff);
    }
}

// Calculate loss
float calculate_loss_attention(Attention* attn, half* d_y) {
    int total = attn->batch_size * attn->seq_len * attn->d_model;
    int block = 256;
    int grid  = (total + block - 1) / block;

    CHECK_CUDA(cudaMemset(attn->d_loss_result, 0, sizeof(float)));

    compute_loss_and_gradient_kernel_attention<<<grid, block>>>(
        attn->d_grad_output, attn->d_output, d_y,
        attn->d_loss_result, total
    );

    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, attn->d_loss_result,
                          sizeof(float), cudaMemcpyDeviceToHost));
    return total_loss / total;
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    int weight_size = attn->d_model * attn->d_model;
    CHECK_CUDA(cudaMemset(attn->d_W_q_grad, 0, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_grad, 0, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_grad, 0, weight_size * sizeof(half)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_grad, 0, weight_size * sizeof(half)));
}

// Backward pass
void backward_pass_attention(Attention* attn, half* d_X, half* d_grad_X) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Step 6 backward: output projection
    // ∂L/∂W_o = Zᵀ(∂L/∂Y)
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              attn->d_attn_output, attn->seq_flat_layout,
              attn->d_grad_output,  attn->seq_flat_layout,
              &beta, attn->d_W_o_grad, attn->weight_layout);

    // ∂L/∂Z = (∂L/∂Y)W_oᵀ
    LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              attn->d_grad_output,  attn->seq_flat_layout,
              attn->d_W_o,          attn->weight_layout,
              &beta, attn->d_grad_attn_output, attn->seq_flat_layout);

    // Step 5 backward: attention output, per head, batched over batch_size
    for (int h = 0; h < attn->num_heads; ++h) {
        half* grad_Z_h = attn->d_grad_attn_output + h * attn->head_dim;
        half* V_h      = attn->d_V + h * attn->head_dim;
        half* grad_A_h = attn->d_grad_weights + (size_t)h * attn->batch_size *
                                             attn->seq_len * attn->seq_len;
        half* A_h      = attn->d_attn_weights + (size_t)h * attn->batch_size *
                                             attn->seq_len * attn->seq_len;
        half* grad_V_h = attn->d_grad_V + h * attn->head_dim;

        // ∂L/∂A_h = (∂L/∂Z_h)V_hᵀ
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  grad_Z_h, attn->head_seq_layout,
                  V_h,      attn->head_seq_layout,
                  &beta, grad_A_h, attn->attn_batch_layout);

        // ∂L/∂V_h = A_hᵀ(∂L/∂Z_h)
        LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
                  A_h,      attn->attn_batch_layout,
                  grad_Z_h, attn->head_seq_layout,
                  &beta, grad_V_h, attn->head_seq_layout);
    }

    // Step 4 backward: softmax over all (head, batch) matrices
    int n_matrices = attn->num_heads * attn->batch_size;
    dim3 grid(n_matrices, attn->seq_len);
    if (attn->is_causal) {
        softmax_causal_backward_kernel_attention<<<grid, 1>>>(
            attn->d_grad_scores,
            attn->d_grad_weights,
            attn->d_attn_weights,
            n_matrices, attn->seq_len
        );
    } else {
        softmax_backward_kernel_attention<<<grid, 1>>>(
            attn->d_grad_scores,
            attn->d_grad_weights,
            attn->d_attn_weights,
            n_matrices, attn->seq_len
        );
    }

    // Step 3 backward: attention scores per head
    for (int h = 0; h < attn->num_heads; ++h) {
        half* grad_S_h = attn->d_grad_scores + (size_t)h * attn->batch_size *
                                             attn->seq_len * attn->seq_len;
        half* Q_h      = attn->d_Q + h * attn->head_dim;
        half* K_h      = attn->d_K + h * attn->head_dim;
        half* grad_Q_h = attn->d_grad_Q + h * attn->head_dim;
        half* grad_K_h = attn->d_grad_K + h * attn->head_dim;

        // ∂L/∂Q_h = (∂L/∂S_h)K_h / sqrt(head_dim)
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_N, &attn->scale,
                  grad_S_h, attn->attn_batch_layout,
                  K_h,      attn->head_seq_layout,
                  &beta, grad_Q_h, attn->head_seq_layout);

        // ∂L/∂K_h = (∂L/∂S_h)ᵀQ_h / sqrt(head_dim)
        LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &attn->scale,
                  grad_S_h, attn->attn_batch_layout,
                  Q_h,      attn->head_seq_layout,
                  &beta, grad_K_h, attn->head_seq_layout);
    }

    // Step 2 backward: inverse RoPE
    if (attn->use_rope) {
        dim3 grid_rope(attn->batch_size, attn->seq_len);
        rope_backward_kernel_attention<<<grid_rope, attn->d_model / 2>>>(
            attn->d_grad_Q, attn->d_grad_K,
            attn->batch_size, attn->seq_len, attn->d_model
        );
    }

    // Step 1 backward: input projections
    // ∂L/∂W_q = Xᵀ(∂L/∂Q)
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X,          attn->seq_flat_layout,
              attn->d_grad_Q, attn->seq_flat_layout,
              &beta, attn->d_W_q_grad, attn->weight_layout);

    // ∂L/∂W_k = Xᵀ(∂L/∂K)
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X,          attn->seq_flat_layout,
              attn->d_grad_K, attn->seq_flat_layout,
              &beta, attn->d_W_k_grad, attn->weight_layout);

    // ∂L/∂W_v = Xᵀ(∂L/∂V)
    LT_MATMUL(attn, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X,          attn->seq_flat_layout,
              attn->d_grad_V, attn->seq_flat_layout,
              &beta, attn->d_W_v_grad, attn->weight_layout);

    // ∂L/∂X if requested
    if (d_grad_X != NULL) {
        // (∂L/∂Q)W_qᵀ
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_Q, attn->seq_flat_layout,
                  attn->d_W_q,    attn->weight_layout,
                  &beta, d_grad_X, attn->seq_flat_layout);

        // + (∂L/∂K)W_kᵀ
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_K, attn->seq_flat_layout,
                  attn->d_W_k,    attn->weight_layout,
                  &alpha, d_grad_X, attn->seq_flat_layout);

        // + (∂L/∂V)W_vᵀ
        LT_MATMUL(attn, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  attn->d_grad_V, attn->seq_flat_layout,
                  attn->d_W_v,    attn->weight_layout,
                  &alpha, d_grad_X, attn->seq_flat_layout);
    }
}

// AdamW kernel
__global__ static void adamw_update_kernel_attention(half* weight, half* grad,
                                                     float* m, float* v,
                                                     float beta1, float beta2,
                                                     float epsilon,
                                                     float learning_rate,
                                                     float weight_decay,
                                                     float alpha_t,
                                                     int size,
                                                     int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = __half2float(grad[idx]) / batch_size;

        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        float w = __half2float(weight[idx]);
        weight[idx] = __float2half(w * (1.0f - learning_rate * weight_decay) - update);
    }
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate, int batch_size) {
    attn->t++;

    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);

    int weight_size = attn->d_model * attn->d_model;
    int block       = 256;
    int grid        = (weight_size + block - 1) / block;

    half*  weights[]  = {attn->d_W_q, attn->d_W_k, attn->d_W_v, attn->d_W_o};
    half*  grads[]    = {attn->d_W_q_grad, attn->d_W_k_grad,
                         attn->d_W_v_grad, attn->d_W_o_grad};
    float* m_arr[]    = {attn->d_W_q_m, attn->d_W_k_m,
                         attn->d_W_v_m, attn->d_W_o_m};
    float* v_arr[]    = {attn->d_W_q_v, attn->d_W_k_v,
                         attn->d_W_v_v, attn->d_W_o_v};

    for (int i = 0; i < 4; i++) {
        adamw_update_kernel_attention<<<grid, block>>>(
            weights[i], grads[i], m_arr[i], v_arr[i],
            attn->beta1, attn->beta2, attn->epsilon,
            learning_rate, attn->weight_decay,
            alpha_t, weight_size, batch_size
        );
    }
}

// Reset optimizer state
void reset_optimizer_attention(Attention* attn) {
    int weight_size = attn->d_model * attn->d_model;

    CHECK_CUDA(cudaMemset(attn->d_W_q_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_q_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_k_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_v_v, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(attn->d_W_o_v, 0, weight_size * sizeof(float)));

    attn->t = 0;
}

// Serialize attention to file (keeps same format as before; num_heads is not stored)
void serialize_attention(Attention* attn, FILE* file) {
    fwrite(&attn->d_model,   sizeof(int),  1, file);
    fwrite(&attn->is_causal, sizeof(bool), 1, file);
    fwrite(&attn->use_rope,  sizeof(bool), 1, file);

    int weight_size = attn->d_model * attn->d_model;

    float* h_W_q = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o = (float*)malloc(weight_size * sizeof(float));

    CHECK_CUDA(cudaMemcpy(h_W_q, attn->d_W_q,
                          weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k, attn->d_W_k,
                          weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v, attn->d_W_v,
                          weight_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o, attn->d_W_o,
                          weight_size * sizeof(half), cudaMemcpyDeviceToHost));

    for (int i = weight_size - 1; i >= 0; i--) {
        h_W_q[i] = __half2float(((half*)h_W_q)[i]);
        h_W_k[i] = __half2float(((half*)h_W_k)[i]);
        h_W_v[i] = __half2float(((half*)h_W_v)[i]);
        h_W_o[i] = __half2float(((half*)h_W_o)[i]);
    }

    fwrite(h_W_q, sizeof(float), weight_size, file);
    fwrite(h_W_k, sizeof(float), weight_size, file);
    fwrite(h_W_v, sizeof(float), weight_size, file);
    fwrite(h_W_o, sizeof(float), weight_size, file);

    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);

    fwrite(&attn->t, sizeof(int), 1, file);

    float* h_W_q_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_q_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_m = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o_v = (float*)malloc(weight_size * sizeof(float));

    CHECK_CUDA(cudaMemcpy(h_W_q_m, attn->d_W_q_m,
                          weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_q_v, attn->d_W_q_v,
                          weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k_m, attn->d_W_k_m,
                          weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_k_v, attn->d_W_k_v,
                          weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v_m, attn->d_W_v_m,
                          weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_v_v, attn->d_W_v_v,
                          weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o_m, attn->d_W_o_m,
                          weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_o_v, attn->d_W_o_v,
                          weight_size * sizeof(float), cudaMemcpyDeviceToHost));

    fwrite(h_W_q_m, sizeof(float), weight_size, file);
    fwrite(h_W_q_v, sizeof(float), weight_size, file);
    fwrite(h_W_k_m, sizeof(float), weight_size, file);
    fwrite(h_W_k_v, sizeof(float), weight_size, file);
    fwrite(h_W_v_m, sizeof(float), weight_size, file);
    fwrite(h_W_v_v, sizeof(float), weight_size, file);
    fwrite(h_W_o_m, sizeof(float), weight_size, file);
    fwrite(h_W_o_v, sizeof(float), weight_size, file);

    free(h_W_q_m); free(h_W_q_v); free(h_W_k_m); free(h_W_k_v);
    free(h_W_v_m); free(h_W_v_v); free(h_W_o_m); free(h_W_o_v);
}

// Deserialize attention from file
Attention* deserialize_attention(FILE* file, int batch_size, int seq_len,
                                 int num_heads, cublasLtHandle_t cublaslt_handle) {
    int  d_model;
    bool is_causal, use_rope;
    fread(&d_model,   sizeof(int),  1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    fread(&use_rope,  sizeof(bool), 1, file);

    Attention* attn = init_attention(seq_len, d_model, num_heads,
                                     batch_size, is_causal, use_rope,
                                     cublaslt_handle);

    int weight_size = d_model * d_model;

    float* h_W_q = (float*)malloc(weight_size * sizeof(float));
    float* h_W_k = (float*)malloc(weight_size * sizeof(float));
    float* h_W_v = (float*)malloc(weight_size * sizeof(float));
    float* h_W_o = (float*)malloc(weight_size * sizeof(float));

    fread(h_W_q, sizeof(float), weight_size, file);
    fread(h_W_k, sizeof(float), weight_size, file);
    fread(h_W_v, sizeof(float), weight_size, file);
    fread(h_W_o, sizeof(float), weight_size, file);

    for (int i = 0; i < weight_size; i++) {
        ((half*)h_W_q)[i] = __float2half(h_W_q[i]);
        ((half*)h_W_k)[i] = __float2half(h_W_k[i]);
        ((half*)h_W_v)[i] = __float2half(h_W_v[i]);
        ((half*)h_W_o)[i] = __float2half(h_W_o[i]);
    }

    CHECK_CUDA(cudaMemcpy(attn->d_W_q, h_W_q,
                          weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k, h_W_k,
                          weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v, h_W_v,
                          weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o, h_W_o,
                          weight_size * sizeof(half), cudaMemcpyHostToDevice));

    free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);

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

    CHECK_CUDA(cudaMemcpy(attn->d_W_q_m, h_W_q_m,
                          weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_q_v, h_W_q_v,
                          weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k_m, h_W_k_m,
                          weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_k_v, h_W_k_v,
                          weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v_m, h_W_v_m,
                          weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_v_v, h_W_v_v,
                          weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o_m, h_W_o_m,
                          weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_W_o_v, h_W_o_v,
                          weight_size * sizeof(float), cudaMemcpyHostToDevice));

    free(h_W_q_m); free(h_W_q_v); free(h_W_k_m); free(h_W_k_v);
    free(h_W_v_m); free(h_W_v_v); free(h_W_o_m); free(h_W_o_v);

    return attn;
}