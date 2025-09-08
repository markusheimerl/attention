#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLAS Error checking macro
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLASLt Error checking macro
#ifndef CHECK_CUBLASLT
#define CHECK_CUBLASLT(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLASLt error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    // Device weights for attention mechanism
    float* d_W_q;      // Query projection [d_model x d_model]
    float* d_W_k;      // Key projection [d_model x d_model]
    float* d_W_v;      // Value projection [d_model x d_model]
    float* d_W_o;      // Output projection [d_model x d_model]
    
    // Device gradients
    float* d_W_q_grad; // [d_model x d_model]
    float* d_W_k_grad; // [d_model x d_model]
    float* d_W_v_grad; // [d_model x d_model]
    float* d_W_o_grad; // [d_model x d_model]
    
    // Adam parameters
    float* d_W_q_m, *d_W_q_v; // First and second moments for W_q
    float* d_W_k_m, *d_W_k_v; // First and second moments for W_k
    float* d_W_v_m, *d_W_v_v; // First and second moments for W_v
    float* d_W_o_m, *d_W_o_v; // First and second moments for W_o
    float beta1;               // Exponential decay rate for first moment
    float beta2;               // Exponential decay rate for second moment
    float epsilon;             // Small constant for numerical stability
    int t;                     // Time step
    float weight_decay;        // Weight decay parameter for AdamW
    
    // Forward pass buffers
    float* d_Q;            // Query matrix [batch_size x seq_len x d_model]
    float* d_K;            // Key matrix [batch_size x seq_len x d_model]
    float* d_V;            // Value matrix [batch_size x seq_len x d_model]
    float* d_scores;       // Attention scores [batch_size x seq_len x seq_len]
    float* d_attn_weights; // Attention weights [batch_size x seq_len x seq_len]
    float* d_attn_output;  // Attention output [batch_size x seq_len x d_model]
    float* d_output;       // Final output [batch_size x seq_len x d_model]
    
    // Backward pass buffers
    float* d_grad_output;      // [batch_size x seq_len x d_model]
    float* d_grad_attn_output; // [batch_size x seq_len x d_model]
    float* d_grad_weights;     // [batch_size x seq_len x seq_len]
    float* d_grad_scores;      // [batch_size x seq_len x seq_len]
    float* d_grad_Q;           // [batch_size x seq_len x d_model]
    float* d_grad_K;           // [batch_size x seq_len x d_model]
    float* d_grad_V;           // [batch_size x seq_len x d_model]

    // cuBLAS and cuBLASLt handles
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    
    // cuBLASLt descriptors and layouts
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatmulDesc_t matmul_NT_desc;  // No transpose A, transpose B
    cublasLtMatmulDesc_t matmul_TN_desc;  // Transpose A, no transpose B
    
    // Matrix layouts for weight operations
    cublasLtMatrixLayout_t W_layout;      // Weight matrices [d_model x d_model]
    cublasLtMatrixLayout_t seq_layout;    // Sequence data [seq_len x d_model]
    cublasLtMatrixLayout_t W_grad_layout; // Weight gradients [d_model x d_model]
    
    // Matrix layouts for attention operations
    cublasLtMatrixLayout_t Q_layout;      // Query [seq_len x d_model] batched
    cublasLtMatrixLayout_t K_layout;      // Key [seq_len x d_model] batched
    cublasLtMatrixLayout_t V_layout;      // Value [seq_len x d_model] batched
    cublasLtMatrixLayout_t scores_layout; // Attention scores [seq_len x seq_len] batched
    cublasLtMatrixLayout_t weights_layout; // Attention weights [seq_len x seq_len] batched
    
    // Broadcast layouts for weight matrices
    cublasLtMatrixLayout_t W_q_broadcast_layout; // Weight matrices for broadcasting
    cublasLtMatrixLayout_t W_k_broadcast_layout;
    cublasLtMatrixLayout_t W_v_broadcast_layout; 
    cublasLtMatrixLayout_t W_o_broadcast_layout;

    // Special layouts for weight gradient computation (input batched, output single)
    cublasLtMatrixLayout_t X_for_wgrad_layout;     // X layout for weight grad computation
    cublasLtMatrixLayout_t grad_QKV_for_wgrad_layout; // grad_Q/K/V layout for weight grad computation
    cublasLtMatrixLayout_t attn_out_for_wgrad_layout;  // attn_output layout for weight grad computation
    cublasLtMatrixLayout_t grad_out_for_wgrad_layout;  // grad_output layout for weight grad computation
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    float scale;  // 1/sqrt(d_model) for scaled dot-product attention
} Attention;

// CUDA kernel prototypes
__global__ void softmax_forward_kernel_attention(float* weights, float* scores, int batch_size, int seq_len);
__global__ void softmax_backward_kernel_attention(float* grad_scores, float* grad_weights, float* weights, int batch_size, int seq_len);
__global__ void adamw_update_kernel_attention(float* weight, float* grad, float* m, float* v, float beta1, float beta2, float epsilon, float learning_rate, float weight_decay, float alpha_t, int size, int batch_size);

// Function prototypes
Attention* init_attention(int seq_len, int d_model, int batch_size, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* d_X);
float calculate_loss_attention(Attention* attn, float* d_y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* d_X, float* d_grad_X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle);

#endif