#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
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

// cuBLASLt matrix multiplication macro
#ifndef LT_MATMUL
#define LT_MATMUL(attn, opA, opB, alpha, A, layA, B, layB, beta, C, layC) do { \
    cublasOperation_t _opA = opA, _opB = opB; \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(attn->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSA, &_opA, sizeof(_opA))); \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(attn->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSB, &_opB, sizeof(_opB))); \
    CHECK_CUBLASLT(cublasLtMatmul(attn->cublaslt_handle, attn->matmul_desc, \
                                  alpha, A, layA, B, layB, \
                                  beta, C, layC, \
                                  C, layC, NULL, NULL, 0, 0)); \
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
    
    // Buffers
    float* d_Q;            // Query matrix [batch_size x seq_len x d_model]
    float* d_K;            // Key matrix [batch_size x seq_len x d_model]
    float* d_V;            // Value matrix [batch_size x seq_len x d_model]
    float* d_scores;       // Attention scores [batch_size x seq_len x seq_len]
    float* d_attn_weights; // Attention weights [batch_size x seq_len x seq_len]
    float* d_attn_output;  // Attention output [batch_size x seq_len x d_model]
    float* d_output;       // Final output [batch_size x seq_len x d_model]

    // Loss computation buffer
    float* d_loss_result;      // [1]

    // cuBLASLt handle and descriptor
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;
    
    // Matrix layouts
    cublasLtMatrixLayout_t weight_layout;     // [d_model x d_model]
    cublasLtMatrixLayout_t seq_flat_layout;   // [batch_size * seq_len x d_model]
    cublasLtMatrixLayout_t seq_batch_layout;  // [seq_len x d_model] batched
    cublasLtMatrixLayout_t attn_batch_layout; // [seq_len x seq_len] batched
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    float scale;
    bool is_causal;
} Attention;

// Function prototypes
Attention* init_attention(int seq_len, int d_model, int batch_size, bool is_causal, cublasLtHandle_t cublaslt_handle);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* d_X);
float calculate_loss_attention(Attention* attn, float* d_y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* d_X, float* d_grad_X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);

#endif