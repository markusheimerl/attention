#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
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

typedef struct {
    // Device pointers for weights and gradients
    float** d_W_q;      // [num_layers][d_model x d_model] - Query weights
    float** d_W_k;      // [num_layers][d_model x d_model] - Key weights  
    float** d_W_v;      // [num_layers][d_model x d_model] - Value weights
    float** d_W_o;      // [num_layers][d_model x d_model] - Output weights
    float** d_W_q_grad; // [num_layers][d_model x d_model]
    float** d_W_k_grad; // [num_layers][d_model x d_model]
    float** d_W_v_grad; // [num_layers][d_model x d_model]
    float** d_W_o_grad; // [num_layers][d_model x d_model]
    
    // Device pointers for Adam parameters
    float** d_W_q_m;    // First moment estimates for W_q
    float** d_W_q_v;    // Second moment estimates for W_q
    float** d_W_k_m;    // First moment estimates for W_k
    float** d_W_k_v;    // Second moment estimates for W_k
    float** d_W_v_m;    // First moment estimates for W_v
    float** d_W_v_v;    // Second moment estimates for W_v
    float** d_W_o_m;    // First moment estimates for W_o
    float** d_W_o_v;    // Second moment estimates for W_o
    float beta1;        // Exponential decay rate for first moment estimates
    float beta2;        // Exponential decay rate for second moment estimates
    float epsilon;      // Small constant for numerical stability
    int t;              // Time step
    float weight_decay; // Weight decay parameter for AdamW regularization
    
    // Device pointers for layer outputs and working buffers
    float** d_Q;              // [num_layers][batch_size * seq_len x d_model] - Queries
    float** d_K;              // [num_layers][batch_size * seq_len x d_model] - Keys
    float** d_V;              // [num_layers][batch_size * seq_len x d_model] - Values
    float** d_attn_scores;    // [num_layers][batch_size * seq_len x seq_len] - Attention scores
    float** d_attn_weights;   // [num_layers][batch_size * seq_len x seq_len] - Attention weights (softmax)
    float** d_attn_output;    // [num_layers][batch_size * seq_len x d_model] - Attention output
    float** d_layer_output;   // [num_layers][batch_size * seq_len x d_model] - Final layer output
    
    // Device pointers for gradient buffers
    float** d_dQ;             // [num_layers][batch_size * seq_len x d_model]
    float** d_dK;             // [num_layers][batch_size * seq_len x d_model]  
    float** d_dV;             // [num_layers][batch_size * seq_len x d_model]
    float** d_d_attn_scores;  // [num_layers][batch_size * seq_len x seq_len]
    float** d_d_attn_weights; // [num_layers][batch_size * seq_len x seq_len]
    float** d_d_attn_output;  // [num_layers][batch_size * seq_len x d_model]
    float** d_d_layer_output; // [num_layers][batch_size * seq_len x d_model]

    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int d_model;      // Model dimension (feature_dim)
    int seq_len;      // Sequence length
    int num_layers;   // Number of attention layers
    int batch_size;   // Batch size
} Attention;

// CUDA kernel prototypes
__global__ void softmax_forward_kernel_attention(float* weights, float* scores, int batch_size, int seq_len);
__global__ void softmax_backward_kernel_attention(float* d_scores, float* d_weights, float* weights, int batch_size, int seq_len);
__global__ void adamw_update_kernel_attention(float* weight, float* grad, float* m, float* v, float beta1, float beta2, float epsilon, float learning_rate, float weight_decay, float alpha_t, int size, int batch_size, int seq_len);

// Function prototypes
Attention* init_attention(int d_model, int seq_len, int num_layers, int batch_size, cublasHandle_t cublas_handle);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* d_X);
float calculate_loss_attention(Attention* attn, float* d_y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* d_X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle);

#endif