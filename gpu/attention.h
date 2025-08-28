#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

typedef struct {
    // Device pointers for weights and gradients
    float* d_W_q;      // [d_model x d_model] - Query weights
    float* d_W_k;      // [d_model x d_model] - Key weights  
    float* d_W_v;      // [d_model x d_model] - Value weights
    float* d_W_o;      // [d_model x d_model] - Output weights
    float* d_W_q_grad; // [d_model x d_model] - Gradient accumulator
    float* d_W_k_grad; // [d_model x d_model] - Gradient accumulator
    float* d_W_v_grad; // [d_model x d_model] - Gradient accumulator
    float* d_W_o_grad; // [d_model x d_model] - Gradient accumulator
    
    // Device pointers for Adam parameters
    float* d_W_q_m;    // First moment estimates for W_q
    float* d_W_q_v;    // Second moment estimates for W_q
    float* d_W_k_m;    // First moment estimates for W_k
    float* d_W_k_v;    // Second moment estimates for W_k
    float* d_W_v_m;    // First moment estimates for W_v
    float* d_W_v_v;    // Second moment estimates for W_v
    float* d_W_o_m;    // First moment estimates for W_o
    float* d_W_o_v;    // Second moment estimates for W_o
    float beta1;       // Exponential decay rate for first moment estimates
    float beta2;       // Exponential decay rate for second moment estimates
    float epsilon;     // Small constant for numerical stability
    int t;             // Time step
    float weight_decay; // Weight decay parameter for AdamW regularization
    
    // Device pointers for forward pass buffers
    float* d_Q;              // [batch_size * seq_len x d_model] - Query activations
    float* d_K;              // [batch_size * seq_len x d_model] - Key activations
    float* d_V;              // [batch_size * seq_len x d_model] - Value activations
    float* d_attn_scores;    // [batch_size * seq_len x seq_len] - Raw attention scores
    float* d_attn_weights;   // [batch_size * seq_len x seq_len] - Softmax attention weights
    float* d_attn_output;    // [batch_size * seq_len x d_model] - Weighted value sum
    float* d_layer_output;   // [batch_size * seq_len x d_model] - Final output after W_o
    
    // Device pointers for backward pass buffers
    float* d_grad_Q;         // [batch_size * seq_len x d_model] - Gradient w.r.t. Q
    float* d_grad_K;         // [batch_size * seq_len x d_model] - Gradient w.r.t. K
    float* d_grad_V;         // [batch_size * seq_len x d_model] - Gradient w.r.t. V
    float* d_grad_scores;    // [batch_size * seq_len x seq_len] - Gradient w.r.t. scores
    float* d_grad_weights;   // [batch_size * seq_len x seq_len] - Gradient w.r.t. weights
    float* d_grad_attn_out;  // [batch_size * seq_len x d_model] - Gradient w.r.t. attention output
    float* d_error_output;   // [batch_size * seq_len x d_model] - Final output error
    
    // Dimensions
    int d_model;      // Model dimension (feature_dim)
    int seq_len;      // Sequence length
    int batch_size;   // Batch size
} Attention;

// CUDA kernel prototypes
__global__ void rowmajor_gemm_kernel(float* C, const float* A, const float* B, 
                                     int M, int N, int K, float alpha, float beta);
__global__ void rowmajor_gemm_transB_kernel(float* C, const float* A, const float* B,
                                            int M, int N, int K, float alpha, float beta);
__global__ void rowmajor_gemm_transA_kernel(float* C, const float* A, const float* B,
                                            int M, int N, int K, float alpha, float beta);
__global__ void softmax_kernel(float* weights, const float* scores, int batch_size, int seq_len);
__global__ void softmax_backward_kernel(float* d_scores, const float* d_weights, const float* weights, 
                                        int batch_size, int seq_len);
__global__ void adamw_update_kernel(float* weight, const float* grad, float* m, float* v,
                                   float beta1, float beta2, float epsilon, float learning_rate,
                                   float weight_decay, float alpha_t, int size, int batch_norm);

// Function prototypes
Attention* init_attention(int d_model, int seq_len, int batch_size);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* d_X);
float calculate_loss_attention(Attention* attn, float* d_y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* d_X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size);

#endif