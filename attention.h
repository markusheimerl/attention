#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights and gradients
    float* W_q;      // [d_model x d_model] - Query weights
    float* W_k;      // [d_model x d_model] - Key weights  
    float* W_v;      // [d_model x d_model] - Value weights
    float* W_o;      // [d_model x d_model] - Output weights
    float* W_q_grad; // [d_model x d_model] - Gradient accumulator
    float* W_k_grad; // [d_model x d_model] - Gradient accumulator
    float* W_v_grad; // [d_model x d_model] - Gradient accumulator
    float* W_o_grad; // [d_model x d_model] - Gradient accumulator
    
    // Adam parameters
    float* W_q_m;    // First moment estimates for W_q
    float* W_q_v;    // Second moment estimates for W_q
    float* W_k_m;    // First moment estimates for W_k
    float* W_k_v;    // Second moment estimates for W_k
    float* W_v_m;    // First moment estimates for W_v
    float* W_v_v;    // Second moment estimates for W_v
    float* W_o_m;    // First moment estimates for W_o
    float* W_o_v;    // Second moment estimates for W_o
    float beta1;     // Exponential decay rate for first moment estimates
    float beta2;     // Exponential decay rate for second moment estimates
    float epsilon;   // Small constant for numerical stability
    int t;           // Time step
    float weight_decay; // Weight decay parameter for AdamW regularization
    
    // Forward pass buffers
    float* Q;              // [batch_size * seq_len x d_model] - Query activations
    float* K;              // [batch_size * seq_len x d_model] - Key activations
    float* V;              // [batch_size * seq_len x d_model] - Value activations
    float* attn_scores;    // [batch_size * seq_len x seq_len] - Raw attention scores
    float* attn_weights;   // [batch_size * seq_len x seq_len] - Softmax attention weights
    float* attn_output;    // [batch_size * seq_len x d_model] - Weighted value sum
    float* layer_output;   // [batch_size * seq_len x d_model] - Final output after W_o
    
    // Backward pass buffers
    float* grad_Q;         // [batch_size * seq_len x d_model] - Gradient w.r.t. Q
    float* grad_K;         // [batch_size * seq_len x d_model] - Gradient w.r.t. K
    float* grad_V;         // [batch_size * seq_len x d_model] - Gradient w.r.t. V
    float* grad_scores;    // [batch_size * seq_len x seq_len] - Gradient w.r.t. scores
    float* grad_weights;   // [batch_size * seq_len x seq_len] - Gradient w.r.t. weights
    float* grad_attn_out;  // [batch_size * seq_len x d_model] - Gradient w.r.t. attention output
    float* error_output;   // [batch_size * seq_len x d_model] - Final output error
    
    // Dimensions
    int d_model;      // Model dimension (feature_dim)
    int seq_len;      // Sequence length
    int batch_size;   // Batch size
} Attention;

// Function prototypes
Attention* init_attention(int d_model, int seq_len, int batch_size);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* X);
float calculate_loss_attention(Attention* attn, float* y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size);

#endif