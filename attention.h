#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights and gradients
    float** W_q;      // [num_layers][d_model x d_model] - Query weights
    float** W_k;      // [num_layers][d_model x d_model] - Key weights  
    float** W_v;      // [num_layers][d_model x d_model] - Value weights
    float** W_o;      // [num_layers][d_model x d_model] - Output weights
    float** W_q_grad; // [num_layers][d_model x d_model]
    float** W_k_grad; // [num_layers][d_model x d_model]
    float** W_v_grad; // [num_layers][d_model x d_model]
    float** W_o_grad; // [num_layers][d_model x d_model]
    
    // Adam parameters
    float** W_q_m;    // First moment estimates for W_q
    float** W_q_v;    // Second moment estimates for W_q
    float** W_k_m;    // First moment estimates for W_k
    float** W_k_v;    // Second moment estimates for W_k
    float** W_v_m;    // First moment estimates for W_v
    float** W_v_v;    // Second moment estimates for W_v
    float** W_o_m;    // First moment estimates for W_o
    float** W_o_v;    // Second moment estimates for W_o
    float beta1;      // Exponential decay rate for first moment estimates
    float beta2;      // Exponential decay rate for second moment estimates
    float epsilon;    // Small constant for numerical stability
    int t;            // Time step
    float weight_decay; // Weight decay parameter for AdamW regularization
    
    // Layer outputs and working buffers
    float** Q;              // [num_layers][batch_size * seq_len x d_model] - Queries
    float** K;              // [num_layers][batch_size * seq_len x d_model] - Keys
    float** V;              // [num_layers][batch_size * seq_len x d_model] - Values
    float** attn_scores;    // [num_layers][batch_size * seq_len x seq_len] - Attention scores
    float** attn_weights;   // [num_layers][batch_size * seq_len x seq_len] - Attention weights (softmax)
    float** attn_output;    // [num_layers][batch_size * seq_len x d_model] - Attention output
    float** layer_output;   // [num_layers][batch_size * seq_len x d_model] - Final layer output
    
    // Gradient buffers
    float** dQ;             // [num_layers][batch_size * seq_len x d_model]
    float** dK;             // [num_layers][batch_size * seq_len x d_model]  
    float** dV;             // [num_layers][batch_size * seq_len x d_model]
    float** d_attn_scores;  // [num_layers][batch_size * seq_len x seq_len]
    float** d_attn_weights; // [num_layers][batch_size * seq_len x seq_len]
    float** d_attn_output;  // [num_layers][batch_size * seq_len x d_model]
    float** d_layer_output; // [num_layers][batch_size * seq_len x d_model]
    
    // Dimensions
    int d_model;      // Model dimension (feature_dim)
    int seq_len;      // Sequence length
    int num_layers;   // Number of attention layers
    int batch_size;   // Batch size
} Attention;

// Function prototypes
Attention* init_attention(int d_model, int seq_len, int num_layers, int batch_size);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* X);
float calculate_loss_attention(Attention* attn, float* y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size);

#endif