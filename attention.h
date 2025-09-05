#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights for attention mechanism
    float* W_q;      // Query projection [d_model x d_model]
    float* W_k;      // Key projection [d_model x d_model]
    float* W_v;      // Value projection [d_model x d_model]
    float* W_o;      // Output projection [d_model x d_model]
    
    // Gradients
    float* W_q_grad; // [d_model x d_model]
    float* W_k_grad; // [d_model x d_model]
    float* W_v_grad; // [d_model x d_model]
    float* W_o_grad; // [d_model x d_model]
    
    // Adam parameters
    float* W_q_m, *W_q_v;
    float* W_k_m, *W_k_v;
    float* W_v_m, *W_v_v;
    float* W_o_m, *W_o_v;
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Forward pass buffers
    float* Q;            // [seq_len x (d_model * batch_size)]
    float* K;            // [seq_len x (d_model * batch_size)]
    float* V;            // [seq_len x (d_model * batch_size)]
    float* scores;       // [seq_len x (seq_len * batch_size)]
    float* attn_weights; // [seq_len x (seq_len * batch_size)]
    float* attn_output;  // [seq_len x (d_model * batch_size)]
    float* output;       // [seq_len x (d_model * batch_size)]
    
    // Backward pass buffers
    float* grad_Q;            // [seq_len x (d_model * batch_size)]
    float* grad_K;            // [seq_len x (d_model * batch_size)]
    float* grad_V;            // [seq_len x (d_model * batch_size)]
    float* grad_scores;       // [seq_len x (seq_len * batch_size)]
    float* grad_attn_output;  // [seq_len x (d_model * batch_size)]
    float* grad_output;       // [seq_len x (d_model * batch_size)]
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    float scale;  // 1/sqrt(d_model)
} Attention;

// Function prototypes
Attention* init_attention(int seq_len, int d_model, int batch_size);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* X);
float calculate_loss_attention(Attention* attn, float* y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* X, float* grad_X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size);

#endif