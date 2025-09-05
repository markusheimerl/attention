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
    float* W_q_m, *W_q_v; // First and second moments for W_q
    float* W_k_m, *W_k_v; // First and second moments for W_k
    float* W_v_m, *W_v_v; // First and second moments for W_v
    float* W_o_m, *W_o_v; // First and second moments for W_o
    float beta1;          // Exponential decay rate for first moment
    float beta2;          // Exponential decay rate for second moment
    float epsilon;        // Small constant for numerical stability
    int t;                // Time step
    float weight_decay;   // Weight decay parameter for AdamW
    
    // Forward pass buffers
    float* Q;            // Query matrix [seq_len x (d_model * batch_size)]
    float* K;            // Key matrix [seq_len x (d_model * batch_size)]
    float* V;            // Value matrix [seq_len x (d_model * batch_size)]
    float* scores;       // Attention scores [seq_len x (seq_len * batch_size)]
    float* attn_weights; // Attention weights [seq_len x (seq_len * batch_size)]
    float* attn_output;  // Attention output [seq_len x (d_model * batch_size)]
    float* output;       // Final output [seq_len x (d_model * batch_size)]
    
    // Backward pass buffers
    float* grad_output;      // [seq_len x (d_model * batch_size)]
    float* grad_attn_output; // [seq_len x (d_model * batch_size)]
    float* grad_weights;     // [seq_len x (seq_len * batch_size)]
    float* grad_scores;      // [seq_len x (seq_len * batch_size)]
    float* grad_Q;           // [seq_len x (d_model * batch_size)]
    float* grad_K;           // [seq_len x (d_model * batch_size)]
    float* grad_V;           // [seq_len x (d_model * batch_size)]
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    float scale;  // 1/sqrt(d_model) for scaled dot-product attention
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