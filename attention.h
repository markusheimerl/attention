#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
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
    float* Q;            // Query matrix [batch_size x seq_len x d_model]
    float* K;            // Key matrix [batch_size x seq_len x d_model]
    float* V;            // Value matrix [batch_size x seq_len x d_model]
    float* scores;       // Attention scores [batch_size x seq_len x seq_len]
    float* attn_weights; // Attention weights [batch_size x seq_len x seq_len]
    float* attn_output;  // Attention output [batch_size x seq_len x d_model]
    float* output;       // Final output [batch_size x seq_len x d_model]
    
    // Backward pass buffers
    float* grad_output;      // [batch_size x seq_len x d_model]
    float* grad_attn_output; // [batch_size x seq_len x d_model]
    float* grad_weights;     // [batch_size x seq_len x seq_len]
    float* grad_scores;      // [batch_size x seq_len x seq_len]
    float* grad_Q;           // [batch_size x seq_len x d_model]
    float* grad_K;           // [batch_size x seq_len x d_model]
    float* grad_V;           // [batch_size x seq_len x d_model]
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    float scale;
    bool is_causal;
    bool use_rope;
} Attention;

// Function prototypes
Attention* init_attention(int seq_len, int d_model, int batch_size, bool is_causal, bool use_rope);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* X);
float calculate_loss_attention(Attention* attn, float* y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* X, float* grad_X);
void update_weights_attention(Attention* attn, float learning_rate, int effective_batch_size);
void reset_optimizer_attention(Attention* attn);
void serialize_attention(Attention* attn, FILE* file);
Attention* deserialize_attention(FILE* file, int seq_len, int batch_size);

#endif