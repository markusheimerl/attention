#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights
    float* W_q;      // [input_dim x head_dim]
    float* W_k;      // [input_dim x head_dim] 
    float* W_v;      // [input_dim x head_dim]
    float* W_o;      // [head_dim x output_dim]
    
    // Gradients
    float* W_q_grad; // [input_dim x head_dim]
    float* W_k_grad; // [input_dim x head_dim]
    float* W_v_grad; // [input_dim x head_dim]
    float* W_o_grad; // [head_dim x output_dim]
    
    // AdamW parameters
    float* W_q_m; float* W_q_v; // First and second moments for W_q
    float* W_k_m; float* W_k_v; // First and second moments for W_k
    float* W_v_m; float* W_v_v; // First and second moments for W_v
    float* W_o_m; float* W_o_v; // First and second moments for W_o
    float beta1;         // Exponential decay rate for first moment estimates
    float beta2;         // Exponential decay rate for second moment estimates
    float epsilon;       // Small constant for numerical stability
    int t;               // Time step
    float weight_decay;  // Weight decay parameter for AdamW
    
    // Layer outputs and working buffers
    float* layer_Q;      // [batch_size*seq_len x head_dim]
    float* layer_K;      // [batch_size*seq_len x head_dim]
    float* layer_V;      // [batch_size*seq_len x head_dim]
    float* layer_scores; // [batch_size*seq_len x seq_len]
    float* layer_attn;   // [batch_size*seq_len x seq_len]
    float* layer_context;// [batch_size*seq_len x head_dim]
    float* layer_output; // [batch_size*seq_len x output_dim]
    
    // Error buffers
    float* error_output;  // [batch_size*seq_len x output_dim]
    float* error_context; // [batch_size*seq_len x head_dim]
    float* error_Q;       // [batch_size*seq_len x head_dim]
    float* error_K;       // [batch_size*seq_len x head_dim]
    float* error_V;       // [batch_size*seq_len x head_dim]
    float* error_scores;  // [batch_size*seq_len x seq_len]
    
    // Dimensions
    int input_dim;
    int head_dim;
    int output_dim;
    int seq_len;
    int batch_size;
} Attention;

// Function prototypes
Attention* init_attention(int input_dim, int head_dim, int output_dim, int seq_len, int batch_size);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* X);
float calculate_loss_attention(Attention* attn, float* y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size);

#endif