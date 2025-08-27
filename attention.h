#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights and gradients
    float** W_q;      // [num_layers][hidden_dim x input_dim] Query weights
    float** W_k;      // [num_layers][hidden_dim x input_dim] Key weights  
    float** W_v;      // [num_layers][hidden_dim x input_dim] Value weights
    float** W_o;      // [num_layers][output_dim x hidden_dim] Output projection
    float** W_r;      // [num_layers][output_dim x input_dim] Residual connection
    float** W_q_grad; // [num_layers][hidden_dim x input_dim]
    float** W_k_grad; // [num_layers][hidden_dim x input_dim]
    float** W_v_grad; // [num_layers][hidden_dim x input_dim]
    float** W_o_grad; // [num_layers][output_dim x hidden_dim]
    float** W_r_grad; // [num_layers][output_dim x input_dim]
    
    // AdamW parameters
    float** W_q_m;    // First moment estimates for W_q
    float** W_q_v;    // Second moment estimates for W_q
    float** W_k_m;    // First moment estimates for W_k
    float** W_k_v;    // Second moment estimates for W_k
    float** W_v_m;    // First moment estimates for W_v
    float** W_v_v;    // Second moment estimates for W_v
    float** W_o_m;    // First moment estimates for W_o
    float** W_o_v;    // Second moment estimates for W_o
    float** W_r_m;    // First moment estimates for W_r
    float** W_r_v;    // Second moment estimates for W_r
    float beta1;      // Exponential decay rate for first moment estimates
    float beta2;      // Exponential decay rate for second moment estimates
    float epsilon;    // Small constant for numerical stability
    int t;            // Time step
    float weight_decay; // Weight decay parameter for AdamW regularization
    
    // Layer outputs and working buffers
    float** layer_q;        // [num_layers][seq_len * batch_size x hidden_dim] Queries
    float** layer_k;        // [num_layers][seq_len * batch_size x hidden_dim] Keys
    float** layer_v;        // [num_layers][seq_len * batch_size x hidden_dim] Values
    float** layer_scores;   // [num_layers][seq_len * batch_size x seq_len] Attention scores
    float** layer_weights;  // [num_layers][seq_len * batch_size x seq_len] Attention weights
    float** layer_context;  // [num_layers][seq_len * batch_size x hidden_dim] Context vectors
    float** layer_preact;   // [num_layers][seq_len * batch_size x hidden_dim] Pre-activation
    float** layer_postact;  // [num_layers][seq_len * batch_size x hidden_dim] Post-activation
    float** layer_output;   // [num_layers][seq_len * batch_size x output_dim] Final output
    float** error_context;  // [num_layers][seq_len * batch_size x hidden_dim]
    float** error_weights;  // [num_layers][seq_len * batch_size x seq_len]
    float** error_values;   // [num_layers][seq_len * batch_size x hidden_dim]
    float** error_keys;     // [num_layers][seq_len * batch_size x hidden_dim]
    float** error_queries;  // [num_layers][seq_len * batch_size x hidden_dim]
    float** error_preact;   // [num_layers][seq_len * batch_size x hidden_dim]
    float** error_output;   // [num_layers][seq_len * batch_size x output_dim]
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int seq_len;
    int batch_size;
    int num_layers;
} Attention;

// Function prototypes
Attention* init_attention(int input_dim, int hidden_dim, int output_dim, int seq_len, int batch_size, int num_layers);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* X);
float calculate_loss_attention(Attention* attn, float* y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size);

#endif