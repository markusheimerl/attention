#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Multi-head attention weights for each layer
    float*** W_Q;     // [num_layers][num_heads][input_size x head_dim]
    float*** W_K;     // [num_layers][num_heads][input_size x head_dim]
    float*** W_V;     // [num_layers][num_heads][input_size x head_dim]
    float** W_O;      // [num_layers][num_heads*head_dim x output_size]
    
    // Gradients
    float*** W_Q_grad;
    float*** W_K_grad;
    float*** W_V_grad;
    float** W_O_grad;
    
    // AdamW parameters
    float*** W_Q_m;   // First moment for W_Q
    float*** W_Q_v;   // Second moment for W_Q
    float*** W_K_m;   // First moment for W_K
    float*** W_K_v;   // Second moment for W_K
    float*** W_V_m;   // First moment for W_V
    float*** W_V_v;   // Second moment for W_V
    float** W_O_m;    // First moment for W_O
    float** W_O_v;    // Second moment for W_O
    float beta1;      // Exponential decay rate for first moment estimates
    float beta2;      // Exponential decay rate for second moment estimates
    float epsilon;    // Small constant for numerical stability
    int t;            // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Working buffers for each layer and head
    float*** Q;       // [num_layers][num_heads][seq_len * batch_size * head_dim]
    float*** K;       // [num_layers][num_heads][seq_len * batch_size * head_dim]
    float*** V;       // [num_layers][num_heads][seq_len * batch_size * head_dim]
    float*** attn_weights; // [num_layers][num_heads][batch_size * seq_len * seq_len]
    float*** head_output; // [num_layers][num_heads][seq_len * batch_size * head_dim]
    float** concat_heads; // [num_layers][seq_len * batch_size * num_heads * head_dim]
    float** layer_output; // [num_layers][seq_len * batch_size * output_size]
    float** error_output; // [num_layers][seq_len * batch_size * output_size]
    
    // Error propagation buffers
    float*** error_Q; // [num_layers][num_heads][seq_len * batch_size * head_dim]
    float*** error_K; // [num_layers][num_heads][seq_len * batch_size * head_dim]
    float*** error_V; // [num_layers][num_heads][seq_len * batch_size * head_dim]
    float*** error_head_output; // [num_layers][num_heads][seq_len * batch_size * head_dim]
    float** error_concat_heads; // [num_layers][seq_len * batch_size * num_heads * head_dim]
    
    // Dimensions
    int input_dim;
    int output_dim;
    int head_dim;
    int num_heads;
    int seq_len;
    int batch_size;
    int num_layers;
} Attention;

// Function prototypes
Attention* init_attention(int input_dim, int output_dim, int head_dim, int num_heads, int seq_len, int batch_size, int num_layers);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* X);
float calculate_loss_attention(Attention* attn, float* y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size);

#endif