#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights and gradients for Q, K, V projections
    float* WQ;       // [d_model x d_model]
    float* WK;       // [d_model x d_model]  
    float* WV;       // [d_model x d_model]
    float* WO;       // [d_model x d_model] (output projection)
    float* WQ_grad;  // [d_model x d_model]
    float* WK_grad;  // [d_model x d_model]
    float* WV_grad;  // [d_model x d_model]
    float* WO_grad;  // [d_model x d_model]
    
    // Adam parameters
    float* WQ_m; float* WQ_v; // First and second moments for WQ
    float* WK_m; float* WK_v; // First and second moments for WK
    float* WV_m; float* WV_v; // First and second moments for WV
    float* WO_m; float* WO_v; // First and second moments for WO
    float beta1;      // Exponential decay rate for first moment
    float beta2;      // Exponential decay rate for second moment
    float epsilon;    // Small constant for numerical stability
    int t;            // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Forward pass buffers
    float* Q;         // [d_model x seq_len x batch_size]
    float* K;         // [d_model x seq_len x batch_size]
    float* V;         // [d_model x seq_len x batch_size]
    float* scores;    // [seq_len x seq_len x batch_size]
    float* attn_weights; // [seq_len x seq_len x batch_size]
    float* context;   // [d_model x seq_len x batch_size]
    float* output;    // [d_model x seq_len x batch_size]
    
    // Backward pass buffers
    float* grad_output;   // [d_model x seq_len x batch_size]
    float* grad_context;  // [d_model x seq_len x batch_size]
    float* grad_attn_weights; // [seq_len x seq_len x batch_size]
    float* grad_scores;   // [seq_len x seq_len x batch_size]
    float* grad_Q;        // [d_model x seq_len x batch_size]
    float* grad_K;        // [d_model x seq_len x batch_size]
    float* grad_V;        // [d_model x seq_len x batch_size]
    
    // Dimensions
    int d_model;      // Model dimension
    int seq_len;      // Sequence length
    int batch_size;   // Batch size
    float scale;      // 1/sqrt(d_model) for scaled dot-product attention
} Attention;

// Function prototypes
Attention* init_attention(int d_model, int seq_len, int batch_size);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* X);
float calculate_loss_attention(Attention* attn, float* y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* X, float* grad_X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size);

#endif