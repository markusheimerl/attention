#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Query, Key, Value projection weights
    float* WQ;      // [d_model x d_model]
    float* WK;      // [d_model x d_model]
    float* WV;      // [d_model x d_model]
    float* WO;      // [d_model x d_model]
    
    // Gradients
    float* WQ_grad; // [d_model x d_model]
    float* WK_grad; // [d_model x d_model]
    float* WV_grad; // [d_model x d_model]
    float* WO_grad; // [d_model x d_model]
    
    // Adam parameters
    float* WQ_m;    // First moment for WQ
    float* WQ_v;    // Second moment for WQ
    float* WK_m;    // First moment for WK
    float* WK_v;    // Second moment for WK
    float* WV_m;    // First moment for WV
    float* WV_v;    // Second moment for WV
    float* WO_m;    // First moment for WO
    float* WO_v;    // Second moment for WO
    float beta1;      // Exponential decay rate for first moment
    float beta2;      // Exponential decay rate for second moment
    float epsilon;    // Small constant for numerical stability
    int t;            // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Working buffers
    float* Q;           // [d_model x seq_len x batch_size]
    float* K;           // [d_model x seq_len x batch_size]
    float* V;           // [d_model x seq_len x batch_size]
    float* scores;      // [seq_len x seq_len x batch_size]
    float* attn_weights; // [seq_len x seq_len x batch_size]
    float* attn_output;  // [d_model x seq_len x batch_size]
    float* layer_output; // [d_model x seq_len x batch_size]
    
    // Error buffers
    float* error_output;      // [d_model x seq_len x batch_size]
    float* error_attn_output; // [d_model x seq_len x batch_size]
    float* error_attn_weights;// [seq_len x seq_len x batch_size]
    float* error_scores;      // [seq_len x seq_len x batch_size]
    float* error_V;           // [d_model x seq_len x batch_size]
    float* error_K;           // [d_model x seq_len x batch_size]
    float* error_Q;           // [d_model x seq_len x batch_size]
    
    // Dimensions
    int d_model;
    int seq_len;
    int batch_size;
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