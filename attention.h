#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights and gradients per layer
    float** Wq;       // [num_layers][input_size x hidden_dim]
    float** Wk;       // [num_layers][input_size x hidden_dim]
    float** Wv;       // [num_layers][input_size x hidden_dim]
    float** Wo;       // [num_layers][hidden_dim x output_dim]
    float** D;        // [num_layers][input_size x output_dim]

    float** Wq_grad;  // [num_layers][input_size x hidden_dim]
    float** Wk_grad;  // [num_layers][input_size x hidden_dim]
    float** Wv_grad;  // [num_layers][input_size x hidden_dim]
    float** Wo_grad;  // [num_layers][hidden_dim x output_dim]
    float** D_grad;   // [num_layers][input_size x output_dim]

    // AdamW parameters
    float** Wq_m; float** Wq_v;
    float** Wk_m; float** Wk_v;
    float** Wv_m; float** Wv_v;
    float** Wo_m; float** Wo_v;
    float** D_m;  float** D_v;
    float beta1;
    float beta2;
    float epsilon;
    int   t;
    float weight_decay;

    // Layer buffers
    float** layer_Q;        // [num_layers][seq_len x batch_size x hidden_dim]
    float** layer_K;        // [num_layers][seq_len x batch_size x hidden_dim]
    float** layer_V;        // [num_layers][seq_len x batch_size x hidden_dim]
    float** layer_attn_scores; // [num_layers][batch_size x seq_len x seq_len] (logits)
    float** layer_attn_probs;  // [num_layers][batch_size x seq_len x seq_len] (softmax)
    float** layer_context;  // [num_layers][seq_len x batch_size x hidden_dim]
    float** layer_output;   // [num_layers][seq_len x batch_size x output_dim]
    float** error_output;   // [num_layers][seq_len x batch_size x output_dim] (also used to carry dX to previous layer)

    // Dimensions
    int input_dim;
    int hidden_dim;   // attention (single-head) dimension d_k
    int output_dim;
    int seq_len;
    int batch_size;
    int num_layers;
} ATTN;

// API
ATTN* init_attn(int input_dim, int hidden_dim, int output_dim, int seq_len, int batch_size, int num_layers);
void  free_attn(ATTN* attn);

void  forward_pass_attn(ATTN* attn, float* X_time_major);    // X_time_major: [seq_len x batch_size x input_dim]
float calculate_loss_attn(ATTN* attn, float* y_time_major);  // y_time_major: [seq_len x batch_size x output_dim]
void  zero_gradients_attn(ATTN* attn);
void  backward_pass_attn(ATTN* attn, float* X_time_major);   // X_time_major: [seq_len x batch_size x input_dim]
void  update_weights_attn(ATTN* attn, float learning_rate);

void  save_attn(ATTN* attn, const char* filename);
ATTN* load_attn(const char* filename, int custom_batch_size);

#endif