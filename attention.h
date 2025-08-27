#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights (single-head attention + residual projection)
    float** Wq;  // [num_layers][input_size x head_dim]
    float** Wk;  // [num_layers][input_size x head_dim]
    float** Wv;  // [num_layers][input_size x head_dim]
    float** Wo;  // [num_layers][head_dim   x output_size]
    float** Wr;  // [num_layers][input_size x output_size]

    // Gradients
    float** Wq_grad;
    float** Wk_grad;
    float** Wv_grad;
    float** Wo_grad;
    float** Wr_grad;

    // AdamW parameters
    float** Wq_m; float** Wq_v;
    float** Wk_m; float** Wk_v;
    float** Wv_m; float** Wv_v;
    float** Wo_m; float** Wo_v;
    float** Wr_m; float** Wr_v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;

    // Layer buffers (batch-major: [batch_size x seq_len x ...])
    float** layer_Q;       // [num_layers][batch_size*seq_len*head_dim]
    float** layer_K;       // [num_layers][batch_size*seq_len*head_dim]
    float** layer_V;       // [num_layers][batch_size*seq_len*head_dim]
    float** layer_scores;  // [num_layers][batch_size*seq_len*seq_len] (pre-softmax)
    float** layer_attn;    // [num_layers][batch_size*seq_len*seq_len] (softmax probs)
    float** layer_O;       // [num_layers][batch_size*seq_len*head_dim]
    float** layer_output;  // [num_layers][batch_size*seq_len*output_dim]

    // Error buffers for backprop
    float** error_output;  // [num_layers][batch_size*seq_len*output_dim] (dL/dY)
    float** error_O;       // [num_layers][batch_size*seq_len*head_dim] (dL/dO)
    float** error_Q;       // [num_layers][batch_size*seq_len*head_dim] (dL/dQ)
    float** error_K;       // [num_layers][batch_size*seq_len*head_dim] (dL/dK)
    float** error_V;       // [num_layers][batch_size*seq_len*head_dim] (dL/dV)
    float** error_scores;  // [num_layers][batch_size*seq_len*seq_len] (dL/dScores)

    // Dimensions
    int input_dim;
    int head_dim;
    int output_dim;
    int seq_len;
    int batch_size;
    int num_layers;
} Attention;

// Function prototypes
Attention* init_attention(int input_dim, int head_dim, int output_dim, int seq_len, int batch_size, int num_layers);
void free_attention(Attention* attn);
void forward_pass_attention(Attention* attn, float* X);
float calculate_loss_attention(Attention* attn, float* y);
void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* X);
void update_weights_attention(Attention* attn, float learning_rate);
void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size);

#endif