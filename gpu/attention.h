#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cudnn.h>

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuDNN Error checking macro
#ifndef CHECK_CUDNN
#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    // cuDNN handle
    cudnnHandle_t cudnn_handle;

    // cuDNN attention and descriptors
    cudnnAttnDescriptor_t attn_desc;
    cudnnDropoutDescriptor_t drop_desc;
    cudnnSeqDataDescriptor_t q_desc, k_desc, v_desc, o_desc;

    // Workspace and reserve space
    void* d_workspace;
    void* d_reserve_space;
    size_t workspace_size;
    size_t reserve_size;

    // Weights buffer and gradients (single contiguous buffer managed by cuDNN)
    void* d_weights;     // float buffer
    void* d_wgrad;       // float buffer (same size as d_weights)
    size_t weight_size;  // in bytes

    // AdamW moment buffers for weights (float arrays length weight_size/sizeof(float))
    float* d_m;
    float* d_v;

    // Forward output and gradient wrt output
    float* d_output;        // [batch_size * seq_len * d_model]
    float* d_grad_output;   // same shape

    // Gradients wrt inputs (returned by cudnnMultiHeadAttnBackwardData)
    float* d_dQ;            // [batch_size * seq_len * d_model]
    float* d_dK;            // [batch_size * seq_len * d_model]
    float* d_dV;            // [batch_size * seq_len * d_model]

    // Device arrays for sequence lengths
    int* d_seq_lengths_qo;  // length batch_size*beam (beam=1)
    int* d_seq_lengths_kv;  // length batch_size

    // Host attention window indices
    int* lo_win_idx;        // length seq_len
    int* hi_win_idx;        // length seq_len

    // Dropout state storage
    void* d_dropout_states;

    // Loss accumulation
    float* d_loss_result;   // single float

    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int num_heads;          // now configurable (set to 4 in train.c)
    bool is_causal;

    // AdamW hyperparameters and state
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int t;                  // time step
} Attention;

// API
Attention* init_attention(int seq_len, int d_model, int batch_size, int num_heads, bool is_causal, cudnnHandle_t cudnn_handle);
void free_attention(Attention* attn);

void forward_pass_attention(Attention* attn, float* d_X);
float calculate_loss_attention(Attention* attn, float* d_y);

void zero_gradients_attention(Attention* attn);
void backward_pass_attention(Attention* attn, float* d_X, float* d_grad_X);
void update_weights_attention(Attention* attn, float learning_rate);

void save_attention(Attention* attn, const char* filename);
Attention* load_attention(const char* filename, int custom_batch_size, cudnnHandle_t cudnn_handle);

#endif