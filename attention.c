#include "attention.h"

// Initialize the network with configurable dimensions
Attention* init_attention(int input_dim, int output_dim, int head_dim, int num_heads, int seq_len, int batch_size, int num_layers) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->input_dim = input_dim;
    attn->output_dim = output_dim;
    attn->head_dim = head_dim;
    attn->num_heads = num_heads;
    attn->seq_len = seq_len;
    attn->batch_size = batch_size;
    attn->num_layers = num_layers;
    
    // Initialize AdamW parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    // Allocate arrays of pointers
    attn->W_Q = (float***)malloc(num_layers * sizeof(float**));
    attn->W_K = (float***)malloc(num_layers * sizeof(float**));
    attn->W_V = (float***)malloc(num_layers * sizeof(float**));
    attn->W_O = (float**)malloc(num_layers * sizeof(float*));
    attn->W_Q_grad = (float***)malloc(num_layers * sizeof(float**));
    attn->W_K_grad = (float***)malloc(num_layers * sizeof(float**));
    attn->W_V_grad = (float***)malloc(num_layers * sizeof(float**));
    attn->W_O_grad = (float**)malloc(num_layers * sizeof(float*));
    
    // Allocate AdamW buffers
    attn->W_Q_m = (float***)malloc(num_layers * sizeof(float**));
    attn->W_Q_v = (float***)malloc(num_layers * sizeof(float**));
    attn->W_K_m = (float***)malloc(num_layers * sizeof(float**));
    attn->W_K_v = (float***)malloc(num_layers * sizeof(float**));
    attn->W_V_m = (float***)malloc(num_layers * sizeof(float**));
    attn->W_V_v = (float***)malloc(num_layers * sizeof(float**));
    attn->W_O_m = (float**)malloc(num_layers * sizeof(float*));
    attn->W_O_v = (float**)malloc(num_layers * sizeof(float*));
    
    // Allocate working buffers
    attn->Q = (float***)malloc(num_layers * sizeof(float**));
    attn->K = (float***)malloc(num_layers * sizeof(float**));
    attn->V = (float***)malloc(num_layers * sizeof(float**));
    attn->attn_weights = (float***)malloc(num_layers * sizeof(float**));
    attn->head_output = (float***)malloc(num_layers * sizeof(float**));
    attn->concat_heads = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_output = (float**)malloc(num_layers * sizeof(float*));
    attn->error_output = (float**)malloc(num_layers * sizeof(float*));
    
    // Allocate error propagation buffers
    attn->error_Q = (float***)malloc(num_layers * sizeof(float**));
    attn->error_K = (float***)malloc(num_layers * sizeof(float**));
    attn->error_V = (float***)malloc(num_layers * sizeof(float**));
    attn->error_head_output = (float***)malloc(num_layers * sizeof(float**));
    attn->error_concat_heads = (float**)malloc(num_layers * sizeof(float*));
    
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : output_dim;
        
        // Allocate head arrays
        attn->W_Q[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_K[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_V[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_Q_grad[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_K_grad[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_V_grad[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_Q_m[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_Q_v[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_K_m[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_K_v[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_V_m[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->W_V_v[layer] = (float**)malloc(num_heads * sizeof(float*));
        
        attn->Q[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->K[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->V[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->attn_weights[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->head_output[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->error_Q[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->error_K[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->error_V[layer] = (float**)malloc(num_heads * sizeof(float*));
        attn->error_head_output[layer] = (float**)malloc(num_heads * sizeof(float*));
        
        for (int head = 0; head < num_heads; head++) {
            int wqkv_size = input_size * head_dim;
            
            // Allocate and initialize Q, K, V weights
            attn->W_Q[layer][head] = (float*)malloc(wqkv_size * sizeof(float));
            attn->W_K[layer][head] = (float*)malloc(wqkv_size * sizeof(float));
            attn->W_V[layer][head] = (float*)malloc(wqkv_size * sizeof(float));
            attn->W_Q_grad[layer][head] = (float*)malloc(wqkv_size * sizeof(float));
            attn->W_K_grad[layer][head] = (float*)malloc(wqkv_size * sizeof(float));
            attn->W_V_grad[layer][head] = (float*)malloc(wqkv_size * sizeof(float));
            attn->W_Q_m[layer][head] = (float*)calloc(wqkv_size, sizeof(float));
            attn->W_Q_v[layer][head] = (float*)calloc(wqkv_size, sizeof(float));
            attn->W_K_m[layer][head] = (float*)calloc(wqkv_size, sizeof(float));
            attn->W_K_v[layer][head] = (float*)calloc(wqkv_size, sizeof(float));
            attn->W_V_m[layer][head] = (float*)calloc(wqkv_size, sizeof(float));
            attn->W_V_v[layer][head] = (float*)calloc(wqkv_size, sizeof(float));
            
            // Allocate working buffers
            attn->Q[layer][head] = (float*)malloc(seq_len * batch_size * head_dim * sizeof(float));
            attn->K[layer][head] = (float*)malloc(seq_len * batch_size * head_dim * sizeof(float));
            attn->V[layer][head] = (float*)malloc(seq_len * batch_size * head_dim * sizeof(float));
            attn->attn_weights[layer][head] = (float*)malloc(batch_size * seq_len * seq_len * sizeof(float));
            attn->head_output[layer][head] = (float*)malloc(seq_len * batch_size * head_dim * sizeof(float));
            attn->error_Q[layer][head] = (float*)malloc(seq_len * batch_size * head_dim * sizeof(float));
            attn->error_K[layer][head] = (float*)malloc(seq_len * batch_size * head_dim * sizeof(float));
            attn->error_V[layer][head] = (float*)malloc(seq_len * batch_size * head_dim * sizeof(float));
            attn->error_head_output[layer][head] = (float*)malloc(seq_len * batch_size * head_dim * sizeof(float));
            
            // Initialize weights
            float scale = 1.0f / sqrtf(head_dim);
            for (int i = 0; i < wqkv_size; i++) {
                attn->W_Q[layer][head][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
                attn->W_K[layer][head][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
                attn->W_V[layer][head][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
            }
        }
        
        // Allocate output projection
        int wo_size = (num_heads * head_dim) * output_size;
        attn->W_O[layer] = (float*)malloc(wo_size * sizeof(float));
        attn->W_O_grad[layer] = (float*)malloc(wo_size * sizeof(float));
        attn->W_O_m[layer] = (float*)calloc(wo_size, sizeof(float));
        attn->W_O_v[layer] = (float*)calloc(wo_size, sizeof(float));
        
        // Initialize output projection weights
        float scale_o = 1.0f / sqrtf(num_heads * head_dim);
        for (int i = 0; i < wo_size; i++) {
            attn->W_O[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_o;
        }
        
        // Allocate layer buffers
        attn->concat_heads[layer] = (float*)malloc(seq_len * batch_size * num_heads * head_dim * sizeof(float));
        attn->layer_output[layer] = (float*)malloc(seq_len * batch_size * output_size * sizeof(float));
        attn->error_output[layer] = (float*)malloc(seq_len * batch_size * output_size * sizeof(float));
        attn->error_concat_heads[layer] = (float*)malloc(seq_len * batch_size * num_heads * head_dim * sizeof(float));
    }
    
    return attn;
}

// Free network memory
void free_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        for (int head = 0; head < attn->num_heads; head++) {
            free(attn->W_Q[layer][head]); free(attn->W_K[layer][head]); free(attn->W_V[layer][head]);
            free(attn->W_Q_grad[layer][head]); free(attn->W_K_grad[layer][head]); free(attn->W_V_grad[layer][head]);
            free(attn->W_Q_m[layer][head]); free(attn->W_Q_v[layer][head]);
            free(attn->W_K_m[layer][head]); free(attn->W_K_v[layer][head]);
            free(attn->W_V_m[layer][head]); free(attn->W_V_v[layer][head]);
            free(attn->Q[layer][head]); free(attn->K[layer][head]); free(attn->V[layer][head]);
            free(attn->attn_weights[layer][head]); free(attn->head_output[layer][head]);
            free(attn->error_Q[layer][head]); free(attn->error_K[layer][head]); free(attn->error_V[layer][head]);
            free(attn->error_head_output[layer][head]);
        }
        
        free(attn->W_Q[layer]); free(attn->W_K[layer]); free(attn->W_V[layer]);
        free(attn->W_Q_grad[layer]); free(attn->W_K_grad[layer]); free(attn->W_V_grad[layer]);
        free(attn->W_Q_m[layer]); free(attn->W_Q_v[layer]);
        free(attn->W_K_m[layer]); free(attn->W_K_v[layer]);
        free(attn->W_V_m[layer]); free(attn->W_V_v[layer]);
        free(attn->Q[layer]); free(attn->K[layer]); free(attn->V[layer]);
        free(attn->attn_weights[layer]); free(attn->head_output[layer]);
        free(attn->error_Q[layer]); free(attn->error_K[layer]); free(attn->error_V[layer]);
        free(attn->error_head_output[layer]);
        
        free(attn->W_O[layer]); free(attn->W_O_grad[layer]);
        free(attn->W_O_m[layer]); free(attn->W_O_v[layer]);
        free(attn->concat_heads[layer]); free(attn->layer_output[layer]);
        free(attn->error_output[layer]); free(attn->error_concat_heads[layer]);
    }
    
    free(attn->W_Q); free(attn->W_K); free(attn->W_V); free(attn->W_O);
    free(attn->W_Q_grad); free(attn->W_K_grad); free(attn->W_V_grad); free(attn->W_O_grad);
    free(attn->W_Q_m); free(attn->W_Q_v); free(attn->W_K_m); free(attn->W_K_v);
    free(attn->W_V_m); free(attn->W_V_v); free(attn->W_O_m); free(attn->W_O_v);
    free(attn->Q); free(attn->K); free(attn->V);
    free(attn->attn_weights); free(attn->head_output);
    free(attn->concat_heads); free(attn->layer_output); free(attn->error_output);
    free(attn->error_Q); free(attn->error_K); free(attn->error_V);
    free(attn->error_head_output); free(attn->error_concat_heads);
    free(attn);
}

// Softmax function for attention weights
static void softmax_2d(float* input, float* output, int batch_size, int seq_len) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < seq_len; i++) {
            int row_offset = b * seq_len * seq_len + i * seq_len;
            
            // Find max for numerical stability
            float max_val = input[row_offset];
            for (int j = 1; j < seq_len; j++) {
                if (input[row_offset + j] > max_val) {
                    max_val = input[row_offset + j];
                }
            }
            
            // Compute softmax
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                output[row_offset + j] = expf(input[row_offset + j] - max_val);
                sum += output[row_offset + j];
            }
            
            for (int j = 0; j < seq_len; j++) {
                output[row_offset + j] /= sum;
            }
        }
    }
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    float* input = X;
    
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        // For each head
        for (int head = 0; head < attn->num_heads; head++) {
            // Q = input @ W_Q
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len * attn->batch_size, attn->head_dim, input_size,
                        1.0f, input, input_size,
                        attn->W_Q[layer][head], attn->head_dim,
                        0.0f, attn->Q[layer][head], attn->head_dim);
            
            // K = input @ W_K
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len * attn->batch_size, attn->head_dim, input_size,
                        1.0f, input, input_size,
                        attn->W_K[layer][head], attn->head_dim,
                        0.0f, attn->K[layer][head], attn->head_dim);
            
            // V = input @ W_V
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len * attn->batch_size, attn->head_dim, input_size,
                        1.0f, input, input_size,
                        attn->W_V[layer][head], attn->head_dim,
                        0.0f, attn->V[layer][head], attn->head_dim);
            
            // Compute attention weights and apply to V for each batch
            float scale = 1.0f / sqrtf(attn->head_dim);
            for (int b = 0; b < attn->batch_size; b++) {
                // Get Q, K, V for this batch
                float* Q_b = &attn->Q[layer][head][b * attn->seq_len * attn->head_dim];
                float* K_b = &attn->K[layer][head][b * attn->seq_len * attn->head_dim];
                float* V_b = &attn->V[layer][head][b * attn->seq_len * attn->head_dim];
                float* scores = &attn->attn_weights[layer][head][b * attn->seq_len * attn->seq_len];
                float* head_out = &attn->head_output[layer][head][b * attn->seq_len * attn->head_dim];
                
                // Compute attention scores: Q @ K^T
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            attn->seq_len, attn->seq_len, attn->head_dim,
                            scale, Q_b, attn->head_dim,
                            K_b, attn->head_dim,
                            0.0f, scores, attn->seq_len);
                
                // Apply softmax to attention scores
                softmax_2d(scores, scores, 1, attn->seq_len);
                
                // Apply attention to values: attn_weights @ V
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            attn->seq_len, attn->head_dim, attn->seq_len,
                            1.0f, scores, attn->seq_len,
                            V_b, attn->head_dim,
                            0.0f, head_out, attn->head_dim);
            }
        }
        
        // Concatenate heads
        for (int b = 0; b < attn->batch_size; b++) {
            for (int t = 0; t < attn->seq_len; t++) {
                for (int h = 0; h < attn->num_heads; h++) {
                    int src_offset = b * attn->seq_len * attn->head_dim + t * attn->head_dim;
                    int dst_offset = b * attn->seq_len * (attn->num_heads * attn->head_dim) + 
                                   t * (attn->num_heads * attn->head_dim) + h * attn->head_dim;
                    memcpy(&attn->concat_heads[layer][dst_offset], 
                           &attn->head_output[layer][h][src_offset], 
                           attn->head_dim * sizeof(float));
                }
            }
        }
        
        // Apply output projection
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len * attn->batch_size, output_size, attn->num_heads * attn->head_dim,
                    1.0f, attn->concat_heads[layer], attn->num_heads * attn->head_dim,
                    attn->W_O[layer], output_size,
                    0.0f, attn->layer_output[layer], output_size);
        
        // Set input for next layer
        if (layer < attn->num_layers - 1) {
            input = attn->layer_output[layer];
        }
    }
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* y) {
    // ∂L/∂Y = Y - Y_true
    int last_layer = attn->num_layers - 1;
    float loss = 0.0f;
    for (int i = 0; i < attn->seq_len * attn->batch_size * attn->output_dim; i++) {
        attn->error_output[last_layer][i] = attn->layer_output[last_layer][i] - y[i];
        loss += attn->error_output[last_layer][i] * attn->error_output[last_layer][i];
    }
    return loss / (attn->seq_len * attn->batch_size * attn->output_dim);
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        for (int head = 0; head < attn->num_heads; head++) {
            int wqkv_size = input_size * attn->head_dim;
            memset(attn->W_Q_grad[layer][head], 0, wqkv_size * sizeof(float));
            memset(attn->W_K_grad[layer][head], 0, wqkv_size * sizeof(float));
            memset(attn->W_V_grad[layer][head], 0, wqkv_size * sizeof(float));
        }
        
        int wo_size = (attn->num_heads * attn->head_dim) * output_size;
        memset(attn->W_O_grad[layer], 0, wo_size * sizeof(float));
    }
}

// Backward pass
void backward_pass_attention(Attention* attn, float* X) {
    for (int layer = attn->num_layers - 1; layer >= 0; layer--) {
        float* input = (layer == 0) ? X : attn->layer_output[layer - 1];
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        // ∂L/∂W_O = concat_heads^T @ (∂L/∂Y)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->num_heads * attn->head_dim, output_size, attn->seq_len * attn->batch_size,
                    1.0f, attn->concat_heads[layer], attn->num_heads * attn->head_dim,
                    attn->error_output[layer], output_size,
                    1.0f, attn->W_O_grad[layer], output_size);
        
        // ∂L/∂concat_heads = (∂L/∂Y) @ W_O^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len * attn->batch_size, attn->num_heads * attn->head_dim, output_size,
                    1.0f, attn->error_output[layer], output_size,
                    attn->W_O[layer], output_size,
                    0.0f, attn->error_concat_heads[layer], attn->num_heads * attn->head_dim);
        
        // Distribute error back to heads
        for (int b = 0; b < attn->batch_size; b++) {
            for (int t = 0; t < attn->seq_len; t++) {
                for (int h = 0; h < attn->num_heads; h++) {
                    int src_offset = b * attn->seq_len * (attn->num_heads * attn->head_dim) + 
                                   t * (attn->num_heads * attn->head_dim) + h * attn->head_dim;
                    int dst_offset = b * attn->seq_len * attn->head_dim + t * attn->head_dim;
                    memcpy(&attn->error_head_output[layer][h][dst_offset],
                           &attn->error_concat_heads[layer][src_offset], 
                           attn->head_dim * sizeof(float));
                }
            }
        }
        
        // Backward through each attention head
        for (int head = 0; head < attn->num_heads; head++) {
            float scale = 1.0f / sqrtf(attn->head_dim);
            
            // Initialize error buffers
            memset(attn->error_Q[layer][head], 0, attn->seq_len * attn->batch_size * attn->head_dim * sizeof(float));
            memset(attn->error_K[layer][head], 0, attn->seq_len * attn->batch_size * attn->head_dim * sizeof(float));
            memset(attn->error_V[layer][head], 0, attn->seq_len * attn->batch_size * attn->head_dim * sizeof(float));
            
            for (int b = 0; b < attn->batch_size; b++) {
                float* Q_b = &attn->Q[layer][head][b * attn->seq_len * attn->head_dim];
                float* K_b = &attn->K[layer][head][b * attn->seq_len * attn->head_dim];
                float* V_b = &attn->V[layer][head][b * attn->seq_len * attn->head_dim];
                float* attn_w = &attn->attn_weights[layer][head][b * attn->seq_len * attn->seq_len];
                float* error_head_out = &attn->error_head_output[layer][head][b * attn->seq_len * attn->head_dim];
                float* error_Q_b = &attn->error_Q[layer][head][b * attn->seq_len * attn->head_dim];
                float* error_K_b = &attn->error_K[layer][head][b * attn->seq_len * attn->head_dim];
                float* error_V_b = &attn->error_V[layer][head][b * attn->seq_len * attn->head_dim];
                
                // ∂L/∂V = attn_weights^T @ (∂L/∂head_output)
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            attn->seq_len, attn->head_dim, attn->seq_len,
                            1.0f, attn_w, attn->seq_len,
                            error_head_out, attn->head_dim,
                            0.0f, error_V_b, attn->head_dim);
                
                // ∂L/∂attn_weights = (∂L/∂head_output) @ V^T
                float* error_attn = (float*)malloc(attn->seq_len * attn->seq_len * sizeof(float));
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            attn->seq_len, attn->seq_len, attn->head_dim,
                            1.0f, error_head_out, attn->head_dim,
                            V_b, attn->head_dim,
                            0.0f, error_attn, attn->seq_len);
                
                // Backward through softmax
                float* error_scores = (float*)malloc(attn->seq_len * attn->seq_len * sizeof(float));
                for (int i = 0; i < attn->seq_len; i++) {
                    for (int j = 0; j < attn->seq_len; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < attn->seq_len; k++) {
                            sum += error_attn[i * attn->seq_len + k] * attn_w[i * attn->seq_len + k];
                        }
                        error_scores[i * attn->seq_len + j] = attn_w[i * attn->seq_len + j] * 
                                                             (error_attn[i * attn->seq_len + j] - sum);
                    }
                }
                
                // ∂L/∂Q = (∂L/∂scores) @ K * scale
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            attn->seq_len, attn->head_dim, attn->seq_len,
                            scale, error_scores, attn->seq_len,
                            K_b, attn->head_dim,
                            0.0f, error_Q_b, attn->head_dim);
                
                // ∂L/∂K = (∂L/∂scores)^T @ Q * scale
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            attn->seq_len, attn->head_dim, attn->seq_len,
                            scale, error_scores, attn->seq_len,
                            Q_b, attn->head_dim,
                            0.0f, error_K_b, attn->head_dim);
                
                free(error_attn);
                free(error_scores);
            }
            
            // ∂L/∂W_Q = input^T @ (∂L/∂Q)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        input_size, attn->head_dim, attn->seq_len * attn->batch_size,
                        1.0f, input, input_size,
                        attn->error_Q[layer][head], attn->head_dim,
                        1.0f, attn->W_Q_grad[layer][head], attn->head_dim);
            
            // ∂L/∂W_K = input^T @ (∂L/∂K)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        input_size, attn->head_dim, attn->seq_len * attn->batch_size,
                        1.0f, input, input_size,
                        attn->error_K[layer][head], attn->head_dim,
                        1.0f, attn->W_K_grad[layer][head], attn->head_dim);
            
            // ∂L/∂W_V = input^T @ (∂L/∂V)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        input_size, attn->head_dim, attn->seq_len * attn->batch_size,
                        1.0f, input, input_size,
                        attn->error_V[layer][head], attn->head_dim,
                        1.0f, attn->W_V_grad[layer][head], attn->head_dim);
        }
        
        // Propagate error to previous layer
        if (layer > 0) {
            // Accumulate errors from all heads
            memset(attn->error_output[layer - 1], 0, attn->seq_len * attn->batch_size * attn->output_dim * sizeof(float));
            
            for (int head = 0; head < attn->num_heads; head++) {
                // ∂L/∂input += (∂L/∂Q) @ W_Q^T + (∂L/∂K) @ W_K^T + (∂L/∂V) @ W_V^T
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            attn->seq_len * attn->batch_size, input_size, attn->head_dim,
                            1.0f, attn->error_Q[layer][head], attn->head_dim,
                            attn->W_Q[layer][head], attn->head_dim,
                            1.0f, attn->error_output[layer - 1], input_size);
                
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            attn->seq_len * attn->batch_size, input_size, attn->head_dim,
                            1.0f, attn->error_K[layer][head], attn->head_dim,
                            attn->W_K[layer][head], attn->head_dim,
                            1.0f, attn->error_output[layer - 1], input_size);
                
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            attn->seq_len * attn->batch_size, input_size, attn->head_dim,
                            1.0f, attn->error_V[layer][head], attn->head_dim,
                            attn->W_V[layer][head], attn->head_dim,
                            1.0f, attn->error_output[layer - 1], input_size);
            }
        }
    }
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;
    int total_samples = attn->seq_len * attn->batch_size;
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        // Update Q, K, V weights for each head
        for (int head = 0; head < attn->num_heads; head++) {
            int wqkv_size = input_size * attn->head_dim;
            
            // Update W_Q
            for (int i = 0; i < wqkv_size; i++) {
                float grad = attn->W_Q_grad[layer][head][i] / total_samples;
                attn->W_Q_m[layer][head][i] = attn->beta1 * attn->W_Q_m[layer][head][i] + (1.0f - attn->beta1) * grad;
                attn->W_Q_v[layer][head][i] = attn->beta2 * attn->W_Q_v[layer][head][i] + (1.0f - attn->beta2) * grad * grad;
                float update = alpha_t * attn->W_Q_m[layer][head][i] / (sqrtf(attn->W_Q_v[layer][head][i]) + attn->epsilon);
                attn->W_Q[layer][head][i] = attn->W_Q[layer][head][i] * (1.0f - learning_rate * attn->weight_decay) - update;
            }
            
            // Update W_K
            for (int i = 0; i < wqkv_size; i++) {
                float grad = attn->W_K_grad[layer][head][i] / total_samples;
                attn->W_K_m[layer][head][i] = attn->beta1 * attn->W_K_m[layer][head][i] + (1.0f - attn->beta1) * grad;
                attn->W_K_v[layer][head][i] = attn->beta2 * attn->W_K_v[layer][head][i] + (1.0f - attn->beta2) * grad * grad;
                float update = alpha_t * attn->W_K_m[layer][head][i] / (sqrtf(attn->W_K_v[layer][head][i]) + attn->epsilon);
                attn->W_K[layer][head][i] = attn->W_K[layer][head][i] * (1.0f - learning_rate * attn->weight_decay) - update;
            }
            
            // Update W_V
            for (int i = 0; i < wqkv_size; i++) {
                float grad = attn->W_V_grad[layer][head][i] / total_samples;
                attn->W_V_m[layer][head][i] = attn->beta1 * attn->W_V_m[layer][head][i] + (1.0f - attn->beta1) * grad;
                attn->W_V_v[layer][head][i] = attn->beta2 * attn->W_V_v[layer][head][i] + (1.0f - attn->beta2) * grad * grad;
                float update = alpha_t * attn->W_V_m[layer][head][i] / (sqrtf(attn->W_V_v[layer][head][i]) + attn->epsilon);
                attn->W_V[layer][head][i] = attn->W_V[layer][head][i] * (1.0f - learning_rate * attn->weight_decay) - update;
            }
        }
        
        // Update W_O
        int wo_size = (attn->num_heads * attn->head_dim) * output_size;
        for (int i = 0; i < wo_size; i++) {
            float grad = attn->W_O_grad[layer][i] / total_samples;
            attn->W_O_m[layer][i] = attn->beta1 * attn->W_O_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->W_O_v[layer][i] = attn->beta2 * attn->W_O_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            float update = alpha_t * attn->W_O_m[layer][i] / (sqrtf(attn->W_O_v[layer][i]) + attn->epsilon);
            attn->W_O[layer][i] = attn->W_O[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
    }
}

// Save model weights to binary file
void save_attention(Attention* attn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&attn->input_dim, sizeof(int), 1, file);
    fwrite(&attn->output_dim, sizeof(int), 1, file);
    fwrite(&attn->head_dim, sizeof(int), 1, file);
    fwrite(&attn->num_heads, sizeof(int), 1, file);
    fwrite(&attn->seq_len, sizeof(int), 1, file);
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    fwrite(&attn->num_layers, sizeof(int), 1, file);
    
    // Save weights for each layer
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        for (int head = 0; head < attn->num_heads; head++) {
            int wqkv_size = input_size * attn->head_dim;
            fwrite(attn->W_Q[layer][head], sizeof(float), wqkv_size, file);
            fwrite(attn->W_K[layer][head], sizeof(float), wqkv_size, file);
            fwrite(attn->W_V[layer][head], sizeof(float), wqkv_size, file);
        }
        
        int wo_size = (attn->num_heads * attn->head_dim) * output_size;
        fwrite(attn->W_O[layer], sizeof(float), wo_size, file);
    }
    
    // Save AdamW state
    fwrite(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        for (int head = 0; head < attn->num_heads; head++) {
            int wqkv_size = input_size * attn->head_dim;
            fwrite(attn->W_Q_m[layer][head], sizeof(float), wqkv_size, file);
            fwrite(attn->W_Q_v[layer][head], sizeof(float), wqkv_size, file);
            fwrite(attn->W_K_m[layer][head], sizeof(float), wqkv_size, file);
            fwrite(attn->W_K_v[layer][head], sizeof(float), wqkv_size, file);
            fwrite(attn->W_V_m[layer][head], sizeof(float), wqkv_size, file);
            fwrite(attn->W_V_v[layer][head], sizeof(float), wqkv_size, file);
        }
        
        int wo_size = (attn->num_heads * attn->head_dim) * output_size;
        fwrite(attn->W_O_m[layer], sizeof(float), wo_size, file);
        fwrite(attn->W_O_v[layer], sizeof(float), wo_size, file);
    }

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights from binary file
Attention* load_attention(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, output_dim, head_dim, num_heads, seq_len, stored_batch_size, num_layers;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&head_dim, sizeof(int), 1, file);
    fread(&num_heads, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    Attention* attn = init_attention(input_dim, output_dim, head_dim, num_heads, seq_len, batch_size, num_layers);
    
    // Load weights for each layer
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : output_dim;
        
        for (int head = 0; head < num_heads; head++) {
            int wqkv_size = input_size * head_dim;
            fread(attn->W_Q[layer][head], sizeof(float), wqkv_size, file);
            fread(attn->W_K[layer][head], sizeof(float), wqkv_size, file);
            fread(attn->W_V[layer][head], sizeof(float), wqkv_size, file);
        }
        
        int wo_size = (num_heads * head_dim) * output_size;
        fread(attn->W_O[layer], sizeof(float), wo_size, file);
    }
    
    // Load AdamW state
    fread(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : output_dim;
        
        for (int head = 0; head < num_heads; head++) {
            int wqkv_size = input_size * head_dim;
            fread(attn->W_Q_m[layer][head], sizeof(float), wqkv_size, file);
            fread(attn->W_Q_v[layer][head], sizeof(float), wqkv_size, file);
            fread(attn->W_K_m[layer][head], sizeof(float), wqkv_size, file);
            fread(attn->W_K_v[layer][head], sizeof(float), wqkv_size, file);
            fread(attn->W_V_m[layer][head], sizeof(float), wqkv_size, file);
            fread(attn->W_V_v[layer][head], sizeof(float), wqkv_size, file);
        }
        
        int wo_size = (num_heads * head_dim) * output_size;
        fread(attn->W_O_m[layer], sizeof(float), wo_size, file);
        fread(attn->W_O_v[layer], sizeof(float), wo_size, file);
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return attn;
}