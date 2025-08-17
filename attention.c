#include "attention.h"

// Initialize the attention network
Attention* init_attention(int d_model, int seq_len, int num_layers, int batch_size) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->d_model = d_model;
    attn->seq_len = seq_len;
    attn->num_layers = num_layers;
    attn->batch_size = batch_size;
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    // Allocate arrays of pointers for weights and gradients
    attn->W_q = (float**)malloc(num_layers * sizeof(float*));
    attn->W_k = (float**)malloc(num_layers * sizeof(float*));
    attn->W_v = (float**)malloc(num_layers * sizeof(float*));
    attn->W_o = (float**)malloc(num_layers * sizeof(float*));
    attn->W_q_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->W_k_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->W_v_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->W_o_grad = (float**)malloc(num_layers * sizeof(float*));
    
    // Allocate Adam buffers
    attn->W_q_m = (float**)malloc(num_layers * sizeof(float*));
    attn->W_q_v = (float**)malloc(num_layers * sizeof(float*));
    attn->W_k_m = (float**)malloc(num_layers * sizeof(float*));
    attn->W_k_v = (float**)malloc(num_layers * sizeof(float*));
    attn->W_v_m = (float**)malloc(num_layers * sizeof(float*));
    attn->W_v_v = (float**)malloc(num_layers * sizeof(float*));
    attn->W_o_m = (float**)malloc(num_layers * sizeof(float*));
    attn->W_o_v = (float**)malloc(num_layers * sizeof(float*));
    
    // Allocate layer outputs and working buffers
    attn->Q = (float**)malloc(num_layers * sizeof(float*));
    attn->K = (float**)malloc(num_layers * sizeof(float*));
    attn->V = (float**)malloc(num_layers * sizeof(float*));
    attn->attn_scores = (float**)malloc(num_layers * sizeof(float*));
    attn->attn_weights = (float**)malloc(num_layers * sizeof(float*));
    attn->attn_output = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_output = (float**)malloc(num_layers * sizeof(float*));
    
    // Allocate gradient buffers
    attn->dQ = (float**)malloc(num_layers * sizeof(float*));
    attn->dK = (float**)malloc(num_layers * sizeof(float*));
    attn->dV = (float**)malloc(num_layers * sizeof(float*));
    attn->d_attn_scores = (float**)malloc(num_layers * sizeof(float*));
    attn->d_attn_weights = (float**)malloc(num_layers * sizeof(float*));
    attn->d_attn_output = (float**)malloc(num_layers * sizeof(float*));
    attn->d_layer_output = (float**)malloc(num_layers * sizeof(float*));
    
    for (int layer = 0; layer < num_layers; layer++) {
        int weight_size = d_model * d_model;
        int qkv_size = batch_size * seq_len * d_model;
        int attn_size = batch_size * seq_len * seq_len;
        
        // Allocate and initialize weights
        attn->W_q[layer] = (float*)malloc(weight_size * sizeof(float));
        attn->W_k[layer] = (float*)malloc(weight_size * sizeof(float));
        attn->W_v[layer] = (float*)malloc(weight_size * sizeof(float));
        attn->W_o[layer] = (float*)malloc(weight_size * sizeof(float));
        attn->W_q_grad[layer] = (float*)malloc(weight_size * sizeof(float));
        attn->W_k_grad[layer] = (float*)malloc(weight_size * sizeof(float));
        attn->W_v_grad[layer] = (float*)malloc(weight_size * sizeof(float));
        attn->W_o_grad[layer] = (float*)malloc(weight_size * sizeof(float));
        
        // Allocate Adam buffers
        attn->W_q_m[layer] = (float*)calloc(weight_size, sizeof(float));
        attn->W_q_v[layer] = (float*)calloc(weight_size, sizeof(float));
        attn->W_k_m[layer] = (float*)calloc(weight_size, sizeof(float));
        attn->W_k_v[layer] = (float*)calloc(weight_size, sizeof(float));
        attn->W_v_m[layer] = (float*)calloc(weight_size, sizeof(float));
        attn->W_v_v[layer] = (float*)calloc(weight_size, sizeof(float));
        attn->W_o_m[layer] = (float*)calloc(weight_size, sizeof(float));
        attn->W_o_v[layer] = (float*)calloc(weight_size, sizeof(float));
        
        // Allocate layer outputs and working buffers
        attn->Q[layer] = (float*)malloc(qkv_size * sizeof(float));
        attn->K[layer] = (float*)malloc(qkv_size * sizeof(float));
        attn->V[layer] = (float*)malloc(qkv_size * sizeof(float));
        attn->attn_scores[layer] = (float*)malloc(attn_size * sizeof(float));
        attn->attn_weights[layer] = (float*)malloc(attn_size * sizeof(float));
        attn->attn_output[layer] = (float*)malloc(qkv_size * sizeof(float));
        attn->layer_output[layer] = (float*)malloc(qkv_size * sizeof(float));
        
        // Allocate gradient buffers
        attn->dQ[layer] = (float*)malloc(qkv_size * sizeof(float));
        attn->dK[layer] = (float*)malloc(qkv_size * sizeof(float));
        attn->dV[layer] = (float*)malloc(qkv_size * sizeof(float));
        attn->d_attn_scores[layer] = (float*)malloc(attn_size * sizeof(float));
        attn->d_attn_weights[layer] = (float*)malloc(attn_size * sizeof(float));
        attn->d_attn_output[layer] = (float*)malloc(qkv_size * sizeof(float));
        attn->d_layer_output[layer] = (float*)malloc(qkv_size * sizeof(float));
        
        // Initialize weights
        float scale = 1.0f / sqrtf(d_model);
        for (int i = 0; i < weight_size; i++) {
            attn->W_q[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
            attn->W_k[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
            attn->W_v[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
            attn->W_o[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    
    return attn;
}

// Free attention network memory
void free_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        // Free weights and gradients
        free(attn->W_q[layer]); free(attn->W_k[layer]); 
        free(attn->W_v[layer]); free(attn->W_o[layer]);
        free(attn->W_q_grad[layer]); free(attn->W_k_grad[layer]); 
        free(attn->W_v_grad[layer]); free(attn->W_o_grad[layer]);
        
        // Free Adam buffers
        free(attn->W_q_m[layer]); free(attn->W_q_v[layer]);
        free(attn->W_k_m[layer]); free(attn->W_k_v[layer]);
        free(attn->W_v_m[layer]); free(attn->W_v_v[layer]);
        free(attn->W_o_m[layer]); free(attn->W_o_v[layer]);
        
        // Free layer outputs and working buffers
        free(attn->Q[layer]); free(attn->K[layer]); free(attn->V[layer]);
        free(attn->attn_scores[layer]); free(attn->attn_weights[layer]);
        free(attn->attn_output[layer]); free(attn->layer_output[layer]);
        
        // Free gradient buffers
        free(attn->dQ[layer]); free(attn->dK[layer]); free(attn->dV[layer]);
        free(attn->d_attn_scores[layer]); free(attn->d_attn_weights[layer]);
        free(attn->d_attn_output[layer]); free(attn->d_layer_output[layer]);
    }
    
    // Free pointer arrays
    free(attn->W_q); free(attn->W_k); free(attn->W_v); free(attn->W_o);
    free(attn->W_q_grad); free(attn->W_k_grad); free(attn->W_v_grad); free(attn->W_o_grad);
    free(attn->W_q_m); free(attn->W_q_v); free(attn->W_k_m); free(attn->W_k_v);
    free(attn->W_v_m); free(attn->W_v_v); free(attn->W_o_m); free(attn->W_o_v);
    free(attn->Q); free(attn->K); free(attn->V);
    free(attn->attn_scores); free(attn->attn_weights);
    free(attn->attn_output); free(attn->layer_output);
    free(attn->dQ); free(attn->dK); free(attn->dV);
    free(attn->d_attn_scores); free(attn->d_attn_weights);
    free(attn->d_attn_output); free(attn->d_layer_output);
    free(attn);
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    float* input = X;
    
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int total_seq = attn->batch_size * attn->seq_len;
        
        // Compute Q, K, V: Q = X * W_q, K = X * W_k, V = X * W_v
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_seq, attn->d_model, attn->d_model,
                    1.0f, input, attn->d_model,
                    attn->W_q[layer], attn->d_model,
                    0.0f, attn->Q[layer], attn->d_model);
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_seq, attn->d_model, attn->d_model,
                    1.0f, input, attn->d_model,
                    attn->W_k[layer], attn->d_model,
                    0.0f, attn->K[layer], attn->d_model);
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_seq, attn->d_model, attn->d_model,
                    1.0f, input, attn->d_model,
                    attn->W_v[layer], attn->d_model,
                    0.0f, attn->V[layer], attn->d_model);
        
        // Compute attention scores for each sequence in the batch
        float scale = 1.0f / sqrtf(attn->d_model);
        for (int batch = 0; batch < attn->batch_size; batch++) {
            float* Q_batch = attn->Q[layer] + batch * attn->seq_len * attn->d_model;
            float* K_batch = attn->K[layer] + batch * attn->seq_len * attn->d_model;
            float* scores_batch = attn->attn_scores[layer] + batch * attn->seq_len * attn->seq_len;
            
            // scores = Q * K^T / sqrt(d_model)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        attn->seq_len, attn->seq_len, attn->d_model,
                        scale, Q_batch, attn->d_model,
                        K_batch, attn->d_model,
                        0.0f, scores_batch, attn->seq_len);
        }
        
        // Apply softmax to attention scores
        for (int batch = 0; batch < attn->batch_size; batch++) {
            float* scores_batch = attn->attn_scores[layer] + batch * attn->seq_len * attn->seq_len;
            float* weights_batch = attn->attn_weights[layer] + batch * attn->seq_len * attn->seq_len;
            
            for (int i = 0; i < attn->seq_len; i++) {
                float* row_scores = scores_batch + i * attn->seq_len;
                float* row_weights = weights_batch + i * attn->seq_len;
                
                // Find max for numerical stability
                float max_val = row_scores[0];
                for (int j = 1; j < attn->seq_len; j++) {
                    if (row_scores[j] > max_val) max_val = row_scores[j];
                }
                
                // Compute softmax
                float sum = 0.0f;
                for (int j = 0; j < attn->seq_len; j++) {
                    row_weights[j] = expf(row_scores[j] - max_val);
                    sum += row_weights[j];
                }
                for (int j = 0; j < attn->seq_len; j++) {
                    row_weights[j] /= sum;
                }
            }
        }
        
        // Compute attention output for each sequence in the batch
        for (int batch = 0; batch < attn->batch_size; batch++) {
            float* weights_batch = attn->attn_weights[layer] + batch * attn->seq_len * attn->seq_len;
            float* V_batch = attn->V[layer] + batch * attn->seq_len * attn->d_model;
            float* output_batch = attn->attn_output[layer] + batch * attn->seq_len * attn->d_model;
            
            // attn_output = weights * V
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len, attn->d_model, attn->seq_len,
                        1.0f, weights_batch, attn->seq_len,
                        V_batch, attn->d_model,
                        0.0f, output_batch, attn->d_model);
        }
        
        // Apply output projection: layer_output = attn_output * W_o
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_seq, attn->d_model, attn->d_model,
                    1.0f, attn->attn_output[layer], attn->d_model,
                    attn->W_o[layer], attn->d_model,
                    0.0f, attn->layer_output[layer], attn->d_model);
        
        // Set input for next layer
        input = attn->layer_output[layer];
    }
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* y) {
    int last_layer = attn->num_layers - 1;
    int total_size = attn->batch_size * attn->seq_len * attn->d_model;
    float loss = 0.0f;
    
    // MSE loss: (1/N) * sum((pred - actual)^2)
    for (int i = 0; i < total_size; i++) {
        float diff = attn->layer_output[last_layer][i] - y[i];
        attn->d_layer_output[last_layer][i] = diff; // Store gradient for backprop
        loss += diff * diff;
    }
    
    return loss / total_size;
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int weight_size = attn->d_model * attn->d_model;
        
        memset(attn->W_q_grad[layer], 0, weight_size * sizeof(float));
        memset(attn->W_k_grad[layer], 0, weight_size * sizeof(float));
        memset(attn->W_v_grad[layer], 0, weight_size * sizeof(float));
        memset(attn->W_o_grad[layer], 0, weight_size * sizeof(float));
    }
}

// Backward pass
void backward_pass_attention(Attention* attn, float* X) {
    for (int layer = attn->num_layers - 1; layer >= 0; layer--) {
        float* input = (layer == 0) ? X : attn->layer_output[layer - 1];
        int total_seq = attn->batch_size * attn->seq_len;
        
        // Gradient w.r.t. W_o: dW_o = attn_output^T * d_layer_output
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, total_seq,
                    1.0f, attn->attn_output[layer], attn->d_model,
                    attn->d_layer_output[layer], attn->d_model,
                    1.0f, attn->W_o_grad[layer], attn->d_model);
        
        // Gradient w.r.t. attention output: d_attn_output = d_layer_output * W_o^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    total_seq, attn->d_model, attn->d_model,
                    1.0f, attn->d_layer_output[layer], attn->d_model,
                    attn->W_o[layer], attn->d_model,
                    0.0f, attn->d_attn_output[layer], attn->d_model);
        
        // Backpropagate through attention mechanism for each sequence in batch
        memset(attn->dQ[layer], 0, total_seq * attn->d_model * sizeof(float));
        memset(attn->dK[layer], 0, total_seq * attn->d_model * sizeof(float));
        memset(attn->dV[layer], 0, total_seq * attn->d_model * sizeof(float));
        
        for (int batch = 0; batch < attn->batch_size; batch++) {
            float* d_attn_output_batch = attn->d_attn_output[layer] + batch * attn->seq_len * attn->d_model;
            float* attn_weights_batch = attn->attn_weights[layer] + batch * attn->seq_len * attn->seq_len;
            float* V_batch = attn->V[layer] + batch * attn->seq_len * attn->d_model;
            float* Q_batch = attn->Q[layer] + batch * attn->seq_len * attn->d_model;
            float* K_batch = attn->K[layer] + batch * attn->seq_len * attn->d_model;
            float* d_attn_weights_batch = attn->d_attn_weights[layer] + batch * attn->seq_len * attn->seq_len;
            float* dQ_batch = attn->dQ[layer] + batch * attn->seq_len * attn->d_model;
            float* dK_batch = attn->dK[layer] + batch * attn->seq_len * attn->d_model;
            float* dV_batch = attn->dV[layer] + batch * attn->seq_len * attn->d_model;
            
            // Gradient w.r.t. V: dV = attn_weights^T * d_attn_output
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        attn->seq_len, attn->d_model, attn->seq_len,
                        1.0f, attn_weights_batch, attn->seq_len,
                        d_attn_output_batch, attn->d_model,
                        0.0f, dV_batch, attn->d_model);
            
            // Gradient w.r.t. attention weights: d_attn_weights = d_attn_output * V^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        attn->seq_len, attn->seq_len, attn->d_model,
                        1.0f, d_attn_output_batch, attn->d_model,
                        V_batch, attn->d_model,
                        0.0f, d_attn_weights_batch, attn->seq_len);
            
            // Backpropagate through softmax
            float* d_attn_scores_batch = attn->d_attn_scores[layer] + batch * attn->seq_len * attn->seq_len;
            for (int i = 0; i < attn->seq_len; i++) {
                float* weights_row = attn_weights_batch + i * attn->seq_len;
                float* d_weights_row = d_attn_weights_batch + i * attn->seq_len;
                float* d_scores_row = d_attn_scores_batch + i * attn->seq_len;
                
                // Softmax backward: d_scores[i] = weights[i] * (d_weights[i] - sum_j(d_weights[j] * weights[j]))
                float sum = 0.0f;
                for (int j = 0; j < attn->seq_len; j++) {
                    sum += d_weights_row[j] * weights_row[j];
                }
                for (int j = 0; j < attn->seq_len; j++) {
                    d_scores_row[j] = weights_row[j] * (d_weights_row[j] - sum);
                }
            }
            
            // Gradient w.r.t. Q: dQ = d_attn_scores * K / sqrt(d_model)
            float scale = 1.0f / sqrtf(attn->d_model);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len, attn->d_model, attn->seq_len,
                        scale, d_attn_scores_batch, attn->seq_len,
                        K_batch, attn->d_model,
                        0.0f, dQ_batch, attn->d_model);
            
            // Gradient w.r.t. K: dK = d_attn_scores^T * Q / sqrt(d_model)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        attn->seq_len, attn->d_model, attn->seq_len,
                        scale, d_attn_scores_batch, attn->seq_len,
                        Q_batch, attn->d_model,
                        0.0f, dK_batch, attn->d_model);
        }
        
        // Gradient w.r.t. weight matrices
        // dW_q = input^T * dQ
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, total_seq,
                    1.0f, input, attn->d_model,
                    attn->dQ[layer], attn->d_model,
                    1.0f, attn->W_q_grad[layer], attn->d_model);
        
        // dW_k = input^T * dK
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, total_seq,
                    1.0f, input, attn->d_model,
                    attn->dK[layer], attn->d_model,
                    1.0f, attn->W_k_grad[layer], attn->d_model);
        
        // dW_v = input^T * dV
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, total_seq,
                    1.0f, input, attn->d_model,
                    attn->dV[layer], attn->d_model,
                    1.0f, attn->W_v_grad[layer], attn->d_model);
        
        // Propagate gradient to previous layer
        if (layer > 0) {
            // d_input = dQ * W_q^T + dK * W_k^T + dV * W_v^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        total_seq, attn->d_model, attn->d_model,
                        1.0f, attn->dQ[layer], attn->d_model,
                        attn->W_q[layer], attn->d_model,
                        0.0f, attn->d_layer_output[layer - 1], attn->d_model);
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        total_seq, attn->d_model, attn->d_model,
                        1.0f, attn->dK[layer], attn->d_model,
                        attn->W_k[layer], attn->d_model,
                        1.0f, attn->d_layer_output[layer - 1], attn->d_model);
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        total_seq, attn->d_model, attn->d_model,
                        1.0f, attn->dV[layer], attn->d_model,
                        attn->W_v[layer], attn->d_model,
                        1.0f, attn->d_layer_output[layer - 1], attn->d_model);
        }
    }
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int weight_size = attn->d_model * attn->d_model;
        
        // Update W_q weights
        for (int i = 0; i < weight_size; i++) {
            float grad = attn->W_q_grad[layer][i] / (attn->batch_size * attn->seq_len);
            
            attn->W_q_m[layer][i] = attn->beta1 * attn->W_q_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->W_q_v[layer][i] = attn->beta2 * attn->W_q_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->W_q_m[layer][i] / (sqrtf(attn->W_q_v[layer][i]) + attn->epsilon);
            attn->W_q[layer][i] = attn->W_q[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update W_k weights
        for (int i = 0; i < weight_size; i++) {
            float grad = attn->W_k_grad[layer][i] / (attn->batch_size * attn->seq_len);
            
            attn->W_k_m[layer][i] = attn->beta1 * attn->W_k_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->W_k_v[layer][i] = attn->beta2 * attn->W_k_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->W_k_m[layer][i] / (sqrtf(attn->W_k_v[layer][i]) + attn->epsilon);
            attn->W_k[layer][i] = attn->W_k[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update W_v weights
        for (int i = 0; i < weight_size; i++) {
            float grad = attn->W_v_grad[layer][i] / (attn->batch_size * attn->seq_len);
            
            attn->W_v_m[layer][i] = attn->beta1 * attn->W_v_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->W_v_v[layer][i] = attn->beta2 * attn->W_v_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->W_v_m[layer][i] / (sqrtf(attn->W_v_v[layer][i]) + attn->epsilon);
            attn->W_v[layer][i] = attn->W_v[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update W_o weights
        for (int i = 0; i < weight_size; i++) {
            float grad = attn->W_o_grad[layer][i] / (attn->batch_size * attn->seq_len);
            
            attn->W_o_m[layer][i] = attn->beta1 * attn->W_o_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->W_o_v[layer][i] = attn->beta2 * attn->W_o_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->W_o_m[layer][i] / (sqrtf(attn->W_o_v[layer][i]) + attn->epsilon);
            attn->W_o[layer][i] = attn->W_o[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
    }
}

// Save model weights and Adam state to binary file
void save_attention(Attention* attn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&attn->d_model, sizeof(int), 1, file);
    fwrite(&attn->seq_len, sizeof(int), 1, file);
    fwrite(&attn->num_layers, sizeof(int), 1, file);
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    
    // Save weights for each layer
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int weight_size = attn->d_model * attn->d_model;
        
        fwrite(attn->W_q[layer], sizeof(float), weight_size, file);
        fwrite(attn->W_k[layer], sizeof(float), weight_size, file);
        fwrite(attn->W_v[layer], sizeof(float), weight_size, file);
        fwrite(attn->W_o[layer], sizeof(float), weight_size, file);
    }
    
    // Save Adam state
    fwrite(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int weight_size = attn->d_model * attn->d_model;
        
        fwrite(attn->W_q_m[layer], sizeof(float), weight_size, file);
        fwrite(attn->W_q_v[layer], sizeof(float), weight_size, file);
        fwrite(attn->W_k_m[layer], sizeof(float), weight_size, file);
        fwrite(attn->W_k_v[layer], sizeof(float), weight_size, file);
        fwrite(attn->W_v_m[layer], sizeof(float), weight_size, file);
        fwrite(attn->W_v_v[layer], sizeof(float), weight_size, file);
        fwrite(attn->W_o_m[layer], sizeof(float), weight_size, file);
        fwrite(attn->W_o_v[layer], sizeof(float), weight_size, file);
    }

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights and Adam state from binary file
Attention* load_attention(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int d_model, seq_len, num_layers, stored_batch_size;
    fread(&d_model, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    Attention* attn = init_attention(d_model, seq_len, num_layers, batch_size);
    
    // Load weights for each layer
    for (int layer = 0; layer < num_layers; layer++) {
        int weight_size = d_model * d_model;
        
        fread(attn->W_q[layer], sizeof(float), weight_size, file);
        fread(attn->W_k[layer], sizeof(float), weight_size, file);
        fread(attn->W_v[layer], sizeof(float), weight_size, file);
        fread(attn->W_o[layer], sizeof(float), weight_size, file);
    }
    
    // Load Adam state
    fread(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < num_layers; layer++) {
        int weight_size = d_model * d_model;
        
        fread(attn->W_q_m[layer], sizeof(float), weight_size, file);
        fread(attn->W_q_v[layer], sizeof(float), weight_size, file);
        fread(attn->W_k_m[layer], sizeof(float), weight_size, file);
        fread(attn->W_k_v[layer], sizeof(float), weight_size, file);
        fread(attn->W_v_m[layer], sizeof(float), weight_size, file);
        fread(attn->W_v_v[layer], sizeof(float), weight_size, file);
        fread(attn->W_o_m[layer], sizeof(float), weight_size, file);
        fread(attn->W_o_v[layer], sizeof(float), weight_size, file);
    }

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return attn;
}