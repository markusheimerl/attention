#include "attention.h"

// Initialize the network with configurable dimensions
Attention* init_attention(int input_dim, int hidden_dim, int output_dim, int seq_len, int batch_size, int num_layers) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->input_dim = input_dim;
    attn->hidden_dim = hidden_dim;
    attn->output_dim = output_dim;
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
    attn->W_q = (float**)malloc(num_layers * sizeof(float*));
    attn->W_k = (float**)malloc(num_layers * sizeof(float*));
    attn->W_v = (float**)malloc(num_layers * sizeof(float*));
    attn->W_o = (float**)malloc(num_layers * sizeof(float*));
    attn->W_r = (float**)malloc(num_layers * sizeof(float*));
    attn->W_q_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->W_k_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->W_v_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->W_o_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->W_r_grad = (float**)malloc(num_layers * sizeof(float*));
    
    // Allocate AdamW buffers
    attn->W_q_m = (float**)malloc(num_layers * sizeof(float*));
    attn->W_q_v = (float**)malloc(num_layers * sizeof(float*));
    attn->W_k_m = (float**)malloc(num_layers * sizeof(float*));
    attn->W_k_v = (float**)malloc(num_layers * sizeof(float*));
    attn->W_v_m = (float**)malloc(num_layers * sizeof(float*));
    attn->W_v_v = (float**)malloc(num_layers * sizeof(float*));
    attn->W_o_m = (float**)malloc(num_layers * sizeof(float*));
    attn->W_o_v = (float**)malloc(num_layers * sizeof(float*));
    attn->W_r_m = (float**)malloc(num_layers * sizeof(float*));
    attn->W_r_v = (float**)malloc(num_layers * sizeof(float*));
    
    // Allocate layer outputs and working buffers
    attn->layer_q = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_k = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_v = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_scores = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_weights = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_context = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_preact = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_postact = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_output = (float**)malloc(num_layers * sizeof(float*));
    attn->error_context = (float**)malloc(num_layers * sizeof(float*));
    attn->error_weights = (float**)malloc(num_layers * sizeof(float*));
    attn->error_values = (float**)malloc(num_layers * sizeof(float*));
    attn->error_keys = (float**)malloc(num_layers * sizeof(float*));
    attn->error_queries = (float**)malloc(num_layers * sizeof(float*));
    attn->error_preact = (float**)malloc(num_layers * sizeof(float*));
    attn->error_output = (float**)malloc(num_layers * sizeof(float*));
    
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : output_dim;
        
        int w_qkv_size = hidden_dim * input_size;
        int w_o_size = output_size * hidden_dim;
        int w_r_size = output_size * input_size;
        
        // Allocate and initialize matrices and gradients
        attn->W_q[layer] = (float*)malloc(w_qkv_size * sizeof(float));
        attn->W_k[layer] = (float*)malloc(w_qkv_size * sizeof(float));
        attn->W_v[layer] = (float*)malloc(w_qkv_size * sizeof(float));
        attn->W_o[layer] = (float*)malloc(w_o_size * sizeof(float));
        attn->W_r[layer] = (float*)malloc(w_r_size * sizeof(float));
        attn->W_q_grad[layer] = (float*)malloc(w_qkv_size * sizeof(float));
        attn->W_k_grad[layer] = (float*)malloc(w_qkv_size * sizeof(float));
        attn->W_v_grad[layer] = (float*)malloc(w_qkv_size * sizeof(float));
        attn->W_o_grad[layer] = (float*)malloc(w_o_size * sizeof(float));
        attn->W_r_grad[layer] = (float*)malloc(w_r_size * sizeof(float));
        
        // Allocate AdamW buffers
        attn->W_q_m[layer] = (float*)calloc(w_qkv_size, sizeof(float));
        attn->W_q_v[layer] = (float*)calloc(w_qkv_size, sizeof(float));
        attn->W_k_m[layer] = (float*)calloc(w_qkv_size, sizeof(float));
        attn->W_k_v[layer] = (float*)calloc(w_qkv_size, sizeof(float));
        attn->W_v_m[layer] = (float*)calloc(w_qkv_size, sizeof(float));
        attn->W_v_v[layer] = (float*)calloc(w_qkv_size, sizeof(float));
        attn->W_o_m[layer] = (float*)calloc(w_o_size, sizeof(float));
        attn->W_o_v[layer] = (float*)calloc(w_o_size, sizeof(float));
        attn->W_r_m[layer] = (float*)calloc(w_r_size, sizeof(float));
        attn->W_r_v[layer] = (float*)calloc(w_r_size, sizeof(float));
        
        // Allocate layer outputs and working buffers
        int seq_batch_size = seq_len * batch_size;
        attn->layer_q[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->layer_k[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->layer_v[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->layer_scores[layer] = (float*)malloc(seq_batch_size * seq_len * sizeof(float));
        attn->layer_weights[layer] = (float*)malloc(seq_batch_size * seq_len * sizeof(float));
        attn->layer_context[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->layer_preact[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->layer_postact[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->layer_output[layer] = (float*)malloc(seq_batch_size * output_size * sizeof(float));
        attn->error_context[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->error_weights[layer] = (float*)malloc(seq_batch_size * seq_len * sizeof(float));
        attn->error_values[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->error_keys[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->error_queries[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->error_preact[layer] = (float*)malloc(seq_batch_size * hidden_dim * sizeof(float));
        attn->error_output[layer] = (float*)malloc(seq_batch_size * output_size * sizeof(float));
        
        // Initialize matrices
        float scale_qkv = 1.0f / sqrtf(input_size);
        float scale_o = 1.0f / sqrtf(hidden_dim);
        float scale_r = 1.0f / sqrtf(input_size);
        
        for (int i = 0; i < w_qkv_size; i++) {
            attn->W_q[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_qkv;
            attn->W_k[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_qkv;
            attn->W_v[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_qkv;
        }
        
        for (int i = 0; i < w_o_size; i++) {
            attn->W_o[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_o;
        }
        
        for (int i = 0; i < w_r_size; i++) {
            attn->W_r[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_r;
        }
    }
    
    return attn;
}

// Free network memory
void free_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        free(attn->W_q[layer]); free(attn->W_k[layer]); free(attn->W_v[layer]); 
        free(attn->W_o[layer]); free(attn->W_r[layer]);
        free(attn->W_q_grad[layer]); free(attn->W_k_grad[layer]); free(attn->W_v_grad[layer]); 
        free(attn->W_o_grad[layer]); free(attn->W_r_grad[layer]);
        free(attn->W_q_m[layer]); free(attn->W_q_v[layer]);
        free(attn->W_k_m[layer]); free(attn->W_k_v[layer]);
        free(attn->W_v_m[layer]); free(attn->W_v_v[layer]);
        free(attn->W_o_m[layer]); free(attn->W_o_v[layer]);
        free(attn->W_r_m[layer]); free(attn->W_r_v[layer]);
        free(attn->layer_q[layer]); free(attn->layer_k[layer]); free(attn->layer_v[layer]);
        free(attn->layer_scores[layer]); free(attn->layer_weights[layer]); free(attn->layer_context[layer]);
        free(attn->layer_preact[layer]); free(attn->layer_postact[layer]); free(attn->layer_output[layer]);
        free(attn->error_context[layer]); free(attn->error_weights[layer]); free(attn->error_values[layer]);
        free(attn->error_keys[layer]); free(attn->error_queries[layer]); free(attn->error_preact[layer]);
        free(attn->error_output[layer]);
    }
    
    free(attn->W_q); free(attn->W_k); free(attn->W_v); free(attn->W_o); free(attn->W_r);
    free(attn->W_q_grad); free(attn->W_k_grad); free(attn->W_v_grad); free(attn->W_o_grad); free(attn->W_r_grad);
    free(attn->W_q_m); free(attn->W_q_v); free(attn->W_k_m); free(attn->W_k_v); 
    free(attn->W_v_m); free(attn->W_v_v); free(attn->W_o_m); free(attn->W_o_v);
    free(attn->W_r_m); free(attn->W_r_v);
    free(attn->layer_q); free(attn->layer_k); free(attn->layer_v);
    free(attn->layer_scores); free(attn->layer_weights); free(attn->layer_context);
    free(attn->layer_preact); free(attn->layer_postact); free(attn->layer_output);
    free(attn->error_context); free(attn->error_weights); free(attn->error_values);
    free(attn->error_keys); free(attn->error_queries); free(attn->error_preact); free(attn->error_output);
    free(attn);
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    float* input = X;
    int seq_batch_size = attn->seq_len * attn->batch_size;
    
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        // Q = XW_q, K = XW_k, V = XW_v
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_batch_size, attn->hidden_dim, input_size,
                    1.0f, input, input_size,
                    attn->W_q[layer], attn->hidden_dim,
                    0.0f, attn->layer_q[layer], attn->hidden_dim);
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_batch_size, attn->hidden_dim, input_size,
                    1.0f, input, input_size,
                    attn->W_k[layer], attn->hidden_dim,
                    0.0f, attn->layer_k[layer], attn->hidden_dim);
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_batch_size, attn->hidden_dim, input_size,
                    1.0f, input, input_size,
                    attn->W_v[layer], attn->hidden_dim,
                    0.0f, attn->layer_v[layer], attn->hidden_dim);
        
        // Compute attention scores for each batch
        float scale = 1.0f / sqrtf(attn->hidden_dim);
        for (int b = 0; b < attn->batch_size; b++) {
            float* Q_b = attn->layer_q[layer] + b * attn->seq_len * attn->hidden_dim;
            float* K_b = attn->layer_k[layer] + b * attn->seq_len * attn->hidden_dim;
            float* scores_b = attn->layer_scores[layer] + b * attn->seq_len * attn->seq_len;
            
            // Scores = QK^T / sqrt(d_k)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        attn->seq_len, attn->seq_len, attn->hidden_dim,
                        scale, Q_b, attn->hidden_dim,
                        K_b, attn->hidden_dim,
                        0.0f, scores_b, attn->seq_len);
        }
        
        // Apply softmax to attention scores
        for (int b = 0; b < attn->batch_size; b++) {
            for (int i = 0; i < attn->seq_len; i++) {
                float* scores_row = attn->layer_scores[layer] + b * attn->seq_len * attn->seq_len + i * attn->seq_len;
                float* weights_row = attn->layer_weights[layer] + b * attn->seq_len * attn->seq_len + i * attn->seq_len;
                
                // Find max for numerical stability
                float max_score = scores_row[0];
                for (int j = 1; j < attn->seq_len; j++) {
                    if (scores_row[j] > max_score) max_score = scores_row[j];
                }
                
                // Compute softmax
                float sum_exp = 0.0f;
                for (int j = 0; j < attn->seq_len; j++) {
                    weights_row[j] = expf(scores_row[j] - max_score);
                    sum_exp += weights_row[j];
                }
                
                for (int j = 0; j < attn->seq_len; j++) {
                    weights_row[j] /= sum_exp;
                }
            }
        }
        
        // Compute context vectors for each batch
        for (int b = 0; b < attn->batch_size; b++) {
            float* weights_b = attn->layer_weights[layer] + b * attn->seq_len * attn->seq_len;
            float* V_b = attn->layer_v[layer] + b * attn->seq_len * attn->hidden_dim;
            float* context_b = attn->layer_context[layer] + b * attn->seq_len * attn->hidden_dim;
            
            // Context = Attention_weights * V
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len, attn->hidden_dim, attn->seq_len,
                        1.0f, weights_b, attn->seq_len,
                        V_b, attn->hidden_dim,
                        0.0f, context_b, attn->hidden_dim);
        }
        
        // H = Context (pre-activation for swish)
        memcpy(attn->layer_preact[layer], attn->layer_context[layer], 
               seq_batch_size * attn->hidden_dim * sizeof(float));
        
        // S = Hσ(H) (swish activation)
        for (int i = 0; i < seq_batch_size * attn->hidden_dim; i++) {
            attn->layer_postact[layer][i] = attn->layer_preact[layer][i] / (1.0f + expf(-attn->layer_preact[layer][i]));
        }
        
        // Y = SW_o
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_batch_size, output_size, attn->hidden_dim,
                    1.0f, attn->layer_postact[layer], attn->hidden_dim,
                    attn->W_o[layer], output_size,
                    0.0f, attn->layer_output[layer], output_size);
        
        // Y = Y + XW_r (residual connection)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_batch_size, output_size, input_size,
                    1.0f, input, input_size,
                    attn->W_r[layer], output_size,
                    1.0f, attn->layer_output[layer], output_size);
        
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
    int seq_batch_size = attn->seq_len * attn->batch_size;
    
    for (int i = 0; i < seq_batch_size * attn->output_dim; i++) {
        attn->error_output[last_layer][i] = attn->layer_output[last_layer][i] - y[i];
        loss += attn->error_output[last_layer][i] * attn->error_output[last_layer][i];
    }
    return loss / (seq_batch_size * attn->output_dim);
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        int w_qkv_size = attn->hidden_dim * input_size;
        int w_o_size = output_size * attn->hidden_dim;
        int w_r_size = output_size * input_size;
        
        memset(attn->W_q_grad[layer], 0, w_qkv_size * sizeof(float));
        memset(attn->W_k_grad[layer], 0, w_qkv_size * sizeof(float));
        memset(attn->W_v_grad[layer], 0, w_qkv_size * sizeof(float));
        memset(attn->W_o_grad[layer], 0, w_o_size * sizeof(float));
        memset(attn->W_r_grad[layer], 0, w_r_size * sizeof(float));
    }
}

// Backward pass
void backward_pass_attention(Attention* attn, float* X) {
    int seq_batch_size = attn->seq_len * attn->batch_size;
    
    for (int layer = attn->num_layers - 1; layer >= 0; layer--) {
        float* input = (layer == 0) ? X : attn->layer_output[layer - 1];
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        // ∂L/∂W_o = S^T(∂L/∂Y)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->hidden_dim, output_size, seq_batch_size,
                    1.0f, attn->layer_postact[layer], attn->hidden_dim,
                    attn->error_output[layer], output_size,
                    1.0f, attn->W_o_grad[layer], output_size);
        
        // ∂L/∂W_r = X^T(∂L/∂Y)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    input_size, output_size, seq_batch_size,
                    1.0f, input, input_size,
                    attn->error_output[layer], output_size,
                    1.0f, attn->W_r_grad[layer], output_size);
        
        // ∂L/∂S = (∂L/∂Y)(W_o)^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_batch_size, attn->hidden_dim, output_size,
                    1.0f, attn->error_output[layer], output_size,
                    attn->W_o[layer], output_size,
                    0.0f, attn->error_preact[layer], attn->hidden_dim);
        
        // ∂L/∂H = ∂L/∂S ⊙ [σ(H) + Hσ(H)(1-σ(H))] (swish derivative)
        for (int i = 0; i < seq_batch_size * attn->hidden_dim; i++) {
            float h = attn->layer_preact[layer][i];
            float sigmoid = 1.0f / (1.0f + expf(-h));
            attn->error_context[layer][i] = attn->error_preact[layer][i] * (sigmoid + h * sigmoid * (1.0f - sigmoid));
        }
        
        // Backpropagate through attention mechanism for each batch
        memset(attn->error_values[layer], 0, seq_batch_size * attn->hidden_dim * sizeof(float));
        memset(attn->error_weights[layer], 0, seq_batch_size * attn->seq_len * sizeof(float));
        
        for (int b = 0; b < attn->batch_size; b++) {
            float* error_context_b = attn->error_context[layer] + b * attn->seq_len * attn->hidden_dim;
            float* error_weights_b = attn->error_weights[layer] + b * attn->seq_len * attn->seq_len;
            float* error_values_b = attn->error_values[layer] + b * attn->seq_len * attn->hidden_dim;
            float* weights_b = attn->layer_weights[layer] + b * attn->seq_len * attn->seq_len;
            float* values_b = attn->layer_v[layer] + b * attn->seq_len * attn->hidden_dim;
            
            // ∂L/∂Weights = ∂L/∂Context * V^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        attn->seq_len, attn->seq_len, attn->hidden_dim,
                        1.0f, error_context_b, attn->hidden_dim,
                        values_b, attn->hidden_dim,
                        0.0f, error_weights_b, attn->seq_len);
            
            // ∂L/∂V = Weights^T * ∂L/∂Context
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        attn->seq_len, attn->hidden_dim, attn->seq_len,
                        1.0f, weights_b, attn->seq_len,
                        error_context_b, attn->hidden_dim,
                        0.0f, error_values_b, attn->hidden_dim);
        }
        
        // Backpropagate through softmax
        memset(attn->error_queries[layer], 0, seq_batch_size * attn->hidden_dim * sizeof(float));
        memset(attn->error_keys[layer], 0, seq_batch_size * attn->hidden_dim * sizeof(float));
        
        float scale = 1.0f / sqrtf(attn->hidden_dim);
        for (int b = 0; b < attn->batch_size; b++) {
            float* error_weights_b = attn->error_weights[layer] + b * attn->seq_len * attn->seq_len;
            float* weights_b = attn->layer_weights[layer] + b * attn->seq_len * attn->seq_len;
            float* queries_b = attn->layer_q[layer] + b * attn->seq_len * attn->hidden_dim;
            float* keys_b = attn->layer_k[layer] + b * attn->seq_len * attn->hidden_dim;
            float* error_queries_b = attn->error_queries[layer] + b * attn->seq_len * attn->hidden_dim;
            float* error_keys_b = attn->error_keys[layer] + b * attn->seq_len * attn->hidden_dim;
            
            // Softmax gradient: ∂L/∂scores[i,j] = weights[i,j] * (error_weights[i,j] - sum_k(error_weights[i,k] * weights[i,k]))
            float* error_scores_b = (float*)malloc(attn->seq_len * attn->seq_len * sizeof(float));
            for (int i = 0; i < attn->seq_len; i++) {
                float sum_term = 0.0f;
                for (int k = 0; k < attn->seq_len; k++) {
                    sum_term += error_weights_b[i * attn->seq_len + k] * weights_b[i * attn->seq_len + k];
                }
                for (int j = 0; j < attn->seq_len; j++) {
                    error_scores_b[i * attn->seq_len + j] = weights_b[i * attn->seq_len + j] * 
                        (error_weights_b[i * attn->seq_len + j] - sum_term);
                }
            }
            
            // ∂L/∂Q = ∂L/∂scores * K * scale
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len, attn->hidden_dim, attn->seq_len,
                        scale, error_scores_b, attn->seq_len,
                        keys_b, attn->hidden_dim,
                        0.0f, error_queries_b, attn->hidden_dim);
            
            // ∂L/∂K = ∂L/∂scores^T * Q * scale
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        attn->seq_len, attn->hidden_dim, attn->seq_len,
                        scale, error_scores_b, attn->seq_len,
                        queries_b, attn->hidden_dim,
                        0.0f, error_keys_b, attn->hidden_dim);
            
            free(error_scores_b);
        }
        
        // ∂L/∂W_q = X^T(∂L/∂Q)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    input_size, attn->hidden_dim, seq_batch_size,
                    1.0f, input, input_size,
                    attn->error_queries[layer], attn->hidden_dim,
                    1.0f, attn->W_q_grad[layer], attn->hidden_dim);
        
        // ∂L/∂W_k = X^T(∂L/∂K)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    input_size, attn->hidden_dim, seq_batch_size,
                    1.0f, input, input_size,
                    attn->error_keys[layer], attn->hidden_dim,
                    1.0f, attn->W_k_grad[layer], attn->hidden_dim);
        
        // ∂L/∂W_v = X^T(∂L/∂V)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    input_size, attn->hidden_dim, seq_batch_size,
                    1.0f, input, input_size,
                    attn->error_values[layer], attn->hidden_dim,
                    1.0f, attn->W_v_grad[layer], attn->hidden_dim);
        
        // Propagate error to previous layer
        if (layer > 0) {
            // ∂L/∂X = (∂L/∂Q)(W_q)^T + (∂L/∂K)(W_k)^T + (∂L/∂V)(W_v)^T + (∂L/∂Y)(W_r)^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_batch_size, input_size, attn->hidden_dim,
                        1.0f, attn->error_queries[layer], attn->hidden_dim,
                        attn->W_q[layer], attn->hidden_dim,
                        0.0f, attn->error_output[layer - 1], input_size);
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_batch_size, input_size, attn->hidden_dim,
                        1.0f, attn->error_keys[layer], attn->hidden_dim,
                        attn->W_k[layer], attn->hidden_dim,
                        1.0f, attn->error_output[layer - 1], input_size);
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_batch_size, input_size, attn->hidden_dim,
                        1.0f, attn->error_values[layer], attn->hidden_dim,
                        attn->W_v[layer], attn->hidden_dim,
                        1.0f, attn->error_output[layer - 1], input_size);
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_batch_size, input_size, output_size,
                        1.0f, attn->error_output[layer], output_size,
                        attn->W_r[layer], output_size,
                        1.0f, attn->error_output[layer - 1], input_size);
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
        
        int w_qkv_size = attn->hidden_dim * input_size;
        int w_o_size = output_size * attn->hidden_dim;
        int w_r_size = output_size * input_size;
        
        // Update W_q weights
        for (int i = 0; i < w_qkv_size; i++) {
            float grad = attn->W_q_grad[layer][i] / total_samples;
            
            attn->W_q_m[layer][i] = attn->beta1 * attn->W_q_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->W_q_v[layer][i] = attn->beta2 * attn->W_q_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->W_q_m[layer][i] / (sqrtf(attn->W_q_v[layer][i]) + attn->epsilon);
            attn->W_q[layer][i] = attn->W_q[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update W_k weights
        for (int i = 0; i < w_qkv_size; i++) {
            float grad = attn->W_k_grad[layer][i] / total_samples;
            
            attn->W_k_m[layer][i] = attn->beta1 * attn->W_k_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->W_k_v[layer][i] = attn->beta2 * attn->W_k_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->W_k_m[layer][i] / (sqrtf(attn->W_k_v[layer][i]) + attn->epsilon);
            attn->W_k[layer][i] = attn->W_k[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update W_v weights
        for (int i = 0; i < w_qkv_size; i++) {
            float grad = attn->W_v_grad[layer][i] / total_samples;
            
            attn->W_v_m[layer][i] = attn->beta1 * attn->W_v_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->W_v_v[layer][i] = attn->beta2 * attn->W_v_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->W_v_m[layer][i] / (sqrtf(attn->W_v_v[layer][i]) + attn->epsilon);
            attn->W_v[layer][i] = attn->W_v[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update W_o weights
        for (int i = 0; i < w_o_size; i++) {
            float grad = attn->W_o_grad[layer][i] / total_samples;
            
            attn->W_o_m[layer][i] = attn->beta1 * attn->W_o_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->W_o_v[layer][i] = attn->beta2 * attn->W_o_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->W_o_m[layer][i] / (sqrtf(attn->W_o_v[layer][i]) + attn->epsilon);
            attn->W_o[layer][i] = attn->W_o[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update W_r weights
        for (int i = 0; i < w_r_size; i++) {
            float grad = attn->W_r_grad[layer][i] / total_samples;
            
            attn->W_r_m[layer][i] = attn->beta1 * attn->W_r_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->W_r_v[layer][i] = attn->beta2 * attn->W_r_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->W_r_m[layer][i] / (sqrtf(attn->W_r_v[layer][i]) + attn->epsilon);
            attn->W_r[layer][i] = attn->W_r[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
    }
}

// Save model weights and AdamW state to binary file
void save_attention(Attention* attn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&attn->input_dim, sizeof(int), 1, file);
    fwrite(&attn->hidden_dim, sizeof(int), 1, file);
    fwrite(&attn->output_dim, sizeof(int), 1, file);
    fwrite(&attn->seq_len, sizeof(int), 1, file);
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    fwrite(&attn->num_layers, sizeof(int), 1, file);
    
    // Save weights for each layer
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        int w_qkv_size = attn->hidden_dim * input_size;
        int w_o_size = output_size * attn->hidden_dim;
        int w_r_size = output_size * input_size;
        
        fwrite(attn->W_q[layer], sizeof(float), w_qkv_size, file);
        fwrite(attn->W_k[layer], sizeof(float), w_qkv_size, file);
        fwrite(attn->W_v[layer], sizeof(float), w_qkv_size, file);
        fwrite(attn->W_o[layer], sizeof(float), w_o_size, file);
        fwrite(attn->W_r[layer], sizeof(float), w_r_size, file);
    }
    
    // Save AdamW state
    fwrite(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        int w_qkv_size = attn->hidden_dim * input_size;
        int w_o_size = output_size * attn->hidden_dim;
        int w_r_size = output_size * input_size;
        
        fwrite(attn->W_q_m[layer], sizeof(float), w_qkv_size, file);
        fwrite(attn->W_q_v[layer], sizeof(float), w_qkv_size, file);
        fwrite(attn->W_k_m[layer], sizeof(float), w_qkv_size, file);
        fwrite(attn->W_k_v[layer], sizeof(float), w_qkv_size, file);
        fwrite(attn->W_v_m[layer], sizeof(float), w_qkv_size, file);
        fwrite(attn->W_v_v[layer], sizeof(float), w_qkv_size, file);
        fwrite(attn->W_o_m[layer], sizeof(float), w_o_size, file);
        fwrite(attn->W_o_v[layer], sizeof(float), w_o_size, file);
        fwrite(attn->W_r_m[layer], sizeof(float), w_r_size, file);
        fwrite(attn->W_r_v[layer], sizeof(float), w_r_size, file);
    }

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights and AdamW state from binary file
Attention* load_attention(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, hidden_dim, output_dim, seq_len, stored_batch_size, num_layers;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    Attention* attn = init_attention(input_dim, hidden_dim, output_dim, seq_len, batch_size, num_layers);
    
    // Load weights for each layer
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : output_dim;
        
        int w_qkv_size = hidden_dim * input_size;
        int w_o_size = output_size * hidden_dim;
        int w_r_size = output_size * input_size;
        
        fread(attn->W_q[layer], sizeof(float), w_qkv_size, file);
        fread(attn->W_k[layer], sizeof(float), w_qkv_size, file);
        fread(attn->W_v[layer], sizeof(float), w_qkv_size, file);
        fread(attn->W_o[layer], sizeof(float), w_o_size, file);
        fread(attn->W_r[layer], sizeof(float), w_r_size, file);
    }
    
    // Load AdamW state
    fread(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : output_dim;
        
        int w_qkv_size = hidden_dim * input_size;
        int w_o_size = output_size * hidden_dim;
        int w_r_size = output_size * input_size;
        
        fread(attn->W_q_m[layer], sizeof(float), w_qkv_size, file);
        fread(attn->W_q_v[layer], sizeof(float), w_qkv_size, file);
        fread(attn->W_k_m[layer], sizeof(float), w_qkv_size, file);
        fread(attn->W_k_v[layer], sizeof(float), w_qkv_size, file);
        fread(attn->W_v_m[layer], sizeof(float), w_qkv_size, file);
        fread(attn->W_v_v[layer], sizeof(float), w_qkv_size, file);
        fread(attn->W_o_m[layer], sizeof(float), w_o_size, file);
        fread(attn->W_o_v[layer], sizeof(float), w_o_size, file);
        fread(attn->W_r_m[layer], sizeof(float), w_r_size, file);
        fread(attn->W_r_v[layer], sizeof(float), w_r_size, file);
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return attn;
}