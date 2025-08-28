#include "attention.h"

// Initialize the attention network
Attention* init_attention(int d_model, int seq_len, int batch_size) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->d_model = d_model;
    attn->seq_len = seq_len;
    attn->batch_size = batch_size;
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    int weight_size = d_model * d_model;
    int seq_size = batch_size * seq_len * d_model;
    int attn_size = batch_size * seq_len * seq_len;
    
    // Allocate weights and gradient accumulators
    attn->W_q = (float*)malloc(weight_size * sizeof(float));
    attn->W_k = (float*)malloc(weight_size * sizeof(float));
    attn->W_v = (float*)malloc(weight_size * sizeof(float));
    attn->W_o = (float*)malloc(weight_size * sizeof(float));
    attn->W_q_grad = (float*)malloc(weight_size * sizeof(float));
    attn->W_k_grad = (float*)malloc(weight_size * sizeof(float));
    attn->W_v_grad = (float*)malloc(weight_size * sizeof(float));
    attn->W_o_grad = (float*)malloc(weight_size * sizeof(float));
    
    // Allocate Adam buffers
    attn->W_q_m = (float*)calloc(weight_size, sizeof(float));
    attn->W_q_v = (float*)calloc(weight_size, sizeof(float));
    attn->W_k_m = (float*)calloc(weight_size, sizeof(float));
    attn->W_k_v = (float*)calloc(weight_size, sizeof(float));
    attn->W_v_m = (float*)calloc(weight_size, sizeof(float));
    attn->W_v_v = (float*)calloc(weight_size, sizeof(float));
    attn->W_o_m = (float*)calloc(weight_size, sizeof(float));
    attn->W_o_v = (float*)calloc(weight_size, sizeof(float));
    
    // Allocate forward pass buffers
    attn->Q = (float*)malloc(seq_size * sizeof(float));
    attn->K = (float*)malloc(seq_size * sizeof(float));
    attn->V = (float*)malloc(seq_size * sizeof(float));
    attn->attn_scores = (float*)malloc(attn_size * sizeof(float));
    attn->attn_weights = (float*)malloc(attn_size * sizeof(float));
    attn->attn_output = (float*)malloc(seq_size * sizeof(float));
    attn->layer_output = (float*)malloc(seq_size * sizeof(float));
    
    // Allocate backward pass buffers
    attn->grad_Q = (float*)malloc(seq_size * sizeof(float));
    attn->grad_K = (float*)malloc(seq_size * sizeof(float));
    attn->grad_V = (float*)malloc(seq_size * sizeof(float));
    attn->grad_scores = (float*)malloc(attn_size * sizeof(float));
    attn->grad_weights = (float*)malloc(attn_size * sizeof(float));
    attn->grad_attn_out = (float*)malloc(seq_size * sizeof(float));
    attn->error_output = (float*)malloc(seq_size * sizeof(float));
    
    // Initialize weights
    float scale = 1.0f / sqrtf(d_model);
    for (int i = 0; i < weight_size; i++) {
        attn->W_q[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        attn->W_k[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        attn->W_v[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        attn->W_o[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    
    return attn;
}

// Free network memory
void free_attention(Attention* attn) {
    free(attn->W_q); free(attn->W_k); free(attn->W_v); free(attn->W_o);
    free(attn->W_q_grad); free(attn->W_k_grad); free(attn->W_v_grad); free(attn->W_o_grad);
    free(attn->W_q_m); free(attn->W_q_v); free(attn->W_k_m); free(attn->W_k_v);
    free(attn->W_v_m); free(attn->W_v_v); free(attn->W_o_m); free(attn->W_o_v);
    free(attn->Q); free(attn->K); free(attn->V);
    free(attn->attn_scores); free(attn->attn_weights);
    free(attn->attn_output); free(attn->layer_output);
    free(attn->grad_Q); free(attn->grad_K); free(attn->grad_V);
    free(attn->grad_scores); free(attn->grad_weights);
    free(attn->grad_attn_out); free(attn->error_output);
    free(attn);
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    int total_seq = attn->batch_size * attn->seq_len;
    
    // Q = XW_q, K = XW_k, V = XW_v
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                attn->d_model, total_seq, attn->d_model,
                1.0f, attn->W_q, attn->d_model,
                X, attn->d_model,
                0.0f, attn->Q, attn->d_model);
    
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                attn->d_model, total_seq, attn->d_model,
                1.0f, attn->W_k, attn->d_model,
                X, attn->d_model,
                0.0f, attn->K, attn->d_model);
    
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                attn->d_model, total_seq, attn->d_model,
                1.0f, attn->W_v, attn->d_model,
                X, attn->d_model,
                0.0f, attn->V, attn->d_model);
    
    // Compute attention scores and weights for each batch
    float scale = 1.0f / sqrtf(attn->d_model);
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* Q_batch = attn->Q + batch * attn->seq_len * attn->d_model;
        float* K_batch = attn->K + batch * attn->seq_len * attn->d_model;
        float* scores_batch = attn->attn_scores + batch * attn->seq_len * attn->seq_len;
        float* weights_batch = attn->attn_weights + batch * attn->seq_len * attn->seq_len;
        
        // Attention scores = QK^T / sqrt(d_model)
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->seq_len, attn->seq_len, attn->d_model,
                    scale, K_batch, attn->d_model,
                    Q_batch, attn->d_model,
                    0.0f, scores_batch, attn->seq_len);
        
        // Apply softmax to get attention weights
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
    
    // Compute weighted values for each batch
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* weights_batch = attn->attn_weights + batch * attn->seq_len * attn->seq_len;
        float* V_batch = attn->V + batch * attn->seq_len * attn->d_model;
        float* output_batch = attn->attn_output + batch * attn->seq_len * attn->d_model;
        
        // Attention output = weights * V
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    attn->d_model, attn->seq_len, attn->seq_len,
                    1.0f, V_batch, attn->d_model,
                    weights_batch, attn->seq_len,
                    0.0f, output_batch, attn->d_model);
    }
    
    // Apply output projection: layer_output = attn_output * W_o
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                attn->d_model, total_seq, attn->d_model,
                1.0f, attn->W_o, attn->d_model,
                attn->attn_output, attn->d_model,
                0.0f, attn->layer_output, attn->d_model);
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* y) {
    // ∂L/∂Y = Y - Y_true
    int total_size = attn->batch_size * attn->seq_len * attn->d_model;
    float loss = 0.0f;
    
    for (int i = 0; i < total_size; i++) {
        attn->error_output[i] = attn->layer_output[i] - y[i];
        loss += attn->error_output[i] * attn->error_output[i];
    }
    return loss / total_size;
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    int weight_size = attn->d_model * attn->d_model;
    
    memset(attn->W_q_grad, 0, weight_size * sizeof(float));
    memset(attn->W_k_grad, 0, weight_size * sizeof(float));
    memset(attn->W_v_grad, 0, weight_size * sizeof(float));
    memset(attn->W_o_grad, 0, weight_size * sizeof(float));
}

// Backward pass
void backward_pass_attention(Attention* attn, float* X) {
    int total_seq = attn->batch_size * attn->seq_len;
    
    // ∂L/∂W_o = attn_output^T * error_output
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                attn->d_model, attn->d_model, total_seq,
                1.0f, attn->error_output, attn->d_model,
                attn->attn_output, attn->d_model,
                1.0f, attn->W_o_grad, attn->d_model);
    
    // ∂L/∂attn_output = error_output * W_o^T
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                attn->d_model, total_seq, attn->d_model,
                1.0f, attn->W_o, attn->d_model,
                attn->error_output, attn->d_model,
                0.0f, attn->grad_attn_out, attn->d_model);
    
    // Backpropagate through attention for each batch
    for (int batch = 0; batch < attn->batch_size; batch++) {
        float* d_attn_output_batch = attn->grad_attn_out + batch * attn->seq_len * attn->d_model;
        float* attn_weights_batch = attn->attn_weights + batch * attn->seq_len * attn->seq_len;
        float* V_batch = attn->V + batch * attn->seq_len * attn->d_model;
        float* Q_batch = attn->Q + batch * attn->seq_len * attn->d_model;
        float* K_batch = attn->K + batch * attn->seq_len * attn->d_model;
        float* d_weights_batch = attn->grad_weights + batch * attn->seq_len * attn->seq_len;
        float* d_scores_batch = attn->grad_scores + batch * attn->seq_len * attn->seq_len;
        float* dQ_batch = attn->grad_Q + batch * attn->seq_len * attn->d_model;
        float* dK_batch = attn->grad_K + batch * attn->seq_len * attn->d_model;
        float* dV_batch = attn->grad_V + batch * attn->seq_len * attn->d_model;
        
        // ∂L/∂V = weights^T * d_attn_output
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    attn->d_model, attn->seq_len, attn->seq_len,
                    1.0f, d_attn_output_batch, attn->d_model,
                    attn_weights_batch, attn->seq_len,
                    0.0f, dV_batch, attn->d_model);
        
        // ∂L/∂weights = d_attn_output * V^T
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->seq_len, attn->seq_len, attn->d_model,
                    1.0f, V_batch, attn->d_model,
                    d_attn_output_batch, attn->d_model,
                    0.0f, d_weights_batch, attn->seq_len);
        
        // Backpropagate through softmax
        for (int i = 0; i < attn->seq_len; i++) {
            float* weights_row = attn_weights_batch + i * attn->seq_len;
            float* d_weights_row = d_weights_batch + i * attn->seq_len;
            float* d_scores_row = d_scores_batch + i * attn->seq_len;
            
            // Softmax gradient: d_scores[i] = weights[i] * (d_weights[i] - sum_j(d_weights[j] * weights[j]))
            float sum = 0.0f;
            for (int j = 0; j < attn->seq_len; j++) {
                sum += d_weights_row[j] * weights_row[j];
            }
            for (int j = 0; j < attn->seq_len; j++) {
                d_scores_row[j] = weights_row[j] * (d_weights_row[j] - sum);
            }
        }
        
        // ∂L/∂Q = d_scores * K / sqrt(d_model)
        float scale = 1.0f / sqrtf(attn->d_model);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    attn->d_model, attn->seq_len, attn->seq_len,
                    scale, K_batch, attn->d_model,
                    d_scores_batch, attn->seq_len,
                    0.0f, dQ_batch, attn->d_model);
        
        // ∂L/∂K = d_scores^T * Q / sqrt(d_model)
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    attn->d_model, attn->seq_len, attn->seq_len,
                    scale, Q_batch, attn->d_model,
                    d_scores_batch, attn->seq_len,
                    0.0f, dK_batch, attn->d_model);
    }
    
    // Accumulate weight gradients
    // ∂L/∂W_q = X^T * grad_Q
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                attn->d_model, attn->d_model, total_seq,
                1.0f, attn->grad_Q, attn->d_model,
                X, attn->d_model,
                1.0f, attn->W_q_grad, attn->d_model);
    
    // ∂L/∂W_k = X^T * grad_K
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                attn->d_model, attn->d_model, total_seq,
                1.0f, attn->grad_K, attn->d_model,
                X, attn->d_model,
                1.0f, attn->W_k_grad, attn->d_model);
    
    // ∂L/∂W_v = X^T * grad_V
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                attn->d_model, attn->d_model, total_seq,
                1.0f, attn->grad_V, attn->d_model,
                X, attn->d_model,
                1.0f, attn->W_v_grad, attn->d_model);
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int weight_size = attn->d_model * attn->d_model;
    int total_seq = attn->batch_size * attn->seq_len;
    
    // Update W_q weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->W_q_grad[i] / total_seq;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        attn->W_q_m[i] = attn->beta1 * attn->W_q_m[i] + (1.0f - attn->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        attn->W_q_v[i] = attn->beta2 * attn->W_q_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->W_q_m[i] / (sqrtf(attn->W_q_v[i]) + attn->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        attn->W_q[i] = attn->W_q[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update W_k weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->W_k_grad[i] / total_seq;
        
        attn->W_k_m[i] = attn->beta1 * attn->W_k_m[i] + (1.0f - attn->beta1) * grad;
        attn->W_k_v[i] = attn->beta2 * attn->W_k_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->W_k_m[i] / (sqrtf(attn->W_k_v[i]) + attn->epsilon);
        attn->W_k[i] = attn->W_k[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update W_v weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->W_v_grad[i] / total_seq;
        
        attn->W_v_m[i] = attn->beta1 * attn->W_v_m[i] + (1.0f - attn->beta1) * grad;
        attn->W_v_v[i] = attn->beta2 * attn->W_v_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->W_v_m[i] / (sqrtf(attn->W_v_v[i]) + attn->epsilon);
        attn->W_v[i] = attn->W_v[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update W_o weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->W_o_grad[i] / total_seq;
        
        attn->W_o_m[i] = attn->beta1 * attn->W_o_m[i] + (1.0f - attn->beta1) * grad;
        attn->W_o_v[i] = attn->beta2 * attn->W_o_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->W_o_m[i] / (sqrtf(attn->W_o_v[i]) + attn->epsilon);
        attn->W_o[i] = attn->W_o[i] * (1.0f - learning_rate * attn->weight_decay) - update;
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
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    
    // Save weights
    int weight_size = attn->d_model * attn->d_model;
    fwrite(attn->W_q, sizeof(float), weight_size, file);
    fwrite(attn->W_k, sizeof(float), weight_size, file);
    fwrite(attn->W_v, sizeof(float), weight_size, file);
    fwrite(attn->W_o, sizeof(float), weight_size, file);
    
    // Save Adam state
    fwrite(&attn->t, sizeof(int), 1, file);
    fwrite(attn->W_q_m, sizeof(float), weight_size, file);
    fwrite(attn->W_q_v, sizeof(float), weight_size, file);
    fwrite(attn->W_k_m, sizeof(float), weight_size, file);
    fwrite(attn->W_k_v, sizeof(float), weight_size, file);
    fwrite(attn->W_v_m, sizeof(float), weight_size, file);
    fwrite(attn->W_v_v, sizeof(float), weight_size, file);
    fwrite(attn->W_o_m, sizeof(float), weight_size, file);
    fwrite(attn->W_o_v, sizeof(float), weight_size, file);

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
    int d_model, seq_len, stored_batch_size;
    fread(&d_model, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    Attention* attn = init_attention(d_model, seq_len, batch_size);
    
    // Load weights
    int weight_size = d_model * d_model;
    fread(attn->W_q, sizeof(float), weight_size, file);
    fread(attn->W_k, sizeof(float), weight_size, file);
    fread(attn->W_v, sizeof(float), weight_size, file);
    fread(attn->W_o, sizeof(float), weight_size, file);
    
    // Load Adam state
    fread(&attn->t, sizeof(int), 1, file);
    fread(attn->W_q_m, sizeof(float), weight_size, file);
    fread(attn->W_q_v, sizeof(float), weight_size, file);
    fread(attn->W_k_m, sizeof(float), weight_size, file);
    fread(attn->W_k_v, sizeof(float), weight_size, file);
    fread(attn->W_v_m, sizeof(float), weight_size, file);
    fread(attn->W_v_v, sizeof(float), weight_size, file);
    fread(attn->W_o_m, sizeof(float), weight_size, file);
    fread(attn->W_o_v, sizeof(float), weight_size, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return attn;
}