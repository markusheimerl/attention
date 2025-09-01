#include "attention.h"

// Initialize the attention layer
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
    int qkv_buffer_size = d_model * seq_len * batch_size;
    int score_buffer_size = seq_len * seq_len * batch_size;
    
    // Allocate weights and gradients
    attn->WQ = (float*)malloc(weight_size * sizeof(float));
    attn->WK = (float*)malloc(weight_size * sizeof(float));
    attn->WV = (float*)malloc(weight_size * sizeof(float));
    attn->WO = (float*)malloc(weight_size * sizeof(float));
    attn->WQ_grad = (float*)malloc(weight_size * sizeof(float));
    attn->WK_grad = (float*)malloc(weight_size * sizeof(float));
    attn->WV_grad = (float*)malloc(weight_size * sizeof(float));
    attn->WO_grad = (float*)malloc(weight_size * sizeof(float));
    
    // Allocate Adam buffers
    attn->WQ_m = (float*)calloc(weight_size, sizeof(float));
    attn->WQ_v = (float*)calloc(weight_size, sizeof(float));
    attn->WK_m = (float*)calloc(weight_size, sizeof(float));
    attn->WK_v = (float*)calloc(weight_size, sizeof(float));
    attn->WV_m = (float*)calloc(weight_size, sizeof(float));
    attn->WV_v = (float*)calloc(weight_size, sizeof(float));
    attn->WO_m = (float*)calloc(weight_size, sizeof(float));
    attn->WO_v = (float*)calloc(weight_size, sizeof(float));
    
    // Allocate forward pass buffers
    attn->Q = (float*)malloc(qkv_buffer_size * sizeof(float));
    attn->K = (float*)malloc(qkv_buffer_size * sizeof(float));
    attn->V = (float*)malloc(qkv_buffer_size * sizeof(float));
    attn->scores = (float*)malloc(score_buffer_size * sizeof(float));
    attn->attn_weights = (float*)malloc(score_buffer_size * sizeof(float));
    attn->attn_output = (float*)malloc(qkv_buffer_size * sizeof(float));
    attn->layer_output = (float*)malloc(qkv_buffer_size * sizeof(float));
    
    // Allocate backward pass buffers
    attn->error_output = (float*)malloc(qkv_buffer_size * sizeof(float));
    attn->error_attn_output = (float*)malloc(qkv_buffer_size * sizeof(float));
    attn->error_attn_weights = (float*)malloc(score_buffer_size * sizeof(float));
    attn->error_scores = (float*)malloc(score_buffer_size * sizeof(float));
    attn->error_V = (float*)malloc(qkv_buffer_size * sizeof(float));
    attn->error_K = (float*)malloc(qkv_buffer_size * sizeof(float));
    attn->error_Q = (float*)malloc(qkv_buffer_size * sizeof(float));
    
    // Initialize weights with Xavier initialization
    float scale = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < weight_size; i++) {
        attn->WQ[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        attn->WK[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        attn->WV[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        attn->WO[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    
    return attn;
}

// Free attention memory
void free_attention(Attention* attn) {
    free(attn->WQ); free(attn->WK); free(attn->WV); free(attn->WO);
    free(attn->WQ_grad); free(attn->WK_grad); free(attn->WV_grad); free(attn->WO_grad);
    free(attn->WQ_m); free(attn->WQ_v); free(attn->WK_m); free(attn->WK_v);
    free(attn->WV_m); free(attn->WV_v); free(attn->WO_m); free(attn->WO_v);
    free(attn->Q); free(attn->K); free(attn->V);
    free(attn->scores); free(attn->attn_weights);
    free(attn->attn_output); free(attn->layer_output);
    free(attn->error_output); free(attn->error_attn_output);
    free(attn->error_attn_weights); free(attn->error_scores);
    free(attn->error_V); free(attn->error_K); free(attn->error_Q);
    free(attn);
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    int d = attn->d_model;
    int s = attn->seq_len;
    int b = attn->batch_size;
    
    // Compute Q = WQ * X
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                d, s * b, d,
                1.0f, attn->WQ, d,
                X, d,
                0.0f, attn->Q, d);
    
    // Compute K = WK * X  
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                d, s * b, d,
                1.0f, attn->WK, d,
                X, d,
                0.0f, attn->K, d);
    
    // Compute V = WV * X
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                d, s * b, d,
                1.0f, attn->WV, d,
                X, d,
                0.0f, attn->V, d);
    
    // Compute attention scores: scores = Q^T * K / sqrt(d_model)
    float scale = 1.0f / sqrtf(d);
    for (int batch = 0; batch < b; batch++) {
        float* Q_batch = attn->Q + batch * d * s;
        float* K_batch = attn->K + batch * d * s;
        float* scores_batch = attn->scores + batch * s * s;
        
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    s, s, d,
                    scale, Q_batch, d,
                    K_batch, d,
                    0.0f, scores_batch, s);
    }
    
    // Apply softmax to get attention weights
    for (int batch = 0; batch < b; batch++) {
        float* scores_batch = attn->scores + batch * s * s;
        float* weights_batch = attn->attn_weights + batch * s * s;
        
        for (int i = 0; i < s; i++) {
            float* score_row = scores_batch + i * s;
            float* weight_row = weights_batch + i * s;
            
            // Find max for numerical stability
            float max_val = score_row[0];
            for (int j = 1; j < s; j++) {
                if (score_row[j] > max_val) max_val = score_row[j];
            }
            
            // Compute softmax
            float sum = 0.0f;
            for (int j = 0; j < s; j++) {
                weight_row[j] = expf(score_row[j] - max_val);
                sum += weight_row[j];
            }
            for (int j = 0; j < s; j++) {
                weight_row[j] /= sum;
            }
        }
    }
    
    // Compute attention output: attn_output = V * attn_weights^T
    for (int batch = 0; batch < b; batch++) {
        float* V_batch = attn->V + batch * d * s;
        float* weights_batch = attn->attn_weights + batch * s * s;
        float* output_batch = attn->attn_output + batch * d * s;
        
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    d, s, s,
                    1.0f, V_batch, d,
                    weights_batch, s,
                    0.0f, output_batch, d);
    }
    
    // Final linear projection: layer_output = WO * attn_output
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                d, s * b, d,
                1.0f, attn->WO, d,
                attn->attn_output, d,
                0.0f, attn->layer_output, d);
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* y) {
    // ∂L/∂output = output - y_true
    float loss = 0.0f;
    int total_size = attn->d_model * attn->seq_len * attn->batch_size;
    
    for (int i = 0; i < total_size; i++) {
        attn->error_output[i] = attn->layer_output[i] - y[i];
        loss += attn->error_output[i] * attn->error_output[i];
    }
    return loss / total_size;
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    int weight_size = attn->d_model * attn->d_model;
    
    memset(attn->WQ_grad, 0, weight_size * sizeof(float));
    memset(attn->WK_grad, 0, weight_size * sizeof(float));
    memset(attn->WV_grad, 0, weight_size * sizeof(float));
    memset(attn->WO_grad, 0, weight_size * sizeof(float));
}

// Backward pass
void backward_pass_attention(Attention* attn, float* X, float* grad_X) {
    int d = attn->d_model;
    int s = attn->seq_len;
    int b = attn->batch_size;
    
    // ∂L/∂WO = error_output * attn_output^T
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                d, d, s * b,
                1.0f, attn->error_output, d,
                attn->attn_output, d,
                1.0f, attn->WO_grad, d);
    
    // ∂L/∂attn_output = WO^T * error_output
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                d, s * b, d,
                1.0f, attn->WO, d,
                attn->error_output, d,
                0.0f, attn->error_attn_output, d);
    
    // Backward through attention computation
    for (int batch = 0; batch < b; batch++) {
        float* V_batch = attn->V + batch * d * s;
        float* weights_batch = attn->attn_weights + batch * s * s;
        float* error_attn_batch = attn->error_attn_output + batch * d * s;
        float* error_V_batch = attn->error_V + batch * d * s;
        float* error_weights_batch = attn->error_attn_weights + batch * s * s;
        
        // ∂L/∂V = error_attn_output * attn_weights
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    d, s, s,
                    1.0f, error_attn_batch, d,
                    weights_batch, s,
                    0.0f, error_V_batch, d);
        
        // ∂L/∂attn_weights = V^T * error_attn_output
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    s, s, d,
                    1.0f, V_batch, d,
                    error_attn_batch, d,
                    0.0f, error_weights_batch, s);
    }
    
    // Backward through softmax
    for (int batch = 0; batch < b; batch++) {
        float* weights_batch = attn->attn_weights + batch * s * s;
        float* error_weights_batch = attn->error_attn_weights + batch * s * s;
        float* error_scores_batch = attn->error_scores + batch * s * s;
        
        for (int i = 0; i < s; i++) {
            float* weight_row = weights_batch + i * s;
            float* error_weight_row = error_weights_batch + i * s;
            float* error_score_row = error_scores_batch + i * s;
            
            // Compute sum for softmax gradient
            float sum = 0.0f;
            for (int j = 0; j < s; j++) {
                sum += error_weight_row[j] * weight_row[j];
            }
            
            // Apply softmax gradient formula
            for (int j = 0; j < s; j++) {
                error_score_row[j] = weight_row[j] * (error_weight_row[j] - sum);
            }
        }
    }
    
    // Backward through attention scores
    float scale = 1.0f / sqrtf(d);
    for (int batch = 0; batch < b; batch++) {
        float* Q_batch = attn->Q + batch * d * s;
        float* K_batch = attn->K + batch * d * s;
        float* error_scores_batch = attn->error_scores + batch * s * s;
        float* error_Q_batch = attn->error_Q + batch * d * s;
        float* error_K_batch = attn->error_K + batch * d * s;
        
        // ∂L/∂Q = K * error_scores^T * scale
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    d, s, s,
                    scale, K_batch, d,
                    error_scores_batch, s,
                    0.0f, error_Q_batch, d);
        
        // ∂L/∂K = Q * error_scores * scale
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    d, s, s,
                    scale, Q_batch, d,
                    error_scores_batch, s,
                    0.0f, error_K_batch, d);
    }
    
    // ∂L/∂WQ = error_Q * X^T
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                d, d, s * b,
                1.0f, attn->error_Q, d,
                X, d,
                1.0f, attn->WQ_grad, d);
    
    // ∂L/∂WK = error_K * X^T
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                d, d, s * b,
                1.0f, attn->error_K, d,
                X, d,
                1.0f, attn->WK_grad, d);
    
    // ∂L/∂WV = error_V * X^T
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                d, d, s * b,
                1.0f, attn->error_V, d,
                X, d,
                1.0f, attn->WV_grad, d);
    
    if (grad_X != NULL) {
        // ∂L/∂X = WQ^T * error_Q + WK^T * error_K + WV^T * error_V
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    d, s * b, d,
                    1.0f, attn->WQ, d,
                    attn->error_Q, d,
                    0.0f, grad_X, d);
        
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    d, s * b, d,
                    1.0f, attn->WK, d,
                    attn->error_K, d,
                    1.0f, grad_X, d);
        
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    d, s * b, d,
                    1.0f, attn->WV, d,
                    attn->error_V, d,
                    1.0f, grad_X, d);
    }
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;  // Increment time step
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int weight_size = attn->d_model * attn->d_model;
    
    // Update WQ weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->WQ_grad[i] / attn->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        attn->WQ_m[i] = attn->beta1 * attn->WQ_m[i] + (1.0f - attn->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        attn->WQ_v[i] = attn->beta2 * attn->WQ_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->WQ_m[i] / (sqrtf(attn->WQ_v[i]) + attn->epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        attn->WQ[i] = attn->WQ[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update WK weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->WK_grad[i] / attn->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        attn->WK_m[i] = attn->beta1 * attn->WK_m[i] + (1.0f - attn->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        attn->WK_v[i] = attn->beta2 * attn->WK_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->WK_m[i] / (sqrtf(attn->WK_v[i]) + attn->epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        attn->WK[i] = attn->WK[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update WV weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->WV_grad[i] / attn->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        attn->WV_m[i] = attn->beta1 * attn->WV_m[i] + (1.0f - attn->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        attn->WV_v[i] = attn->beta2 * attn->WV_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->WV_m[i] / (sqrtf(attn->WV_v[i]) + attn->epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        attn->WV[i] = attn->WV[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update WO weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->WO_grad[i] / attn->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        attn->WO_m[i] = attn->beta1 * attn->WO_m[i] + (1.0f - attn->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        attn->WO_v[i] = attn->beta2 * attn->WO_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->WO_m[i] / (sqrtf(attn->WO_v[i]) + attn->epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        attn->WO[i] = attn->WO[i] * (1.0f - learning_rate * attn->weight_decay) - update;
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
    
    int weight_size = attn->d_model * attn->d_model;
    
    // Save weights
    fwrite(attn->WQ, sizeof(float), weight_size, file);
    fwrite(attn->WK, sizeof(float), weight_size, file);
    fwrite(attn->WV, sizeof(float), weight_size, file);
    fwrite(attn->WO, sizeof(float), weight_size, file);
    
    // Save Adam state
    fwrite(&attn->t, sizeof(int), 1, file);
    fwrite(attn->WQ_m, sizeof(float), weight_size, file);
    fwrite(attn->WQ_v, sizeof(float), weight_size, file);
    fwrite(attn->WK_m, sizeof(float), weight_size, file);
    fwrite(attn->WK_v, sizeof(float), weight_size, file);
    fwrite(attn->WV_m, sizeof(float), weight_size, file);
    fwrite(attn->WV_v, sizeof(float), weight_size, file);
    fwrite(attn->WO_m, sizeof(float), weight_size, file);
    fwrite(attn->WO_v, sizeof(float), weight_size, file);

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
    
    // Initialize attention
    Attention* attn = init_attention(d_model, seq_len, batch_size);
    
    int weight_size = d_model * d_model;
    
    // Load weights
    fread(attn->WQ, sizeof(float), weight_size, file);
    fread(attn->WK, sizeof(float), weight_size, file);
    fread(attn->WV, sizeof(float), weight_size, file);
    fread(attn->WO, sizeof(float), weight_size, file);
    
    // Load Adam state
    fread(&attn->t, sizeof(int), 1, file);
    fread(attn->WQ_m, sizeof(float), weight_size, file);
    fread(attn->WQ_v, sizeof(float), weight_size, file);
    fread(attn->WK_m, sizeof(float), weight_size, file);
    fread(attn->WK_v, sizeof(float), weight_size, file);
    fread(attn->WV_m, sizeof(float), weight_size, file);
    fread(attn->WV_v, sizeof(float), weight_size, file);
    fread(attn->WO_m, sizeof(float), weight_size, file);
    fread(attn->WO_v, sizeof(float), weight_size, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return attn;
}