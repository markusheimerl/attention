#include "attention.h"

// Initialize the attention module
Attention* init_attention(int d_model, int seq_len, int batch_size) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->d_model = d_model;
    attn->seq_len = seq_len;
    attn->batch_size = batch_size;
    attn->scale = 1.0f / sqrtf(d_model);
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    int weight_size = d_model * d_model;
    int qkv_size = d_model * seq_len * batch_size;
    int scores_size = seq_len * seq_len * batch_size;
    
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
    attn->Q = (float*)malloc(qkv_size * sizeof(float));
    attn->K = (float*)malloc(qkv_size * sizeof(float));
    attn->V = (float*)malloc(qkv_size * sizeof(float));
    attn->scores = (float*)malloc(scores_size * sizeof(float));
    attn->attn_weights = (float*)malloc(scores_size * sizeof(float));
    attn->context = (float*)malloc(qkv_size * sizeof(float));
    attn->output = (float*)malloc(qkv_size * sizeof(float));
    
    // Allocate backward pass buffers
    attn->grad_output = (float*)malloc(qkv_size * sizeof(float));
    attn->grad_context = (float*)malloc(qkv_size * sizeof(float));
    attn->grad_attn_weights = (float*)malloc(scores_size * sizeof(float));
    attn->grad_scores = (float*)malloc(scores_size * sizeof(float));
    attn->grad_Q = (float*)malloc(qkv_size * sizeof(float));
    attn->grad_K = (float*)malloc(qkv_size * sizeof(float));
    attn->grad_V = (float*)malloc(qkv_size * sizeof(float));
    
    // Initialize weights with smaller scale for attention
    float scale_init = sqrtf(1.0f / d_model);
    
    for (int i = 0; i < weight_size; i++) {
        attn->WQ[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_init;
        attn->WK[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_init;
        attn->WV[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_init;
        attn->WO[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_init;
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
    free(attn->scores); free(attn->attn_weights); free(attn->context); free(attn->output);
    free(attn->grad_output); free(attn->grad_context);
    free(attn->grad_attn_weights); free(attn->grad_scores);
    free(attn->grad_Q); free(attn->grad_K); free(attn->grad_V);
    free(attn);
}

// Softmax function for attention weights
void softmax_2d(float* input, float* output, int rows, int cols, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < rows; i++) {
            // Find max for numerical stability
            float max_val = input[b * rows * cols + i * cols + 0];
            for (int j = 1; j < cols; j++) {
                float val = input[b * rows * cols + i * cols + j];
                if (val > max_val) max_val = val;
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                float exp_val = expf(input[b * rows * cols + i * cols + j] - max_val);
                output[b * rows * cols + i * cols + j] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for (int j = 0; j < cols; j++) {
                output[b * rows * cols + i * cols + j] /= sum;
            }
        }
    }
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    int batch_seq = attn->seq_len * attn->batch_size;
    
    // Compute Q = WQ * X, K = WK * X, V = WV * X
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                attn->d_model, batch_seq, attn->d_model,
                1.0f, attn->WQ, attn->d_model,
                X, attn->d_model,
                0.0f, attn->Q, attn->d_model);
                
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                attn->d_model, batch_seq, attn->d_model,
                1.0f, attn->WK, attn->d_model,
                X, attn->d_model,
                0.0f, attn->K, attn->d_model);
                
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                attn->d_model, batch_seq, attn->d_model,
                1.0f, attn->WV, attn->d_model,
                X, attn->d_model,
                0.0f, attn->V, attn->d_model);
    
    // Compute attention scores = Q^T * K for each batch
    for (int b = 0; b < attn->batch_size; b++) {
        float* Q_batch = &attn->Q[b * attn->d_model * attn->seq_len];
        float* K_batch = &attn->K[b * attn->d_model * attn->seq_len];
        float* scores_batch = &attn->scores[b * attn->seq_len * attn->seq_len];
        
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->seq_len, attn->seq_len, attn->d_model,
                    attn->scale, Q_batch, attn->d_model,
                    K_batch, attn->d_model,
                    0.0f, scores_batch, attn->seq_len);
    }
    
    // Apply softmax to get attention weights
    softmax_2d(attn->scores, attn->attn_weights, attn->seq_len, attn->seq_len, attn->batch_size);
    
    // Compute context = V * attention_weights for each batch
    for (int b = 0; b < attn->batch_size; b++) {
        float* V_batch = &attn->V[b * attn->d_model * attn->seq_len];
        float* attn_weights_batch = &attn->attn_weights[b * attn->seq_len * attn->seq_len];
        float* context_batch = &attn->context[b * attn->d_model * attn->seq_len];
        
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    attn->d_model, attn->seq_len, attn->seq_len,
                    1.0f, V_batch, attn->d_model,
                    attn_weights_batch, attn->seq_len,
                    0.0f, context_batch, attn->d_model);
    }
    
    // Final output projection: output = WO * context
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                attn->d_model, batch_seq, attn->d_model,
                1.0f, attn->WO, attn->d_model,
                attn->context, attn->d_model,
                0.0f, attn->output, attn->d_model);
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* y) {
    // Compute gradient: grad_output = output - y
    float loss = 0.0f;
    int total_size = attn->d_model * attn->seq_len * attn->batch_size;
    
    for (int i = 0; i < total_size; i++) {
        attn->grad_output[i] = attn->output[i] - y[i];
        loss += attn->grad_output[i] * attn->grad_output[i];
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

// Softmax backward pass
void softmax_backward(float* grad_input, float* grad_output, float* softmax_output, 
                      int rows, int cols, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = b * rows * cols + i * cols + j;
                float sum = 0.0f;
                
                // Compute sum for softmax derivative
                for (int k = 0; k < cols; k++) {
                    int k_idx = b * rows * cols + i * cols + k;
                    sum += grad_output[k_idx] * softmax_output[k_idx];
                }
                
                grad_input[idx] = softmax_output[idx] * (grad_output[idx] - sum);
            }
        }
    }
}

// Backward pass
void backward_pass_attention(Attention* attn, float* X, float* grad_X) {
    int batch_seq = attn->seq_len * attn->batch_size;
    
    // Gradient w.r.t. WO: WO_grad = grad_output * context^T
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                attn->d_model, attn->d_model, batch_seq,
                1.0f, attn->grad_output, attn->d_model,
                attn->context, attn->d_model,
                1.0f, attn->WO_grad, attn->d_model);
    
    // Gradient w.r.t. context: grad_context = WO^T * grad_output
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                attn->d_model, batch_seq, attn->d_model,
                1.0f, attn->WO, attn->d_model,
                attn->grad_output, attn->d_model,
                0.0f, attn->grad_context, attn->d_model);
    
    // Backward through attention computation for each batch
    for (int b = 0; b < attn->batch_size; b++) {
        float* V_batch = &attn->V[b * attn->d_model * attn->seq_len];
        float* attn_weights_batch = &attn->attn_weights[b * attn->seq_len * attn->seq_len];
        float* grad_context_batch = &attn->grad_context[b * attn->d_model * attn->seq_len];
        float* grad_attn_weights_batch = &attn->grad_attn_weights[b * attn->seq_len * attn->seq_len];
        float* grad_V_batch = &attn->grad_V[b * attn->d_model * attn->seq_len];
        
        // Gradient w.r.t. attention weights: grad_attn_weights = V^T * grad_context
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->seq_len, attn->seq_len, attn->d_model,
                    1.0f, V_batch, attn->d_model,
                    grad_context_batch, attn->d_model,
                    0.0f, grad_attn_weights_batch, attn->seq_len);
        
        // Gradient w.r.t. V: grad_V = grad_context * attn_weights^T
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    attn->d_model, attn->seq_len, attn->seq_len,
                    1.0f, grad_context_batch, attn->d_model,
                    attn_weights_batch, attn->seq_len,
                    0.0f, grad_V_batch, attn->d_model);
    }
    
    // Gradient w.r.t. scores (softmax backward)
    softmax_backward(attn->grad_scores, attn->grad_attn_weights, attn->attn_weights,
                     attn->seq_len, attn->seq_len, attn->batch_size);
    
    // Gradients w.r.t. Q and K for each batch
    for (int b = 0; b < attn->batch_size; b++) {
        float* Q_batch = &attn->Q[b * attn->d_model * attn->seq_len];
        float* K_batch = &attn->K[b * attn->d_model * attn->seq_len];
        float* grad_scores_batch = &attn->grad_scores[b * attn->seq_len * attn->seq_len];
        float* grad_Q_batch = &attn->grad_Q[b * attn->d_model * attn->seq_len];
        float* grad_K_batch = &attn->grad_K[b * attn->d_model * attn->seq_len];
        
        // grad_Q = K * grad_scores^T * scale
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    attn->d_model, attn->seq_len, attn->seq_len,
                    attn->scale, K_batch, attn->d_model,
                    grad_scores_batch, attn->seq_len,
                    0.0f, grad_Q_batch, attn->d_model);
        
        // grad_K = Q * grad_scores * scale
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    attn->d_model, attn->seq_len, attn->seq_len,
                    attn->scale, Q_batch, attn->d_model,
                    grad_scores_batch, attn->seq_len,
                    0.0f, grad_K_batch, attn->d_model);
    }
    
    // Gradients w.r.t. weight matrices
    // WQ_grad = grad_Q * X^T
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                attn->d_model, attn->d_model, batch_seq,
                1.0f, attn->grad_Q, attn->d_model,
                X, attn->d_model,
                1.0f, attn->WQ_grad, attn->d_model);
    
    // WK_grad = grad_K * X^T
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                attn->d_model, attn->d_model, batch_seq,
                1.0f, attn->grad_K, attn->d_model,
                X, attn->d_model,
                1.0f, attn->WK_grad, attn->d_model);
    
    // WV_grad = grad_V * X^T
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                attn->d_model, attn->d_model, batch_seq,
                1.0f, attn->grad_V, attn->d_model,
                X, attn->d_model,
                1.0f, attn->WV_grad, attn->d_model);
    
    if (grad_X != NULL) {
        // grad_X = WQ^T * grad_Q + WK^T * grad_K + WV^T * grad_V
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, batch_seq, attn->d_model,
                    1.0f, attn->WQ, attn->d_model,
                    attn->grad_Q, attn->d_model,
                    0.0f, grad_X, attn->d_model);
        
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, batch_seq, attn->d_model,
                    1.0f, attn->WK, attn->d_model,
                    attn->grad_K, attn->d_model,
                    1.0f, grad_X, attn->d_model);
        
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, batch_seq, attn->d_model,
                    1.0f, attn->WV, attn->d_model,
                    attn->grad_V, attn->d_model,
                    1.0f, grad_X, attn->d_model);
    }
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;  // Increment time step
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int weight_size = attn->d_model * attn->d_model;
    
    // Update all weight matrices (WQ, WK, WV, WO)
    float* weights[] = {attn->WQ, attn->WK, attn->WV, attn->WO};
    float* grads[] = {attn->WQ_grad, attn->WK_grad, attn->WV_grad, attn->WO_grad};
    float* m[] = {attn->WQ_m, attn->WK_m, attn->WV_m, attn->WO_m};
    float* v[] = {attn->WQ_v, attn->WK_v, attn->WV_v, attn->WO_v};
    
    for (int w = 0; w < 4; w++) {
        for (int i = 0; i < weight_size; i++) {
            float grad = grads[w][i];  // Don't divide by batch_size here
            
            // m = β₁m + (1-β₁)(∂L/∂W)
            m[w][i] = attn->beta1 * m[w][i] + (1.0f - attn->beta1) * grad;
            // v = β₂v + (1-β₂)(∂L/∂W)²
            v[w][i] = attn->beta2 * v[w][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * m[w][i] / (sqrtf(v[w][i]) + attn->epsilon);
            // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
            weights[w][i] = weights[w][i] * (1.0f - learning_rate * attn->weight_decay) - update;
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
    
    // Initialize attention module
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