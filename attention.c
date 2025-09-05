#include "attention.h"

// Initialize attention layer
Attention* init_attention(int seq_len, int d_model, int batch_size) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->seq_len = seq_len;
    attn->d_model = d_model;
    attn->batch_size = batch_size;
    attn->scale = 1.0f / sqrtf(d_model);
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.001f;
    
    int weight_size = d_model * d_model;
    int seq_batch_size = seq_len * d_model * batch_size;
    int attn_size = seq_len * seq_len * batch_size;
    
    // Allocate weights and gradients
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
    attn->Q = (float*)malloc(seq_batch_size * sizeof(float));
    attn->K = (float*)malloc(seq_batch_size * sizeof(float));
    attn->V = (float*)malloc(seq_batch_size * sizeof(float));
    attn->scores = (float*)malloc(attn_size * sizeof(float));
    attn->attn_weights = (float*)malloc(attn_size * sizeof(float));
    attn->attn_output = (float*)malloc(seq_batch_size * sizeof(float));
    attn->output = (float*)malloc(seq_batch_size * sizeof(float));
    
    // Allocate backward pass buffers
    attn->grad_Q = (float*)malloc(seq_batch_size * sizeof(float));
    attn->grad_K = (float*)malloc(seq_batch_size * sizeof(float));
    attn->grad_V = (float*)malloc(seq_batch_size * sizeof(float));
    attn->grad_scores = (float*)malloc(attn_size * sizeof(float));
    attn->grad_attn_output = (float*)malloc(seq_batch_size * sizeof(float));
    attn->grad_output = (float*)malloc(seq_batch_size * sizeof(float));
    
    // Initialize weights with Xavier initialization
    float scale = sqrtf(6.0f / (d_model + d_model));
    
    for (int i = 0; i < weight_size; i++) {
        attn->W_q[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        attn->W_k[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        attn->W_v[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        attn->W_o[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
    
    return attn;
}

// Free attention memory
void free_attention(Attention* attn) {
    free(attn->W_q); free(attn->W_k); free(attn->W_v); free(attn->W_o);
    free(attn->W_q_grad); free(attn->W_k_grad); free(attn->W_v_grad); free(attn->W_o_grad);
    free(attn->W_q_m); free(attn->W_q_v); free(attn->W_k_m); free(attn->W_k_v);
    free(attn->W_v_m); free(attn->W_v_v); free(attn->W_o_m); free(attn->W_o_v);
    free(attn->Q); free(attn->K); free(attn->V);
    free(attn->scores); free(attn->attn_weights);
    free(attn->attn_output); free(attn->output);
    free(attn->grad_Q); free(attn->grad_K); free(attn->grad_V);
    free(attn->grad_scores); free(attn->grad_attn_output); free(attn->grad_output);
    free(attn);
}

// Softmax over each column (for each batch element)
static void softmax_columns(float* output, float* input, int rows, int cols) {
    for (int c = 0; c < cols; c++) {
        // Find max for numerical stability
        float max_val = -1e30f;
        for (int r = 0; r < rows; r++) {
            float val = input[r + rows * c];
            if (val > max_val) max_val = val;
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int r = 0; r < rows; r++) {
            float exp_val = expf(input[r + rows * c] - max_val);
            output[r + rows * c] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int r = 0; r < rows; r++) {
            output[r + rows * c] /= sum;
        }
    }
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    int ncols = attn->d_model * attn->batch_size;
    
    // Q = XW_q^T (treating each sequence position separately)
    for (int b = 0; b < attn->batch_size; b++) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, X + attn->seq_len * attn->d_model * b, attn->seq_len,
                    attn->W_q, attn->d_model,
                    0.0f, attn->Q + attn->seq_len * attn->d_model * b, attn->seq_len);
        
        // K = XW_k^T
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, X + attn->seq_len * attn->d_model * b, attn->seq_len,
                    attn->W_k, attn->d_model,
                    0.0f, attn->K + attn->seq_len * attn->d_model * b, attn->seq_len);
        
        // V = XW_v^T
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, X + attn->seq_len * attn->d_model * b, attn->seq_len,
                    attn->W_v, attn->d_model,
                    0.0f, attn->V + attn->seq_len * attn->d_model * b, attn->seq_len);
    }
    
    // Scores = QK^T / sqrt(d_model)
    for (int b = 0; b < attn->batch_size; b++) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->seq_len, attn->d_model,
                    attn->scale, 
                    attn->Q + attn->seq_len * attn->d_model * b, attn->seq_len,
                    attn->K + attn->seq_len * attn->d_model * b, attn->seq_len,
                    0.0f, attn->scores + attn->seq_len * attn->seq_len * b, attn->seq_len);
    }
    
    // Attention weights = softmax(scores)
    softmax_columns(attn->attn_weights, attn->scores, attn->seq_len, attn->seq_len * attn->batch_size);
    
    // Attention output = weights * V
    for (int b = 0; b < attn->batch_size; b++) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->seq_len,
                    1.0f, 
                    attn->attn_weights + attn->seq_len * attn->seq_len * b, attn->seq_len,
                    attn->V + attn->seq_len * attn->d_model * b, attn->seq_len,
                    0.0f, attn->attn_output + attn->seq_len * attn->d_model * b, attn->seq_len);
        
        // Output = attn_output * W_o^T
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, 
                    attn->attn_output + attn->seq_len * attn->d_model * b, attn->seq_len,
                    attn->W_o, attn->d_model,
                    0.0f, attn->output + attn->seq_len * attn->d_model * b, attn->seq_len);
    }
}

// Calculate MSE loss
float calculate_loss_attention(Attention* attn, float* y) {
    int total_elements = attn->seq_len * attn->d_model * attn->batch_size;
    
    // Compute error: grad_output = output - y
    cblas_scopy(total_elements, attn->output, 1, attn->grad_output, 1);
    cblas_saxpy(total_elements, -1.0f, y, 1, attn->grad_output, 1);
    
    // Calculate MSE
    float loss = cblas_sdot(total_elements, attn->grad_output, 1, attn->grad_output, 1);
    return loss / total_elements;
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
void backward_pass_attention(Attention* attn, float* X, float* grad_X) {
    // Gradient through output projection
    for (int b = 0; b < attn->batch_size; b++) {
        // grad_W_o += attn_output^T * grad_output
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, attn->seq_len,
                    1.0f, 
                    attn->grad_output + attn->seq_len * attn->d_model * b, attn->seq_len,
                    attn->attn_output + attn->seq_len * attn->d_model * b, attn->seq_len,
                    1.0f, attn->W_o_grad, attn->d_model);
        
        // grad_attn_output = grad_output * W_o
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, 
                    attn->grad_output + attn->seq_len * attn->d_model * b, attn->seq_len,
                    attn->W_o, attn->d_model,
                    0.0f, attn->grad_attn_output + attn->seq_len * attn->d_model * b, attn->seq_len);
    }
    
    // Gradient through attention
    for (int b = 0; b < attn->batch_size; b++) {
        // grad_V = attn_weights^T * grad_attn_output
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->seq_len,
                    1.0f, 
                    attn->attn_weights + attn->seq_len * attn->seq_len * b, attn->seq_len,
                    attn->grad_attn_output + attn->seq_len * attn->d_model * b, attn->seq_len,
                    0.0f, attn->grad_V + attn->seq_len * attn->d_model * b, attn->seq_len);
        
        // grad_attn_weights = grad_attn_output * V^T
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->seq_len, attn->d_model,
                    1.0f, 
                    attn->grad_attn_output + attn->seq_len * attn->d_model * b, attn->seq_len,
                    attn->V + attn->seq_len * attn->d_model * b, attn->seq_len,
                    0.0f, attn->grad_scores + attn->seq_len * attn->seq_len * b, attn->seq_len);
    }
    
    // Gradient through softmax
    for (int b = 0; b < attn->batch_size; b++) {
        for (int c = 0; c < attn->seq_len; c++) {
            float* weights_col = attn->attn_weights + attn->seq_len * (attn->seq_len * b + c);
            float* grad_col = attn->grad_scores + attn->seq_len * (attn->seq_len * b + c);
            
            float dot = cblas_sdot(attn->seq_len, weights_col, 1, grad_col, 1);
            
            for (int r = 0; r < attn->seq_len; r++) {
                grad_col[r] = weights_col[r] * (grad_col[r] - dot);
            }
        }
    }
    
    // Scale gradient by 1/sqrt(d_model)
    cblas_sscal(attn->seq_len * attn->seq_len * attn->batch_size, attn->scale, attn->grad_scores, 1);
    
    // Gradient through QK^T
    for (int b = 0; b < attn->batch_size; b++) {
        // grad_Q = grad_scores * K
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->seq_len,
                    1.0f, 
                    attn->grad_scores + attn->seq_len * attn->seq_len * b, attn->seq_len,
                    attn->K + attn->seq_len * attn->d_model * b, attn->seq_len,
                    0.0f, attn->grad_Q + attn->seq_len * attn->d_model * b, attn->seq_len);
        
        // grad_K = grad_scores^T * Q
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->seq_len,
                    1.0f, 
                    attn->grad_scores + attn->seq_len * attn->seq_len * b, attn->seq_len,
                    attn->Q + attn->seq_len * attn->d_model * b, attn->seq_len,
                    0.0f, attn->grad_K + attn->seq_len * attn->d_model * b, attn->seq_len);
    }
    
    // Gradient through projections
    for (int b = 0; b < attn->batch_size; b++) {
        // grad_W_q += X^T * grad_Q
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, attn->seq_len,
                    1.0f, 
                    attn->grad_Q + attn->seq_len * attn->d_model * b, attn->seq_len,
                    X + attn->seq_len * attn->d_model * b, attn->seq_len,
                    1.0f, attn->W_q_grad, attn->d_model);
        
        // grad_W_k += X^T * grad_K
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, attn->seq_len,
                    1.0f, 
                    attn->grad_K + attn->seq_len * attn->d_model * b, attn->seq_len,
                    X + attn->seq_len * attn->d_model * b, attn->seq_len,
                    1.0f, attn->W_k_grad, attn->d_model);
        
        // grad_W_v += X^T * grad_V
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, attn->seq_len,
                    1.0f, 
                    attn->grad_V + attn->seq_len * attn->d_model * b, attn->seq_len,
                    X + attn->seq_len * attn->d_model * b, attn->seq_len,
                    1.0f, attn->W_v_grad, attn->d_model);
        
        if (grad_X != NULL) {
            // grad_X = grad_Q * W_q + grad_K * W_k + grad_V * W_v
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len, attn->d_model, attn->d_model,
                        1.0f, 
                        attn->grad_Q + attn->seq_len * attn->d_model * b, attn->seq_len,
                        attn->W_q, attn->d_model,
                        0.0f, grad_X + attn->seq_len * attn->d_model * b, attn->seq_len);
            
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len, attn->d_model, attn->d_model,
                        1.0f, 
                        attn->grad_K + attn->seq_len * attn->d_model * b, attn->seq_len,
                        attn->W_k, attn->d_model,
                        1.0f, grad_X + attn->seq_len * attn->d_model * b, attn->seq_len);
            
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len, attn->d_model, attn->d_model,
                        1.0f, 
                        attn->grad_V + attn->seq_len * attn->d_model * b, attn->seq_len,
                        attn->W_v, attn->d_model,
                        1.0f, grad_X + attn->seq_len * attn->d_model * b, attn->seq_len);
        }
    }
}

// Helper function to update weights with AdamW
static void adamw_update(float* weight, float* grad, float* m, float* v,
                         float beta1, float beta2, float epsilon,
                         float learning_rate, float weight_decay,
                         float alpha_t, int size, int batch_size) {
    for (int i = 0; i < size; i++) {
        float g = grad[i] / batch_size;
        
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[i] / (sqrtf(v[i]) + epsilon);
        weight[i] = weight[i] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int weight_size = attn->d_model * attn->d_model;
    
    adamw_update(attn->W_q, attn->W_q_grad, attn->W_q_m, attn->W_q_v,
                 attn->beta1, attn->beta2, attn->epsilon, learning_rate,
                 attn->weight_decay, alpha_t, weight_size, attn->batch_size);
    
    adamw_update(attn->W_k, attn->W_k_grad, attn->W_k_m, attn->W_k_v,
                 attn->beta1, attn->beta2, attn->epsilon, learning_rate,
                 attn->weight_decay, alpha_t, weight_size, attn->batch_size);
    
    adamw_update(attn->W_v, attn->W_v_grad, attn->W_v_m, attn->W_v_v,
                 attn->beta1, attn->beta2, attn->epsilon, learning_rate,
                 attn->weight_decay, alpha_t, weight_size, attn->batch_size);
    
    adamw_update(attn->W_o, attn->W_o_grad, attn->W_o_m, attn->W_o_v,
                 attn->beta1, attn->beta2, attn->epsilon, learning_rate,
                 attn->weight_decay, alpha_t, weight_size, attn->batch_size);
}

// Save model
void save_attention(Attention* attn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&attn->seq_len, sizeof(int), 1, file);
    fwrite(&attn->d_model, sizeof(int), 1, file);
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    
    int weight_size = attn->d_model * attn->d_model;
    
    // Save weights
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

// Load model
Attention* load_attention(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int seq_len, d_model, stored_batch_size;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    Attention* attn = init_attention(seq_len, d_model, batch_size);
    
    int weight_size = d_model * d_model;
    
    // Load weights
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