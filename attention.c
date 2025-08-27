#include "attention.h"

// Initialize the network with configurable dimensions
Attention* init_attention(int input_dim, int head_dim, int output_dim, int seq_len, int batch_size) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->input_dim = input_dim;
    attn->head_dim = head_dim;
    attn->output_dim = output_dim;
    attn->seq_len = seq_len;
    attn->batch_size = batch_size;
    
    // Initialize AdamW parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    // Calculate sizes
    int w_q_size = input_dim * head_dim;
    int w_k_size = input_dim * head_dim;
    int w_v_size = input_dim * head_dim;
    int w_o_size = head_dim * output_dim;
    int total_tokens = batch_size * seq_len;
    
    // Allocate and initialize weights and gradients
    attn->W_q = (float*)malloc(w_q_size * sizeof(float));
    attn->W_k = (float*)malloc(w_k_size * sizeof(float));
    attn->W_v = (float*)malloc(w_v_size * sizeof(float));
    attn->W_o = (float*)malloc(w_o_size * sizeof(float));
    attn->W_q_grad = (float*)malloc(w_q_size * sizeof(float));
    attn->W_k_grad = (float*)malloc(w_k_size * sizeof(float));
    attn->W_v_grad = (float*)malloc(w_v_size * sizeof(float));
    attn->W_o_grad = (float*)malloc(w_o_size * sizeof(float));
    
    // Allocate AdamW buffers
    attn->W_q_m = (float*)calloc(w_q_size, sizeof(float));
    attn->W_q_v = (float*)calloc(w_q_size, sizeof(float));
    attn->W_k_m = (float*)calloc(w_k_size, sizeof(float));
    attn->W_k_v = (float*)calloc(w_k_size, sizeof(float));
    attn->W_v_m = (float*)calloc(w_v_size, sizeof(float));
    attn->W_v_v = (float*)calloc(w_v_size, sizeof(float));
    attn->W_o_m = (float*)calloc(w_o_size, sizeof(float));
    attn->W_o_v = (float*)calloc(w_o_size, sizeof(float));
    
    // Allocate layer outputs and working buffers
    attn->layer_Q = (float*)malloc(total_tokens * head_dim * sizeof(float));
    attn->layer_K = (float*)malloc(total_tokens * head_dim * sizeof(float));
    attn->layer_V = (float*)malloc(total_tokens * head_dim * sizeof(float));
    attn->layer_scores = (float*)malloc(total_tokens * seq_len * sizeof(float));
    attn->layer_attn = (float*)malloc(total_tokens * seq_len * sizeof(float));
    attn->layer_context = (float*)malloc(total_tokens * head_dim * sizeof(float));
    attn->layer_output = (float*)malloc(total_tokens * output_dim * sizeof(float));
    attn->error_output = (float*)malloc(total_tokens * output_dim * sizeof(float));
    attn->error_context = (float*)malloc(total_tokens * head_dim * sizeof(float));
    attn->error_Q = (float*)malloc(total_tokens * head_dim * sizeof(float));
    attn->error_K = (float*)malloc(total_tokens * head_dim * sizeof(float));
    attn->error_V = (float*)malloc(total_tokens * head_dim * sizeof(float));
    attn->error_scores = (float*)malloc(total_tokens * seq_len * sizeof(float));
    
    // Initialize weights
    float scale_input = 1.0f / sqrtf(input_dim);
    float scale_head = 1.0f / sqrtf(head_dim);
    
    for (int i = 0; i < w_q_size; i++) {
        attn->W_q[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_input;
    }
    
    for (int i = 0; i < w_k_size; i++) {
        attn->W_k[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_input;
    }
    
    for (int i = 0; i < w_v_size; i++) {
        attn->W_v[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_input;
    }
    
    for (int i = 0; i < w_o_size; i++) {
        attn->W_o[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_head;
    }
    
    return attn;
}

// Free network memory
void free_attention(Attention* attn) {
    free(attn->W_q); free(attn->W_k); free(attn->W_v); free(attn->W_o);
    free(attn->W_q_grad); free(attn->W_k_grad); free(attn->W_v_grad); free(attn->W_o_grad);
    free(attn->W_q_m); free(attn->W_q_v);
    free(attn->W_k_m); free(attn->W_k_v);
    free(attn->W_v_m); free(attn->W_v_v);
    free(attn->W_o_m); free(attn->W_o_v);
    free(attn->layer_Q); free(attn->layer_K); free(attn->layer_V);
    free(attn->layer_scores); free(attn->layer_attn); free(attn->layer_context);
    free(attn->layer_output);
    free(attn->error_output); free(attn->error_context);
    free(attn->error_Q); free(attn->error_K); free(attn->error_V);
    free(attn->error_scores);
    free(attn);
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    int total_tokens = attn->batch_size * attn->seq_len;
    
    // Q = X W_q
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, attn->head_dim, attn->input_dim,
                1.0f, X, attn->input_dim,
                attn->W_q, attn->head_dim,
                0.0f, attn->layer_Q, attn->head_dim);
    
    // K = X W_k
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, attn->head_dim, attn->input_dim,
                1.0f, X, attn->input_dim,
                attn->W_k, attn->head_dim,
                0.0f, attn->layer_K, attn->head_dim);
    
    // V = X W_v
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, attn->head_dim, attn->input_dim,
                1.0f, X, attn->input_dim,
                attn->W_v, attn->head_dim,
                0.0f, attn->layer_V, attn->head_dim);
    
    // For each sequence in batch, compute attention
    float scale = 1.0f / sqrtf((float)attn->head_dim);
    
    for (int b = 0; b < attn->batch_size; b++) {
        float* Q_seq = &attn->layer_Q[b * attn->seq_len * attn->head_dim];
        float* K_seq = &attn->layer_K[b * attn->seq_len * attn->head_dim];
        float* V_seq = &attn->layer_V[b * attn->seq_len * attn->head_dim];
        float* scores_seq = &attn->layer_scores[b * attn->seq_len * attn->seq_len];
        float* attn_seq = &attn->layer_attn[b * attn->seq_len * attn->seq_len];
        float* context_seq = &attn->layer_context[b * attn->seq_len * attn->head_dim];
        
        // Scores = Q K^T / sqrt(head_dim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->seq_len, attn->head_dim,
                    scale, Q_seq, attn->head_dim,
                    K_seq, attn->head_dim,
                    0.0f, scores_seq, attn->seq_len);
        
        // Apply softmax row-wise to get attention weights
        for (int i = 0; i < attn->seq_len; i++) {
            float* row = &scores_seq[i * attn->seq_len];
            float max_val = row[0];
            for (int j = 1; j < attn->seq_len; j++) {
                if (row[j] > max_val) max_val = row[j];
            }
            
            float sum = 0.0f;
            for (int j = 0; j < attn->seq_len; j++) {
                row[j] = expf(row[j] - max_val);
                sum += row[j];
            }
            
            float inv_sum = 1.0f / (sum + 1e-12f);
            for (int j = 0; j < attn->seq_len; j++) {
                row[j] *= inv_sum;
            }
        }
        
        // Copy attention weights
        memcpy(attn_seq, scores_seq, attn->seq_len * attn->seq_len * sizeof(float));
        
        // Context = Attention * V
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->head_dim, attn->seq_len,
                    1.0f, attn_seq, attn->seq_len,
                    V_seq, attn->head_dim,
                    0.0f, context_seq, attn->head_dim);
    }
    
    // Output = Context W_o
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, attn->output_dim, attn->head_dim,
                1.0f, attn->layer_context, attn->head_dim,
                attn->W_o, attn->output_dim,
                0.0f, attn->layer_output, attn->output_dim);
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* y) {
    // ∂L/∂Y = Y - Y_true
    float loss = 0.0f;
    int total_size = attn->batch_size * attn->seq_len * attn->output_dim;
    
    for (int i = 0; i < total_size; i++) {
        attn->error_output[i] = attn->layer_output[i] - y[i];
        loss += attn->error_output[i] * attn->error_output[i];
    }
    return loss / total_size;
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    int w_q_size = attn->input_dim * attn->head_dim;
    int w_k_size = attn->input_dim * attn->head_dim;
    int w_v_size = attn->input_dim * attn->head_dim;
    int w_o_size = attn->head_dim * attn->output_dim;
    
    memset(attn->W_q_grad, 0, w_q_size * sizeof(float));
    memset(attn->W_k_grad, 0, w_k_size * sizeof(float));
    memset(attn->W_v_grad, 0, w_v_size * sizeof(float));
    memset(attn->W_o_grad, 0, w_o_size * sizeof(float));
}

// Backward pass
void backward_pass_attention(Attention* attn, float* X) {
    int total_tokens = attn->batch_size * attn->seq_len;
    
    // ∂L/∂W_o += Context^T * error_output
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                attn->head_dim, attn->output_dim, total_tokens,
                1.0f, attn->layer_context, attn->head_dim,
                attn->error_output, attn->output_dim,
                1.0f, attn->W_o_grad, attn->output_dim);
    
    // ∂L/∂Context = error_output * W_o^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                total_tokens, attn->head_dim, attn->output_dim,
                1.0f, attn->error_output, attn->output_dim,
                attn->W_o, attn->output_dim,
                0.0f, attn->error_context, attn->head_dim);
    
    // For each sequence in batch, compute attention gradients
    for (int b = 0; b < attn->batch_size; b++) {
        float* Q_seq = &attn->layer_Q[b * attn->seq_len * attn->head_dim];
        float* K_seq = &attn->layer_K[b * attn->seq_len * attn->head_dim];
        float* V_seq = &attn->layer_V[b * attn->seq_len * attn->head_dim];
        float* attn_seq = &attn->layer_attn[b * attn->seq_len * attn->seq_len];
        float* error_context_seq = &attn->error_context[b * attn->seq_len * attn->head_dim];
        float* error_V_seq = &attn->error_V[b * attn->seq_len * attn->head_dim];
        float* error_scores_seq = &attn->error_scores[b * attn->seq_len * attn->seq_len];
        float* error_Q_seq = &attn->error_Q[b * attn->seq_len * attn->head_dim];
        float* error_K_seq = &attn->error_K[b * attn->seq_len * attn->head_dim];
        
        // ∂L/∂V = Attention^T * error_context
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->seq_len, attn->head_dim, attn->seq_len,
                    1.0f, attn_seq, attn->seq_len,
                    error_context_seq, attn->head_dim,
                    0.0f, error_V_seq, attn->head_dim);
        
        // ∂L/∂Attention = error_context * V^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->seq_len, attn->head_dim,
                    1.0f, error_context_seq, attn->head_dim,
                    V_seq, attn->head_dim,
                    0.0f, error_scores_seq, attn->seq_len);
        
        // Convert ∂L/∂Attention to ∂L/∂Scores via softmax jacobian
        for (int i = 0; i < attn->seq_len; i++) {
            float* attn_row = &attn_seq[i * attn->seq_len];
            float* error_row = &error_scores_seq[i * attn->seq_len];
            
            float sum = 0.0f;
            for (int j = 0; j < attn->seq_len; j++) {
                sum += error_row[j] * attn_row[j];
            }
            
            for (int j = 0; j < attn->seq_len; j++) {
                error_row[j] = attn_row[j] * (error_row[j] - sum);
            }
        }
        
        float scale = 1.0f / sqrtf((float)attn->head_dim);
        
        // ∂L/∂Q = error_scores * K / sqrt(head_dim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->head_dim, attn->seq_len,
                    scale, error_scores_seq, attn->seq_len,
                    K_seq, attn->head_dim,
                    0.0f, error_Q_seq, attn->head_dim);
        
        // ∂L/∂K = error_scores^T * Q / sqrt(head_dim)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->seq_len, attn->head_dim, attn->seq_len,
                    scale, error_scores_seq, attn->seq_len,
                    Q_seq, attn->head_dim,
                    0.0f, error_K_seq, attn->head_dim);
    }
    
    // ∂L/∂W_q += X^T * error_Q
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                attn->input_dim, attn->head_dim, total_tokens,
                1.0f, X, attn->input_dim,
                attn->error_Q, attn->head_dim,
                1.0f, attn->W_q_grad, attn->head_dim);
    
    // ∂L/∂W_k += X^T * error_K
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                attn->input_dim, attn->head_dim, total_tokens,
                1.0f, X, attn->input_dim,
                attn->error_K, attn->head_dim,
                1.0f, attn->W_k_grad, attn->head_dim);
    
    // ∂L/∂W_v += X^T * error_V
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                attn->input_dim, attn->head_dim, total_tokens,
                1.0f, X, attn->input_dim,
                attn->error_V, attn->head_dim,
                1.0f, attn->W_v_grad, attn->head_dim);
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;
    int total_samples = attn->batch_size * attn->seq_len;
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int w_q_size = attn->input_dim * attn->head_dim;
    int w_k_size = attn->input_dim * attn->head_dim;
    int w_v_size = attn->input_dim * attn->head_dim;
    int w_o_size = attn->head_dim * attn->output_dim;
    
    // Update W_q weights
    for (int i = 0; i < w_q_size; i++) {
        float grad = attn->W_q_grad[i] / total_samples;
        
        attn->W_q_m[i] = attn->beta1 * attn->W_q_m[i] + (1.0f - attn->beta1) * grad;
        attn->W_q_v[i] = attn->beta2 * attn->W_q_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->W_q_m[i] / (sqrtf(attn->W_q_v[i]) + attn->epsilon);
        attn->W_q[i] = attn->W_q[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update W_k weights
    for (int i = 0; i < w_k_size; i++) {
        float grad = attn->W_k_grad[i] / total_samples;
        
        attn->W_k_m[i] = attn->beta1 * attn->W_k_m[i] + (1.0f - attn->beta1) * grad;
        attn->W_k_v[i] = attn->beta2 * attn->W_k_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->W_k_m[i] / (sqrtf(attn->W_k_v[i]) + attn->epsilon);
        attn->W_k[i] = attn->W_k[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update W_v weights
    for (int i = 0; i < w_v_size; i++) {
        float grad = attn->W_v_grad[i] / total_samples;
        
        attn->W_v_m[i] = attn->beta1 * attn->W_v_m[i] + (1.0f - attn->beta1) * grad;
        attn->W_v_v[i] = attn->beta2 * attn->W_v_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->W_v_m[i] / (sqrtf(attn->W_v_v[i]) + attn->epsilon);
        attn->W_v[i] = attn->W_v[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update W_o weights
    for (int i = 0; i < w_o_size; i++) {
        float grad = attn->W_o_grad[i] / total_samples;
        
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
    fwrite(&attn->input_dim, sizeof(int), 1, file);
    fwrite(&attn->head_dim, sizeof(int), 1, file);
    fwrite(&attn->output_dim, sizeof(int), 1, file);
    fwrite(&attn->seq_len, sizeof(int), 1, file);
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    
    // Calculate sizes
    int w_q_size = attn->input_dim * attn->head_dim;
    int w_k_size = attn->input_dim * attn->head_dim;
    int w_v_size = attn->input_dim * attn->head_dim;
    int w_o_size = attn->head_dim * attn->output_dim;
    
    // Save weights
    fwrite(attn->W_q, sizeof(float), w_q_size, file);
    fwrite(attn->W_k, sizeof(float), w_k_size, file);
    fwrite(attn->W_v, sizeof(float), w_v_size, file);
    fwrite(attn->W_o, sizeof(float), w_o_size, file);
    
    // Save Adam state
    fwrite(&attn->t, sizeof(int), 1, file);
    fwrite(attn->W_q_m, sizeof(float), w_q_size, file);
    fwrite(attn->W_q_v, sizeof(float), w_q_size, file);
    fwrite(attn->W_k_m, sizeof(float), w_k_size, file);
    fwrite(attn->W_k_v, sizeof(float), w_k_size, file);
    fwrite(attn->W_v_m, sizeof(float), w_v_size, file);
    fwrite(attn->W_v_v, sizeof(float), w_v_size, file);
    fwrite(attn->W_o_m, sizeof(float), w_o_size, file);
    fwrite(attn->W_o_v, sizeof(float), w_o_size, file);

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
    int input_dim, head_dim, output_dim, seq_len, stored_batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&head_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    Attention* attn = init_attention(input_dim, head_dim, output_dim, seq_len, batch_size);
    
    // Calculate sizes
    int w_q_size = input_dim * head_dim;
    int w_k_size = input_dim * head_dim;
    int w_v_size = input_dim * head_dim;
    int w_o_size = head_dim * output_dim;
    
    // Load weights
    fread(attn->W_q, sizeof(float), w_q_size, file);
    fread(attn->W_k, sizeof(float), w_k_size, file);
    fread(attn->W_v, sizeof(float), w_v_size, file);
    fread(attn->W_o, sizeof(float), w_o_size, file);
    
    // Load Adam state
    fread(&attn->t, sizeof(int), 1, file);
    fread(attn->W_q_m, sizeof(float), w_q_size, file);
    fread(attn->W_q_v, sizeof(float), w_q_size, file);
    fread(attn->W_k_m, sizeof(float), w_k_size, file);
    fread(attn->W_k_v, sizeof(float), w_k_size, file);
    fread(attn->W_v_m, sizeof(float), w_v_size, file);
    fread(attn->W_v_v, sizeof(float), w_v_size, file);
    fread(attn->W_o_m, sizeof(float), w_o_size, file);
    fread(attn->W_o_v, sizeof(float), w_o_size, file);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return attn;
}