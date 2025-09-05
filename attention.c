#include "attention.h"

// Initialize the attention layer
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
    attn->weight_decay = 0.01f;
    
    int weight_size = d_model * d_model;
    int seq_batch_size = seq_len * d_model * batch_size;
    int attn_matrix_size = seq_len * seq_len * batch_size;
    
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
    attn->scores = (float*)malloc(attn_matrix_size * sizeof(float));
    attn->attn_weights = (float*)malloc(attn_matrix_size * sizeof(float));
    attn->attn_output = (float*)malloc(seq_batch_size * sizeof(float));
    attn->output = (float*)malloc(seq_batch_size * sizeof(float));
    
    // Allocate backward pass buffers
    attn->grad_output = (float*)malloc(seq_batch_size * sizeof(float));
    attn->grad_attn_output = (float*)malloc(seq_batch_size * sizeof(float));
    attn->grad_weights = (float*)malloc(attn_matrix_size * sizeof(float));
    attn->grad_scores = (float*)malloc(attn_matrix_size * sizeof(float));
    attn->grad_Q = (float*)malloc(seq_batch_size * sizeof(float));
    attn->grad_K = (float*)malloc(seq_batch_size * sizeof(float));
    attn->grad_V = (float*)malloc(seq_batch_size * sizeof(float));
    
    // Initialize weights
    float scale_W = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < weight_size; i++) {
        attn->W_q[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
        attn->W_k[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
        attn->W_v[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
        attn->W_o[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W;
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
    free(attn->grad_output); free(attn->grad_attn_output); free(attn->grad_weights);
    free(attn->grad_scores); free(attn->grad_Q); free(attn->grad_K); free(attn->grad_V);
    free(attn);
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    (void)attn;
    (void)X;
    // TODO: Implement forward pass
    // 1. Q = X * W_q, K = X * W_k, V = X * W_v
    // 2. scores = Q * K^T / sqrt(d_model)
    // 3. attn_weights = softmax(scores)
    // 4. attn_output = attn_weights * V
    // 5. output = attn_output * W_o
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* y) {
    (void)attn;
    (void)y;
    // TODO: Implement loss calculation (MSE)
    // Calculate grad_output = output - y and return MSE
    return 0.0f;
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
    (void)attn;
    (void)X;
    (void)grad_X;
    // TODO: Implement backward pass
    // 1. Gradient through output projection: grad_W_o, grad_attn_output
    // 2. Gradient through attention: grad_weights, grad_V
    // 3. Gradient through softmax: grad_scores
    // 4. Gradient through scaled dot-product: grad_Q, grad_K
    // 5. Gradient through projections: grad_W_q, grad_W_k, grad_W_v
    // 6. If grad_X != NULL, compute gradient w.r.t. input
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;  // Increment time step
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int weight_size = attn->d_model * attn->d_model;
    
    // Update W_q weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->W_q_grad[i] / attn->batch_size;
        
        attn->W_q_m[i] = attn->beta1 * attn->W_q_m[i] + (1.0f - attn->beta1) * grad;
        attn->W_q_v[i] = attn->beta2 * attn->W_q_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->W_q_m[i] / (sqrtf(attn->W_q_v[i]) + attn->epsilon);
        attn->W_q[i] = attn->W_q[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update W_k weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->W_k_grad[i] / attn->batch_size;
        
        attn->W_k_m[i] = attn->beta1 * attn->W_k_m[i] + (1.0f - attn->beta1) * grad;
        attn->W_k_v[i] = attn->beta2 * attn->W_k_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->W_k_m[i] / (sqrtf(attn->W_k_v[i]) + attn->epsilon);
        attn->W_k[i] = attn->W_k[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update W_v weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->W_v_grad[i] / attn->batch_size;
        
        attn->W_v_m[i] = attn->beta1 * attn->W_v_m[i] + (1.0f - attn->beta1) * grad;
        attn->W_v_v[i] = attn->beta2 * attn->W_v_v[i] + (1.0f - attn->beta2) * grad * grad;
        
        float update = alpha_t * attn->W_v_m[i] / (sqrtf(attn->W_v_v[i]) + attn->epsilon);
        attn->W_v[i] = attn->W_v[i] * (1.0f - learning_rate * attn->weight_decay) - update;
    }
    
    // Update W_o weights
    for (int i = 0; i < weight_size; i++) {
        float grad = attn->W_o_grad[i] / attn->batch_size;
        
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

// Load model weights and Adam state from binary file
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
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize attention layer
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