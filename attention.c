#include "attention.h"

// Initialize the attention layer
Attention* init_attention(int seq_len, int d_model, int batch_size, bool is_causal, bool use_rope) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    
    // Store dimensions
    attn->seq_len = seq_len;
    attn->d_model = d_model;
    attn->batch_size = batch_size;
    attn->scale = 1.0f / sqrtf(d_model);
    attn->is_causal = is_causal;
    attn->use_rope = use_rope;
    
    // Initialize Adam parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;
    
    int weight_size = d_model * d_model;
    int seq_batch_size = batch_size * seq_len * d_model;
    int attn_matrix_size = batch_size * seq_len * seq_len;
    
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
    
    // Alias/Allocate backward pass buffers
    attn->grad_output = attn->output;
    attn->grad_attn_output = attn->attn_output;
    attn->grad_weights = (float*)malloc(attn_matrix_size * sizeof(float));
    attn->grad_scores = attn->scores;
    attn->grad_Q = attn->Q;
    attn->grad_K = attn->K;
    attn->grad_V = attn->V;
    
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
    free(attn->grad_weights);
    free(attn);
}

// Softmax forward pass
static void softmax_forward_attention(float* weights, float* scores, int batch_size, int seq_len) {
    for (int b = 0; b < batch_size; b++) {
        float* scores_b = &scores[b * seq_len * seq_len];
        float* weights_b = &weights[b * seq_len * seq_len];
        
        for (int i = 0; i < seq_len; i++) {
            // Find max for numerical stability
            float max_val = -1e30f;
            for (int j = 0; j < seq_len; j++) {
                float val = scores_b[i * seq_len + j];
                if (val > max_val) max_val = val;
            }
            
            // A_ij = exp(S_ij)/∑_k exp(S_ik)
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                float exp_val = expf(scores_b[i * seq_len + j] - max_val);
                weights_b[i * seq_len + j] = exp_val;
                sum_exp += exp_val;
            }
            
            // Normalize
            for (int j = 0; j < seq_len; j++) {
                weights_b[i * seq_len + j] /= sum_exp;
            }
        }
    }
}

// Causal softmax forward pass
static void softmax_causal_forward_attention(float* weights, float* scores, int batch_size, int seq_len) {
    for (int b = 0; b < batch_size; b++) {
        float* scores_b = &scores[b * seq_len * seq_len];
        float* weights_b = &weights[b * seq_len * seq_len];
        
        for (int i = 0; i < seq_len; i++) {
            // Find max for numerical stability (only consider positions <= i)
            float max_val = -1e30f;
            for (int j = 0; j <= i; j++) {
                float val = scores_b[i * seq_len + j];
                if (val > max_val) max_val = val;
            }
            
            // A_ij = exp(S_ij)/∑_k exp(S_ik) for j <= i, 0 for j > i
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                if (j <= i) {
                    float exp_val = expf(scores_b[i * seq_len + j] - max_val);
                    weights_b[i * seq_len + j] = exp_val;
                    sum_exp += exp_val;
                } else {
                    weights_b[i * seq_len + j] = 0.0f;
                }
            }
            
            // Normalize only the valid positions
            for (int j = 0; j <= i; j++) {
                weights_b[i * seq_len + j] /= sum_exp;
            }
        }
    }
}

// Softmax backward pass
static void softmax_backward_attention(float* grad_scores, float* grad_weights, float* weights, int batch_size, int seq_len) {
    for (int b = 0; b < batch_size; b++) {
        float* grad_weights_b = &grad_weights[b * seq_len * seq_len];
        float* weights_b = &weights[b * seq_len * seq_len];
        float* grad_scores_b = &grad_scores[b * seq_len * seq_len];
        
        for (int i = 0; i < seq_len; i++) {
            // Compute ∑_j ∂L/∂A⊙A
            float sum_term = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                int idx = i * seq_len + k;
                sum_term += grad_weights_b[idx] * weights_b[idx];
            }
            
            // ∂L/∂S = A⊙(∂L/∂A - ∑_j ∂L/∂A⊙A)
            for (int j = 0; j < seq_len; j++) {
                int idx = i * seq_len + j;
                grad_scores_b[idx] = weights_b[idx] * (grad_weights_b[idx] - sum_term);
            }
        }
    }
}

// Causal softmax backward pass
static void softmax_causal_backward_attention(float* grad_scores, float* grad_weights, float* weights, int batch_size, int seq_len) {
    for (int b = 0; b < batch_size; b++) {
        float* grad_weights_b = &grad_weights[b * seq_len * seq_len];
        float* weights_b = &weights[b * seq_len * seq_len];
        float* grad_scores_b = &grad_scores[b * seq_len * seq_len];
        
        for (int i = 0; i < seq_len; i++) {
            // Compute ∑_j ∂L/∂A⊙A (only for j <= i)
            float sum_term = 0.0f;
            for (int k = 0; k <= i; k++) {
                int idx = i * seq_len + k;
                sum_term += grad_weights_b[idx] * weights_b[idx];
            }
            
            // ∂L/∂S = A⊙(∂L/∂A - ∑_j ∂L/∂A⊙A) for j <= i, 0 for j > i
            for (int j = 0; j < seq_len; j++) {
                int idx = i * seq_len + j;
                if (j <= i) {
                    grad_scores_b[idx] = weights_b[idx] * (grad_weights_b[idx] - sum_term);
                } else {
                    grad_scores_b[idx] = 0.0f;
                }
            }
        }
    }
}

// RoPE forward pass
static void rope_forward_attention(float* Q, float* K, int batch_size, int seq_len, int d_model) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            for (int d_pair = 0; d_pair < d_model / 2; d_pair++) {
                int d = d_pair * 2;
                
                // Compute rotation angle
                float theta = powf(10000.0f, -((float)d / (float)d_model));
                float angle = (float)t * theta;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                
                int idx = b * seq_len * d_model + t * d_model + d;
                
                // Rotate Q
                float q0 = Q[idx];
                float q1 = Q[idx + 1];
                Q[idx] = q0 * cos_a - q1 * sin_a;
                Q[idx + 1] = q0 * sin_a + q1 * cos_a;
                
                // Rotate K
                float k0 = K[idx];
                float k1 = K[idx + 1];
                K[idx] = k0 * cos_a - k1 * sin_a;
                K[idx + 1] = k0 * sin_a + k1 * cos_a;
            }
        }
    }
}

// RoPE backward pass
static void rope_backward_attention(float* grad_Q, float* grad_K, int batch_size, int seq_len, int d_model) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            for (int d_pair = 0; d_pair < d_model / 2; d_pair++) {
                int d = d_pair * 2;
                
                // Compute rotation angle
                float theta = powf(10000.0f, -((float)d / (float)d_model));
                float angle = (float)t * theta;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                
                int idx = b * seq_len * d_model + t * d_model + d;
                
                // Inverse rotate grad_Q
                float gq0 = grad_Q[idx];
                float gq1 = grad_Q[idx + 1];
                grad_Q[idx] = gq0 * cos_a + gq1 * sin_a;
                grad_Q[idx + 1] = -gq0 * sin_a + gq1 * cos_a;
                
                // Inverse rotate grad_K
                float gk0 = grad_K[idx];
                float gk1 = grad_K[idx + 1];
                grad_K[idx] = gk0 * cos_a + gk1 * sin_a;
                grad_K[idx + 1] = -gk0 * sin_a + gk1 * cos_a;
            }
        }
    }
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    // Step 1: Compute Q, K, V for each batch separately
    // Q = XW_q, K = XW_k, V = XW_v
    for (int b = 0; b < attn->batch_size; b++) {
        float* X_b = &X[b * attn->seq_len * attn->d_model];
        float* Q_b = &attn->Q[b * attn->seq_len * attn->d_model];
        float* K_b = &attn->K[b * attn->seq_len * attn->d_model];
        float* V_b = &attn->V[b * attn->seq_len * attn->d_model];
        
        // Q = XW_q
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, X_b, attn->d_model,
                    attn->W_q, attn->d_model,
                    0.0f, Q_b, attn->d_model);
        
        // K = XW_k
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, X_b, attn->d_model,
                    attn->W_k, attn->d_model,
                    0.0f, K_b, attn->d_model);
        
        // V = XW_v
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, X_b, attn->d_model,
                    attn->W_v, attn->d_model,
                    0.0f, V_b, attn->d_model);
    }
    
    // Step 2: Apply RoPE to Q and K
    if (attn->use_rope) {
        rope_forward_attention(attn->Q, attn->K, attn->batch_size, attn->seq_len, attn->d_model);
    }
    
    // Step 3: Compute attention scores
    // S = QKᵀ/√d_model
    for (int b = 0; b < attn->batch_size; b++) {
        float* Q_b = &attn->Q[b * attn->seq_len * attn->d_model];
        float* K_b = &attn->K[b * attn->seq_len * attn->d_model];
        float* scores_b = &attn->scores[b * attn->seq_len * attn->seq_len];
        
        // S = QKᵀ/√d_model
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->seq_len, attn->d_model,
                    attn->scale, Q_b, attn->d_model,
                    K_b, attn->d_model,
                    0.0f, scores_b, attn->seq_len);
    }
    
    // Step 4: Apply softmax row-wise
    if (attn->is_causal) {
        softmax_causal_forward_attention(attn->attn_weights, attn->scores, attn->batch_size, attn->seq_len);
    } else {
        softmax_forward_attention(attn->attn_weights, attn->scores, attn->batch_size, attn->seq_len);
    }
    
    // Step 5: Compute attention output
    // Z = AV
    for (int b = 0; b < attn->batch_size; b++) {
        float* weights_b = &attn->attn_weights[b * attn->seq_len * attn->seq_len];
        float* V_b = &attn->V[b * attn->seq_len * attn->d_model];
        float* attn_output_b = &attn->attn_output[b * attn->seq_len * attn->d_model];
        
        // Z = AV
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->seq_len,
                    1.0f, weights_b, attn->seq_len,
                    V_b, attn->d_model,
                    0.0f, attn_output_b, attn->d_model);
    }
    
    // Step 6: Apply output projection
    // Y = ZW_o
    for (int b = 0; b < attn->batch_size; b++) {
        float* attn_output_b = &attn->attn_output[b * attn->seq_len * attn->d_model];
        float* output_b = &attn->output[b * attn->seq_len * attn->d_model];
        
        // Y = ZW_o
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, attn_output_b, attn->d_model,
                    attn->W_o, attn->d_model,
                    0.0f, output_b, attn->d_model);
    }
}

// Calculate loss
float calculate_loss_attention(Attention* attn, float* y) {
    int total_elements = attn->batch_size * attn->seq_len * attn->d_model;
    
    // ∂L/∂Y = Y - Y_true
    cblas_scopy(total_elements, attn->output, 1, attn->grad_output, 1);
    cblas_saxpy(total_elements, -1.0f, y, 1, attn->grad_output, 1);
    
    // Calculate MSE loss
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
    // Step 6 (backward): Gradient through output projection
    // ∂L/∂W_o = Zᵀ(∂L/∂Y), ∂L/∂Z = (∂L/∂Y)W_oᵀ
    for (int b = 0; b < attn->batch_size; b++) {
        float* grad_output_b = &attn->grad_output[b * attn->seq_len * attn->d_model];
        float* attn_output_b = &attn->attn_output[b * attn->seq_len * attn->d_model];
        float* grad_attn_output_b = &attn->grad_attn_output[b * attn->seq_len * attn->d_model];
        
        // ∂L/∂W_o += Zᵀ(∂L/∂Y)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, attn->seq_len,
                    1.0f, attn_output_b, attn->d_model,
                    grad_output_b, attn->d_model,
                    1.0f, attn->W_o_grad, attn->d_model);
        
        // ∂L/∂Z = (∂L/∂Y)W_oᵀ
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, grad_output_b, attn->d_model,
                    attn->W_o, attn->d_model,
                    0.0f, grad_attn_output_b, attn->d_model);
    }
    
    // Step 5 (backward): Gradient through attention output computation
    // ∂L/∂A = (∂L/∂Z)Vᵀ, ∂L/∂V = Aᵀ(∂L/∂Z)
    for (int b = 0; b < attn->batch_size; b++) {
        float* grad_attn_output_b = &attn->grad_attn_output[b * attn->seq_len * attn->d_model];
        float* weights_b = &attn->attn_weights[b * attn->seq_len * attn->seq_len];
        float* V_b = &attn->V[b * attn->seq_len * attn->d_model];
        float* grad_weights_b = &attn->grad_weights[b * attn->seq_len * attn->seq_len];
        float* grad_V_b = &attn->grad_V[b * attn->seq_len * attn->d_model];
        
        // ∂L/∂A = (∂L/∂Z)Vᵀ
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    attn->seq_len, attn->seq_len, attn->d_model,
                    1.0f, grad_attn_output_b, attn->d_model,
                    V_b, attn->d_model,
                    0.0f, grad_weights_b, attn->seq_len);
        
        // ∂L/∂V = Aᵀ(∂L/∂Z)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->seq_len,
                    1.0f, weights_b, attn->seq_len,
                    grad_attn_output_b, attn->d_model,
                    0.0f, grad_V_b, attn->d_model);
    }
    
    // Step 4 (backward): Gradient through softmax
    if (attn->is_causal) {
        softmax_causal_backward_attention(attn->grad_scores, attn->grad_weights, attn->attn_weights, attn->batch_size, attn->seq_len);
    } else {
        softmax_backward_attention(attn->grad_scores, attn->grad_weights, attn->attn_weights, attn->batch_size, attn->seq_len);
    }
    
    // Step 3 (backward): Gradient through attention scores
    // ∂L/∂Q = (∂L/∂S)K/√d_model, ∂L/∂K = (∂L/∂S)ᵀQ/√d_model
    for (int b = 0; b < attn->batch_size; b++) {
        float* grad_scores_b = &attn->grad_scores[b * attn->seq_len * attn->seq_len];
        float* Q_b = &attn->Q[b * attn->seq_len * attn->d_model];
        float* K_b = &attn->K[b * attn->seq_len * attn->d_model];
        float* grad_Q_b = &attn->grad_Q[b * attn->seq_len * attn->d_model];
        float* grad_K_b = &attn->grad_K[b * attn->seq_len * attn->d_model];
        
        // ∂L/∂Q = (∂L/∂S)K/√d_model
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->seq_len,
                    attn->scale, grad_scores_b, attn->seq_len,
                    K_b, attn->d_model,
                    0.0f, grad_Q_b, attn->d_model);
        
        // ∂L/∂K = (∂L/∂S)ᵀQ/√d_model
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->seq_len,
                    attn->scale, grad_scores_b, attn->seq_len,
                    Q_b, attn->d_model,
                    0.0f, grad_K_b, attn->d_model);
    }
    
    // Step 2 (backward): Apply inverse RoPE to grad_Q and grad_K
    if (attn->use_rope) {
        rope_backward_attention(attn->grad_Q, attn->grad_K, attn->batch_size, attn->seq_len, attn->d_model);
    }
    
    // Step 1 (backward): Gradient through linear projections
    // ∂L/∂W_q = Xᵀ(∂L/∂Q), ∂L/∂W_k = Xᵀ(∂L/∂K), ∂L/∂W_v = Xᵀ(∂L/∂V)
    // ∂L/∂X = (∂L/∂Q)W_qᵀ + (∂L/∂K)W_kᵀ + (∂L/∂V)W_vᵀ
    for (int b = 0; b < attn->batch_size; b++) {
        float* X_b = &X[b * attn->seq_len * attn->d_model];
        float* grad_Q_b = &attn->grad_Q[b * attn->seq_len * attn->d_model];
        float* grad_K_b = &attn->grad_K[b * attn->seq_len * attn->d_model];
        float* grad_V_b = &attn->grad_V[b * attn->seq_len * attn->d_model];
        
        // ∂L/∂W_q += Xᵀ(∂L/∂Q)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, attn->seq_len,
                    1.0f, X_b, attn->d_model,
                    grad_Q_b, attn->d_model,
                    1.0f, attn->W_q_grad, attn->d_model);
        
        // ∂L/∂W_k += Xᵀ(∂L/∂K)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, attn->seq_len,
                    1.0f, X_b, attn->d_model,
                    grad_K_b, attn->d_model,
                    1.0f, attn->W_k_grad, attn->d_model);
        
        // ∂L/∂W_v += Xᵀ(∂L/∂V)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->d_model, attn->d_model, attn->seq_len,
                    1.0f, X_b, attn->d_model,
                    grad_V_b, attn->d_model,
                    1.0f, attn->W_v_grad, attn->d_model);
        
        // ∂L/∂X = (∂L/∂Q)W_qᵀ + (∂L/∂K)W_kᵀ + (∂L/∂V)W_vᵀ
        if (grad_X != NULL) {
            float* grad_X_b = &grad_X[b * attn->seq_len * attn->d_model];
            
            // ∂L/∂X = (∂L/∂Q)W_qᵀ
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        attn->seq_len, attn->d_model, attn->d_model,
                        1.0f, grad_Q_b, attn->d_model,
                        attn->W_q, attn->d_model,
                        0.0f, grad_X_b, attn->d_model);
            
            // ∂L/∂X += (∂L/∂K)W_kᵀ
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        attn->seq_len, attn->d_model, attn->d_model,
                        1.0f, grad_K_b, attn->d_model,
                        attn->W_k, attn->d_model,
                        1.0f, grad_X_b, attn->d_model);
            
            // ∂L/∂X += (∂L/∂V)W_vᵀ
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        attn->seq_len, attn->d_model, attn->d_model,
                        1.0f, grad_V_b, attn->d_model,
                        attn->W_v, attn->d_model,
                        1.0f, grad_X_b, attn->d_model);
        }
    }
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate, int effective_batch_size) {
    attn->t++;
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int weight_size = attn->d_model * attn->d_model;
    
    // Update all weight matrices (W_q, W_k, W_v, W_o)
    float* weights[] = {attn->W_q, attn->W_k, attn->W_v, attn->W_o};
    float* grads[] = {attn->W_q_grad, attn->W_k_grad, attn->W_v_grad, attn->W_o_grad};
    float* m_arrays[] = {attn->W_q_m, attn->W_k_m, attn->W_v_m, attn->W_o_m};
    float* v_arrays[] = {attn->W_q_v, attn->W_k_v, attn->W_v_v, attn->W_o_v};
    
    for (int w = 0; w < 4; w++) {
        for (int i = 0; i < weight_size; i++) {
            float grad = grads[w][i] / effective_batch_size;
            
            // m = β₁m + (1-β₁)(∂L/∂W)
            m_arrays[w][i] = attn->beta1 * m_arrays[w][i] + (1.0f - attn->beta1) * grad;
            // v = β₂v + (1-β₂)(∂L/∂W)²
            v_arrays[w][i] = attn->beta2 * v_arrays[w][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * m_arrays[w][i] / (sqrtf(v_arrays[w][i]) + attn->epsilon);
            // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
            weights[w][i] = weights[w][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
    }
}

// Reset optimizer state
void reset_optimizer_attention(Attention* attn) {
    int weight_size = attn->d_model * attn->d_model;
    
    // Reset Adam moment estimates to zero
    memset(attn->W_q_m, 0, weight_size * sizeof(float));
    memset(attn->W_q_v, 0, weight_size * sizeof(float));
    memset(attn->W_k_m, 0, weight_size * sizeof(float));
    memset(attn->W_k_v, 0, weight_size * sizeof(float));
    memset(attn->W_v_m, 0, weight_size * sizeof(float));
    memset(attn->W_v_v, 0, weight_size * sizeof(float));
    memset(attn->W_o_m, 0, weight_size * sizeof(float));
    memset(attn->W_o_v, 0, weight_size * sizeof(float));
    
    // Reset time step
    attn->t = 0;
}

// Serialize attention to a file
void serialize_attention(Attention* attn, FILE* file) {
    // Write dimensions
    fwrite(&attn->seq_len, sizeof(int), 1, file);
    fwrite(&attn->d_model, sizeof(int), 1, file);
    fwrite(&attn->is_causal, sizeof(bool), 1, file);
    fwrite(&attn->use_rope, sizeof(bool), 1, file);
    
    int weight_size = attn->d_model * attn->d_model;
    
    // Write weights
    fwrite(attn->W_q, sizeof(float), weight_size, file);
    fwrite(attn->W_k, sizeof(float), weight_size, file);
    fwrite(attn->W_v, sizeof(float), weight_size, file);
    fwrite(attn->W_o, sizeof(float), weight_size, file);
    
    // Write optimizer state
    fwrite(&attn->t, sizeof(int), 1, file);
    fwrite(attn->W_q_m, sizeof(float), weight_size, file);
    fwrite(attn->W_q_v, sizeof(float), weight_size, file);
    fwrite(attn->W_k_m, sizeof(float), weight_size, file);
    fwrite(attn->W_k_v, sizeof(float), weight_size, file);
    fwrite(attn->W_v_m, sizeof(float), weight_size, file);
    fwrite(attn->W_v_v, sizeof(float), weight_size, file);
    fwrite(attn->W_o_m, sizeof(float), weight_size, file);
    fwrite(attn->W_o_v, sizeof(float), weight_size, file);
}

// Deserialize attention from a file
Attention* deserialize_attention(FILE* file, int batch_size) {
    // Read dimensions
    int seq_len, d_model;
    bool is_causal, use_rope;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    fread(&use_rope, sizeof(bool), 1, file);
    
    // Initialize attention layer
    Attention* attn = init_attention(seq_len, d_model, batch_size, is_causal, use_rope);
    
    int weight_size = d_model * d_model;
    
    // Read weights
    fread(attn->W_q, sizeof(float), weight_size, file);
    fread(attn->W_k, sizeof(float), weight_size, file);
    fread(attn->W_v, sizeof(float), weight_size, file);
    fread(attn->W_o, sizeof(float), weight_size, file);
    
    // Read optimizer state
    fread(&attn->t, sizeof(int), 1, file);
    fread(attn->W_q_m, sizeof(float), weight_size, file);
    fread(attn->W_q_v, sizeof(float), weight_size, file);
    fread(attn->W_k_m, sizeof(float), weight_size, file);
    fread(attn->W_k_v, sizeof(float), weight_size, file);
    fread(attn->W_v_m, sizeof(float), weight_size, file);
    fread(attn->W_v_v, sizeof(float), weight_size, file);
    fread(attn->W_o_m, sizeof(float), weight_size, file);
    fread(attn->W_o_v, sizeof(float), weight_size, file);
    
    return attn;
}