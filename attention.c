#include "attention.h"

// Initialize the network with configurable dimensions
Attention* init_attention(int input_dim, int head_dim, int output_dim, int seq_len, int batch_size, int num_layers) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));

    // Store dimensions
    attn->input_dim = input_dim;
    attn->head_dim = head_dim;
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
    attn->Wq = (float**)malloc(num_layers * sizeof(float*));
    attn->Wk = (float**)malloc(num_layers * sizeof(float*));
    attn->Wv = (float**)malloc(num_layers * sizeof(float*));
    attn->Wo = (float**)malloc(num_layers * sizeof(float*));
    attn->Wr = (float**)malloc(num_layers * sizeof(float*));

    attn->Wq_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->Wk_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->Wv_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->Wo_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->Wr_grad = (float**)malloc(num_layers * sizeof(float*));

    // Allocate AdamW buffers
    attn->Wq_m = (float**)malloc(num_layers * sizeof(float*));
    attn->Wq_v = (float**)malloc(num_layers * sizeof(float*));
    attn->Wk_m = (float**)malloc(num_layers * sizeof(float*));
    attn->Wk_v = (float**)malloc(num_layers * sizeof(float*));
    attn->Wv_m = (float**)malloc(num_layers * sizeof(float*));
    attn->Wv_v = (float**)malloc(num_layers * sizeof(float*));
    attn->Wo_m = (float**)malloc(num_layers * sizeof(float*));
    attn->Wo_v = (float**)malloc(num_layers * sizeof(float*));
    attn->Wr_m = (float**)malloc(num_layers * sizeof(float*));
    attn->Wr_v = (float**)malloc(num_layers * sizeof(float*));

    // Allocate layer outputs and working buffers
    attn->layer_Q = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_K = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_V = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_scores = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_attn = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_O = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_output = (float**)malloc(num_layers * sizeof(float*));

    attn->error_output = (float**)malloc(num_layers * sizeof(float*));
    attn->error_O = (float**)malloc(num_layers * sizeof(float*));
    attn->error_Q = (float**)malloc(num_layers * sizeof(float*));
    attn->error_K = (float**)malloc(num_layers * sizeof(float*));
    attn->error_V = (float**)malloc(num_layers * sizeof(float*));
    attn->error_scores = (float**)malloc(num_layers * sizeof(float*));

    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : output_dim;

        int Wq_size = input_size * head_dim;
        int Wk_size = input_size * head_dim;
        int Wv_size = input_size * head_dim;
        int Wo_size = head_dim * output_size;
        int Wr_size = input_size * output_size;

        // Allocate and initialize matrices and gradients
        attn->Wq[layer] = (float*)malloc(Wq_size * sizeof(float));
        attn->Wk[layer] = (float*)malloc(Wk_size * sizeof(float));
        attn->Wv[layer] = (float*)malloc(Wv_size * sizeof(float));
        attn->Wo[layer] = (float*)malloc(Wo_size * sizeof(float));
        attn->Wr[layer] = (float*)malloc(Wr_size * sizeof(float));
        attn->Wq_grad[layer] = (float*)malloc(Wq_size * sizeof(float));
        attn->Wk_grad[layer] = (float*)malloc(Wk_size * sizeof(float));
        attn->Wv_grad[layer] = (float*)malloc(Wv_size * sizeof(float));
        attn->Wo_grad[layer] = (float*)malloc(Wo_size * sizeof(float));
        attn->Wr_grad[layer] = (float*)malloc(Wr_size * sizeof(float));

        // Allocate AdamW buffers
        attn->Wq_m[layer] = (float*)calloc(Wq_size, sizeof(float));
        attn->Wq_v[layer] = (float*)calloc(Wq_size, sizeof(float));
        attn->Wk_m[layer] = (float*)calloc(Wk_size, sizeof(float));
        attn->Wk_v[layer] = (float*)calloc(Wk_size, sizeof(float));
        attn->Wv_m[layer] = (float*)calloc(Wv_size, sizeof(float));
        attn->Wv_v[layer] = (float*)calloc(Wv_size, sizeof(float));
        attn->Wo_m[layer] = (float*)calloc(Wo_size, sizeof(float));
        attn->Wo_v[layer] = (float*)calloc(Wo_size, sizeof(float));
        attn->Wr_m[layer] = (float*)calloc(Wr_size, sizeof(float));
        attn->Wr_v[layer] = (float*)calloc(Wr_size, sizeof(float));

        // Allocate layer outputs and working buffers
        attn->layer_Q[layer] = (float*)malloc(batch_size * seq_len * head_dim * sizeof(float));
        attn->layer_K[layer] = (float*)malloc(batch_size * seq_len * head_dim * sizeof(float));
        attn->layer_V[layer] = (float*)malloc(batch_size * seq_len * head_dim * sizeof(float));
        attn->layer_scores[layer] = (float*)malloc(batch_size * seq_len * seq_len * sizeof(float));
        attn->layer_attn[layer] = (float*)malloc(batch_size * seq_len * seq_len * sizeof(float));
        attn->layer_O[layer] = (float*)malloc(batch_size * seq_len * head_dim * sizeof(float));
        attn->layer_output[layer] = (float*)malloc(batch_size * seq_len * output_size * sizeof(float));

        attn->error_output[layer] = (float*)malloc(batch_size * seq_len * output_size * sizeof(float));
        attn->error_O[layer] = (float*)malloc(batch_size * seq_len * head_dim * sizeof(float));
        attn->error_Q[layer] = (float*)malloc(batch_size * seq_len * head_dim * sizeof(float));
        attn->error_K[layer] = (float*)malloc(batch_size * seq_len * head_dim * sizeof(float));
        attn->error_V[layer] = (float*)malloc(batch_size * seq_len * head_dim * sizeof(float));
        attn->error_scores[layer] = (float*)malloc(batch_size * seq_len * seq_len * sizeof(float));

        // Initialize weights
        float scale_in = 1.0f / sqrtf((float)input_size);
        float scale_hd = 1.0f / sqrtf((float)head_dim);

        for (int i = 0; i < Wq_size; i++) {
            attn->Wq[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
        }

        for (int i = 0; i < Wk_size; i++) {
            attn->Wk[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
        }

        for (int i = 0; i < Wv_size; i++) {
            attn->Wv[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
        }

        for (int i = 0; i < Wo_size; i++) {
            attn->Wo[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_hd;
        }

        for (int i = 0; i < Wr_size; i++) {
            attn->Wr[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
        }
    }

    return attn;
}

// Free network memory
void free_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        free(attn->Wq[layer]); free(attn->Wk[layer]); free(attn->Wv[layer]); free(attn->Wo[layer]); free(attn->Wr[layer]);
        free(attn->Wq_grad[layer]); free(attn->Wk_grad[layer]); free(attn->Wv_grad[layer]); free(attn->Wo_grad[layer]); free(attn->Wr_grad[layer]);
        free(attn->Wq_m[layer]); free(attn->Wq_v[layer]);
        free(attn->Wk_m[layer]); free(attn->Wk_v[layer]);
        free(attn->Wv_m[layer]); free(attn->Wv_v[layer]);
        free(attn->Wo_m[layer]); free(attn->Wo_v[layer]);
        free(attn->Wr_m[layer]); free(attn->Wr_v[layer]);
        free(attn->layer_Q[layer]); free(attn->layer_K[layer]); free(attn->layer_V[layer]);
        free(attn->layer_scores[layer]); free(attn->layer_attn[layer]); free(attn->layer_O[layer]);
        free(attn->layer_output[layer]);
        free(attn->error_output[layer]); free(attn->error_O[layer]);
        free(attn->error_Q[layer]); free(attn->error_K[layer]); free(attn->error_V[layer]);
        free(attn->error_scores[layer]);
    }

    free(attn->Wq); free(attn->Wk); free(attn->Wv); free(attn->Wo); free(attn->Wr);
    free(attn->Wq_grad); free(attn->Wk_grad); free(attn->Wv_grad); free(attn->Wo_grad); free(attn->Wr_grad);
    free(attn->Wq_m); free(attn->Wq_v);
    free(attn->Wk_m); free(attn->Wk_v);
    free(attn->Wv_m); free(attn->Wv_v);
    free(attn->Wo_m); free(attn->Wo_v);
    free(attn->Wr_m); free(attn->Wr_v);
    free(attn->layer_Q); free(attn->layer_K); free(attn->layer_V);
    free(attn->layer_scores); free(attn->layer_attn); free(attn->layer_O); free(attn->layer_output);
    free(attn->error_output); free(attn->error_O); free(attn->error_Q); free(attn->error_K); free(attn->error_V); free(attn->error_scores);
    free(attn);
}

// Softmax function for attention scores
static void softmax_rows(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float* row = &matrix[i * cols];
        
        // Find max for numerical stability
        float max_val = row[0];
        for (int j = 1; j < cols; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        
        // Normalize
        float inv_sum = 1.0f / (sum + 1e-12f);
        for (int j = 0; j < cols; j++) {
            row[j] *= inv_sum;
        }
    }
}

// Forward pass
void forward_pass_attention(Attention* attn, float* X) {
    float* input = X;
    
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        // Reshape input: [batch_size*seq_len x input_size]
        int total_seq = attn->batch_size * attn->seq_len;
        
        // Q = X Wq
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_seq, attn->head_dim, input_size,
                    1.0f, input, input_size,
                    attn->Wq[layer], attn->head_dim,
                    0.0f, attn->layer_Q[layer], attn->head_dim);
        
        // K = X Wk
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_seq, attn->head_dim, input_size,
                    1.0f, input, input_size,
                    attn->Wk[layer], attn->head_dim,
                    0.0f, attn->layer_K[layer], attn->head_dim);
        
        // V = X Wv
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_seq, attn->head_dim, input_size,
                    1.0f, input, input_size,
                    attn->Wv[layer], attn->head_dim,
                    0.0f, attn->layer_V[layer], attn->head_dim);
        
        // For each batch, compute attention
        for (int b = 0; b < attn->batch_size; b++) {
            float* Q_b = &attn->layer_Q[layer][b * attn->seq_len * attn->head_dim];
            float* K_b = &attn->layer_K[layer][b * attn->seq_len * attn->head_dim];
            float* V_b = &attn->layer_V[layer][b * attn->seq_len * attn->head_dim];
            float* scores_b = &attn->layer_scores[layer][b * attn->seq_len * attn->seq_len];
            float* attn_b = &attn->layer_attn[layer][b * attn->seq_len * attn->seq_len];
            float* O_b = &attn->layer_O[layer][b * attn->seq_len * attn->head_dim];
            
            // Scores = Q K^T / sqrt(head_dim)
            float scale = 1.0f / sqrtf((float)attn->head_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        attn->seq_len, attn->seq_len, attn->head_dim,
                        scale, Q_b, attn->head_dim,
                        K_b, attn->head_dim,
                        0.0f, scores_b, attn->seq_len);
            
            // Copy scores to attention buffer and apply softmax
            memcpy(attn_b, scores_b, attn->seq_len * attn->seq_len * sizeof(float));
            softmax_rows(attn_b, attn->seq_len, attn->seq_len);
            
            // O = Attention V
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len, attn->head_dim, attn->seq_len,
                        1.0f, attn_b, attn->seq_len,
                        V_b, attn->head_dim,
                        0.0f, O_b, attn->head_dim);
        }
        
        // Y = O Wo
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_seq, output_size, attn->head_dim,
                    1.0f, attn->layer_O[layer], attn->head_dim,
                    attn->Wo[layer], output_size,
                    0.0f, attn->layer_output[layer], output_size);
        
        // Y = Y + X Wr (residual connection)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    total_seq, output_size, input_size,
                    1.0f, input, input_size,
                    attn->Wr[layer], output_size,
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
    int total = attn->batch_size * attn->seq_len * attn->output_dim;
    
    for (int i = 0; i < total; i++) {
        attn->error_output[last_layer][i] = attn->layer_output[last_layer][i] - y[i];
        loss += attn->error_output[last_layer][i] * attn->error_output[last_layer][i];
    }
    return loss / total;
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        int Wq_size = input_size * attn->head_dim;
        int Wk_size = input_size * attn->head_dim;
        int Wv_size = input_size * attn->head_dim;
        int Wo_size = attn->head_dim * output_size;
        int Wr_size = input_size * output_size;
        
        memset(attn->Wq_grad[layer], 0, Wq_size * sizeof(float));
        memset(attn->Wk_grad[layer], 0, Wk_size * sizeof(float));
        memset(attn->Wv_grad[layer], 0, Wv_size * sizeof(float));
        memset(attn->Wo_grad[layer], 0, Wo_size * sizeof(float));
        memset(attn->Wr_grad[layer], 0, Wr_size * sizeof(float));
    }
}

// Backward pass
void backward_pass_attention(Attention* attn, float* X) {
    for (int layer = attn->num_layers - 1; layer >= 0; layer--) {
        float* input = (layer == 0) ? X : attn->layer_output[layer - 1];
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        int total_seq = attn->batch_size * attn->seq_len;
        
        // ∂L/∂Wo += O^T (∂L/∂Y)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    attn->head_dim, output_size, total_seq,
                    1.0f, attn->layer_O[layer], attn->head_dim,
                    attn->error_output[layer], output_size,
                    1.0f, attn->Wo_grad[layer], output_size);
        
        // ∂L/∂O = (∂L/∂Y) Wo^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    total_seq, attn->head_dim, output_size,
                    1.0f, attn->error_output[layer], output_size,
                    attn->Wo[layer], output_size,
                    0.0f, attn->error_O[layer], attn->head_dim);
        
        // ∂L/∂Wr += X^T (∂L/∂Y)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    input_size, output_size, total_seq,
                    1.0f, input, input_size,
                    attn->error_output[layer], output_size,
                    1.0f, attn->Wr_grad[layer], output_size);
        
        // Backprop through attention for each batch
        for (int b = 0; b < attn->batch_size; b++) {
            float* Q_b = &attn->layer_Q[layer][b * attn->seq_len * attn->head_dim];
            float* K_b = &attn->layer_K[layer][b * attn->seq_len * attn->head_dim];
            float* V_b = &attn->layer_V[layer][b * attn->seq_len * attn->head_dim];
            float* attn_b = &attn->layer_attn[layer][b * attn->seq_len * attn->seq_len];
            float* dO_b = &attn->error_O[layer][b * attn->seq_len * attn->head_dim];
            float* dQ_b = &attn->error_Q[layer][b * attn->seq_len * attn->head_dim];
            float* dK_b = &attn->error_K[layer][b * attn->seq_len * attn->head_dim];
            float* dV_b = &attn->error_V[layer][b * attn->seq_len * attn->head_dim];
            float* dscores_b = &attn->error_scores[layer][b * attn->seq_len * attn->seq_len];
            
            // ∂L/∂V = Attention^T (∂L/∂O)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        attn->seq_len, attn->head_dim, attn->seq_len,
                        1.0f, attn_b, attn->seq_len,
                        dO_b, attn->head_dim,
                        0.0f, dV_b, attn->head_dim);
            
            // ∂L/∂Attention = (∂L/∂O) V^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        attn->seq_len, attn->seq_len, attn->head_dim,
                        1.0f, dO_b, attn->head_dim,
                        V_b, attn->head_dim,
                        0.0f, dscores_b, attn->seq_len);
            
            // Apply softmax gradient
            for (int i = 0; i < attn->seq_len; i++) {
                float* grad_row = &dscores_b[i * attn->seq_len];
                float* attn_row = &attn_b[i * attn->seq_len];
                
                // Compute sum of grad * attn for this row
                float sum = 0.0f;
                for (int j = 0; j < attn->seq_len; j++) {
                    sum += grad_row[j] * attn_row[j];
                }
                
                // Apply softmax gradient formula
                for (int j = 0; j < attn->seq_len; j++) {
                    grad_row[j] = attn_row[j] * (grad_row[j] - sum);
                }
            }
            
            // ∂L/∂Q = (∂L/∂scores) K / sqrt(head_dim)
            float scale = 1.0f / sqrtf((float)attn->head_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len, attn->head_dim, attn->seq_len,
                        scale, dscores_b, attn->seq_len,
                        K_b, attn->head_dim,
                        0.0f, dQ_b, attn->head_dim);
            
            // ∂L/∂K = (∂L/∂scores)^T Q / sqrt(head_dim)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        attn->seq_len, attn->head_dim, attn->seq_len,
                        scale, dscores_b, attn->seq_len,
                        Q_b, attn->head_dim,
                        0.0f, dK_b, attn->head_dim);
        }
        
        // ∂L/∂Wq += X^T (∂L/∂Q)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    input_size, attn->head_dim, total_seq,
                    1.0f, input, input_size,
                    attn->error_Q[layer], attn->head_dim,
                    1.0f, attn->Wq_grad[layer], attn->head_dim);
        
        // ∂L/∂Wk += X^T (∂L/∂K)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    input_size, attn->head_dim, total_seq,
                    1.0f, input, input_size,
                    attn->error_K[layer], attn->head_dim,
                    1.0f, attn->Wk_grad[layer], attn->head_dim);
        
        // ∂L/∂Wv += X^T (∂L/∂V)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    input_size, attn->head_dim, total_seq,
                    1.0f, input, input_size,
                    attn->error_V[layer], attn->head_dim,
                    1.0f, attn->Wv_grad[layer], attn->head_dim);
        
        // Propagate error to previous layer
        if (layer > 0) {
            // ∂L/∂X = (∂L/∂Q) Wq^T + (∂L/∂K) Wk^T + (∂L/∂V) Wv^T + (∂L/∂Y) Wr^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        total_seq, input_size, attn->head_dim,
                        1.0f, attn->error_Q[layer], attn->head_dim,
                        attn->Wq[layer], attn->head_dim,
                        0.0f, attn->error_output[layer - 1], input_size);
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        total_seq, input_size, attn->head_dim,
                        1.0f, attn->error_K[layer], attn->head_dim,
                        attn->Wk[layer], attn->head_dim,
                        1.0f, attn->error_output[layer - 1], input_size);
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        total_seq, input_size, attn->head_dim,
                        1.0f, attn->error_V[layer], attn->head_dim,
                        attn->Wv[layer], attn->head_dim,
                        1.0f, attn->error_output[layer - 1], input_size);
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        total_seq, input_size, output_size,
                        1.0f, attn->error_output[layer], output_size,
                        attn->Wr[layer], output_size,
                        1.0f, attn->error_output[layer - 1], input_size);
        }
    }
}

// Update weights using AdamW
void update_weights_attention(Attention* attn, float learning_rate) {
    attn->t++;
    int total_samples = attn->batch_size * attn->seq_len;
    
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        int Wq_size = input_size * attn->head_dim;
        int Wk_size = input_size * attn->head_dim;
        int Wv_size = input_size * attn->head_dim;
        int Wo_size = attn->head_dim * output_size;
        int Wr_size = input_size * output_size;
        
        // Update Wq weights
        for (int i = 0; i < Wq_size; i++) {
            float grad = attn->Wq_grad[layer][i] / total_samples;
            
            attn->Wq_m[layer][i] = attn->beta1 * attn->Wq_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->Wq_v[layer][i] = attn->beta2 * attn->Wq_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->Wq_m[layer][i] / (sqrtf(attn->Wq_v[layer][i]) + attn->epsilon);
            attn->Wq[layer][i] = attn->Wq[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update Wk weights
        for (int i = 0; i < Wk_size; i++) {
            float grad = attn->Wk_grad[layer][i] / total_samples;
            
            attn->Wk_m[layer][i] = attn->beta1 * attn->Wk_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->Wk_v[layer][i] = attn->beta2 * attn->Wk_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->Wk_m[layer][i] / (sqrtf(attn->Wk_v[layer][i]) + attn->epsilon);
            attn->Wk[layer][i] = attn->Wk[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update Wv weights
        for (int i = 0; i < Wv_size; i++) {
            float grad = attn->Wv_grad[layer][i] / total_samples;
            
            attn->Wv_m[layer][i] = attn->beta1 * attn->Wv_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->Wv_v[layer][i] = attn->beta2 * attn->Wv_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->Wv_m[layer][i] / (sqrtf(attn->Wv_v[layer][i]) + attn->epsilon);
            attn->Wv[layer][i] = attn->Wv[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update Wo weights
        for (int i = 0; i < Wo_size; i++) {
            float grad = attn->Wo_grad[layer][i] / total_samples;
            
            attn->Wo_m[layer][i] = attn->beta1 * attn->Wo_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->Wo_v[layer][i] = attn->beta2 * attn->Wo_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->Wo_m[layer][i] / (sqrtf(attn->Wo_v[layer][i]) + attn->epsilon);
            attn->Wo[layer][i] = attn->Wo[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
        }
        
        // Update Wr weights
        for (int i = 0; i < Wr_size; i++) {
            float grad = attn->Wr_grad[layer][i] / total_samples;
            
            attn->Wr_m[layer][i] = attn->beta1 * attn->Wr_m[layer][i] + (1.0f - attn->beta1) * grad;
            attn->Wr_v[layer][i] = attn->beta2 * attn->Wr_v[layer][i] + (1.0f - attn->beta2) * grad * grad;
            
            float update = alpha_t * attn->Wr_m[layer][i] / (sqrtf(attn->Wr_v[layer][i]) + attn->epsilon);
            attn->Wr[layer][i] = attn->Wr[layer][i] * (1.0f - learning_rate * attn->weight_decay) - update;
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
    fwrite(&attn->input_dim, sizeof(int), 1, file);
    fwrite(&attn->head_dim, sizeof(int), 1, file);
    fwrite(&attn->output_dim, sizeof(int), 1, file);
    fwrite(&attn->seq_len, sizeof(int), 1, file);
    fwrite(&attn->batch_size, sizeof(int), 1, file);
    fwrite(&attn->num_layers, sizeof(int), 1, file);
    
    // Save weights for each layer
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        int Wq_size = input_size * attn->head_dim;
        int Wk_size = input_size * attn->head_dim;
        int Wv_size = input_size * attn->head_dim;
        int Wo_size = attn->head_dim * output_size;
        int Wr_size = input_size * output_size;
        
        fwrite(attn->Wq[layer], sizeof(float), Wq_size, file);
        fwrite(attn->Wk[layer], sizeof(float), Wk_size, file);
        fwrite(attn->Wv[layer], sizeof(float), Wv_size, file);
        fwrite(attn->Wo[layer], sizeof(float), Wo_size, file);
        fwrite(attn->Wr[layer], sizeof(float), Wr_size, file);
    }
    
    // Save Adam state
    fwrite(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = (layer == attn->num_layers - 1) ? attn->output_dim : attn->output_dim;
        
        int Wq_size = input_size * attn->head_dim;
        int Wk_size = input_size * attn->head_dim;
        int Wv_size = input_size * attn->head_dim;
        int Wo_size = attn->head_dim * output_size;
        int Wr_size = input_size * output_size;
        
        fwrite(attn->Wq_m[layer], sizeof(float), Wq_size, file);
        fwrite(attn->Wq_v[layer], sizeof(float), Wq_size, file);
        fwrite(attn->Wk_m[layer], sizeof(float), Wk_size, file);
        fwrite(attn->Wk_v[layer], sizeof(float), Wk_size, file);
        fwrite(attn->Wv_m[layer], sizeof(float), Wv_size, file);
        fwrite(attn->Wv_v[layer], sizeof(float), Wv_size, file);
        fwrite(attn->Wo_m[layer], sizeof(float), Wo_size, file);
        fwrite(attn->Wo_v[layer], sizeof(float), Wo_size, file);
        fwrite(attn->Wr_m[layer], sizeof(float), Wr_size, file);
        fwrite(attn->Wr_v[layer], sizeof(float), Wr_size, file);
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
    int input_dim, head_dim, output_dim, seq_len, stored_batch_size, num_layers;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&head_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    Attention* attn = init_attention(input_dim, head_dim, output_dim, seq_len, batch_size, num_layers);
    
    // Load weights for each layer
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : output_dim;
        
        int Wq_size = input_size * head_dim;
        int Wk_size = input_size * head_dim;
        int Wv_size = input_size * head_dim;
        int Wo_size = head_dim * output_size;
        int Wr_size = input_size * output_size;
        
        fread(attn->Wq[layer], sizeof(float), Wq_size, file);
        fread(attn->Wk[layer], sizeof(float), Wk_size, file);
        fread(attn->Wv[layer], sizeof(float), Wv_size, file);
        fread(attn->Wo[layer], sizeof(float), Wo_size, file);
        fread(attn->Wr[layer], sizeof(float), Wr_size, file);
    }
    
    // Load Adam state
    fread(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : output_dim;
        
        int Wq_size = input_size * head_dim;
        int Wk_size = input_size * head_dim;
        int Wv_size = input_size * head_dim;
        int Wo_size = head_dim * output_size;
        int Wr_size = input_size * output_size;
        
        fread(attn->Wq_m[layer], sizeof(float), Wq_size, file);
        fread(attn->Wq_v[layer], sizeof(float), Wq_size, file);
        fread(attn->Wk_m[layer], sizeof(float), Wk_size, file);
        fread(attn->Wk_v[layer], sizeof(float), Wk_size, file);
        fread(attn->Wv_m[layer], sizeof(float), Wv_size, file);
        fread(attn->Wv_v[layer], sizeof(float), Wv_size, file);
        fread(attn->Wo_m[layer], sizeof(float), Wo_size, file);
        fread(attn->Wo_v[layer], sizeof(float), Wo_size, file);
        fread(attn->Wr_m[layer], sizeof(float), Wr_size, file);
        fread(attn->Wr_v[layer], sizeof(float), Wr_size, file);
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return attn;
}