#include "attention.h"

// Softmax over rows of a square matrix (in-place on row pointer)
static void softmax_row(float* row, int n) {
    float maxv = row[0];
    for (int i = 1; i < n; i++) if (row[i] > maxv) maxv = row[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        row[i] = expf(row[i] - maxv);
        sum += row[i];
    }
    float inv = 1.0f / (sum + 1e-12f);
    for (int i = 0; i < n; i++) {
        row[i] *= inv;
    }
}

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

    // AdamW parameters
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;

    // Allocate pointer arrays
    int L = num_layers;
    attn->Wq = (float**)malloc(L * sizeof(float*));
    attn->Wk = (float**)malloc(L * sizeof(float*));
    attn->Wv = (float**)malloc(L * sizeof(float*));
    attn->Wo = (float**)malloc(L * sizeof(float*));
    attn->Wr = (float**)malloc(L * sizeof(float*));

    attn->Wq_grad = (float**)malloc(L * sizeof(float*));
    attn->Wk_grad = (float**)malloc(L * sizeof(float*));
    attn->Wv_grad = (float**)malloc(L * sizeof(float*));
    attn->Wo_grad = (float**)malloc(L * sizeof(float*));
    attn->Wr_grad = (float**)malloc(L * sizeof(float*));

    attn->Wq_m = (float**)malloc(L * sizeof(float*));
    attn->Wq_v = (float**)malloc(L * sizeof(float*));
    attn->Wk_m = (float**)malloc(L * sizeof(float*));
    attn->Wk_v = (float**)malloc(L * sizeof(float*));
    attn->Wv_m = (float**)malloc(L * sizeof(float*));
    attn->Wv_v = (float**)malloc(L * sizeof(float*));
    attn->Wo_m = (float**)malloc(L * sizeof(float*));
    attn->Wo_v = (float**)malloc(L * sizeof(float*));
    attn->Wr_m = (float**)malloc(L * sizeof(float*));
    attn->Wr_v = (float**)malloc(L * sizeof(float*));

    attn->layer_Q = (float**)malloc(L * sizeof(float*));
    attn->layer_K = (float**)malloc(L * sizeof(float*));
    attn->layer_V = (float**)malloc(L * sizeof(float*));
    attn->layer_scores = (float**)malloc(L * sizeof(float*));
    attn->layer_attn = (float**)malloc(L * sizeof(float*));
    attn->layer_O = (float**)malloc(L * sizeof(float*));
    attn->layer_output = (float**)malloc(L * sizeof(float*));

    attn->error_output = (float**)malloc(L * sizeof(float*));
    attn->error_O = (float**)malloc(L * sizeof(float*));
    attn->error_Q = (float**)malloc(L * sizeof(float*));
    attn->error_K = (float**)malloc(L * sizeof(float*));
    attn->error_V = (float**)malloc(L * sizeof(float*));
    attn->error_scores = (float**)malloc(L * sizeof(float*));

    for (int layer = 0; layer < L; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = output_dim; // consistent per layer

        int Wq_size = input_size * head_dim;
        int Wk_size = input_size * head_dim;
        int Wv_size = input_size * head_dim;
        int Wo_size = head_dim * output_size;
        int Wr_size = input_size * output_size;

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

        // Buffers
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

        for (int i = 0; i < Wq_size; i++) attn->Wq[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
        for (int i = 0; i < Wk_size; i++) attn->Wk[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
        for (int i = 0; i < Wv_size; i++) attn->Wv[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
        for (int i = 0; i < Wo_size; i++) attn->Wo[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_hd;
        for (int i = 0; i < Wr_size; i++) attn->Wr[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
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
    free(attn->Wq_m); free(attn->Wq_v); free(attn->Wk_m); free(attn->Wk_v); free(attn->Wv_m); free(attn->Wv_v);
    free(attn->Wo_m); free(attn->Wo_v); free(attn->Wr_m); free(attn->Wr_v);
    free(attn->layer_Q); free(attn->layer_K); free(attn->layer_V);
    free(attn->layer_scores); free(attn->layer_attn); free(attn->layer_O); free(attn->layer_output);
    free(attn->error_output); free(attn->error_O); free(attn->error_Q); free(attn->error_K); free(attn->error_V); free(attn->error_scores);
    free(attn);
}

// Forward pass (single-head, bidirectional attention within each sequence, no mask)
void forward_pass_attention(Attention* attn, float* X) {
    int B = attn->batch_size;
    int T = attn->seq_len;
    int Din = attn->input_dim;
    int Dhd = attn->head_dim;
    int Dout = attn->output_dim;

    // Per layer
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? Din : Dout;
        int output_size = Dout;

        for (int b = 0; b < B; b++) {
            // Pointers for this batch
            float* Xb = (layer == 0)
                ? (X + b * T * input_size)
                : (&attn->layer_output[layer - 1][b * T * output_size]);

            float* Qb = &attn->layer_Q[layer][b * T * Dhd];
            float* Kb = &attn->layer_K[layer][b * T * Dhd];
            float* Vb = &attn->layer_V[layer][b * T * Dhd];
            float* Sb = &attn->layer_scores[layer][b * T * T];
            float* Ab = &attn->layer_attn[layer][b * T * T];
            float* Ob = &attn->layer_O[layer][b * T * Dhd];
            float* Yb = &attn->layer_output[layer][b * T * output_size];

            // Q = X Wq
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, Dhd, input_size,
                        1.0f, Xb, input_size,
                        attn->Wq[layer], Dhd,
                        0.0f, Qb, Dhd);

            // K = X Wk
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, Dhd, input_size,
                        1.0f, Xb, input_size,
                        attn->Wk[layer], Dhd,
                        0.0f, Kb, Dhd);

            // V = X Wv
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, Dhd, input_size,
                        1.0f, Xb, input_size,
                        attn->Wv[layer], Dhd,
                        0.0f, Vb, Dhd);

            // Scores = (Q K^T) / sqrt(Dhd)
            float scale = 1.0f / sqrtf((float)Dhd);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, T, Dhd,
                        scale, Qb, Dhd,
                        Kb, Dhd,
                        0.0f, Sb, T);

            // Attn = softmax(scores) row-wise
            for (int i = 0; i < T; i++) {
                softmax_row(&Sb[i * T], T);
            }
            // Store to Ab
            memcpy(Ab, Sb, T * T * sizeof(float));

            // O = Attn V
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, Dhd, T,
                        1.0f, Ab, T,
                        Vb, Dhd,
                        0.0f, Ob, Dhd);

            // Y = O Wo + X Wr
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, output_size, Dhd,
                        1.0f, Ob, Dhd,
                        attn->Wo[layer], output_size,
                        0.0f, Yb, output_size);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, output_size, input_size,
                        1.0f, Xb, input_size,
                        attn->Wr[layer], output_size,
                        1.0f, Yb, output_size);
        }
    }
}

// Calculate loss (MSE) and fill error_output at last layer
float calculate_loss_attention(Attention* attn, float* y) {
    int last = attn->num_layers - 1;
    int total = attn->batch_size * attn->seq_len * attn->output_dim;
    float loss = 0.0f;

    for (int i = 0; i < total; i++) {
        float diff = attn->layer_output[last][i] - y[i];
        attn->error_output[last][i] = diff;
        loss += diff * diff;
    }
    return loss / (float)total;
}

// Zero gradients
void zero_gradients_attention(Attention* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = attn->output_dim;
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
    int B = attn->batch_size;
    int T = attn->seq_len;
    int Din = attn->input_dim;
    int Dhd = attn->head_dim;
    int Dout = attn->output_dim;

    for (int layer = attn->num_layers - 1; layer >= 0; layer--) {
        int input_size = (layer == 0) ? Din : Dout;
        int output_size = Dout;

        for (int b = 0; b < B; b++) {
            // Pointers
            float* Xb = (layer == 0)
                ? (X + b * T * input_size)
                : (&attn->layer_output[layer - 1][b * T * input_size]);

            float* Qb = &attn->layer_Q[layer][b * T * Dhd];
            float* Kb = &attn->layer_K[layer][b * T * Dhd];
            float* Vb = &attn->layer_V[layer][b * T * Dhd];
            float* Ab = &attn->layer_attn[layer][b * T * T];
            float* Ob = &attn->layer_O[layer][b * T * Dhd];
            float* dY = &attn->error_output[layer][b * T * output_size];

            float* dOb = &attn->error_O[layer][b * T * Dhd];
            float* dQb = &attn->error_Q[layer][b * T * Dhd];
            float* dKb = &attn->error_K[layer][b * T * Dhd];
            float* dVb = &attn->error_V[layer][b * T * Dhd];
            float* dSb = &attn->error_scores[layer][b * T * T];

            float* Yprev_grad = (layer > 0) ? (&attn->error_output[layer - 1][b * T * input_size]) : NULL;

            // Gradients for Wo: dWo += O^T dY
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        Dhd, output_size, T,
                        1.0f, Ob, Dhd,
                        dY, output_size,
                        1.0f, attn->Wo_grad[layer], output_size);

            // dO = dY Wo^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, Dhd, output_size,
                        1.0f, dY, output_size,
                        attn->Wo[layer], output_size,
                        0.0f, dOb, Dhd);

            // Gradients for Wr: dWr += X^T dY
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        input_size, output_size, T,
                        1.0f, Xb, input_size,
                        dY, output_size,
                        1.0f, attn->Wr_grad[layer], output_size);

            // dX from residual path: dX_res = dY Wr^T
            if (layer > 0) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            T, input_size, output_size,
                            1.0f, dY, output_size,
                            attn->Wr[layer], output_size,
                            0.0f, Yprev_grad, input_size);
            }

            // dV = A^T dO
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        T, Dhd, T,
                        1.0f, Ab, T,
                        dOb, Dhd,
                        0.0f, dVb, Dhd);

            // dA = dO V^T  (store in dSb temporarily)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, T, Dhd,
                        1.0f, dOb, Dhd,
                        Vb, Dhd,
                        0.0f, dSb, T);

            // Convert dA to dScores via softmax Jacobian: dZ_j = s_j * (g_j - sum_k g_k s_k)
            for (int i = 0; i < T; i++) {
                float* g = &dSb[i * T];
                float* s = &Ab[i * T];
                float dot = 0.0f;
                for (int j = 0; j < T; j++) dot += g[j] * s[j];
                for (int j = 0; j < T; j++) {
                    g[j] = s[j] * (g[j] - dot); // now g is dScores row
                }
            }

            // dQ = dScores K / sqrt(d)
            float scale = 1.0f / sqrtf((float)Dhd);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, Dhd, T,
                        scale, dSb, T,
                        Kb, Dhd,
                        0.0f, dQb, Dhd);

            // dK = dScores^T Q / sqrt(d)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        T, Dhd, T,
                        scale, dSb, T,
                        Qb, Dhd,
                        0.0f, dKb, Dhd);

            // Accumulate weight gradients for Wq, Wk, Wv
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        input_size, Dhd, T,
                        1.0f, Xb, input_size,
                        dQb, Dhd,
                        1.0f, attn->Wq_grad[layer], Dhd);

            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        input_size, Dhd, T,
                        1.0f, Xb, input_size,
                        dKb, Dhd,
                        1.0f, attn->Wk_grad[layer], Dhd);

            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        input_size, Dhd, T,
                        1.0f, Xb, input_size,
                        dVb, Dhd,
                        1.0f, attn->Wv_grad[layer], Dhd);

            // Backprop to previous layer input: accumulate into Yprev_grad
            if (layer > 0) {
                // Yprev_grad already has dY * Wr^T; now add from Q/K/V paths
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            T, input_size, Dhd,
                            1.0f, dQb, Dhd,
                            attn->Wq[layer], Dhd,
                            1.0f, Yprev_grad, input_size);

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            T, input_size, Dhd,
                            1.0f, dKb, Dhd,
                            attn->Wk[layer], Dhd,
                            1.0f, Yprev_grad, input_size);

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            T, input_size, Dhd,
                            1.0f, dVb, Dhd,
                            attn->Wv[layer], Dhd,
                            1.0f, Yprev_grad, input_size);
            }
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
        int output_size = attn->output_dim;
        int Wq_size = input_size * attn->head_dim;
        int Wk_size = input_size * attn->head_dim;
        int Wv_size = input_size * attn->head_dim;
        int Wo_size = attn->head_dim * output_size;
        int Wr_size = input_size * output_size;

        float* Ws[5] = { attn->Wq[layer], attn->Wk[layer], attn->Wv[layer], attn->Wo[layer], attn->Wr[layer] };
        float* Gs[5] = { attn->Wq_grad[layer], attn->Wk_grad[layer], attn->Wv_grad[layer], attn->Wo_grad[layer], attn->Wr_grad[layer] };
        float* Ms[5] = { attn->Wq_m[layer], attn->Wk_m[layer], attn->Wv_m[layer], attn->Wo_m[layer], attn->Wr_m[layer] };
        float* Vs[5] = { attn->Wq_v[layer], attn->Wk_v[layer], attn->Wv_v[layer], attn->Wo_v[layer], attn->Wr_v[layer] };
        int sizes[5] = { Wq_size, Wk_size, Wv_size, Wo_size, Wr_size };

        for (int s = 0; s < 5; s++) {
            float* W = Ws[s];
            float* G = Gs[s];
            float* M = Ms[s];
            float* V = Vs[s];
            int n = sizes[s];

            for (int i = 0; i < n; i++) {
                float g = G[i] / (float)total_samples;
                M[i] = attn->beta1 * M[i] + (1.0f - attn->beta1) * g;
                V[i] = attn->beta2 * V[i] + (1.0f - attn->beta2) * g * g;
                float update = alpha_t * M[i] / (sqrtf(V[i]) + attn->epsilon);
                W[i] = W[i] * (1.0f - learning_rate * attn->weight_decay) - update;
            }
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

    // Save weights
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = attn->output_dim;

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

    // Save AdamW state
    fwrite(&attn->t, sizeof(int), 1, file);
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int input_size = (layer == 0) ? attn->input_dim : attn->output_dim;
        int output_size = attn->output_dim;

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

    int input_dim, head_dim, output_dim, seq_len, stored_batch_size, num_layers;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&head_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);

    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;

    Attention* attn = init_attention(input_dim, head_dim, output_dim, seq_len, batch_size, num_layers);

    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = output_dim;

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

    fread(&attn->t, sizeof(int), 1, file);

    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : output_dim;
        int output_size = output_dim;

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