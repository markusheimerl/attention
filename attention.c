
#include "attention.h"

// Helpers

static void softmax_rows(const float* logits, float* probs, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float* row_in = logits + i * cols;
        float* row_out = probs + i * cols;

        float maxv = row_in[0];
        for (int j = 1; j < cols; j++) {
            if (row_in[j] > maxv) maxv = row_in[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float e = expf(row_in[j] - maxv);
            row_out[j] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) {
            row_out[j] *= inv_sum;
        }
    }
}

// Given dP and P (both rows x cols), compute dL row-wise where L is logits before softmax.
// For each row r: dL = (dP - (sum(dP .* P)))* ⊙ P
static void softmax_backward_rows(const float* dP, const float* P, float* dL, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float* g = dP + i * cols;
        const float* p = P + i * cols;
        float* dl = dL + i * cols;
        float s = 0.0f;
        for (int j = 0; j < cols; j++) s += g[j] * p[j];
        for (int j = 0; j < cols; j++) dl[j] = (g[j] - s) * p[j];
    }
}

// Copy from time-major [T x B x C] selecting batch b into contiguous [T x C]
static void gather_time_major_batch(float* dst, const float* src, int T, int B, int C, int b) {
    for (int t = 0; t < T; t++) {
        const float* row = src + t * B * C + b * C;
        memcpy(dst + t * C, row, C * sizeof(float));
    }
}

// Scatter to time-major [T x B x C] for batch b from contiguous [T x C] (overwrite)
static void scatter_time_major_batch(float* dst, const float* src, int T, int B, int C, int b) {
    for (int t = 0; t < T; t++) {
        float* row = dst + t * B * C + b * C;
        memcpy(row, src + t * C, C * sizeof(float));
    }
}

// Scatter-add to time-major [T x B x C] for batch b from contiguous [T x C]
static void scatter_add_time_major_batch(float* dst, const float* src, int T, int B, int C, int b) {
    for (int t = 0; t < T; t++) {
        float* row = dst + t * B * C + b * C;
        const float* srow = src + t * C;
        for (int j = 0; j < C; j++) row[j] += srow[j];
    }
}

ATTN* init_attn(int input_dim, int hidden_dim, int output_dim, int seq_len, int batch_size, int num_layers) {
    ATTN* attn = (ATTN*)malloc(sizeof(ATTN));

    attn->input_dim = input_dim;
    attn->hidden_dim = hidden_dim;
    attn->output_dim = output_dim;
    attn->seq_len = seq_len;
    attn->batch_size = batch_size;
    attn->num_layers = num_layers;

    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->t = 0;
    attn->weight_decay = 0.01f;

    // Allocate pointer arrays
    attn->Wq = (float**)malloc(num_layers * sizeof(float*));
    attn->Wk = (float**)malloc(num_layers * sizeof(float*));
    attn->Wv = (float**)malloc(num_layers * sizeof(float*));
    attn->Wo = (float**)malloc(num_layers * sizeof(float*));
    attn->D  = (float**)malloc(num_layers * sizeof(float*));

    attn->Wq_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->Wk_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->Wv_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->Wo_grad = (float**)malloc(num_layers * sizeof(float*));
    attn->D_grad  = (float**)malloc(num_layers * sizeof(float*));

    attn->Wq_m = (float**)malloc(num_layers * sizeof(float*));
    attn->Wq_v = (float**)malloc(num_layers * sizeof(float*));
    attn->Wk_m = (float**)malloc(num_layers * sizeof(float*));
    attn->Wk_v = (float**)malloc(num_layers * sizeof(float*));
    attn->Wv_m = (float**)malloc(num_layers * sizeof(float*));
    attn->Wv_v = (float**)malloc(num_layers * sizeof(float*));
    attn->Wo_m = (float**)malloc(num_layers * sizeof(float*));
    attn->Wo_v = (float**)malloc(num_layers * sizeof(float*));
    attn->D_m  = (float**)malloc(num_layers * sizeof(float*));
    attn->D_v  = (float**)malloc(num_layers * sizeof(float*));

    attn->layer_Q = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_K = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_V = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_attn_scores = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_attn_probs  = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_context = (float**)malloc(num_layers * sizeof(float*));
    attn->layer_output  = (float**)malloc(num_layers * sizeof(float*));
    attn->error_output  = (float**)malloc(num_layers * sizeof(float*));

    int T = seq_len;
    int B = batch_size;

    for (int layer = 0; layer < num_layers; layer++) {
        int in  = (layer == 0) ? input_dim : output_dim;
        int out = output_dim;
        int Dk  = hidden_dim;

        int size_Wq = in * Dk;
        int size_Wk = in * Dk;
        int size_Wv = in * Dk;
        int size_Wo = Dk * out;
        int size_D  = in * out;

        // Allocate weights and gradients
        attn->Wq[layer] = (float*)malloc(size_Wq * sizeof(float));
        attn->Wk[layer] = (float*)malloc(size_Wk * sizeof(float));
        attn->Wv[layer] = (float*)malloc(size_Wv * sizeof(float));
        attn->Wo[layer] = (float*)malloc(size_Wo * sizeof(float));
        attn->D[layer]  = (float*)malloc(size_D  * sizeof(float));

        attn->Wq_grad[layer] = (float*)malloc(size_Wq * sizeof(float));
        attn->Wk_grad[layer] = (float*)malloc(size_Wk * sizeof(float));
        attn->Wv_grad[layer] = (float*)malloc(size_Wv * sizeof(float));
        attn->Wo_grad[layer] = (float*)malloc(size_Wo * sizeof(float));
        attn->D_grad[layer]  = (float*)malloc(size_D  * sizeof(float));

        attn->Wq_m[layer] = (float*)calloc(size_Wq, sizeof(float));
        attn->Wq_v[layer] = (float*)calloc(size_Wq, sizeof(float));
        attn->Wk_m[layer] = (float*)calloc(size_Wk, sizeof(float));
        attn->Wk_v[layer] = (float*)calloc(size_Wk, sizeof(float));
        attn->Wv_m[layer] = (float*)calloc(size_Wv, sizeof(float));
        attn->Wv_v[layer] = (float*)calloc(size_Wv, sizeof(float));
        attn->Wo_m[layer] = (float*)calloc(size_Wo, sizeof(float));
        attn->Wo_v[layer] = (float*)calloc(size_Wo, sizeof(float));
        attn->D_m[layer]  = (float*)calloc(size_D,  sizeof(float));
        attn->D_v[layer]  = (float*)calloc(size_D,  sizeof(float));

        // Buffers
        attn->layer_Q[layer] = (float*)malloc(T * B * Dk * sizeof(float));
        attn->layer_K[layer] = (float*)malloc(T * B * Dk * sizeof(float));
        attn->layer_V[layer] = (float*)malloc(T * B * Dk * sizeof(float));
        attn->layer_attn_scores[layer] = (float*)malloc(B * T * T * sizeof(float));
        attn->layer_attn_probs[layer]  = (float*)malloc(B * T * T * sizeof(float));
        attn->layer_context[layer] = (float*)malloc(T * B * Dk * sizeof(float));
        attn->layer_output[layer]  = (float*)malloc(T * B * out * sizeof(float));
        attn->error_output[layer]  = (float*)malloc(T * B * out * sizeof(float));

        // Initialize weights
        float scale_in = 1.0f / sqrtf((float)in);
        float scale_dk = 1.0f / sqrtf((float)Dk);
        for (int i = 0; i < size_Wq; i++) attn->Wq[layer][i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_in;
        for (int i = 0; i < size_Wk; i++) attn->Wk[layer][i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_in;
        for (int i = 0; i < size_Wv; i++) attn->Wv[layer][i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_in;
        for (int i = 0; i < size_Wo; i++) attn->Wo[layer][i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_dk;
        for (int i = 0; i < size_D;  i++) attn->D[layer][i]  = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_in;
    }

    return attn;
}

void free_attn(ATTN* attn) {
    if (!attn) return;

    for (int layer = 0; layer < attn->num_layers; layer++) {
        free(attn->Wq[layer]); free(attn->Wk[layer]); free(attn->Wv[layer]); free(attn->Wo[layer]); free(attn->D[layer]);
        free(attn->Wq_grad[layer]); free(attn->Wk_grad[layer]); free(attn->Wv_grad[layer]); free(attn->Wo_grad[layer]); free(attn->D_grad[layer]);
        free(attn->Wq_m[layer]); free(attn->Wq_v[layer]); free(attn->Wk_m[layer]); free(attn->Wk_v[layer]); free(attn->Wv_m[layer]); free(attn->Wv_v[layer]);
        free(attn->Wo_m[layer]); free(attn->Wo_v[layer]); free(attn->D_m[layer]); free(attn->D_v[layer]);

        free(attn->layer_Q[layer]); free(attn->layer_K[layer]); free(attn->layer_V[layer]);
        free(attn->layer_attn_scores[layer]); free(attn->layer_attn_probs[layer]);
        free(attn->layer_context[layer]); free(attn->layer_output[layer]);
        free(attn->error_output[layer]);
    }

    free(attn->Wq); free(attn->Wk); free(attn->Wv); free(attn->Wo); free(attn->D);
    free(attn->Wq_grad); free(attn->Wk_grad); free(attn->Wv_grad); free(attn->Wo_grad); free(attn->D_grad);
    free(attn->Wq_m); free(attn->Wq_v); free(attn->Wk_m); free(attn->Wk_v); free(attn->Wv_m); free(attn->Wv_v);
    free(attn->Wo_m); free(attn->Wo_v); free(attn->D_m); free(attn->D_v);

    free(attn->layer_Q); free(attn->layer_K); free(attn->layer_V);
    free(attn->layer_attn_scores); free(attn->layer_attn_probs);
    free(attn->layer_context); free(attn->layer_output);
    free(attn->error_output);

    free(attn);
}

void forward_pass_attn(ATTN* attn, float* X_time_major) {
    int T = attn->seq_len;
    int B = attn->batch_size;

    for (int layer = 0; layer < attn->num_layers; layer++) {
        int in  = (layer == 0) ? attn->input_dim : attn->output_dim;
        int out = attn->output_dim;
        int Dk  = attn->hidden_dim;
        float scale = 1.0f / sqrtf((float)Dk);

        float* X_in = (layer == 0) ? X_time_major : attn->layer_output[layer - 1];

        // Temporary per-batch buffers
        float* Xb = (float*)malloc(T * in  * sizeof(float));
        float* Qb = (float*)malloc(T * Dk  * sizeof(float));
        float* Kb = (float*)malloc(T * Dk  * sizeof(float));
        float* Vb = (float*)malloc(T * Dk  * sizeof(float));
        float* Lb = (float*)malloc(T * T   * sizeof(float));
        float* Pb = (float*)malloc(T * T   * sizeof(float));
        float* Cb = (float*)malloc(T * Dk  * sizeof(float));
        float* Yb = (float*)malloc(T * out * sizeof(float));

        for (int b = 0; b < B; b++) {
            gather_time_major_batch(Xb, X_in, T, B, in, b);

            // Q = X Wq; K = X Wk; V = X Wv
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, Dk, in, 1.0f, Xb, in, attn->Wq[layer], Dk, 0.0f, Qb, Dk);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, Dk, in, 1.0f, Xb, in, attn->Wk[layer], Dk, 0.0f, Kb, Dk);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, Dk, in, 1.0f, Xb, in, attn->Wv[layer], Dk, 0.0f, Vb, Dk);

            // Store Q,K,V in time-major buffers
            for (int t = 0; t < T; t++) {
                memcpy(attn->layer_Q[layer] + t * B * Dk + b * Dk, Qb + t * Dk, Dk * sizeof(float));
                memcpy(attn->layer_K[layer] + t * B * Dk + b * Dk, Kb + t * Dk, Dk * sizeof(float));
                memcpy(attn->layer_V[layer] + t * B * Dk + b * Dk, Vb + t * Dk, Dk * sizeof(float));
            }

            // L = (Q K^T)/sqrt(Dk)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, Dk, scale, Qb, Dk, Kb, Dk, 0.0f, Lb, T);

            // P = softmax(L) row-wise
            softmax_rows(Lb, Pb, T, T);

            // Store scores and probs
            memcpy(attn->layer_attn_scores[layer] + b * T * T, Lb, T * T * sizeof(float));
            memcpy(attn->layer_attn_probs[layer]  + b * T * T, Pb, T * T * sizeof(float));

            // C = P V
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, Dk, T, 1.0f, Pb, T, Vb, Dk, 0.0f, Cb, Dk);

            // Y = C Wo + X D
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, out, Dk, 1.0f, Cb, Dk, attn->Wo[layer], out, 0.0f, Yb, out);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, out, in, 1.0f, Xb, in, attn->D[layer], out, 1.0f, Yb, out);

            // Store context and output (time-major)
            scatter_time_major_batch(attn->layer_context[layer], Cb, T, B, Dk, b);
            scatter_time_major_batch(attn->layer_output[layer],  Yb, T, B, out, b);
        }

        free(Xb); free(Qb); free(Kb); free(Vb); free(Lb); free(Pb); free(Cb); free(Yb);
    }
}

float calculate_loss_attn(ATTN* attn, float* y_time_major) {
    int last = attn->num_layers - 1;
    int T = attn->seq_len;
    int B = attn->batch_size;
    int out = attn->output_dim;

    float loss = 0.0f;
    int N = T * B * out;
    for (int i = 0; i < N; i++) {
        float diff = attn->layer_output[last][i] - y_time_major[i];
        attn->error_output[last][i] = diff;
        loss += diff * diff;
    }
    return loss / (float)N;
}

void zero_gradients_attn(ATTN* attn) {
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int in  = (layer == 0) ? attn->input_dim : attn->output_dim;
        int out = attn->output_dim;
        int Dk  = attn->hidden_dim;

        memset(attn->Wq_grad[layer], 0, in * Dk * sizeof(float));
        memset(attn->Wk_grad[layer], 0, in * Dk * sizeof(float));
        memset(attn->Wv_grad[layer], 0, in * Dk * sizeof(float));
        memset(attn->Wo_grad[layer], 0, Dk * out * sizeof(float));
        memset(attn->D_grad[layer],  0, in * out * sizeof(float));
    }
}

void backward_pass_attn(ATTN* attn, float* X_time_major) {
    int T = attn->seq_len;
    int B = attn->batch_size;

    for (int layer = attn->num_layers - 1; layer >= 0; layer--) {
        int in  = (layer == 0) ? attn->input_dim : attn->output_dim;
        int out = attn->output_dim;
        int Dk  = attn->hidden_dim;
        float inv_sqrt = 1.0f / sqrtf((float)Dk);

        float* X_in = (layer == 0) ? X_time_major : attn->layer_output[layer - 1];
        if (layer > 0) {
            // We'll overwrite this with dX, so zero it first
            memset(attn->error_output[layer - 1], 0, T * B * in * sizeof(float));
        }

        // Temp buffers for one batch slice
        float* Xb   = (float*)malloc(T * in  * sizeof(float));
        float* dXb  = (float*)malloc(T * in  * sizeof(float));
        float* Qb   = (float*)malloc(T * Dk  * sizeof(float));
        float* Kb   = (float*)malloc(T * Dk  * sizeof(float));
        float* Vb   = (float*)malloc(T * Dk  * sizeof(float));
        float* Pb   = (float*)malloc(T * T   * sizeof(float));
        float* Cb   = (float*)malloc(T * Dk  * sizeof(float));
        float* dYb  = (float*)malloc(T * out * sizeof(float));
        float* dCb  = (float*)malloc(T * Dk  * sizeof(float));
        float* dPb  = (float*)malloc(T * T   * sizeof(float));
        float* dLb  = (float*)malloc(T * T   * sizeof(float));
        float* dQb  = (float*)malloc(T * Dk  * sizeof(float));
        float* dKb  = (float*)malloc(T * Dk  * sizeof(float));
        float* dVb  = (float*)malloc(T * Dk  * sizeof(float));

        for (int b = 0; b < B; b++) {
            gather_time_major_batch(Xb,  X_in,                        T, B, in,  b);
            gather_time_major_batch(Cb,  attn->layer_context[layer],  T, B, Dk,  b);
            gather_time_major_batch(dYb, attn->error_output[layer],   T, B, out, b);

            // Gather Q, K, V and P
            for (int t = 0; t < T; t++) {
                memcpy(Qb + t * Dk, attn->layer_Q[layer] + t * B * Dk + b * Dk, Dk * sizeof(float));
                memcpy(Kb + t * Dk, attn->layer_K[layer] + t * B * Dk + b * Dk, Dk * sizeof(float));
                memcpy(Vb + t * Dk, attn->layer_V[layer] + t * B * Dk + b * Dk, Dk * sizeof(float));
            }
            memcpy(Pb, attn->layer_attn_probs[layer] + b * T * T, T * T * sizeof(float));

            // dXb = 0
            memset(dXb, 0, T * in * sizeof(float));

            // Gradients wrt Wo and D; and partials
            // Wo_grad += C^T dY
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Dk, out, T, 1.0f, Cb, Dk, dYb, out, 1.0f, attn->Wo_grad[layer], out);
            // dC = dY Wo^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, Dk, out, 1.0f, dYb, out, attn->Wo[layer], out, 0.0f, dCb, Dk);
            // D_grad += X^T dY
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, out, T, 1.0f, Xb, in, dYb, out, 1.0f, attn->D_grad[layer], out);
            // dX += dY D^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, in, out, 1.0f, dYb, out, attn->D[layer], out, 1.0f, dXb, in);

            // Through C = P V
            // dP = dC V^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, Dk, 1.0f, dCb, Dk, Vb, Dk, 0.0f, dPb, T);
            // dV = P^T dC
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, T, Dk, T, 1.0f, Pb, T, dCb, Dk, 0.0f, dVb, Dk);

            // Accumulate Wv_grad and dX from V path
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, Dk, T, 1.0f, Xb, in, dVb, Dk, 1.0f, attn->Wv_grad[layer], Dk);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, in, Dk, 1.0f, dVb, Dk, attn->Wv[layer], Dk, 1.0f, dXb, in);

            // Through softmax: dL from dP and P
            softmax_backward_rows(dPb, Pb, dLb, T, T);

            // Through L = (Q K^T) / sqrt(Dk)
            // dQ = dL K
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, Dk, T, inv_sqrt, dLb, T, Kb, Dk, 0.0f, dQb, Dk);
            // dK = dL^T Q
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, T, Dk, T, inv_sqrt, dLb, T, Qb, Dk, 0.0f, dKb, Dk);

            // Accumulate Wq_grad, Wk_grad and dX from Q/K paths
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, Dk, T, 1.0f, Xb, in, dQb, Dk, 1.0f, attn->Wq_grad[layer], Dk);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in, Dk, T, 1.0f, Xb, in, dKb, Dk, 1.0f, attn->Wk_grad[layer], Dk);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, in, Dk, 1.0f, dQb, Dk, attn->Wq[layer], Dk, 1.0f, dXb, in);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, in, Dk, 1.0f, dKb, Dk, attn->Wk[layer], Dk, 1.0f, dXb, in);

            // Propagate to previous layer as error_output
            if (layer > 0) {
                scatter_add_time_major_batch(attn->error_output[layer - 1], dXb, T, B, in, b);
            }
        }

        free(Xb); free(dXb); free(Qb); free(Kb); free(Vb); free(Pb); free(Cb); free(dYb);
        free(dCb); free(dPb); free(dLb); free(dQb); free(dKb); free(dVb);
    }
}

void update_weights_attn(ATTN* attn, float learning_rate) {
    attn->t++;
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    int total_samples = attn->seq_len * attn->batch_size;

    for (int layer = 0; layer < attn->num_layers; layer++) {
        int in  = (layer == 0) ? attn->input_dim : attn->output_dim;
        int out = attn->output_dim;
        int Dk  = attn->hidden_dim;

        int size_Wq = in * Dk, size_Wk = in * Dk, size_Wv = in * Dk, size_Wo = Dk * out, size_D = in * out;

        for (int i = 0; i < size_Wq; i++) {
            float g = attn->Wq_grad[layer][i] / (float)total_samples;
            attn->Wq_m[layer][i] = attn->beta1 * attn->Wq_m[layer][i] + (1.0f - attn->beta1) * g;
            attn->Wq_v[layer][i] = attn->beta2 * attn->Wq_v[layer][i] + (1.0f - attn->beta2) * g * g;
            float upd = alpha_t * attn->Wq_m[layer][i] / (sqrtf(attn->Wq_v[layer][i]) + attn->epsilon);
            attn->Wq[layer][i] = attn->Wq[layer][i] * (1.0f - learning_rate * attn->weight_decay) - upd;
        }
        for (int i = 0; i < size_Wk; i++) {
            float g = attn->Wk_grad[layer][i] / (float)total_samples;
            attn->Wk_m[layer][i] = attn->beta1 * attn->Wk_m[layer][i] + (1.0f - attn->beta1) * g;
            attn->Wk_v[layer][i] = attn->beta2 * attn->Wk_v[layer][i] + (1.0f - attn->beta2) * g * g;
            float upd = alpha_t * attn->Wk_m[layer][i] / (sqrtf(attn->Wk_v[layer][i]) + attn->epsilon);
            attn->Wk[layer][i] = attn->Wk[layer][i] * (1.0f - learning_rate * attn->weight_decay) - upd;
        }
        for (int i = 0; i < size_Wv; i++) {
            float g = attn->Wv_grad[layer][i] / (float)total_samples;
            attn->Wv_m[layer][i] = attn->beta1 * attn->Wv_m[layer][i] + (1.0f - attn->beta1) * g;
            attn->Wv_v[layer][i] = attn->beta2 * attn->Wv_v[layer][i] + (1.0f - attn->beta2) * g * g;
            float upd = alpha_t * attn->Wv_m[layer][i] / (sqrtf(attn->Wv_v[layer][i]) + attn->epsilon);
            attn->Wv[layer][i] = attn->Wv[layer][i] * (1.0f - learning_rate * attn->weight_decay) - upd;
        }
        for (int i = 0; i < size_Wo; i++) {
            float g = attn->Wo_grad[layer][i] / (float)total_samples;
            attn->Wo_m[layer][i] = attn->beta1 * attn->Wo_m[layer][i] + (1.0f - attn->beta1) * g;
            attn->Wo_v[layer][i] = attn->beta2 * attn->Wo_v[layer][i] + (1.0f - attn->beta2) * g * g;
            float upd = alpha_t * attn->Wo_m[layer][i] / (sqrtf(attn->Wo_v[layer][i]) + attn->epsilon);
            attn->Wo[layer][i] = attn->Wo[layer][i] * (1.0f - learning_rate * attn->weight_decay) - upd;
        }
        for (int i = 0; i < size_D; i++) {
            float g = attn->D_grad[layer][i] / (float)total_samples;
            attn->D_m[layer][i] = attn->beta1 * attn->D_m[layer][i] + (1.0f - attn->beta1) * g;
            attn->D_v[layer][i] = attn->beta2 * attn->D_v[layer][i] + (1.0f - attn->beta2) * g * g;
            float upd = alpha_t * attn->D_m[layer][i] / (sqrtf(attn->D_v[layer][i]) + attn->epsilon);
            attn->D[layer][i] = attn->D[layer][i] * (1.0f - learning_rate * attn->weight_decay) - upd;
        }
    }
}

void save_attn(ATTN* attn, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("Error opening file for writing: %s\n", filename); return; }

    fwrite(&attn->input_dim,  sizeof(int), 1, f);
    fwrite(&attn->hidden_dim, sizeof(int), 1, f);
    fwrite(&attn->output_dim, sizeof(int), 1, f);
    fwrite(&attn->seq_len,    sizeof(int), 1, f);
    fwrite(&attn->batch_size, sizeof(int), 1, f);
    fwrite(&attn->num_layers, sizeof(int), 1, f);

    for (int layer = 0; layer < attn->num_layers; layer++) {
        int in  = (layer == 0) ? attn->input_dim : attn->output_dim;
        int out = attn->output_dim;
        int Dk  = attn->hidden_dim;

        fwrite(attn->Wq[layer], sizeof(float), in * Dk, f);
        fwrite(attn->Wk[layer], sizeof(float), in * Dk, f);
        fwrite(attn->Wv[layer], sizeof(float), in * Dk, f);
        fwrite(attn->Wo[layer], sizeof(float), Dk * out, f);
        fwrite(attn->D[layer],  sizeof(float), in * out, f);
    }

    fwrite(&attn->t, sizeof(int), 1, f);
    for (int layer = 0; layer < attn->num_layers; layer++) {
        int in  = (layer == 0) ? attn->input_dim : attn->output_dim;
        int out = attn->output_dim;
        int Dk  = attn->hidden_dim;
        int size_Wq = in * Dk, size_Wk = in * Dk, size_Wv = in * Dk, size_Wo = Dk * out, size_D = in * out;

        fwrite(attn->Wq_m[layer], sizeof(float), size_Wq, f);
        fwrite(attn->Wq_v[layer], sizeof(float), size_Wq, f);
        fwrite(attn->Wk_m[layer], sizeof(float), size_Wk, f);
        fwrite(attn->Wk_v[layer], sizeof(float), size_Wk, f);
        fwrite(attn->Wv_m[layer], sizeof(float), size_Wv, f);
        fwrite(attn->Wv_v[layer], sizeof(float), size_Wv, f);
        fwrite(attn->Wo_m[layer], sizeof(float), size_Wo, f);
        fwrite(attn->Wo_v[layer], sizeof(float), size_Wo, f);
        fwrite(attn->D_m[layer],  sizeof(float), size_D,  f);
        fwrite(attn->D_v[layer],  sizeof(float), size_D,  f);
    }

    fclose(f);
    printf("Model saved to %s\n", filename);
}

ATTN* load_attn(const char* filename, int custom_batch_size) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Error opening file for reading: %s\n", filename); return NULL; }

    int input_dim, hidden_dim, output_dim, seq_len, stored_batch_size, num_layers;
    fread(&input_dim,  sizeof(int), 1, f);
    fread(&hidden_dim, sizeof(int), 1, f);
    fread(&output_dim, sizeof(int), 1, f);
    fread(&seq_len,    sizeof(int), 1, f);
    fread(&stored_batch_size, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);

    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    ATTN* attn = init_attn(input_dim, hidden_dim, output_dim, seq_len, batch_size, num_layers);

    for (int layer = 0; layer < num_layers; layer++) {
        int in  = (layer == 0) ? input_dim : output_dim;
        int out = output_dim;
        int Dk  = hidden_dim;

        fread(attn->Wq[layer], sizeof(float), in * Dk, f);
        fread(attn->Wk[layer], sizeof(float), in * Dk, f);
        fread(attn->Wv[layer], sizeof(float), in * Dk, f);
        fread(attn->Wo[layer], sizeof(float), Dk * out, f);
        fread(attn->D[layer],  sizeof(float), in * out, f);
    }

    fread(&attn->t, sizeof(int), 1, f);
    for (int layer = 0; layer < num_layers; layer++) {
        int in  = (layer == 0) ? input_dim : output_dim;
        int out = output_dim;
        int Dk  = hidden_dim;
        int size_Wq = in * Dk, size_Wk = in * Dk, size_Wv = in * Dk, size_Wo = Dk * out, size_D = in * out;

        fread(attn->Wq_m[layer], sizeof(float), size_Wq, f);
        fread(attn->Wq_v[layer], sizeof(float), size_Wq, f);
        fread(attn->Wk_m[layer], sizeof(float), size_Wk, f);
        fread(attn->Wk_v[layer], sizeof(float), size_Wk, f);
        fread(attn->Wv_m[layer], sizeof(float), size_Wv, f);
        fread(attn->Wv_v[layer], sizeof(float), size_Wv, f);
        fread(attn->Wo_m[layer], sizeof(float), size_Wo, f);
        fread(attn->Wo_v[layer], sizeof(float), size_Wo, f);
        fread(attn->D_m[layer],  sizeof(float), size_D,  f);
        fread(attn->D_v[layer],  sizeof(float), size_D,  f);
    }

    fclose(f);
    printf("Model loaded from %s\n", filename);
    return attn;
}
