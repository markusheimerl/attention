#include "attention.h"
#include <limits.h>

// CUDA kernel for computing loss and gradient
__global__ static void compute_loss_and_gradient_kernel_attention(float* grad_output, const float* predictions, const float* targets, float* loss_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        grad_output[idx] = diff;
        atomicAdd(loss_result, diff * diff);
    }
}

// CUDA kernel for AdamW update on a flat weights buffer
__global__ static void adamw_update_flat_kernel(float* weights, const float* grads, float* m, float* v,
                                                float beta1, float beta2, float epsilon, float lr,
                                                float weight_decay, float alpha_t, int n, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grads[i] / batch_size;
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float update = alpha_t * m[i] / (sqrtf(v[i]) + epsilon);
        weights[i] = weights[i] * (1.0f - lr * weight_decay) - update;
    }
}

static void set_seqdesc_BTnV(cudnnSeqDataDescriptor_t desc, int batch_size, int seq_len, int beam_size, int vec_len, const int* h_seq_lengths, size_t seq_length_array_size) {
    // axes order to match memory layout [batch, time, beam, vect] where beam=1 and fastest dim is vect
    cudnnSeqDataAxis_t axes[4] = {
        CUDNN_SEQDATA_BATCH_DIM,
        CUDNN_SEQDATA_TIME_DIM,
        CUDNN_SEQDATA_BEAM_DIM,
        CUDNN_SEQDATA_VECT_DIM
    };
    int dimA[4];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = batch_size;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = seq_len;
    dimA[CUDNN_SEQDATA_BEAM_DIM]  = beam_size;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = vec_len;

    CHECK_CUDNN(cudnnSetSeqDataDescriptor(desc,
                                          CUDNN_DATA_FLOAT,
                                          4,
                                          dimA,
                                          axes,
                                          seq_length_array_size,
                                          h_seq_lengths,
                                          NULL));
}

// Initialize the attention layer
Attention* init_attention(int seq_len, int d_model, int batch_size, bool is_causal, cudnnHandle_t cudnn_handle) {
    Attention* attn = (Attention*)calloc(1, sizeof(Attention));
    attn->seq_len = seq_len;
    attn->d_model = d_model;
    attn->batch_size = batch_size;
    attn->num_heads = 4; // simple single-head config
    attn->is_causal = is_causal;

    // AdamW defaults
    attn->beta1 = 0.9f;
    attn->beta2 = 0.999f;
    attn->epsilon = 1e-8f;
    attn->weight_decay = 0.01f;
    attn->t = 0;

    attn->cudnn_handle = cudnn_handle;

    // Create cuDNN descriptors
    CHECK_CUDNN(cudnnCreateAttnDescriptor(&attn->attn_desc));
    CHECK_CUDNN(cudnnCreateDropoutDescriptor(&attn->drop_desc));
    CHECK_CUDNN(cudnnCreateSeqDataDescriptor(&attn->q_desc));
    CHECK_CUDNN(cudnnCreateSeqDataDescriptor(&attn->k_desc));
    CHECK_CUDNN(cudnnCreateSeqDataDescriptor(&attn->v_desc));
    CHECK_CUDNN(cudnnCreateSeqDataDescriptor(&attn->o_desc));

    // Dropout (disabled)
    size_t dropout_state_size = 0;
    CHECK_CUDNN(cudnnDropoutGetStatesSize(attn->cudnn_handle, &dropout_state_size));
    CHECK_CUDA(cudaMalloc(&attn->d_dropout_states, dropout_state_size));
    CHECK_CUDNN(cudnnSetDropoutDescriptor(attn->drop_desc, attn->cudnn_handle, 0.0f, attn->d_dropout_states, dropout_state_size, 0));

    // Set attention descriptor
    // Use full projections inside cuDNN. With 1 head, per-head proj sizes equal d_model.
    int qSize = d_model, kSize = d_model, vSize = d_model;
    int qProjSize = d_model, kProjSize = d_model, vProjSize = d_model, oProjSize = d_model;
    int qoMaxSeqLength = seq_len, kvMaxSeqLength = seq_len;
    int maxBatchSize = batch_size, maxBeamSize = 1;

    CHECK_CUDNN(cudnnSetAttnDescriptor(attn->attn_desc,
                                       CUDNN_ATTN_QUERYMAP_ONE_TO_ONE,
                                       attn->num_heads,
                                       1.0 / sqrt((double)d_model),
                                       CUDNN_DATA_FLOAT,
                                       CUDNN_DATA_FLOAT,
                                       CUDNN_DEFAULT_MATH,
                                       attn->drop_desc,
                                       NULL, // postDropoutDesc
                                       qSize, kSize, vSize,
                                       qProjSize, kProjSize, vProjSize, oProjSize,
                                       qoMaxSeqLength, kvMaxSeqLength,
                                       maxBatchSize, maxBeamSize));

    // Query weight sizes and allocate buffers
    CHECK_CUDNN(cudnnGetMultiHeadAttnBuffers(attn->cudnn_handle,
                                             attn->attn_desc,
                                             &attn->weight_size,
                                             &attn->workspace_size,
                                             &attn->reserve_size));

    if (attn->weight_size > 0) {
        CHECK_CUDA(cudaMalloc(&attn->d_weights, attn->weight_size));
        CHECK_CUDA(cudaMalloc(&attn->d_wgrad,  attn->weight_size));
        // Adam moments
        int n = (int)(attn->weight_size / sizeof(float));
        CHECK_CUDA(cudaMalloc(&attn->d_m, attn->weight_size));
        CHECK_CUDA(cudaMalloc(&attn->d_v, attn->weight_size));
        CHECK_CUDA(cudaMemset(attn->d_m, 0, attn->weight_size));
        CHECK_CUDA(cudaMemset(attn->d_v, 0, attn->weight_size));

        // Initialize weights
        float* hW = (float*)malloc(attn->weight_size);
        float scale = 1.0f / sqrtf((float)d_model);
        for (size_t i = 0; i < attn->weight_size / sizeof(float); i++) {
            hW[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        }
        CHECK_CUDA(cudaMemcpy(attn->d_weights, hW, attn->weight_size, cudaMemcpyHostToDevice));
        free(hW);
    } else {
        attn->d_weights = NULL;
        attn->d_wgrad = NULL;
        attn->d_m = NULL;
        attn->d_v = NULL;
    }

    // Workspace and reserve
    if (attn->workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&attn->d_workspace, attn->workspace_size));
    }
    if (attn->reserve_size > 0) {
        CHECK_CUDA(cudaMalloc(&attn->d_reserve_space, attn->reserve_size));
    }

    // Output and grad_output
    size_t elems = (size_t)batch_size * seq_len * d_model;
    CHECK_CUDA(cudaMalloc(&attn->d_output, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_grad_output, elems * sizeof(float)));

    // Backward data buffers (required by cudnnMultiHeadAttnBackwardData)
    CHECK_CUDA(cudaMalloc(&attn->d_dQ, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_dK, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn->d_dV, elems * sizeof(float)));

    // Loss accumulator
    CHECK_CUDA(cudaMalloc(&attn->d_loss_result, sizeof(float)));

    // Device arrays for sequence lengths
    CHECK_CUDA(cudaMalloc(&attn->d_seq_lengths_qo, batch_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&attn->d_seq_lengths_kv, batch_size * sizeof(int)));
    int* h_seq_qo = (int*)malloc(batch_size * sizeof(int));
    int* h_seq_kv = (int*)malloc(batch_size * sizeof(int));
    for (int i = 0; i < batch_size; i++) {
        h_seq_qo[i] = seq_len;
        h_seq_kv[i] = seq_len;
    }
    CHECK_CUDA(cudaMemcpy(attn->d_seq_lengths_qo, h_seq_qo, batch_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(attn->d_seq_lengths_kv, h_seq_kv, batch_size * sizeof(int), cudaMemcpyHostToDevice));

    // Attention window setup
    attn->lo_win_idx = (int*)malloc(seq_len * sizeof(int));
    attn->hi_win_idx = (int*)malloc(seq_len * sizeof(int));
    for (int t = 0; t < seq_len; t++) {
        if (is_causal) {
            attn->lo_win_idx[t] = 0;
            attn->hi_win_idx[t] = t + 1; // [0, t] inclusive --> hi exclusive => t+1
        } else {
            attn->lo_win_idx[t] = 0;
            attn->hi_win_idx[t] = seq_len; // attend to all tokens
        }
    }

    // Set SeqData descriptors to match memory layout X: [batch, time, vect] contiguous with last-dim fastest
    // We choose axes order [BATCH, TIME, BEAM, VECT] and dims accordingly.
    // For Q and O, seqLength array size is batch_size (beam=1). For K and V, also batch_size (self-attention).
    set_seqdesc_BTnV(attn->q_desc, batch_size, seq_len, 1, d_model, h_seq_qo, (size_t)batch_size);
    set_seqdesc_BTnV(attn->o_desc, batch_size, seq_len, 1, d_model, h_seq_qo, (size_t)batch_size);
    set_seqdesc_BTnV(attn->k_desc, batch_size, seq_len, 1, d_model, h_seq_kv, (size_t)batch_size);
    set_seqdesc_BTnV(attn->v_desc, batch_size, seq_len, 1, d_model, h_seq_kv, (size_t)batch_size);

    free(h_seq_qo);
    free(h_seq_kv);

    return attn;
}

void free_attention(Attention* attn) {
    if (!attn) return;

    cudnnDestroyAttnDescriptor(attn->attn_desc);
    cudnnDestroyDropoutDescriptor(attn->drop_desc);
    cudnnDestroySeqDataDescriptor(attn->q_desc);
    cudnnDestroySeqDataDescriptor(attn->k_desc);
    cudnnDestroySeqDataDescriptor(attn->v_desc);
    cudnnDestroySeqDataDescriptor(attn->o_desc);

    if (attn->d_dropout_states) cudaFree(attn->d_dropout_states);
    if (attn->d_workspace) cudaFree(attn->d_workspace);
    if (attn->d_reserve_space) cudaFree(attn->d_reserve_space);

    if (attn->d_weights) cudaFree(attn->d_weights);
    if (attn->d_wgrad) cudaFree(attn->d_wgrad);
    if (attn->d_m) cudaFree(attn->d_m);
    if (attn->d_v) cudaFree(attn->d_v);

    if (attn->d_output) cudaFree(attn->d_output);
    if (attn->d_grad_output) cudaFree(attn->d_grad_output);

    if (attn->d_dQ) cudaFree(attn->d_dQ);
    if (attn->d_dK) cudaFree(attn->d_dK);
    if (attn->d_dV) cudaFree(attn->d_dV);

    if (attn->d_seq_lengths_qo) cudaFree(attn->d_seq_lengths_qo);
    if (attn->d_seq_lengths_kv) cudaFree(attn->d_seq_lengths_kv);

    if (attn->d_loss_result) cudaFree(attn->d_loss_result);

    free(attn->lo_win_idx);
    free(attn->hi_win_idx);

    free(attn);
}

// Forward pass: Q=K=V=X, weights handled by cuDNN
void forward_pass_attention(Attention* attn, float* d_X) {
    CHECK_CUDNN(cudnnMultiHeadAttnForward(attn->cudnn_handle,
                                          attn->attn_desc,
                                          -1, // process all time steps
                                          attn->lo_win_idx,
                                          attn->hi_win_idx,
                                          attn->d_seq_lengths_qo, // devSeqLengthsQO
                                          attn->d_seq_lengths_kv, // devSeqLengthsKV
                                          attn->q_desc, d_X,
                                          NULL, // residuals
                                          attn->k_desc, d_X,
                                          attn->v_desc, d_X,
                                          attn->o_desc, attn->d_output,
                                          attn->weight_size, attn->d_weights,
                                          attn->workspace_size, attn->d_workspace,
                                          attn->reserve_size, attn->d_reserve_space));
}

// Calculate MSE loss and fill d_grad_output
float calculate_loss_attention(Attention* attn, float* d_y) {
    int total = attn->batch_size * attn->seq_len * attn->d_model;
    int block = 256;
    int grid = (total + block - 1) / block;
    CHECK_CUDA(cudaMemset(attn->d_loss_result, 0, sizeof(float)));
    compute_loss_and_gradient_kernel_attention<<<grid, block>>>(attn->d_grad_output, attn->d_output, d_y, attn->d_loss_result, total);
    float host_loss = 0.0f;
    CHECK_CUDA(cudaMemcpy(&host_loss, attn->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    return host_loss / total;
}

void zero_gradients_attention(Attention* attn) {
    if (attn->d_wgrad && attn->weight_size > 0) {
        CHECK_CUDA(cudaMemset(attn->d_wgrad, 0, attn->weight_size));
    }
}

// Backward pass: compute wgrad via cuDNN; dX is optional (not needed by caller)
void backward_pass_attention(Attention* attn, float* d_X, float* d_grad_X_unused) {
    // Backward Data (to compute dQ/dK/dV; not used by caller, but may be required by cudnn for wgrad)
    CHECK_CUDNN(cudnnMultiHeadAttnBackwardData(attn->cudnn_handle,
                                               attn->attn_desc,
                                               attn->lo_win_idx,
                                               attn->hi_win_idx,
                                               attn->d_seq_lengths_qo, // devSeqLengthsDQDO
                                               attn->d_seq_lengths_kv, // devSeqLengthsDKDV
                                               attn->o_desc, attn->d_grad_output,
                                               attn->q_desc, attn->d_dQ, d_X,
                                               attn->k_desc, attn->d_dK, d_X,
                                               attn->v_desc, attn->d_dV, d_X,
                                               attn->weight_size, attn->d_weights,
                                               attn->workspace_size, attn->d_workspace,
                                               attn->reserve_size, attn->d_reserve_space));

    // Backward Weights (compute d(weights))
    CHECK_CUDNN(cudnnMultiHeadAttnBackwardWeights(attn->cudnn_handle,
                                                  attn->attn_desc,
                                                  CUDNN_WGRAD_MODE_SET,
                                                  attn->q_desc, d_X,
                                                  attn->k_desc, d_X,
                                                  attn->v_desc, d_X,
                                                  attn->o_desc, attn->d_grad_output,
                                                  attn->weight_size, attn->d_weights, attn->d_wgrad,
                                                  attn->workspace_size, attn->d_workspace,
                                                  attn->reserve_size, attn->d_reserve_space));
}

// AdamW update on flattened weight buffer
void update_weights_attention(Attention* attn, float learning_rate) {
    if (!attn->d_weights || !attn->d_wgrad) return;

    attn->t++;
    float beta1_t = powf(attn->beta1, attn->t);
    float beta2_t = powf(attn->beta2, attn->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);

    int n = (int)(attn->weight_size / sizeof(float));
    int block = 256;
    int grid = (n + block - 1) / block;

    adamw_update_flat_kernel<<<grid, block>>>((float*)attn->d_weights, (float*)attn->d_wgrad, attn->d_m, attn->d_v,
                                              attn->beta1, attn->beta2, attn->epsilon, learning_rate,
                                              attn->weight_decay, alpha_t, n, attn->batch_size);
}

// Save model: store geometry and full cuDNN weights buffer + Adam moments + t
void save_attention(Attention* attn, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Error opening %s for write\n", filename); return; }

    fwrite(&attn->seq_len, sizeof(int), 1, f);
    fwrite(&attn->d_model, sizeof(int), 1, f);
    fwrite(&attn->batch_size, sizeof(int), 1, f);
    fwrite(&attn->is_causal, sizeof(bool), 1, f);
    fwrite(&attn->num_heads, sizeof(int), 1, f);

    // weight buffer
    fwrite(&attn->weight_size, sizeof(size_t), 1, f);
    if (attn->weight_size > 0) {
        float* hW = (float*)malloc(attn->weight_size);
        float* hM = (float*)malloc(attn->weight_size);
        float* hV = (float*)malloc(attn->weight_size);
        CHECK_CUDA(cudaMemcpy(hW, attn->d_weights, attn->weight_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hM, attn->d_m,       attn->weight_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hV, attn->d_v,       attn->weight_size, cudaMemcpyDeviceToHost));
        fwrite(hW, 1, attn->weight_size, f);
        fwrite(hM, 1, attn->weight_size, f);
        fwrite(hV, 1, attn->weight_size, f);
        free(hW); free(hM); free(hV);
    }

    fwrite(&attn->t, sizeof(int), 1, f);

    fclose(f);
    printf("Model saved to %s\n", filename);
}

// Load model: reconstruct descriptor and replace weight buffer and Adam moments
Attention* load_attention(const char* filename, int custom_batch_size, cudnnHandle_t cudnn_handle) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Error opening %s for read\n", filename); return NULL; }

    int seq_len, d_model, stored_batch_size, num_heads;
    bool is_causal;
    fread(&seq_len, sizeof(int), 1, f);
    fread(&d_model, sizeof(int), 1, f);
    fread(&stored_batch_size, sizeof(int), 1, f);
    fread(&is_causal, sizeof(bool), 1, f);
    fread(&num_heads, sizeof(int), 1, f);

    int batch_size = custom_batch_size > 0 ? custom_batch_size : stored_batch_size;

    Attention* attn = init_attention(seq_len, d_model, batch_size, is_causal, cudnn_handle);

    size_t weight_size_file = 0;
    fread(&weight_size_file, sizeof(size_t), 1, f);
    if (weight_size_file != attn->weight_size) {
        fprintf(stderr, "Weight size mismatch: file %zu vs runtime %zu\n", weight_size_file, attn->weight_size);
        fclose(f);
        free_attention(attn);
        return NULL;
    }

    if (attn->weight_size > 0) {
        float* hW = (float*)malloc(attn->weight_size);
        float* hM = (float*)malloc(attn->weight_size);
        float* hV = (float*)malloc(attn->weight_size);
        fread(hW, 1, attn->weight_size, f);
        fread(hM, 1, attn->weight_size, f);
        fread(hV, 1, attn->weight_size, f);
        CHECK_CUDA(cudaMemcpy(attn->d_weights, hW, attn->weight_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_m,       hM, attn->weight_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(attn->d_v,       hV, attn->weight_size, cudaMemcpyHostToDevice));
        free(hW); free(hM); free(hV);
    }

    fread(&attn->t, sizeof(int), 1, f);
    fclose(f);
    printf("Model loaded from %s\n", filename);
    return attn;
}