#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include "../data.h"
#include "attention.h"

int main() {
    srand(time(NULL));

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Parameters
    const int seq_len     = 128;
    const int d_model     = 64;
    const int num_heads   = 8;   // multi-head
    const int num_samples = 1024;
    const int batch_size  = 32;
    
    // Generate synthetic data
    float *X, *y;
    generate_data(&X, &y, seq_len, num_samples, d_model, -5.0f, 5.0f);

    // Convert to half precision
    size_t total = (size_t)num_samples * seq_len * d_model;
    half *h_X = (half*)malloc(total * sizeof(half));
    half *h_y = (half*)malloc(total * sizeof(half));
    for (size_t i = 0; i < total; i++) h_X[i] = __float2half(X[i]);
    for (size_t i = 0; i < total; i++) h_y[i] = __float2half(y[i]);
    
    // Initialize attention layer
    Attention* attn = init_attention(seq_len, d_model, num_heads,
                                     batch_size, false, false,
                                     cublaslt_handle);
    
    // Training parameters
    const int   num_epochs    = 50;
    const float learning_rate = 0.001f;
    const int   num_batches   = num_samples / batch_size;
    
    // Device batch buffers
    size_t batch_elems = (size_t)batch_size * seq_len * d_model;
    half *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, batch_elems * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y, batch_elems * sizeof(half)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            size_t off = (size_t)batch * batch_elems;

            CHECK_CUDA(cudaMemcpy(d_X, &h_X[off],
                                  batch_elems * sizeof(half),
                                  cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, &h_y[off],
                                  batch_elems * sizeof(half),
                                  cudaMemcpyHostToDevice));
            
            forward_pass_attention(attn, d_X);
            float loss = calculate_loss_attention(attn, d_y);
            epoch_loss += loss;

            if (epoch == num_epochs) continue;

            zero_gradients_attention(attn);
            backward_pass_attention(attn, d_X, NULL);
            update_weights_attention(attn, learning_rate, batch_size);
        }
        
        epoch_loss /= num_batches;
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n",
                   epoch, num_epochs, epoch_loss);
        }
    }

    // Save model and data
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname),
             "%Y%m%d_%H%M%S_attention.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname),
             "%Y%m%d_%H%M%S_attention_data.csv", localtime(&now));

    FILE* model_file = fopen(model_fname, "wb");
    serialize_attention(attn, model_file);
    fclose(model_file);
    printf("Model saved to %s\n", model_fname);
    
    save_data(X, y, seq_len, num_samples, d_model, data_fname);
    
    // Reload model and verify
    printf("\nVerifying saved model...\n");
    model_file = fopen(model_fname, "rb");
    Attention* loaded_attn = deserialize_attention(model_file,
                                                   batch_size, seq_len,
                                                   num_heads, cublaslt_handle);
    fclose(model_file);
    printf("Model loaded from %s\n", model_fname);

    // Forward on first batch
    CHECK_CUDA(cudaMemcpy(d_X, h_X,
                          batch_elems * sizeof(half),
                          cudaMemcpyHostToDevice));
    forward_pass_attention(loaded_attn, d_X);
    
    // Copy predictions back
    half* h_output = (half*)malloc(batch_elems * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h_output, loaded_attn->d_output,
                          batch_elems * sizeof(half),
                          cudaMemcpyDeviceToHost));

    // Simple per-feature metrics (R^2 and MAE) on first batch
    printf("Feature\tR²\t\tMAE\t\tSample Predictions\n");
    printf("-------\t--------\t--------\t--------------------------------\n");

    for (int d = 0; d < d_model; d++) {
        float y_mean = 0.0f;
        int total_elements = batch_size * seq_len;
        
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                size_t idx = (size_t)b * seq_len * d_model + t * d_model + d;
                y_mean += __half2float(h_y[idx]);
            }
        }
        y_mean /= total_elements;
        
        float ss_res = 0.0f, ss_tot = 0.0f, mae = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                size_t idx = (size_t)b * seq_len * d_model + t * d_model + d;
                float pred   = __half2float(h_output[idx]);
                float actual = __half2float(h_y[idx]);
                float diff   = pred - actual;
                ss_res += diff * diff;
                ss_tot += (actual - y_mean) * (actual - y_mean);
                mae    += fabsf(diff);
            }
        }
        float r2 = 1.0f - (ss_res / ss_tot);
        mae /= total_elements;
        
        printf("d%d\t%.6f\t%.3f\t\t", d, r2, mae);
        for (int sample = 0; sample < 3; sample++) {
            size_t idx = (size_t)0 * seq_len * d_model + sample * d_model + d;
            float pred   = __half2float(h_output[idx]);
            float actual = __half2float(h_y[idx]);
            printf("%.2f/%.2f ", pred, actual);
        }
        printf("\n");
    }
    
    // Cleanup
    free(X);
    free(y);
    free(h_X);
    free(h_y);
    free(h_output);
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_y));
    free_attention(attn);
    free_attention(loaded_attn);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}