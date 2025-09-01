#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../data.h"
#include "attention.h"

int main() {
    srand(time(NULL));

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    // Parameters
    const int seq_len = 16;
    const int feature_dim = 8;
    const int num_samples = 65536;
    const int batch_size = 512;
    
    // Generate synthetic data
    float *X, *y;
    generate_data(&X, &y, num_samples, seq_len, feature_dim);

    // Initialize network
    Attention* attn = init_attention(feature_dim, seq_len, batch_size, false, cublas_handle);
    
    // Training parameters
    const int num_epochs = 50;
    const float learning_rate = 0.001f;
    const int num_batches = num_samples / batch_size;
    
    // Allocate batch buffers
    int seq_size = batch_size * seq_len * feature_dim;
    float* X_batch = (float*)malloc(seq_size * sizeof(float));
    float* y_batch = (float*)malloc(seq_size * sizeof(float));
    
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, seq_size * sizeof(float)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            
            // Copy batch data
            for (int sample = 0; sample < batch_size; sample++) {
                for (int seq = 0; seq < seq_len; seq++) {
                    for (int feat = 0; feat < feature_dim; feat++) {
                        int src_idx = (start_idx + sample) * seq_len * feature_dim + seq * feature_dim + feat;
                        int dst_idx = sample * seq_len * feature_dim + seq * feature_dim + feat;
                        X_batch[dst_idx] = X[src_idx];
                        y_batch[dst_idx] = y[src_idx];
                    }
                }
            }
            
            // Copy to device
            CHECK_CUDA(cudaMemcpy(d_X, X_batch, seq_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, y_batch, seq_size * sizeof(float), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_attention(attn, d_X);
            
            // Calculate loss
            float loss = calculate_loss_attention(attn, d_y);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_attention(attn);
            backward_pass_attention(attn, d_X, NULL);
            
            // Update weights
            update_weights_attention(attn, learning_rate);
        }
        
        epoch_loss /= num_batches;

        // Print progress
        if (epoch > 0 && epoch % 2 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));

    // Save model and data with timestamped filenames
    save_attention(attn, model_fname);
    save_data(X, y, num_samples, seq_len, feature_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    Attention* loaded_attn = load_attention(model_fname, batch_size, cublas_handle);
    
    // Evaluate on first batch
    for (int sample = 0; sample < batch_size; sample++) {
        for (int seq = 0; seq < seq_len; seq++) {
            for (int feat = 0; feat < feature_dim; feat++) {
                int src_idx = sample * seq_len * feature_dim + seq * feature_dim + feat;
                int dst_idx = sample * seq_len * feature_dim + seq * feature_dim + feat;
                X_batch[dst_idx] = X[src_idx];
                y_batch[dst_idx] = y[src_idx];
            }
        }
    }
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_X, X_batch, seq_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y_batch, seq_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Forward pass with loaded model
    forward_pass_attention(loaded_attn, d_X);
    
    // Copy predictions back to host
    CHECK_CUDA(cudaMemcpy(y_batch, loaded_attn->d_layer_output, 
                         seq_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Evaluate model performance on first batch
    printf("Output\tR²\t\tMAE\t\tSample Predictions\n");
    printf("------\t--------\t--------\t--------------------------------\n");

    for (int i = 0; i < feature_dim; i++) {
        // Calculate mean for R²
        float y_mean = 0.0f;
        for (int sample = 0; sample < batch_size; sample++) {
            for (int seq = 0; seq < seq_len; seq++) {
                y_mean += y[sample * seq_len * feature_dim + seq * feature_dim + i];
            }
        }
        y_mean /= (batch_size * seq_len);
        
        // Calculate R² and MAE
        float ss_res = 0.0f, ss_tot = 0.0f, mae = 0.0f;
        for (int sample = 0; sample < batch_size; sample++) {
            for (int seq = 0; seq < seq_len; seq++) {
                float pred = y_batch[sample * seq_len * feature_dim + seq * feature_dim + i];
                float actual = y[sample * seq_len * feature_dim + seq * feature_dim + i];
                float diff = pred - actual;
                
                ss_res += diff * diff;
                ss_tot += (actual - y_mean) * (actual - y_mean);
                mae += fabs(diff);
            }
        }
        
        float r2 = 1.0f - (ss_res / ss_tot);
        mae /= (batch_size * seq_len);
        
        // Print summary
        printf("f%d\t%.6f\t%.3f\t\t", i, r2, mae);
        for (int j = 0; j < 3; j++) {
            float pred = y_batch[j * seq_len * feature_dim + 0 * feature_dim + i];
            float actual = y[j * seq_len * feature_dim + 0 * feature_dim + i];
            printf("%.2f/%.2f ", pred, actual);
        }
        printf("\n");
    }
    
    // Cleanup
    free(X);
    free(y);
    free(X_batch);
    free(y_batch);
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_y));
    free_attention(attn);
    free_attention(loaded_attn);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    return 0;
}