#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "data.h"
#include "attention.h"

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters
    const int seq_len = 16;
    const int feature_dim = 8;
    const int num_samples = 65536;
    const int batch_size = 512;
    
    // Generate synthetic data
    float *X, *y;
    generate_attention_data(&X, &y, num_samples, seq_len, feature_dim);

    // Initialize network
    Attention* attn = init_attention(feature_dim, seq_len, batch_size, false);
    
    // Training parameters
    const int num_epochs = 50;
    const float learning_rate = 0.001f;
    const int num_batches = num_samples / batch_size;
    
    // Allocate batch buffers
    int seq_size = batch_size * seq_len * feature_dim;
    float* X_batch = (float*)malloc(seq_size * sizeof(float));
    float* y_batch = (float*)malloc(seq_size * sizeof(float));
    
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
            
            // Forward pass
            forward_pass_attention(attn, X_batch);
            
            // Calculate loss
            float loss = calculate_loss_attention(attn, y_batch);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_attention(attn);
            backward_pass_attention(attn, X_batch, NULL);
            
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
    Attention* loaded_attn = load_attention(model_fname, batch_size);
    
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
    
    // Forward pass with loaded model
    forward_pass_attention(loaded_attn, X_batch);

    // Evaluate model performance on first batch
    printf("Output\tR²\t\tMAE\t\tSample Predictions\n");
    printf("------\t--------\t--------\t--------------------------------\n");

    for (int i = 0; i < feature_dim; i++) {
        // Calculate mean for R²
        float y_mean = 0.0f;
        for (int sample = 0; sample < batch_size; sample++) {
            for (int seq = 0; seq < seq_len; seq++) {
                y_mean += y_batch[sample * seq_len * feature_dim + seq * feature_dim + i];
            }
        }
        y_mean /= (batch_size * seq_len);
        
        // Calculate R² and MAE
        float ss_res = 0.0f, ss_tot = 0.0f, mae = 0.0f;
        for (int sample = 0; sample < batch_size; sample++) {
            for (int seq = 0; seq < seq_len; seq++) {
                float pred = loaded_attn->layer_output[sample * seq_len * feature_dim + seq * feature_dim + i];
                float actual = y_batch[sample * seq_len * feature_dim + seq * feature_dim + i];
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
            float pred = loaded_attn->layer_output[j * seq_len * feature_dim + 0 * feature_dim + i];
            float actual = y_batch[j * seq_len * feature_dim + 0 * feature_dim + i];
            printf("%.2f/%.2f ", pred, actual);
        }
        printf("\n");
    }
    
    // Cleanup
    free(X);
    free(y);
    free(X_batch);
    free(y_batch);
    free_attention(attn);
    free_attention(loaded_attn);
    
    return 0;
}