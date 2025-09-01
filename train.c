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
    const int d_model = 64;
    const int seq_len = 16;
    const int num_samples = 2048;
    const int batch_size = 256;
    
    // Generate synthetic sequence data
    float *X, *y;
    generate_sequence_data(&X, &y, num_samples, seq_len, d_model, -3.0f, 3.0f);
    
    // Initialize attention layer
    Attention* attn = init_attention(d_model, seq_len, batch_size);
    
    // Training parameters
    const int num_epochs = 5000;
    const float learning_rate = 0.0001f;
    const int num_batches = num_samples / batch_size;
    
    // Allocate batch buffers
    int batch_x_size = batch_size * seq_len * d_model;
    int batch_y_size = batch_size * seq_len * d_model;
    float* X_batch = (float*)malloc(batch_x_size * sizeof(float));
    float* y_batch = (float*)malloc(batch_y_size * sizeof(float));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            
            // Copy batch data
            for (int pos = 0; pos < seq_len; pos++) {
                for (int dim = 0; dim < d_model; dim++) {
                    int src_offset = dim * (seq_len * num_samples) + pos * num_samples + start_idx;
                    int dst_offset = dim * (seq_len * batch_size) + pos * batch_size;
                    memcpy(&X_batch[dst_offset], &X[src_offset], batch_size * sizeof(float));
                    memcpy(&y_batch[dst_offset], &y[src_offset], batch_size * sizeof(float));
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
        if (epoch > 0 && epoch % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_attention.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_attention_data.csv", localtime(&now));

    // Save model and data with timestamped filenames
    save_attention(attn, model_fname);
    save_sequence_data(X, y, num_samples, seq_len, d_model, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    Attention* loaded_attn = load_attention(model_fname, batch_size);
    
    // Evaluate on first batch
    for (int pos = 0; pos < seq_len; pos++) {
        for (int dim = 0; dim < d_model; dim++) {
            int src_offset = dim * (seq_len * num_samples) + pos * num_samples;
            int dst_offset = dim * (seq_len * batch_size) + pos * batch_size;
            memcpy(&X_batch[dst_offset], &X[src_offset], batch_size * sizeof(float));
            memcpy(&y_batch[dst_offset], &y[src_offset], batch_size * sizeof(float));
        }
    }
    
    // Forward pass with loaded model
    forward_pass_attention(loaded_attn, X_batch);

    // Evaluate model performance on first batch
    printf("Position\tR²\t\tMAE\t\tSample Predictions (first dim)\n");
    printf("--------\t--------\t--------\t--------------------------------\n");

    for (int pos = 0; pos < seq_len; pos++) {
        // Calculate metrics for first dimension only (for simplicity)
        int dim = 0;
        int offset = dim * (seq_len * batch_size) + pos * batch_size;
        
        // Calculate mean for R²
        float y_mean = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            y_mean += y_batch[offset + b];
        }
        y_mean /= batch_size;
        
        // Calculate R² and MAE
        float ss_res = 0.0f, ss_tot = 0.0f, mae = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            float pred = loaded_attn->layer_output[offset + b];
            float actual = y_batch[offset + b];
            float diff = pred - actual;
            
            ss_res += diff * diff;
            ss_tot += (actual - y_mean) * (actual - y_mean);
            mae += fabs(diff);
        }
        
        float r2 = 1.0f - (ss_res / ss_tot);
        mae /= batch_size;
        
        // Print summary
        printf("pos%d\t\t%.6f\t%.3f\t\t", pos, r2, mae);
        for (int b = 0; b < 3; b++) {
            float pred = loaded_attn->layer_output[offset + b];
            float actual = y_batch[offset + b];
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