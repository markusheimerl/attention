#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "data.h"
#include "attention.h"

#define min(a,b) ((a)<(b)?(a):(b))

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters
    const int d_model = 64;      // Model dimension
    const int seq_len = 8;       // Sequence length
    const int batch_size = 512;   // Batch size
    const int num_samples = 4096; // Total samples
    
    // For attention, we need 3D input: [d_model x seq_len x batch_size]
    int input_dim = d_model * seq_len;
    int output_dim = d_model * seq_len;
    
    // Generate synthetic data
    float *X, *y;
    generate_data(&X, &y, num_samples, input_dim, output_dim, -3.0f, 3.0f);
    
    // Initialize attention module
    Attention* attn = init_attention(d_model, seq_len, batch_size);
    
    // Training parameters
    const int num_epochs = 5000;
    const float learning_rate = 0.001f;
    const int num_batches = num_samples / batch_size;
    
    // Allocate batch buffers
    float* X_batch = (float*)malloc(batch_size * input_dim * sizeof(float));
    float* y_batch = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Reshape buffers for 3D data: [d_model x seq_len x batch_size]
    float* X_reshaped = (float*)malloc(d_model * seq_len * batch_size * sizeof(float));
    float* y_reshaped = (float*)malloc(d_model * seq_len * batch_size * sizeof(float));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            
            // Copy batch data
            for (int feature = 0; feature < input_dim; feature++) {
                memcpy(&X_batch[feature * batch_size], 
                       &X[feature * num_samples + start_idx], 
                       batch_size * sizeof(float));
            }
            
            for (int out = 0; out < output_dim; out++) {
                memcpy(&y_batch[out * batch_size],
                       &y[out * num_samples + start_idx],
                       batch_size * sizeof(float));
            }
            
            // Reshape from [input_dim x batch_size] to [d_model x seq_len x batch_size]
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < seq_len; s++) {
                    for (int d = 0; d < d_model; d++) {
                        int src_idx = (s * d_model + d) * batch_size + b;
                        int dst_idx = b * d_model * seq_len + s * d_model + d;
                        X_reshaped[dst_idx] = X_batch[src_idx];
                        y_reshaped[dst_idx] = y_batch[src_idx];
                    }
                }
            }
            
            // Forward pass
            forward_pass_attention(attn, X_reshaped);
            
            // Calculate loss
            float loss = calculate_loss_attention(attn, y_reshaped);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_attention(attn);
            backward_pass_attention(attn, X_reshaped, NULL);
            
            // Update weights
            update_weights_attention(attn, learning_rate);
        }
        
        epoch_loss /= num_batches;

        // Print progress
        if (epoch % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_attention.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));

    // Save model and data with timestamped filenames
    save_attention(attn, model_fname);
    save_data(X, y, num_samples, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    Attention* loaded_attn = load_attention(model_fname, batch_size);
    
    // Evaluate on first batch
    for (int feature = 0; feature < input_dim; feature++) {
        memcpy(&X_batch[feature * batch_size], 
               &X[feature * num_samples], 
               batch_size * sizeof(float));
    }
    
    for (int out = 0; out < output_dim; out++) {
        memcpy(&y_batch[out * batch_size],
               &y[out * num_samples],
               batch_size * sizeof(float));
    }
    
    // Reshape for evaluation
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < d_model; d++) {
                int src_idx = (s * d_model + d) * batch_size + b;
                int dst_idx = b * d_model * seq_len + s * d_model + d;
                X_reshaped[dst_idx] = X_batch[src_idx];
                y_reshaped[dst_idx] = y_batch[src_idx];
            }
        }
    }
    
    // Forward pass with loaded model
    forward_pass_attention(loaded_attn, X_reshaped);
    
    // Calculate R² and MAE for a few output dimensions
    printf("Dimension\tR²\t\tMAE\t\tSample Predictions\n");
    printf("---------\t--------\t--------\t--------------------------------\n");

    for (int dim = 0; dim < min(4, output_dim); dim++) {
        // Calculate mean for R²
        float y_mean = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            y_mean += y_batch[dim * batch_size + i];
        }
        y_mean /= batch_size;
        
        // Calculate R² and MAE
        float ss_res = 0.0f, ss_tot = 0.0f, mae = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            // Get prediction from reshaped output
            int out_idx = i * d_model * seq_len + (dim % seq_len) * d_model + (dim / seq_len);
            float pred = loaded_attn->output[out_idx];
            float actual = y_batch[dim * batch_size + i];
            float diff = pred - actual;
            
            ss_res += diff * diff;
            ss_tot += (actual - y_mean) * (actual - y_mean);
            mae += fabs(diff);
        }
        
        float r2 = 1.0f - (ss_res / ss_tot);
        mae /= batch_size;
        
        // Print summary
        printf("dim%d\t\t%.6f\t%.3f\t\t", dim, r2, mae);
        for (int i = 0; i < 3; i++) {
            int out_idx = i * d_model * seq_len + (dim % seq_len) * d_model + (dim / seq_len);
            float pred = loaded_attn->output[out_idx];
            float actual = y_batch[dim * batch_size + i];
            printf("%.2f/%.2f ", pred, actual);
        }
        printf("\n");
    }
    
    // Cleanup
    free(X);
    free(y);
    free(X_batch);
    free(y_batch);
    free(X_reshaped);
    free(y_reshaped);
    free_attention(attn);
    free_attention(loaded_attn);
    
    return 0;
}