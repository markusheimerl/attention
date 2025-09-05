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
    const int seq_len = 128;
    const int d_model = 64;
    const int num_samples = 1024;  // Total number of sequences
    const int batch_size = 32;
    
    // Generate all data upfront
    float *X, *y;
    generate_attention_data(&X, &y, seq_len, num_samples, d_model, -5.0f, 5.0f);
    
    // Initialize attention layer
    Attention* attn = init_attention(seq_len, d_model, batch_size);
    
    // Training parameters
    const int num_epochs = 50;
    const float learning_rate = 0.001f;
    const int num_batches = num_samples / batch_size;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Index into the pre-generated data
            int batch_offset = batch * batch_size * d_model * seq_len;
            
            // Forward pass
            forward_pass_attention(attn, &X[batch_offset]);
            
            // Calculate loss
            float loss = calculate_loss_attention(attn, &y[batch_offset]);
            epoch_loss += loss;
            
            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;
            
            // Backward pass
            zero_gradients_attention(attn);
            backward_pass_attention(attn, &X[batch_offset], NULL);
            
            // Update weights
            update_weights_attention(attn, learning_rate);
        }
        
        epoch_loss /= num_batches;
        
        // Print progress
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
        }
    }
    
    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_attention.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_attention_data.csv", localtime(&now));
    
    // Save model and data
    save_attention(attn, model_fname);
    save_data(X, y, seq_len, num_samples, d_model, data_fname);
    
    // Load model back and verify
    printf("\nVerifying saved model...\n");
    Attention* loaded_attn = load_attention(model_fname, batch_size);
    
    // Forward pass with loaded model on first batch
    forward_pass_attention(loaded_attn, X);
    
    // Evaluate model performance on first batch
    printf("Test Performance (First Batch):\n");
    printf("-------------------------------\n");
    
    int batch_elements = seq_len * d_model * batch_size;
    
    // Calculate overall MSE and MAE
    float mse = 0.0f, mae = 0.0f;
    for (int i = 0; i < batch_elements; i++) {
        float diff = loaded_attn->output[i] - y[i];
        mse += diff * diff;
        mae += fabs(diff);
    }
    mse /= batch_elements;
    mae /= batch_elements;
    
    // Calculate R²
    float y_mean = 0.0f;
    for (int i = 0; i < batch_elements; i++) {
        y_mean += y[i];
    }
    y_mean /= batch_elements;
    
    float ss_res = 0.0f, ss_tot = 0.0f;
    for (int i = 0; i < batch_elements; i++) {
        float diff = loaded_attn->output[i] - y[i];
        ss_res += diff * diff;
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    float r2 = 1.0f - (ss_res / ss_tot);
    
    printf("  MSE: %.8f\n", mse);
    printf("  MAE: %.6f\n", mae);
    printf("  R²:  %.6f\n", r2);
    
    // Print sample predictions for different positions and dimensions
    printf("\nSample predictions (various positions/dimensions, batch 0):\n");
    printf("Pos\tDim\tPredicted\tActual\t\tError\n");
    printf("---\t---\t---------\t------\t\t-----\n");
    
    for (int pos = 0; pos < 5; pos++) {
        for (int dim = 0; dim < 2; dim++) {
            // Index: position + seq_len * (dimension * batch_size + batch_idx)
            int idx = pos + seq_len * (dim * batch_size + 0);
            float pred = loaded_attn->output[idx];
            float actual = y[idx];
            printf("%d\t%d\t%.4f\t\t%.4f\t\t%.4f\n", 
                   pos, dim, pred, actual, fabs(pred - actual));
        }
    }
    
    // Clean up
    free(X);
    free(y);
    free_attention(attn);
    free_attention(loaded_attn);
    
    return 0;
}