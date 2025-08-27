#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "data.h"
#include "attention.h"

// Reshape data from [batch][time][feature] to [time][batch][feature]
void reshape_data_for_batch_processing(float* X, float* y, float** X_reshaped, float** y_reshaped, int num_sequences, int seq_len, int input_dim, int output_dim) {
    *X_reshaped = (float*)malloc(seq_len * num_sequences * input_dim * sizeof(float));
    *y_reshaped = (float*)malloc(seq_len * num_sequences * output_dim * sizeof(float));
    
    for (int t = 0; t < seq_len; t++) {
        for (int b = 0; b < num_sequences; b++) {
            memcpy(&(*X_reshaped)[t * num_sequences * input_dim + b * input_dim], &X[b * seq_len * input_dim + t * input_dim], input_dim * sizeof(float));
            memcpy(&(*y_reshaped)[t * num_sequences * output_dim + b * output_dim], &y[b * seq_len * output_dim + t * output_dim], output_dim * sizeof(float));
        }
    }
}

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters
    const int input_dim = 16;
    const int hidden_dim = 128;
    const int output_dim = 4;
    const int seq_len = 8;
    const int num_sequences = 128;
    const int num_layers = 2;
    const int batch_size = num_sequences;
    
    // Generate synthetic sequence data with bidirectional temporal dependencies
    float *X, *y;
    generate_data(&X, &y, num_sequences, seq_len, input_dim, output_dim, -3.0f, 3.0f, -seq_len);
    
    // Reshape data for batch processing
    float *X_reshaped, *y_reshaped;
    reshape_data_for_batch_processing(X, y, &X_reshaped, &y_reshaped, num_sequences, seq_len, input_dim, output_dim);
    
    // Initialize attention model
    Attention* attn = init_attention(input_dim, hidden_dim, output_dim, seq_len, batch_size, num_layers);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.0001f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Forward pass
        forward_pass_attention(attn, X_reshaped);
        
        // Calculate loss
        float loss = calculate_loss_attention(attn, y_reshaped);

        // Print progress
        if (epoch > 0 && epoch % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, loss);
        }

        // Don't update weights after final evaluation
        if (epoch == num_epochs) break;

        // Backward pass
        zero_gradients_attention(attn);
        backward_pass_attention(attn, X_reshaped);
        
        // Update weights
        update_weights_attention(attn, learning_rate);
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));

    // Save model and data with timestamped filenames
    save_attention(attn, model_fname);
    save_data(X, y, num_sequences, seq_len, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    Attention* loaded_attn = load_attention(model_fname, batch_size);
    
    // Forward pass with loaded model
    forward_pass_attention(loaded_attn, X_reshaped);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss_attention(loaded_attn, y_reshaped);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Calculate R² scores
    printf("\nR² scores:\n");
    int last_layer = loaded_attn->num_layers - 1;
    int total_samples = num_sequences * seq_len;
    for (int i = 0; i < output_dim; i++) {
        float y_mean = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            for (int b = 0; b < num_sequences; b++) {
                int idx = t * num_sequences * output_dim + b * output_dim + i;
                y_mean += y_reshaped[idx];
            }
        }
        y_mean /= total_samples;

        float ss_res = 0.0f;
        float ss_tot = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            for (int b = 0; b < num_sequences; b++) {
                int idx = t * num_sequences * output_dim + b * output_dim + i;
                float diff_res = y_reshaped[idx] - loaded_attn->layer_output[last_layer][idx];
                float diff_tot = y_reshaped[idx] - y_mean;
                ss_res += diff_res * diff_res;
                ss_tot += diff_tot * diff_tot;
            }
        }
        float r2 = 1.0f - (ss_res / ss_tot);
        printf("R² score for output y%d: %.8f\n", i, r2);
    }

    // Print sample predictions from first sequence
    printf("\nSample Predictions (first sequence, first 10 time steps):\n");
    printf("Time\tOutput\t\tPredicted\tActual\t\tDifference\n");
    printf("----------------------------------------------------------------\n");

    for (int i = 0; i < output_dim; i++) {
        printf("\ny%d:\n", i);
        for (int t = 0; t < 10; t++) {
            // First sequence (b=0) in reshaped format
            int idx = t * num_sequences * output_dim + 0 * output_dim + i;
            float pred = loaded_attn->layer_output[last_layer][idx];
            float actual = y_reshaped[idx];
            float diff = pred - actual;
            printf("t=%d\t\t%8.3f\t%8.3f\t%8.3f\n", t, pred, actual, diff);
        }
        
        // Calculate MAE for this output across all sequences and time steps
        float mae = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            for (int b = 0; b < num_sequences; b++) {
                int idx = t * num_sequences * output_dim + b * output_dim + i;
                mae += fabs(loaded_attn->layer_output[last_layer][idx] - y_reshaped[idx]);
            }
        }
        mae /= total_samples;
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
    }
    
    // Cleanup
    free(X);
    free(y);
    free(X_reshaped);
    free(y_reshaped);
    free_attention(attn);
    free_attention(loaded_attn);
    
    return 0;
}