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
    const int feature_dim = 16;
    const int seq_len = 8;
    const int num_sequences = 128;
    const int num_layers = 2;
    const int batch_size = num_sequences;
    
    // Generate synthetic sequence data
    float *X, *y;
    generate_synthetic_data(&X, &y, num_sequences, seq_len, feature_dim, feature_dim, -3.0f, 3.0f);
    
    // Initialize attention network
    Attention* attn = init_attention(feature_dim, seq_len, num_layers, batch_size);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.0001f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Forward pass
        forward_pass_attention(attn, X);
        
        // Calculate loss
        float loss = calculate_loss_attention(attn, y);

        // Print progress
        if (epoch > 0 && epoch % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, loss);
        }

        // Don't update weights after final evaluation
        if (epoch == num_epochs) break;

        // Backward pass
        zero_gradients_attention(attn);
        backward_pass_attention(attn, X);
        
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
    save_data(X, y, num_sequences, seq_len, feature_dim, feature_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    Attention* loaded_attn = load_attention(model_fname, batch_size);
    
    // Forward pass with loaded model
    forward_pass_attention(loaded_attn, X);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss_attention(loaded_attn, y);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Calculate R² scores
    printf("\nR² scores:\n");
    int last_layer = loaded_attn->num_layers - 1;
    int total_samples = num_sequences * seq_len;
    for (int i = 0; i < feature_dim; i++) {
        float y_mean = 0.0f;
        for (int seq = 0; seq < num_sequences; seq++) {
            for (int t = 0; t < seq_len; t++) {
                int idx = seq * seq_len * feature_dim + t * feature_dim + i;
                y_mean += y[idx];
            }
        }
        y_mean /= total_samples;

        float ss_res = 0.0f;
        float ss_tot = 0.0f;
        for (int seq = 0; seq < num_sequences; seq++) {
            for (int t = 0; t < seq_len; t++) {
                int idx = seq * seq_len * feature_dim + t * feature_dim + i;
                float diff_res = y[idx] - loaded_attn->layer_output[last_layer][idx];
                float diff_tot = y[idx] - y_mean;
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

    for (int i = 0; i < feature_dim; i++) {
        printf("\ny%d:\n", i);
        int seq = 0; // First sequence
        for (int t = 0; t < seq_len && t < 10; t++) {
            int idx = seq * seq_len * feature_dim + t * feature_dim + i;
            float pred = loaded_attn->layer_output[last_layer][idx];
            float actual = y[idx];
            float diff = pred - actual;
            printf("t=%d\t\t%8.3f\t%8.3f\t%8.3f\n", t, pred, actual, diff);
        }
        
        // Calculate MAE for this output across all sequences and time steps
        float mae = 0.0f;
        for (int seq = 0; seq < num_sequences; seq++) {
            for (int t = 0; t < seq_len; t++) {
                int idx = seq * seq_len * feature_dim + t * feature_dim + i;
                mae += fabs(loaded_attn->layer_output[last_layer][idx] - y[idx]);
            }
        }
        mae /= total_samples;
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
    }
    
    // Cleanup
    free(X);
    free(y);
    free_attention(attn);
    free_attention(loaded_attn);
    
    return 0;
}