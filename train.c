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
    const int seq_len = 16;          // Sequence length
    const int feature_dim = 8;       // Feature dimension (d_model)
    const int num_samples = 65536;   // Number of training samples
    const int batch_size = 512;      // Batch size for training
    
    // Generate attention task data
    float *X, *y;
    generate_attention_data(&X, &y, num_samples, seq_len, feature_dim);
    
    // Print some sample data for inspection
    printf("Sample data for inspection:\n");
    printf("============================\n");
    for (int i = 0; i < 3; i++) {
        print_sample_data(X, y, i, seq_len, feature_dim);
    }
    
    // Initialize attention network
    Attention* attn = init_attention(feature_dim, seq_len, batch_size);
    
    // Training parameters
    const int num_epochs = 50;
    const float learning_rate = 0.001f;
    const int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    printf("Starting training...\n");
    printf("Architecture: d_model=%d, seq_len=%d, batch_size=%d, num_samples=%d, num_batches=%d\n\n", 
           feature_dim, seq_len, batch_size, num_samples, num_batches);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float total_loss = 0.0f;
        
        // Process each batch
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = (start_idx + batch_size > num_samples) ? num_samples : start_idx + batch_size;
            int current_batch_size = end_idx - start_idx;
            
            // Skip incomplete batches for simplicity
            if (current_batch_size < batch_size) continue;
            
            // Extract batch data
            float* X_batch = X + start_idx * seq_len * feature_dim;
            float* y_batch = y + start_idx * seq_len * feature_dim;
            
            // Forward pass
            forward_pass_attention(attn, X_batch);
            
            // Calculate loss
            float loss = calculate_loss_attention(attn, y_batch);
            total_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_attention(attn);
            backward_pass_attention(attn, X_batch);
            
            // Update weights
            update_weights_attention(attn, learning_rate);
        }

        // Print progress
        if (epoch % 2 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.8f\n", epoch, num_epochs, total_loss / num_batches);
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
    
    // Evaluate on a sample batch for verification
    forward_pass_attention(loaded_attn, X);
    float verification_loss = calculate_loss_attention(loaded_attn, y);
    printf("Loss with loaded model (sample batch): %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Calculate accuracy for attention task on all data
    int correct_predictions = 0;
    int total_predictions = 0;
    
    // Process evaluation in batches
    for (int batch = 0; batch < num_batches; batch++) {
        int start_idx = batch * batch_size;
        int end_idx = (start_idx + batch_size > num_samples) ? num_samples : start_idx + batch_size;
        int current_batch_size = end_idx - start_idx;
        
        // Skip incomplete batches for simplicity
        if (current_batch_size < batch_size) continue;
        
        // Extract batch data
        float* X_batch = X + start_idx * seq_len * feature_dim;
        
        // Forward pass
        forward_pass_attention(loaded_attn, X_batch);
        
        // Evaluate this batch
        for (int sample = 0; sample < current_batch_size; sample++) {
            int global_sample = start_idx + sample;
            
            // Find the expected max row from input
            int expected_max_row = 0;
            float max_val = X[global_sample * seq_len * feature_dim + 0];
            for (int seq = 1; seq < seq_len; seq++) {
                float current_val = X[global_sample * seq_len * feature_dim + seq * feature_dim + 0];
                if (current_val > max_val) {
                    max_val = current_val;
                    expected_max_row = seq;
                }
            }
            
            // Check if model output matches expected pattern
            int sample_correct = 1;
            float tolerance = 0.5f;
            
            for (int seq = 0; seq < seq_len && sample_correct; seq++) {
                for (int feat = 0; feat < feature_dim && sample_correct; feat++) {
                    float predicted = loaded_attn->layer_output[sample * seq_len * feature_dim + seq * feature_dim + feat];
                    float expected = X[global_sample * seq_len * feature_dim + expected_max_row * feature_dim + feat];
                    
                    if (fabsf(predicted - expected) > tolerance) {
                        sample_correct = 0;
                    }
                }
            }
            
            if (sample_correct) {
                correct_predictions++;
            }
            total_predictions++;
        }
    }
    
    printf("Attention Task Accuracy: %d/%d (%.1f%%)\n", 
           correct_predictions, total_predictions, 
           (100.0f * correct_predictions) / total_predictions);

    // Print sample predictions (first batch)
    printf("\nSample Predictions (first 5 samples):\n");
    printf("=====================================\n");

    forward_pass_attention(loaded_attn, X);
    
    for (int sample = 0; sample < 5; sample++) {
        printf("\nSample %d:\n", sample);
        printf("Input:\n");
        
        // Print input
        for (int seq = 0; seq < seq_len; seq++) {
            printf("  [");
            for (int feat = 0; feat < feature_dim; feat++) {
                printf("%6.2f", X[sample * seq_len * feature_dim + seq * feature_dim + feat]);
                if (feat < feature_dim - 1) printf(", ");
            }
            printf("]\n");
        }
        
        // Find expected max row
        int expected_max_row = 0;
        float max_val = X[sample * seq_len * feature_dim + 0];
        for (int seq = 1; seq < seq_len; seq++) {
            float current_val = X[sample * seq_len * feature_dim + seq * feature_dim + 0];
            if (current_val > max_val) {
                max_val = current_val;
                expected_max_row = seq;
            }
        }
        
        printf("Expected max row: %d (value: %.2f)\n", expected_max_row, max_val);
        printf("Model Output:\n");
        
        // Print model output
        for (int seq = 0; seq < seq_len; seq++) {
            printf("  [");
            for (int feat = 0; feat < feature_dim; feat++) {
                float pred = loaded_attn->layer_output[sample * seq_len * feature_dim + seq * feature_dim + feat];
                printf("%6.2f", pred);
                if (feat < feature_dim - 1) printf(", ");
            }
            printf("]\n");
        }
        
        printf("Target Output:\n");
        
        // Print target output
        for (int seq = 0; seq < seq_len; seq++) {
            printf("  [");
            for (int feat = 0; feat < feature_dim; feat++) {
                printf("%6.2f", y[sample * seq_len * feature_dim + seq * feature_dim + feat]);
                if (feat < feature_dim - 1) printf(", ");
            }
            printf("]\n");
        }
    }
    
    // Calculate MSE per feature on first batch
    printf("\nMSE per feature (first batch):\n");
    for (int feat = 0; feat < feature_dim; feat++) {
        float mse = 0.0f;
        for (int sample = 0; sample < batch_size; sample++) {
            for (int seq = 0; seq < seq_len; seq++) {
                float pred = loaded_attn->layer_output[sample * seq_len * feature_dim + seq * feature_dim + feat];
                float actual = y[sample * seq_len * feature_dim + seq * feature_dim + feat];
                float diff = pred - actual;
                mse += diff * diff;
            }
        }
        mse /= (batch_size * seq_len);
        printf("Feature %d MSE: %.6f\n", feat, mse);
    }
    
    // Cleanup
    free(X);
    free(y);
    free_attention(attn);
    free_attention(loaded_attn);
    
    return 0;
}