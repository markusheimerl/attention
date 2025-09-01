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
    
    // Print sample data for inspection
    printf("Sample data for inspection:\n");
    printf("============================\n");
    for (int i = 0; i < 3; i++) {
        print_sample_data(X, y, i, seq_len, feature_dim);
    }
    
    // Initialize network
    Attention* attn = init_attention(feature_dim, seq_len, batch_size, false);
    
    // Training parameters
    const int num_epochs = 50;
    const float learning_rate = 0.001f;
    const int num_batches = num_samples / batch_size;
    
    printf("Starting training...\n");
    printf("Architecture: d_model=%d, seq_len=%d, batch_size=%d, num_samples=%d, num_batches=%d\n\n", 
           attn->d_model, attn->seq_len, batch_size, num_samples, num_batches);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            
            float* X_batch = X + start_idx * seq_len * feature_dim;
            float* y_batch = y + start_idx * seq_len * feature_dim;
            
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
        if (epoch % 2 == 0) {
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
    float* X_batch = X;
    float* y_batch = y;
    
    // Forward pass with loaded model
    forward_pass_attention(loaded_attn, X_batch);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss_attention(loaded_attn, y_batch);
    printf("Loss with loaded model (first batch): %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Generate new evaluation dataset
    printf("Generating new evaluation dataset...\n");
    const int eval_samples = 2048;
    float *X_eval, *y_eval;
    generate_attention_data(&X_eval, &y_eval, eval_samples, seq_len, feature_dim);
    
    // Evaluate accuracy on new data
    const int eval_batches = eval_samples / batch_size;
    int correct_predictions = 0, total_predictions = 0;
    
    for (int batch = 0; batch < eval_batches; batch++) {
        int start_idx = batch * batch_size;
        float* X_eval_batch = X_eval + start_idx * seq_len * feature_dim;
        
        forward_pass_attention(loaded_attn, X_eval_batch);
        
        for (int sample = 0; sample < batch_size; sample++) {
            int global_sample = start_idx + sample;
            
            // Find expected max row
            int expected_max_row = 0;
            float max_val = X_eval[global_sample * seq_len * feature_dim + 0];
            for (int seq = 1; seq < seq_len; seq++) {
                float val = X_eval[global_sample * seq_len * feature_dim + seq * feature_dim + 0];
                if (val > max_val) {
                    max_val = val;
                    expected_max_row = seq;
                }
            }
            
            // Check prediction accuracy
            int sample_correct = 1;
            float tolerance = 0.5f;
            
            for (int seq = 0; seq < seq_len && sample_correct; seq++) {
                for (int feat = 0; feat < feature_dim && sample_correct; feat++) {
                    float predicted = loaded_attn->layer_output[sample * seq_len * feature_dim + seq * feature_dim + feat];
                    float expected = X_eval[global_sample * seq_len * feature_dim + expected_max_row * feature_dim + feat];
                    
                    if (fabsf(predicted - expected) > tolerance) {
                        sample_correct = 0;
                    }
                }
            }
            
            if (sample_correct) correct_predictions++;
            total_predictions++;
        }
    }
    
    printf("Attention Task Accuracy on NEW data: %d/%d (%.1f%%)\n", 
           correct_predictions, total_predictions, 
           (100.0f * correct_predictions) / total_predictions);

    // Print sample predictions
    printf("\nSample Predictions from NEW evaluation data (first 5 samples):\n");
    printf("=============================================================\n");

    forward_pass_attention(loaded_attn, X_eval);
    
    for (int sample = 0; sample < 5; sample++) {
        printf("\nSample %d:\n", sample);
        printf("Input:\n");
        
        for (int seq = 0; seq < seq_len; seq++) {
            printf("  [");
            for (int feat = 0; feat < feature_dim; feat++) {
                printf("%6.2f", X_eval[sample * seq_len * feature_dim + seq * feature_dim + feat]);
                if (feat < feature_dim - 1) printf(", ");
            }
            printf("]\n");
        }
        
        // Find expected max row
        int expected_max_row = 0;
        float max_val = X_eval[sample * seq_len * feature_dim + 0];
        for (int seq = 1; seq < seq_len; seq++) {
            float val = X_eval[sample * seq_len * feature_dim + seq * feature_dim + 0];
            if (val > max_val) {
                max_val = val;
                expected_max_row = seq;
            }
        }
        
        printf("Expected max row: %d (value: %.2f)\n", expected_max_row, max_val);
        printf("Model Output:\n");
        
        for (int seq = 0; seq < seq_len; seq++) {
            printf("  [");
            for (int feat = 0; feat < feature_dim; feat++) {
                printf("%6.2f", loaded_attn->layer_output[sample * seq_len * feature_dim + seq * feature_dim + feat]);
                if (feat < feature_dim - 1) printf(", ");
            }
            printf("]\n");
        }
        
        printf("Target Output:\n");
        
        for (int seq = 0; seq < seq_len; seq++) {
            printf("  [");
            for (int feat = 0; feat < feature_dim; feat++) {
                printf("%6.2f", y_eval[sample * seq_len * feature_dim + seq * feature_dim + feat]);
                if (feat < feature_dim - 1) printf(", ");
            }
            printf("]\n");
        }
    }
    
    // Calculate MSE per feature
    printf("\nMSE per feature (first evaluation batch):\n");
    for (int feat = 0; feat < feature_dim; feat++) {
        float mse = 0.0f;
        for (int sample = 0; sample < batch_size; sample++) {
            for (int seq = 0; seq < seq_len; seq++) {
                float pred = loaded_attn->layer_output[sample * seq_len * feature_dim + seq * feature_dim + feat];
                float actual = y_eval[sample * seq_len * feature_dim + seq * feature_dim + feat];
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
    free(X_eval);
    free(y_eval);
    free_attention(attn);
    free_attention(loaded_attn);
    
    return 0;
}