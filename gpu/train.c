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
    const int seq_len = 16;          // Sequence length
    const int feature_dim = 4;      // Feature dimension (d_model)
    const int num_layers = 2;       // Number of attention layers
    const int num_samples = 1024;   // Number of training samples
    const int batch_size = num_samples; // Full batch training
    
    // Generate attention task data
    float *X, *y;
    generate_attention_data(&X, &y, num_samples, seq_len, feature_dim);
    
    // Print some sample data for inspection
    printf("Sample data for inspection:\n");
    printf("============================\n");
    for (int i = 0; i < 3; i++) {
        print_sample_data(X, y, i, seq_len, feature_dim);
    }

    // Allocate device memory for input and output and copy data
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, batch_size * seq_len * feature_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, batch_size * seq_len * feature_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, batch_size * seq_len * feature_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, batch_size * seq_len * feature_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize attention network
    Attention* attn = init_attention(feature_dim, seq_len, num_layers, batch_size, cublas_handle);
    
    // Training parameters
    const int num_epochs = 20000;
    const float learning_rate = 0.0003f;
    
    printf("Starting training...\n");
    printf("Architecture: %d layers, d_model=%d, seq_len=%d, batch_size=%d\n\n", 
           num_layers, feature_dim, seq_len, batch_size);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Forward pass
        forward_pass_attention(attn, d_X);
        
        // Calculate loss
        float loss = calculate_loss_attention(attn, d_y);

        // Print progress
        if (epoch % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, loss);
        }

        // Don't update weights after final evaluation
        if (epoch == num_epochs) break;

        // Backward pass
        zero_gradients_attention(attn);
        backward_pass_attention(attn, d_X);
        
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
    save_data(X, y, num_samples, seq_len, feature_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    Attention* loaded_attn = load_attention(model_fname, batch_size, cublas_handle);
    
    // Forward pass with loaded model
    forward_pass_attention(loaded_attn, d_X);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss_attention(loaded_attn, d_y);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Allocate host memory for predictions
    float* predictions = (float*)malloc(num_samples * seq_len * feature_dim * sizeof(float));

    // Copy predictions from device to host (from last layer)
    int last_layer = loaded_attn->num_layers - 1;
    CHECK_CUDA(cudaMemcpy(predictions, loaded_attn->d_layer_output[last_layer], 
                         num_samples * seq_len * feature_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Calculate accuracy for attention task
    int correct_predictions = 0;
    int total_predictions = num_samples;
    
    for (int sample = 0; sample < num_samples; sample++) {
        // Find the expected max row from input
        int expected_max_row = 0;
        float max_val = X[sample * seq_len * feature_dim + 0];
        for (int seq = 1; seq < seq_len; seq++) {
            float current_val = X[sample * seq_len * feature_dim + seq * feature_dim + 0];
            if (current_val > max_val) {
                max_val = current_val;
                expected_max_row = seq;
            }
        }
        
        // Check if model output matches expected pattern (all rows should be the max row)
        int sample_correct = 1;
        float tolerance = 0.5f;  // Allow some tolerance for floating point comparison
        
        for (int seq = 0; seq < seq_len && sample_correct; seq++) {
            for (int feat = 0; feat < feature_dim && sample_correct; feat++) {
                float predicted = predictions[sample * seq_len * feature_dim + seq * feature_dim + feat];
                float expected = X[sample * seq_len * feature_dim + expected_max_row * feature_dim + feat];
                
                if (fabsf(predicted - expected) > tolerance) {
                    sample_correct = 0;
                }
            }
        }
        
        if (sample_correct) {
            correct_predictions++;
        }
    }
    
    printf("Attention Task Accuracy: %d/%d (%.1f%%)\n", 
           correct_predictions, total_predictions, 
           (100.0f * correct_predictions) / total_predictions);

    // Print sample predictions
    printf("\nSample Predictions (first 5 samples):\n");
    printf("=====================================\n");

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
                float pred = predictions[sample * seq_len * feature_dim + seq * feature_dim + feat];
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
    
    // Calculate MSE per feature
    printf("\nMSE per feature:\n");
    for (int feat = 0; feat < feature_dim; feat++) {
        float mse = 0.0f;
        for (int sample = 0; sample < num_samples; sample++) {
            for (int seq = 0; seq < seq_len; seq++) {
                float pred = predictions[sample * seq_len * feature_dim + seq * feature_dim + feat];
                float actual = y[sample * seq_len * feature_dim + seq * feature_dim + feat];
                float diff = pred - actual;
                mse += diff * diff;
            }
        }
        mse /= (num_samples * seq_len);
        printf("Feature %d MSE: %.6f\n", feat, mse);
    }
    
    // Cleanup
    free(X);
    free(y);
    free(predictions);
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_y));
    free_attention(attn);
    free_attention(loaded_attn);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    return 0;
}