#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"

int main() {
    srand(time(NULL));

    // Parameters
    const int num_samples = 1000;
    const int seq_len = 4;      // Sequence length (number of rows)
    const int feature_dim = 3;   // Feature dimension (number of columns)
    
    printf("Attention Data Generation\n");
    printf("========================\n");
    printf("Task: Find row with maximum value in first column, output that row for all positions\n\n");
    
    // Generate attention task data
    float *X, *y;
    generate_attention_data(&X, &y, num_samples, seq_len, feature_dim);
    
    // Print some sample data for inspection
    printf("Sample data for inspection:\n");
    printf("============================\n");
    for (int i = 0; i < 5; i++) {
        print_sample_data(X, y, i, seq_len, feature_dim);
    }
    
    // Get timestamp for filename
    char data_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_attention_data.csv", localtime(&now));
    
    // Save data with timestamped filename
    save_data(X, y, num_samples, seq_len, feature_dim, data_fname);
    
    // Calculate and print some statistics
    printf("\nData Statistics:\n");
    printf("================\n");
    
    // Calculate input data range
    float min_val = X[0], max_val = X[0];
    for (int i = 0; i < num_samples * seq_len * feature_dim; i++) {
        if (X[i] < min_val) min_val = X[i];
        if (X[i] > max_val) max_val = X[i];
    }
    printf("Input data range: [%.3f, %.3f]\n", min_val, max_val);
    
    // Verify data correctness for a few samples
    int correct_samples = 0;
    for (int sample = 0; sample < num_samples; sample++) {
        // Find max row in input
        int max_row = 0;
        float max_val_sample = X[sample * seq_len * feature_dim + 0];
        for (int seq = 1; seq < seq_len; seq++) {
            float current_val = X[sample * seq_len * feature_dim + seq * feature_dim + 0];
            if (current_val > max_val_sample) {
                max_val_sample = current_val;
                max_row = seq;
            }
        }
        
        // Check if output matches expected pattern
        int sample_correct = 1;
        for (int seq = 0; seq < seq_len && sample_correct; seq++) {
            for (int feat = 0; feat < feature_dim && sample_correct; feat++) {
                float expected = X[sample * seq_len * feature_dim + max_row * feature_dim + feat];
                float actual = y[sample * seq_len * feature_dim + seq * feature_dim + feat];
                if (fabsf(expected - actual) > 1e-6) {
                    sample_correct = 0;
                }
            }
        }
        
        if (sample_correct) correct_samples++;
    }
    
    printf("Data verification: %d/%d samples correct (%.1f%%)\n", 
           correct_samples, num_samples, (100.0f * correct_samples) / num_samples);
    
    printf("\nData generation completed successfully!\n");
    printf("Total data points: %d samples × %d sequence positions × %d features = %d input values\n",
           num_samples, seq_len, feature_dim, num_samples * seq_len * feature_dim);
    
    // Cleanup
    free(X);
    free(y);
    
    return 0;
}