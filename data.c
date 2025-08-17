#include "data.h"

void generate_attention_data(float** X, float** y, int num_samples, int seq_len, int feature_dim) {
    // Allocate memory
    *X = (float*)malloc(num_samples * seq_len * feature_dim * sizeof(float));
    *y = (float*)malloc(num_samples * seq_len * feature_dim * sizeof(float));
    
    printf("Generating attention task data...\n");
    printf("Task: Find row with max value in column 0, output that row for all positions\n");
    printf("Input shape: [%d samples, %d sequence length, %d features]\n", num_samples, seq_len, feature_dim);
    printf("Output shape: [%d samples, %d sequence length, %d features]\n\n", num_samples, seq_len, feature_dim);
    
    for (int sample = 0; sample < num_samples; sample++) {
        // Generate random input sequence
        for (int seq = 0; seq < seq_len; seq++) {
            for (int feat = 0; feat < feature_dim; feat++) {
                float rand_val = (float)rand() / (float)RAND_MAX;
                // Generate values between -5.0 and 10.0 to ensure variety
                (*X)[sample * seq_len * feature_dim + seq * feature_dim + feat] = -5.0f + rand_val * 15.0f;
            }
        }
        
        // Find the row with maximum value in column 0 (feature 0)
        int max_row = 0;
        float max_val = (*X)[sample * seq_len * feature_dim + 0 * feature_dim + 0];
        
        for (int seq = 1; seq < seq_len; seq++) {
            float current_val = (*X)[sample * seq_len * feature_dim + seq * feature_dim + 0];
            if (current_val > max_val) {
                max_val = current_val;
                max_row = seq;
            }
        }
        
        // Set output: all rows should be copies of the max row
        for (int seq = 0; seq < seq_len; seq++) {
            for (int feat = 0; feat < feature_dim; feat++) {
                // Copy the max row to all output positions
                float max_row_value = (*X)[sample * seq_len * feature_dim + max_row * feature_dim + feat];
                (*y)[sample * seq_len * feature_dim + seq * feature_dim + feat] = max_row_value;
            }
        }
    }
    
    printf("Data generation completed.\n\n");
}

void save_data(float* X, float* y, int num_samples, int seq_len, int feature_dim, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write header
    fprintf(file, "sample,seq_pos,");
    for (int i = 0; i < feature_dim; i++) {
        fprintf(file, "x%d,", i);
    }
    for (int i = 0; i < feature_dim - 1; i++) {
        fprintf(file, "y%d,", i);
    }
    fprintf(file, "y%d\n", feature_dim - 1);
    
    // Write data
    for (int sample = 0; sample < num_samples; sample++) {
        for (int seq = 0; seq < seq_len; seq++) {
            fprintf(file, "%d,%d,", sample, seq);
            
            // Input features
            for (int feat = 0; feat < feature_dim; feat++) {
                fprintf(file, "%.6f,", X[sample * seq_len * feature_dim + seq * feature_dim + feat]);
            }
            
            // Output values
            for (int feat = 0; feat < feature_dim - 1; feat++) {
                fprintf(file, "%.6f,", y[sample * seq_len * feature_dim + seq * feature_dim + feat]);
            }
            fprintf(file, "%.6f\n", y[sample * seq_len * feature_dim + seq * feature_dim + (feature_dim - 1)]);
        }
    }
    
    fclose(file);
    printf("Data saved to %s\n", filename);
}

void print_sample_data(float* X, float* y, int sample_idx, int seq_len, int feature_dim) {
    printf("Sample %d:\n", sample_idx);
    printf("Input:\n");
    
    for (int seq = 0; seq < seq_len; seq++) {
        printf("  [");
        for (int feat = 0; feat < feature_dim; feat++) {
            printf("%6.2f", X[sample_idx * seq_len * feature_dim + seq * feature_dim + feat]);
            if (feat < feature_dim - 1) printf(", ");
        }
        printf("]\n");
    }
    
    // Find which row had the maximum in column 0
    int max_row = 0;
    float max_val = X[sample_idx * seq_len * feature_dim + 0 * feature_dim + 0];
    for (int seq = 1; seq < seq_len; seq++) {
        float current_val = X[sample_idx * seq_len * feature_dim + seq * feature_dim + 0];
        if (current_val > max_val) {
            max_val = current_val;
            max_row = seq;
        }
    }
    
    printf("Max value %.2f found in row %d (column 0)\n", max_val, max_row);
    printf("Expected Output (row %d repeated):\n", max_row);
    
    for (int seq = 0; seq < seq_len; seq++) {
        printf("  [");
        for (int feat = 0; feat < feature_dim; feat++) {
            printf("%6.2f", y[sample_idx * seq_len * feature_dim + seq * feature_dim + feat]);
            if (feat < feature_dim - 1) printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}