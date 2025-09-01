#include "data.h"

void generate_data(float** X, float** y, int num_samples, int seq_len, int feature_dim) {
    *X = (float*)malloc(num_samples * seq_len * feature_dim * sizeof(float));
    *y = (float*)malloc(num_samples * seq_len * feature_dim * sizeof(float));
    
    printf("Generating attention task data: [%d, %d, %d]\n", num_samples, seq_len, feature_dim);
    
    for (int sample = 0; sample < num_samples; sample++) {
        int base = sample * seq_len * feature_dim;
        
        // Generate random input and find max row
        int max_row = 0;
        float max_val = (*X)[base] = -5.0f + ((float)rand() / (float)RAND_MAX) * 15.0f;
        
        for (int i = 1; i < seq_len * feature_dim; i++) {
            (*X)[base + i] = -5.0f + ((float)rand() / (float)RAND_MAX) * 15.0f;
            if (i % feature_dim == 0 && (*X)[base + i] > max_val) {
                max_val = (*X)[base + i];
                max_row = i / feature_dim;
            }
        }
        
        // Copy max row to all output positions
        for (int seq = 0; seq < seq_len; seq++) {
            for (int feat = 0; feat < feature_dim; feat++) {
                (*y)[base + seq * feature_dim + feat] = (*X)[base + max_row * feature_dim + feat];
            }
        }
    }
    printf("Data generation completed.\n");
}

void save_data(float* X, float* y, int num_samples, int seq_len, int feature_dim, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    
    // Header
    fprintf(file, "sample,seq_pos,");
    for (int i = 0; i < feature_dim; i++) fprintf(file, "x%d,", i);
    for (int i = 0; i < feature_dim - 1; i++) fprintf(file, "y%d,", i);
    fprintf(file, "y%d\n", feature_dim - 1);
    
    // Data
    for (int sample = 0; sample < num_samples; sample++) {
        for (int seq = 0; seq < seq_len; seq++) {
            int idx = sample * seq_len * feature_dim + seq * feature_dim;
            fprintf(file, "%d,%d,", sample, seq);
            
            for (int feat = 0; feat < feature_dim; feat++) {
                fprintf(file, "%.6f,", X[idx + feat]);
            }
            for (int feat = 0; feat < feature_dim - 1; feat++) {
                fprintf(file, "%.6f,", y[idx + feat]);
            }
            fprintf(file, "%.6f\n", y[idx + feature_dim - 1]);
        }
    }
    
    fclose(file);
    printf("Data saved to %s\n", filename);
}