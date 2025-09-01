#include "data.h"

void generate_sequence_data(float** X, float** y, int num_samples, int seq_len, int d_model, float range_min, float range_max) {
    int total_x = num_samples * seq_len * d_model;
    int total_y = num_samples * seq_len * d_model;
    
    *X = (float*)malloc(total_x * sizeof(float));
    *y = (float*)malloc(total_y * sizeof(float));
    
    // Generate sequence data in format: [d_model × seq_len × num_samples]
    for (int sample = 0; sample < num_samples; sample++) {
        for (int pos = 0; pos < seq_len; pos++) {
            for (int dim = 0; dim < d_model; dim++) {
                int idx = dim * (seq_len * num_samples) + pos * num_samples + sample;
                (*X)[idx] = range_min + 
                    ((float)rand() / (float)RAND_MAX) * (range_max - range_min);
            }
        }
    }
    
    // Generate target data in format: [d_model × seq_len × num_samples]
    for (int sample = 0; sample < num_samples; sample++) {
        for (int pos = 0; pos < seq_len; pos++) {
            for (int dim = 0; dim < d_model; dim++) {
                int idx = dim * (seq_len * num_samples) + pos * num_samples + sample;
                (*y)[idx] = range_min + 
                    ((float)rand() / (float)RAND_MAX) * (range_max - range_min);
            }
        }
    }
    
    printf("Generated random sequence data: %d samples, %d sequence length, %d dimensions\n", 
           num_samples, seq_len, d_model);
}

void save_sequence_data(float* X, float* y, int num_samples, int seq_len, int d_model, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) { 
        printf("Error: cannot write %s\n", filename); 
        return; 
    }
    
    // Header
    for (int pos = 0; pos < seq_len; pos++) {
        for (int dim = 0; dim < d_model; dim++) {
            fprintf(f, "x%d_%d,", pos, dim);
        }
    }
    for (int pos = 0; pos < seq_len; pos++) {
        for (int dim = 0; dim < d_model; dim++) {
            fprintf(f, "y%d_%d%s", pos, dim, 
                   (pos == seq_len-1 && dim == d_model-1) ? "\n" : ",");
        }
    }
    
    // Data
    for (int s = 0; s < num_samples; s++) {
        // Input sequence
        for (int pos = 0; pos < seq_len; pos++) {
            for (int dim = 0; dim < d_model; dim++) {
                int idx = dim * (seq_len * num_samples) + pos * num_samples + s;
                fprintf(f, "%.6f,", X[idx]);
            }
        }
        // Target sequence
        for (int pos = 0; pos < seq_len; pos++) {
            for (int dim = 0; dim < d_model; dim++) {
                int idx = dim * (seq_len * num_samples) + pos * num_samples + s;
                fprintf(f, "%.6f%s", y[idx], 
                       (pos == seq_len-1 && dim == d_model-1) ? "\n" : ",");
            }
        }
    }
    
    fclose(f);
    printf("Data saved to %s\n", filename);
}