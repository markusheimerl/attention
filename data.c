#include "data.h"

void generate_data(half** X, half** y, int seq_len, int num_samples, int d_model,
                   float range_min, float range_max) {
    // Row-major layout: [num_samples x seq_len x d_model]
    const int total = num_samples * seq_len * d_model;
    
    // Allocate temporary float buffers for computation
    float* X_float = (float*)malloc(total * sizeof(float));
    float* y_float = (float*)malloc(total * sizeof(float));
    
    // Allocate half output
    *X = (half*)malloc(total * sizeof(half));
    *y = (half*)malloc(total * sizeof(half));
    
    // Fill X with random data
    float range = range_max - range_min;
    for (int i = 0; i < total; i++) {
        X_float[i] = range_min + ((float)rand() / (float)RAND_MAX) * range;
    }
    
    // Create attention matrix A: [seq_len × seq_len]
    float* A = (float*)malloc(seq_len * seq_len * sizeof(float));
    float a_scale = 1.0f / sqrtf(seq_len);
    
    for (int i = 0; i < seq_len * seq_len; i++) {
        A[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * a_scale;
    }
    
    // Row-wise softmax on A
    for (int i = 0; i < seq_len; i++) {
        float max_val = -1e30f;
        for (int j = 0; j < seq_len; j++) {
            float v = A[i * seq_len + j];
            if (v > max_val) max_val = v;
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float e = expf(A[i * seq_len + j] - max_val);
            A[i * seq_len + j] = e;
            sum += e;
        }
        
        for (int j = 0; j < seq_len; j++) {
            A[i * seq_len + j] /= sum;
        }
    }
    
    // Apply attention transformation for each batch: Y_b = A * X_b
    for (int b = 0; b < num_samples; b++) {
        float* X_b = &X_float[b * seq_len * d_model];
        float* Y_b = &y_float[b * seq_len * d_model];
        
        // Manual matrix multiplication: Y_b = A * X_b
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < d_model; d++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    sum += A[i * seq_len + j] * X_b[j * d_model + d];
                }
                Y_b[i * d_model + d] = sum;
            }
        }
    }
    
    // Add noise
    float noise_scale = range * 0.001f;
    for (int i = 0; i < total; i++) {
        float noise = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * noise_scale;
        y_float[i] += noise;
    }
    
    // Convert to half precision
    for (int i = 0; i < total; i++) {
        (*X)[i] = __float2half(X_float[i]);
        (*y)[i] = __float2half(y_float[i]);
    }
    
    free(X_float);
    free(y_float);
    free(A);
    
    printf("Generated attention data: %d samples, length %d, d_model %d (FP16)\n", 
           num_samples, seq_len, d_model);
}

void save_data(half* X, half* y, int seq_len, int num_samples, int d_model,
               const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;
    
    // Header: batch_id, seq_pos, then features, then targets
    fprintf(f, "batch_id,seq_pos,");
    for (int d = 0; d < d_model; d++) {
        fprintf(f, "x_d%d,", d);
    }
    for (int d = 0; d < d_model; d++) {
        fprintf(f, "y_d%d%s", d, d == d_model-1 ? "\n" : ",");
    }
    
    // Data: one row per (batch, sequence_position)
    for (int b = 0; b < num_samples; b++) {
        for (int t = 0; t < seq_len; t++) {
            fprintf(f, "%d,%d,", b, t);
            
            // X features for this (batch, position)
            for (int d = 0; d < d_model; d++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                fprintf(f, "%.6f,", __half2float(X[idx]));
            }
            
            // Y features for this (batch, position)
            for (int d = 0; d < d_model; d++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                fprintf(f, "%.6f%s", __half2float(y[idx]), d == d_model-1 ? "\n" : ",");
            }
        }
    }
    
    fclose(f);
    printf("Data saved to %s\n", filename);
}