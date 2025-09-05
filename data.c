#include "data.h"

void generate_attention_data(float** X, float** y, int seq_len, int num_samples, int d_model,
                           float range_min, float range_max) {
    const int ncols = d_model * num_samples;
    const int total = seq_len * ncols;
    
    *X = (float*)malloc(total * sizeof(float));
    *y = (float*)malloc(total * sizeof(float));
    
    // Fill X with random data
    float range = range_max - range_min;
    for (int i = 0; i < total; i++) {
        (*X)[i] = range_min + ((float)rand() / RAND_MAX) * range;
    }
    
    // Create attention matrix A: [seq_len × seq_len]
    float* A = (float*)malloc(seq_len * seq_len * sizeof(float));
    float a_scale = 1.0f / sqrtf(seq_len);
    
    for (int i = 0; i < seq_len * seq_len; i++) {
        A[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * a_scale;
    }
    
    // Row-wise softmax
    for (int i = 0; i < seq_len; i++) {
        float max_val = -1e30f;
        for (int j = 0; j < seq_len; j++) {
            float v = A[i + seq_len * j];
            if (v > max_val) max_val = v;
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float e = expf(A[i + seq_len * j] - max_val);
            A[i + seq_len * j] = e;
            sum += e;
        }
        
        for (int j = 0; j < seq_len; j++) {
            A[i + seq_len * j] /= sum;
        }
    }
    
    // Y = A * X
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                seq_len, ncols, seq_len,
                1.0f, A, seq_len,
                *X, seq_len,
                0.0f, *y, seq_len);
    
    // Add noise
    float noise_scale = range * 0.01f;
    for (int i = 0; i < total; i++) {
        float noise = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        (*y)[i] += noise;
    }
    
    free(A);
    
    printf("Generated attention data: %d samples, length %d, d_model %d\n", 
           num_samples, seq_len, d_model);
}

void save_data(float* X, float* y, int seq_len, int num_samples, int d_model,
               const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;
    
    const int ncols = d_model * num_samples;
    
    // Header
    for (int d = 0; d < d_model; d++) {
        for (int s = 0; s < num_samples; s++) {
            fprintf(f, "x_d%d_s%d,", d, s);
        }
    }
    for (int d = 0; d < d_model; d++) {
        for (int s = 0; s < num_samples; s++) {
            fprintf(f, "y_d%d_s%d%s", d, s, 
                   (d == d_model-1 && s == num_samples-1) ? "\n" : ",");
        }
    }
    
    // Data
    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < d_model; d++) {
            for (int s = 0; s < num_samples; s++) {
                int c = d * num_samples + s;
                fprintf(f, "%.6f,", X[t + seq_len * c]);
            }
        }
        for (int d = 0; d < d_model; d++) {
            for (int s = 0; s < num_samples; s++) {
                int c = d * num_samples + s;
                fprintf(f, "%.6f%s", y[t + seq_len * c],
                       (d == d_model-1 && s == num_samples-1) ? "\n" : ",");
            }
        }
    }
    
    fclose(f);
    printf("Data saved to %s\n", filename);
}