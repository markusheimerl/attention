#include "data.h"

static const char* basis_names[] = {
    "sin", "cos", "tanh", "exp", "log", "poly", "sinh", "sinprod", "max", "min", "copy"
};

static float basis(int op, float x, float y) {
    switch (op % 11) {
        case 0: return sinf(2.0f * x);
        case 1: return cosf(1.5f * x);
        case 2: return tanhf(x + y);
        case 3: return expf(-x * x);
        case 4: return logf(fabsf(x) + 1.0f);
        case 5: return x * x * y;
        case 6: return sinhf(x - y);
        case 7: return x * sinf(M_PI * y);
        case 8: return fmaxf(x, y);  // max of two inputs
        case 9: return fminf(x, y);  // min of two inputs
        case 10: return x;           // copy first input
    }
    return 0.0f;
}

static void print_term(float coeff, int op, int idx1, int idx2, int offset1, int offset2, int sign, int first_term) {
    if (!first_term) {
        printf(sign ? " - " : " + ");
    } else if (sign) {
        printf("-");
    }
    
    if (fabsf(coeff - 1.0f) > 1e-6) printf("%.3f*", coeff);
    
    const char* op_name = basis_names[op % 11];
    
    if (op % 11 == 10) {
        // Copy - single input
        if (offset1 == 0) {
            printf("x%d", idx1);
        } else {
            printf("x%d[t%+d]", idx1, offset1);
        }
    } else {
        // Two-input functions
        printf("%s(", op_name);
        
        if (offset1 == 0) {
            printf("x%d", idx1);
        } else {
            printf("x%d[t%+d]", idx1, offset1);
        }
        
        printf(",");
        
        if (offset2 == 0) {
            printf("x%d", idx2);
        } else {
            printf("x%d[t%+d]", idx2, offset2);
        }
        
        printf(")");
    }
}

static void print_function(int output_idx, const float* params, int n_terms) {
    printf("y%d =", output_idx);
    
    for (int i = 0; i < n_terms; i++) {
        int base = i * 7;
        float coeff = params[base];
        int op = (int)params[base + 1];
        int idx1 = (int)params[base + 2];
        int idx2 = (int)params[base + 3];
        int offset1 = (int)params[base + 4];
        int offset2 = (int)params[base + 5];
        int sign = (int)params[base + 6] % 2;
        
        print_term(coeff, op, idx1, idx2, offset1, offset2, sign, i == 0);
    }
    printf("\n");
}

static float eval_function(const float* params, int n_terms, const float* X, 
                          int sample, int t, int input_dim, int seq_len) {
    float result = 0.0f;
    
    for (int i = 0; i < n_terms; i++) {
        int base = i * 7;
        float coeff = params[base];
        int op = (int)params[base + 1];
        int idx1 = (int)params[base + 2] % input_dim;
        int idx2 = (int)params[base + 3] % input_dim;
        int offset1 = (int)params[base + 4];
        int offset2 = (int)params[base + 5];
        int sign = (int)params[base + 6] % 2;
        
        // Calculate effective time indices
        int eff_t1 = t + offset1;
        int eff_t2 = t + offset2;
        
        // Clamp to valid sequence bounds
        eff_t1 = fmaxf(0, fminf(seq_len - 1, eff_t1));
        eff_t2 = fmaxf(0, fminf(seq_len - 1, eff_t2));
        
        // Extract values from temporal pool
        int base_idx1 = sample * seq_len * input_dim + eff_t1 * input_dim;
        int base_idx2 = sample * seq_len * input_dim + eff_t2 * input_dim;
        float x = X[base_idx1 + idx1];
        float y = X[base_idx2 + idx2];
        
        float term = coeff * basis(op, x, y);
        result += sign ? -term : term;
    }
    
    return result;
}

void generate_data(float** X, float** y, int num_samples, int seq_len, int input_dim, int output_dim,
                   float range_min, float range_max, int max_offset) {
    int total_x = num_samples * seq_len * input_dim;
    int total_y = num_samples * seq_len * output_dim;
    
    *X = (float*)malloc(total_x * sizeof(float));
    *y = (float*)malloc(total_y * sizeof(float));
    
    // Generate input data
    for (int i = 0; i < total_x; i++) {
        (*X)[i] = range_min + ((float)rand() / (float)RAND_MAX) * (range_max - range_min);
    }
    
    // Generate function parameters for each output
    const int n_terms = 6 + rand() % 7; // 6-12 terms per function
    const int params_per_term = 7;
    float* all_params = (float*)malloc(output_dim * n_terms * params_per_term * sizeof(float));
    
    for (int out = 0; out < output_dim; out++) {
        for (int term = 0; term < n_terms; term++) {
            int base = (out * n_terms + term) * params_per_term;
            all_params[base + 0] = 0.1f + 0.4f * ((float)rand() / (float)RAND_MAX); // coefficient
            all_params[base + 1] = (float)(rand() % 11);                      // operation
            all_params[base + 2] = (float)(rand() % input_dim);               // input idx 1
            all_params[base + 3] = (float)(rand() % input_dim);               // input idx 2
            
            // Generate temporal offsets based on max_offset mode
            if (max_offset == 0) {
                // No sequence dependencies
                all_params[base + 4] = 0.0f;
                all_params[base + 5] = 0.0f;
            } else if (max_offset > 0) {
                // Bidirectional: [-max_offset, +max_offset]
                all_params[base + 4] = (float)(rand() % (2 * max_offset + 1) - max_offset);
                all_params[base + 5] = (float)(rand() % (2 * max_offset + 1) - max_offset);
            } else {
                // Causal: [max_offset, 0] (max_offset is negative)
                all_params[base + 4] = (float)(rand() % (-max_offset + 1) + max_offset);
                all_params[base + 5] = (float)(rand() % (-max_offset + 1) + max_offset);
            }
            
            all_params[base + 6] = (float)(rand() % 2);                       // add/subtract
        }
    }
    
    // Print generated functions
    printf("\nGenerated synthetic functions:\n");
    for (int out = 0; out < output_dim; out++) {
        float* params = &all_params[out * n_terms * params_per_term];
        print_function(out, params, n_terms);
    }
    printf("\n");
    
    // Generate outputs
    for (int sample = 0; sample < num_samples; sample++) {
        for (int t = 0; t < seq_len; t++) {
            for (int out = 0; out < output_dim; out++) {
                int y_idx = sample * seq_len * output_dim + t * output_dim + out;
                float* params = &all_params[out * n_terms * params_per_term];
                (*y)[y_idx] = eval_function(params, n_terms, *X, sample, t, input_dim, seq_len);
            }
        }
    }
    
    free(all_params);
}

void save_data(float* X, float* y, int num_samples, int seq_len, int input_dim, int output_dim,
               const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) { printf("Error: cannot write %s\n", filename); return; }
    
    // Header
    for (int i = 0; i < input_dim; i++) fprintf(f, "x%d,", i);
    for (int i = 0; i < output_dim; i++) fprintf(f, "y%d%s", i, i == output_dim-1 ? "\n" : ",");
    
    // Data
    for (int s = 0; s < num_samples; s++) {
        for (int t = 0; t < seq_len; t++) {
            int x_base = s * seq_len * input_dim + t * input_dim;
            int y_base = s * seq_len * output_dim + t * output_dim;
            
            for (int i = 0; i < input_dim; i++) fprintf(f, "%.6f,", X[x_base + i]);
            for (int i = 0; i < output_dim; i++) 
                fprintf(f, "%.6f%s", y[y_base + i], i == output_dim-1 ? "\n" : ",");
        }
        if (s < num_samples-1 && seq_len > 1) fprintf(f, "\n");
    }
    
    fclose(f);
    printf("Data saved to %s\n", filename);
}