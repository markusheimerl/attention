#include "data.h"

static float evaluate_attention_function(int num_terms, const float* coefficients, const int* operations,
                              const int* idx1, const int* idx2, const int* add_subtract, 
                              const int* position_deps, const float* X, int seq, int t, int seq_len, int input_dim) {
    float result = 0.0f;
    
    for (int i = 0; i < num_terms; i++) {
        float coefficient = coefficients[i];
        int operation = operations[i];
        int input_idx1 = idx1[i];
        int input_idx2 = idx2[i];
        int add_sub = add_subtract[i];
        int pos_dep = position_deps[i];
        
        // Create position-dependent interactions that benefit from attention
        int other_t = (t + pos_dep) % seq_len;  // Circular reference to other positions
        
        // Get input values from current timestep and position-dependent timestep
        int x_base_idx = seq * seq_len * input_dim + t * input_dim;
        int x_other_idx = seq * seq_len * input_dim + other_t * input_dim;
        const float* x_curr = &X[x_base_idx];
        const float* x_other = &X[x_other_idx];
        
        float term_value = 0.0f;
        
        switch (operation) {
            case 0: term_value = coefficient * sinf(x_curr[input_idx1] + x_other[input_idx2]); break;
            case 1: term_value = coefficient * cosf(x_curr[input_idx1] * x_other[input_idx2]); break;
            case 2: term_value = coefficient * tanhf(x_curr[input_idx1] - x_other[input_idx1]); break;
            case 3: term_value = coefficient * expf(-powf(x_curr[input_idx1] + x_other[input_idx2], 2)); break;
            case 4: term_value = coefficient * logf(fabsf(x_curr[input_idx1] * x_other[input_idx2]) + 1.0f); break;
            case 5: term_value = coefficient * (x_curr[input_idx1] * x_other[input_idx1] + x_curr[input_idx2]); break;
            case 6: term_value = coefficient * sinhf(x_curr[input_idx1]) * cosf(x_other[input_idx2]); break;
            case 7: term_value = coefficient * x_curr[input_idx1] * sinf(M_PI * x_other[input_idx2]); break;
        }
        
        if (add_sub == 0) {
            result += term_value;
        } else {
            result -= term_value;
        }
    }
    
    return result;
}

static void print_symbolic_function(int output_idx, int num_terms, const float* coefficients, 
                                  const int* operations, const int* idx1, const int* idx2, 
                                  const int* add_subtract, const int* position_deps) {
    printf("y%d = ", output_idx);
    
    for (int i = 0; i < num_terms; i++) {
        float coeff = coefficients[i];
        int op = operations[i];
        int in1 = idx1[i];
        int in2 = idx2[i];
        int add_sub = add_subtract[i];
        int pos_dep = position_deps[i];
        
        // Print sign
        if (i == 0) {
            if (add_sub == 1) printf("-");
        } else {
            printf(" %s ", (add_sub == 0) ? "+" : "-");
        }
        
        // Print coefficient if not 1.0
        if (fabsf(coeff - 1.0f) > 1e-6) {
            printf("%.3f*", coeff);
        }
        
        // Print operation with position dependencies
        switch (op) {
            case 0: printf("sin(x%d[t] + x%d[t+%d])", in1, in2, pos_dep); break;
            case 1: printf("cos(x%d[t] * x%d[t+%d])", in1, in2, pos_dep); break;
            case 2: printf("tanh(x%d[t] - x%d[t+%d])", in1, in1, pos_dep); break;
            case 3: printf("exp(-(x%d[t] + x%d[t+%d])^2)", in1, in2, pos_dep); break;
            case 4: printf("log(|x%d[t] * x%d[t+%d]| + 1)", in1, in2, pos_dep); break;
            case 5: printf("x%d[t] * x%d[t+%d] + x%d[t]", in1, in1, pos_dep, in2); break;
            case 6: printf("sinh(x%d[t]) * cos(x%d[t+%d])", in1, in2, pos_dep); break;
            case 7: printf("x%d[t] * sin(π * x%d[t+%d])", in1, in2, pos_dep); break;
        }
    }
    printf("\n");
}

void generate_synthetic_data(float** X, float** y, int num_sequences, int seq_len, int input_dim, int output_dim, 
                           float input_min, float input_max) {
    // Allocate memory
    *X = (float*)malloc(num_sequences * seq_len * input_dim * sizeof(float));
    *y = (float*)malloc(num_sequences * seq_len * output_dim * sizeof(float));
    
    // Generate random input data
    for (int i = 0; i < num_sequences * seq_len * input_dim; i++) {
        float rand_val = (float)rand() / (float)RAND_MAX;
        (*X)[i] = input_min + rand_val * (input_max - input_min);
    }
    
    // Create function parameters for each output dimension
    int* num_terms_per_output = (int*)malloc(output_dim * sizeof(int));
    float** coefficients = (float**)malloc(output_dim * sizeof(float*));
    int** operations = (int**)malloc(output_dim * sizeof(int*));
    int** idx1 = (int**)malloc(output_dim * sizeof(int*));
    int** idx2 = (int**)malloc(output_dim * sizeof(int*));
    int** add_subtract = (int**)malloc(output_dim * sizeof(int*));
    int** position_deps = (int**)malloc(output_dim * sizeof(int*));
    
    for (int output_idx = 0; output_idx < output_dim; output_idx++) {
        // Random number of terms between 6 and 10
        int num_terms = 6 + (rand() % 5);
        num_terms_per_output[output_idx] = num_terms;
        
        // Allocate arrays for this function's terms
        coefficients[output_idx] = (float*)malloc(num_terms * sizeof(float));
        operations[output_idx] = (int*)malloc(num_terms * sizeof(int));
        idx1[output_idx] = (int*)malloc(num_terms * sizeof(int));
        idx2[output_idx] = (int*)malloc(num_terms * sizeof(int));
        add_subtract[output_idx] = (int*)malloc(num_terms * sizeof(int));
        position_deps[output_idx] = (int*)malloc(num_terms * sizeof(int));

        // Generate random terms
        for (int term = 0; term < num_terms; term++) {
            coefficients[output_idx][term] = 0.1f + 0.3f * ((float)rand() / (float)RAND_MAX);
            operations[output_idx][term] = rand() % 8;
            idx1[output_idx][term] = rand() % input_dim;
            idx2[output_idx][term] = rand() % input_dim;
            add_subtract[output_idx][term] = rand() % 2;
            position_deps[output_idx][term] = 1 + (rand() % (seq_len / 2)); // Dependencies across positions
        }
    }
    
    // Print symbolic representation of generated functions
    printf("\nGenerated synthetic functions with position dependencies:\n");
    for (int output_idx = 0; output_idx < output_dim; output_idx++) {
        print_symbolic_function(output_idx, num_terms_per_output[output_idx], 
                              coefficients[output_idx], operations[output_idx], 
                              idx1[output_idx], idx2[output_idx], add_subtract[output_idx],
                              position_deps[output_idx]);
    }
    printf("\n");
    
    // Generate output data by evaluating each function
    for (int seq = 0; seq < num_sequences; seq++) {
        for (int t = 0; t < seq_len; t++) {
            for (int j = 0; j < output_dim; j++) {
                int y_idx = seq * seq_len * output_dim + t * output_dim + j;
                (*y)[y_idx] = evaluate_attention_function(num_terms_per_output[j], 
                                                        coefficients[j], operations[j], 
                                                        idx1[j], idx2[j], add_subtract[j],
                                                        position_deps[j], *X, seq, t, seq_len, input_dim);
            }
        }
    }
    
    // Clean up
    for (int i = 0; i < output_dim; i++) {
        free(coefficients[i]);
        free(operations[i]);
        free(idx1[i]);
        free(idx2[i]);
        free(add_subtract[i]);
        free(position_deps[i]);
    }
    free(num_terms_per_output);
    free(coefficients);
    free(operations);
    free(idx1);
    free(idx2);
    free(add_subtract);
    free(position_deps);
}

void save_data(float* X, float* y, int num_sequences, int seq_len, int input_dim, int output_dim, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write header
    for (int i = 0; i < input_dim; i++) {
        fprintf(file, "x%d,", i);
    }
    for (int i = 0; i < output_dim - 1; i++) {
        fprintf(file, "y%d,", i);
    }
    fprintf(file, "y%d\n", output_dim - 1);
    
    // Write data
    for (int seq = 0; seq < num_sequences; seq++) {
        for (int t = 0; t < seq_len; t++) {
            int x_idx = seq * seq_len * input_dim + t * input_dim;
            int y_idx = seq * seq_len * output_dim + t * output_dim;
            
            // Input features
            for (int j = 0; j < input_dim; j++) {
                fprintf(file, "%.17f,", X[x_idx + j]);
            }
            // Output values
            for (int j = 0; j < output_dim - 1; j++) {
                fprintf(file, "%.17f,", y[y_idx + j]);
            }
            fprintf(file, "%.17f\n", y[y_idx + output_dim - 1]);
        }

        if (seq < num_sequences - 1) {
            fprintf(file, "\n");
        }
    }
    
    fclose(file);
    printf("Data saved to %s\n", filename);
}