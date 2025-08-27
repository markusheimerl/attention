#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void generate_attention_data(float** X, float** y, int num_samples, int seq_len, int feature_dim);
void save_data(float* X, float* y, int num_samples, int seq_len, int feature_dim, const char* filename);
void print_sample_data(float* X, float* y, int sample_idx, int seq_len, int feature_dim);

#endif