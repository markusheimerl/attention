#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>

void generate_data(float** X, float** y, int num_samples, int seq_len, int feature_dim);
void save_data(float* X, float* y, int num_samples, int seq_len, int feature_dim, const char* filename);

#endif