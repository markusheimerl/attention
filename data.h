#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>

void generate_data(float** X, float** y, int num_samples, int input_dim, int output_dim, float range_min, float range_max);
void save_data(float* X, float* y, int num_samples, int input_dim, int output_dim, const char* filename);

#endif