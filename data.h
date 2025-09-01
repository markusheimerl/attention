#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>

void generate_sequence_data(float** X, float** y, int num_samples, int seq_len, int d_model, float range_min, float range_max);
void save_sequence_data(float* X, float* y, int num_samples, int seq_len, int d_model, const char* filename);

#endif