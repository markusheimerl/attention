#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_fp16.h>

void generate_data(half** X, half** y, int seq_len, int num_samples, int d_model, float range_min, float range_max);
void save_data(half* X, half* y, int seq_len, int num_samples, int d_model, const char* filename);

#endif