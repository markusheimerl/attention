#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// max_offset modes:
// = 0: no temporal dependencies (current timestep only)
// > 0: bidirectional dependencies [-max_offset, +max_offset]
// < 0: causal dependencies [max_offset, 0] (past only)
void generate_data(float** X, float** y, int num_samples, int seq_len, int input_dim, int output_dim, float range_min, float range_max, int max_offset);
void save_data(float* X, float* y, int num_samples, int seq_len, int input_dim, int output_dim, const char* filename);

#endif