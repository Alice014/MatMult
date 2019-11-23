__global__ void mmkernel(float* c, float* a, float* b, int n, int m, int p) {
	int idxCol = blockIdx.x * blockDim.x + threadIdx.x; 
	int idxRow = blockIdx.y * blockDim.y + threadIdx.y; 
    float sum = 0.0f; 
    
    if (!(idxCol < m && idxRow < n)) {
        return;
    }
	for (int k = 0; k < p; k++) 
		sum += a[idxRow * p + k] * b[k * m + idxCol];
	c[idxRow * m +idxCol] = sum;
}

#define kernel mmkernel
#include "main.h"