/*  Sunil
 *  Umasankar
 *  suniluma
 */

#ifndef A3_HPP
#define A3_HPP

#include <cuda.h>
// we halve the number of blocks
__global__ void reduce2(int* gin, int* gout) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = gin[i] + gin[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) gout[blockIdx.x] = sdata[0];
}

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    dim3 dimGrid(1, 1);
    dim3 dimBlock(n, n);
    reduce2<<<dimGrid, dimBlock>>>(x,y);
} // gaussian_kde

#endif // A3_HPP
