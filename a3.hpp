/*  Sunil
 *  Umasankar
 *  suniluma
 */

#ifndef A3_HPP
#define A3_HPP
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

//CONSTANTS
#define THREADS_PER_BLOCK 512
#define PI 3.14

using namespace std;


//Handle Error from Stackoverflow
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


void print(const vector<float>& array) {
    cout<<endl;
    for (int i = 0; i < array.size(); i++) {
        //if(array[i] != 0)
            cout<<array[i]<<" ";
    }
    cout<<endl<<endl;
}


void print_v2(const vector<float>& array1, const vector<float>& array2) {
    cout<<endl;
    float a = 0, b = 0;
    for (int i = 0; i < array1.size(); i++) {
        //if(array[i] != 0)
            cout<<array1[i]<<"\t"<<array2[i]<<endl;
            a += array1[i];
            b += array2[i];
    }
    cout<<"Sum "<<a<<" "<<b<<endl;

}


//Template from CUDA by Example
__global__ void add(float *a, float *b, float h, int n, int index) {

    float constant = 1.0/sqrt(2.0*PI)/(float)n/h;
    __shared__ float shared_array[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int shared_index = threadIdx.x;
    float temp = 0.0f;
    while (tid < n) {
        temp += constant*exp(-(a[index] - a[tid])*(a[index] - a[tid])/2.0/h/h);
        tid += blockDim.x * gridDim.x;
    }
    shared_array[shared_index] = temp;

    __syncthreads();

    int reduction_index = (blockDim.x)/2;
    while (reduction_index != 0) {
        if (shared_index < reduction_index)
            shared_array[shared_index] += shared_array[shared_index + reduction_index];
        __syncthreads();
        reduction_index = reduction_index/2;

    }

    if (shared_index == 1)
    b[blockIdx.x] = shared_array[0];
}

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {

    int size = n * sizeof(float);
    float *inp;
    float *out;
    float constant = 1.0/sqrt(2.0*PI)/(float)n/h;
    int number_of_blocks = (n+THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    vector<float> result(n,0.0);

    HANDLE_ERROR(cudaMalloc( (void**)&inp, size ));
    HANDLE_ERROR(cudaMalloc( (void**)&out, size ));
    HANDLE_ERROR( cudaMemcpy( inp, x.data(), size, cudaMemcpyHostToDevice ) );


    for(int i = 0; i < n; i++) {
        add<<<number_of_blocks, THREADS_PER_BLOCK>>>(inp,out, h, n, i);
        HANDLE_ERROR(cudaMemcpy(result.data(), out, size, cudaMemcpyDeviceToHost));

        float seq_red = 0.0;
        for(int j = 0; j < number_of_blocks; j++) {
            seq_red += result[j];
        }
        y[i] = seq_red;
    }
    HANDLE_ERROR(cudaFree(inp));
    HANDLE_ERROR(cudaFree(out));

    /* END OF CODE */


// SEQUENTIAL CODE FOR TESTING TIME TAKEN
//If you want to run the following, please comment out the parallel part of the code above and uncomment below
/*
    vector<float> by;

    for(int i = 0; i < n; i ++) {
        float temp = 0.0f;
        for (int j = 0; j < n; j++) {
            temp += constant*(float)exp(-(x[i] - x[j])*(x[i] - x[j])/2.0/h/h);
        }

        by.push_back(temp);
    }

    print_v2(by,y);
    cout<<number_of_blocks<<endl;
*/

} // gaussian_kde

#endif // A3_HPP
