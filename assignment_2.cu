#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_common.cuh"
#include <cmath>
#include <stdio.h>
 
__global__ void sum_array_gpu(int *a, int *b, int *c, int *d, int size)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid < size) {
        c[gid] = a[gid] + b[gid] + c[gid];
    }
}

void sum_array_cpu(int *a, int *b, int *c, int *results, int size) {
    for (int i = 0; i < size; i++) {
        results[i] = a[i] + b[i] + c[i];
    }
}

int main()
{
    int size = pow(2, 22);
    int block_size = 128;

    int NO_BYTES = size * sizeof(int);

    // host pointers
    int *h_a, *h_b, *h_c, *gpu_results, *cpu_results;

    // allocate memory for host pointers
    h_a = (int*)malloc(NO_BYTES);
    h_b = (int*)malloc(NO_BYTES);
    h_c = (int*)malloc(NO_BYTES);
    gpu_results = (int*)malloc(NO_BYTES);
    cpu_results = (int*)malloc(NO_BYTES);
    
    // initial host pointer
    time_t t;
    srand( (unsigned)time(&t) );

    for (int i = 0; i< size; i++) {
        h_a[i] = (int)(rand() & 0xFF);
    }

    for (int i = 0; i< size; i++) {
        h_b[i] = (int)(rand() & 0xFF);
    }

    for (int i = 0; i< size; i++) {
        h_c[i] = (int)(rand() & 0xFF);
    }

    memset(gpu_results,0,NO_BYTES);
    memset(cpu_results,0,NO_BYTES);

    // summation in CPU
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_array_cpu(h_a, h_b, h_c, cpu_results, size);
    cpu_end = clock();

    // devices pointer
    int *d_a, *d_b, *d_c, *results;
    gpuErrchk(cudaMalloc((int **)&d_a, NO_BYTES));
    gpuErrchk(cudaMalloc((int **)&d_b, NO_BYTES));
    gpuErrchk(cudaMalloc((int **)&d_c, NO_BYTES));
    gpuErrchk(cudaMalloc((int **)&results, NO_BYTES));
    
    // memory transfer from host to device
    clock_t htod_start, htod_end;
    htod_start = clock();
    cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, NO_BYTES, cudaMemcpyHostToDevice);
    htod_end = clock();

    dim3 block(block_size);

    dim3 grid((size/block.x) + 1);
    
    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    sum_array_gpu <<<grid, block>>>(d_a, d_b, d_c, results, size);
    gpu_end = clock();
    cudaDeviceSynchronize();
    
    // memory transfer back to host
    clock_t dtoh_start, dtoh_end;
    dtoh_start = clock();
    cudaMemcpy(gpu_results, results, NO_BYTES, cudaMemcpyDeviceToHost);
    dtoh_end = clock();

    printf("Sum array CPU execution time: %4.6f \n", (double)((double)(cpu_end-cpu_start)/CLOCKS_PER_SEC));

    printf("Memory transfer from host to device time: %4.6f \n", (double)((double)(htod_end-htod_start)/CLOCKS_PER_SEC));

    printf("Memory transfer from device to host time: %4.6f \n", (double)((double)(dtoh_end-dtoh_start)/CLOCKS_PER_SEC));

    printf("Sum array GPU execution time: %4.6f \n", (double)((double)(gpu_end-gpu_start)/CLOCKS_PER_SEC));

    cudaFree(results);
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
    
    free(gpu_results);
    free(h_c);
    free(h_b);
    free(h_a);
    
    cudaDeviceReset();
}