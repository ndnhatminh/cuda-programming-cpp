#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 
#include <stdio.h>
 
__global__ void unique_idx_calc_threadIdx(int * input)
{
    int tid = threadIdx.x;
    printf("threadIdx : %d, value : %d \n", tid, input[tid]);
}

__global__ void unique_gid_calculation(int * input)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int block_offset = blockDim.x * blockDim.y * blockIdx.x;
    int row_offset = gridDim.x * blockDim.x * blockDim.y;
    int gid = row_offset + block_offset + tid;

    printf("blockIdx.x: %d, threadIdx.x: %d, gid: %d, value: %d \n", blockIdx.x, threadIdx.x, gid, input[gid]);
}
int main()
{
    int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = {23, 9, 4, 53, 65, 12, 1, 33, 24, 34, 77, 12, 14, 17, 67, 90};

    for (int i = 0; i< array_size; i++)
    {
        printf("%d ", h_data[i]);
    }
    printf("\n \n");

    int * d_data;   
    cudaMalloc((void**)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(2,2,2);

    dim3 grid(2,2,2);
 
    unique_gid_calculation <<<grid, block>>>(d_data);
 
    cudaDeviceSynchronize();
    
    cudaDeviceReset();
}