/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
#include <cuda.h>

#define KERNEL_LENGTH (2*KERNEL_RADIUS+1)

#ifdef CONSTANT_MEM
__constant__ int dim[3];
__constant__ float d_Kernel[KERNEL_LENGTH];
#endif

////////////////////////////////////////////////////////////////////////////////
// Slice convolution filter
////////////////////////////////////////////////////////////////////////////////
#if CONSTANT_MEM
__global__ void convolutionSlicesKernel(
    float *d_Dst,
	const float *d_Src) {
#else
__global__ void convolutionSlicesKernel(
    float *d_Dst,
    const float *d_Src,
	const int *dim,
	const float *d_Kernel) {
#endif
    __shared__ float s_Data[BLOCKDIM_Y][BLOCKDIM_X][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_Z + 1];

    // Separate blockIdx.xy
    const int blockIdx_x = blockIdx.x % (dim[0] / BLOCKDIM_X);
    const int blockIdx_y = blockIdx.x / (dim[0] / BLOCKDIM_X);

    //Offset to the upper halo edge
    const int baseX = blockIdx_x * BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx_y * BLOCKDIM_Y + threadIdx.y;
    const int baseZ = (blockIdx.y * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_Z + threadIdx.z;

	/*
	printf("Thread (%i,%i,%i), block (%i,%i) converted to block (%i,%i,%i), point (%i,%i,%i)\n",
			threadIdx.x, threadIdx.y, threadIdx.z,
			blockIdx.x, blockIdx.y,
			blockIdx_x, blockIdx_y, blockIdx.y,
			baseX, baseY, baseZ);
	*/
	const int ind = (baseZ * dim[1] + baseY) * dim[0] + baseX;
    d_Src += ind;
    d_Dst += ind;

	const int tilt = dim[0]*dim[1];

    //Main data
    #pragma unroll
    for(int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
        s_Data[threadIdx.y][threadIdx.x][threadIdx.z + i * BLOCKDIM_Z] = d_Src[i * BLOCKDIM_Z * tilt];

    //Back halo
    #pragma unroll
    for(int i = 0; i < HALO_STEPS; i++)
        s_Data[threadIdx.y][threadIdx.x][threadIdx.z + i * BLOCKDIM_Z] = (baseZ >= -i * BLOCKDIM_Z) ? d_Src[i * BLOCKDIM_Z * tilt] : 0;

    //Front halo
    #pragma unroll
    for(int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++)
        s_Data[threadIdx.y][threadIdx.x][threadIdx.z + i * BLOCKDIM_Z]= (dim[2] - baseZ > i * BLOCKDIM_Z) ? d_Src[i * BLOCKDIM_Z * tilt] : 0;

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for(int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++){
        float sum = 0;
        #pragma unroll
        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += d_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x][threadIdx.z + i * BLOCKDIM_Z + j];

        d_Dst[i * BLOCKDIM_Z * tilt] = sum;
    }
}

/*
void convolutionSlicesGPU(
    float *d_Dst,
    float *d_Src,
	float *d_Kernel,
    int imageW,
    int imageH,
    int imageD
) throw (const char *) {
    if (BLOCKDIM_Z * HALO_STEPS < KERNEL_RADIUS) throw("convolutionSlicesGPU(): halo too small for kernel size");
    if (imageW % BLOCKDIM_X != 0) throw("convolutionSlicesGPU(): image width not dividable by number of threads");
    if (imageH % BLOCKDIM_Y != 0) throw("convolutionSlicesGPU(): image height not dividable by number of threads");
    if (imageD % (RESULT_STEPS * BLOCKDIM_Z) != 0) throw("convolutionSlicesGPU(): image depth not dividable by number of threads");

    dim3 blocks(imageW / BLOCKDIM_X * imageH / BLOCKDIM_Y,
                imageD / (RESULT_STEPS * BLOCKDIM_Z));
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

	
	//printf("Executing blocks (%i,%i) with threads (%i,%i,%i)\n",
	//		blocks.x, blocks.y, threads.x, threads.y, threads.z);

    convolutionSlicesKernel<BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z,
							RESULT_STEPS, HALO_STEPS, KERNEL_RADIUS><<<blocks, threads>>>(
								d_Dst, d_Src, d_Kernel, imageW, imageH, imageD, imageW, imageW * imageH);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw (strcat("convolutionSlicesKernel() CUDA error: ", cudaGetErrorString(err)));
}
*/
