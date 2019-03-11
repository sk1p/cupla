/* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/** @file Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <iostream> //std:cout
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_to_cupla.hpp>
//Timer for test purpose
#include <chrono>
#include <boost/lexical_cast.hpp>
#include <vector>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

#define DETECTOR_SIZE (128 * 128)
//#define SCAN_SIZE (128 * 64)
#define SCAN_SIZE (3 * 3)
#define NUM_MASKS 2

#define RESULT_SIZE (NUM_MASKS * SCAN_SIZE)
#define DATASET_SIZE (SCAN_SIZE * DETECTOR_SIZE)
#define MASKS_SIZE (NUM_MASKS * DETECTOR_SIZE / 8)

typedef float result_t;
typedef float pixel_t;
typedef char mask_t;

struct maskedSum
{
    template <typename T_Acc, class PixelType>
    ALPAKA_FN_HOST_ACC void operator()(T_Acc const &acc,
                                       const PixelType *images, PixelType *result, char *masks,
                                       int numMasks, int scanSize) const
    {
        int mask = blockDim.x * blockIdx.x * elemDim.x + threadIdx.x * elemDim.x;
        int f = blockDim.y * blockIdx.y * elemDim.y + threadIdx.y * elemDim.y;
        /*int numThreadsY = blockDim.y * gridDim.y;
        int numThreadsX = blockDim.x * gridDim.x;*/
        int resultIdx = mask * scanSize + f;

        printf("resultIdx=%d elemDim.x=%d elemDim.y=%d\n", resultIdx, elemDim.x, elemDim.y);

        if (mask >= numMasks || f >= scanSize)
        {
            return;
        }

        PixelType res = 0;
        for (int p = 0; p < DETECTOR_SIZE / 8; p++)
        {
            mask_t maskbyte = masks[(DETECTOR_SIZE / 8) * mask + p];
            int kMax = min(8, DETECTOR_SIZE - p);
            for (int k = 0; k < 8; k++)
            {
                res += (maskbyte & 1 << (7 - k)) ? images[DETECTOR_SIZE * f + p + k] : 0;
            }
            /*
            PixelType r0 = (maskbyte & 0x80) ? images[DETECTOR_SIZE * f + p + 0] : 0;
            PixelType r1 = (maskbyte & 0x40) ? images[DETECTOR_SIZE * f + p + 1] : 0;
            PixelType r2 = (maskbyte & 0x20) ? images[DETECTOR_SIZE * f + p + 2] : 0;
            PixelType r3 = (maskbyte & 0x10) ? images[DETECTOR_SIZE * f + p + 3] : 0;
            PixelType r4 = (maskbyte & 0x08) ? images[DETECTOR_SIZE * f + p + 4] : 0;
            PixelType r5 = (maskbyte & 0x04) ? images[DETECTOR_SIZE * f + p + 5] : 0;
            PixelType r6 = (maskbyte & 0x02) ? images[DETECTOR_SIZE * f + p + 6] : 0;
            PixelType r7 = (maskbyte & 0x01) ? images[DETECTOR_SIZE * f + p + 7] : 0;
            res += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
            */
        }
        result[resultIdx] = res;
    }
};

/**
 * Host main routine
 */
int main(int argc, char *argv[])
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // init results:
    result_t *results = (result_t *)malloc(sizeof(result_t) * RESULT_SIZE);
    mask_t *maskbuf = (mask_t *)malloc(sizeof(mask_t) * MASKS_SIZE);
    pixel_t *dataset = (pixel_t *)malloc(sizeof(pixel_t) * DATASET_SIZE);

    // Verify that allocations succeeded
    if (dataset == NULL || maskbuf == NULL || results == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < RESULT_SIZE; i++)
    {
        results[i] = 0;
    }

    // init masks
    for (int i = 0; i < MASKS_SIZE; i++)
    {
        maskbuf[i] = 0xFF;
    }

    // init source data buffer
    for (int i = 0; i < DATASET_SIZE; i++)
    {
        dataset[i] = 1;
    }

    // Allocate the device input vector dataset
    pixel_t *d_dataset = NULL;
    err = cudaMalloc((void **)&d_dataset, sizeof(pixel_t) * DATASET_SIZE);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector dataset (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector maskbuf
    mask_t *d_maskbuf = NULL;
    err = cudaMalloc((void **)&d_maskbuf, sizeof(mask_t) * MASKS_SIZE);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector masks (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector results
    float *d_results = NULL;
    err = cudaMalloc((void **)&d_results, sizeof(result_t) * RESULT_SIZE);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector results (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors dataset and masks in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_dataset, dataset, DATASET_SIZE * sizeof(pixel_t), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector dataset from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_maskbuf, maskbuf, MASKS_SIZE * sizeof(mask_t), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector maskbuf from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the CUDA Kernel
    //dim3 threadsPerBlock(16, 8);
    dim3 threadsPerBlock(2, 1);
    //dim3 threadsPerBlock(1, 1);

    dim3 blocksPerGrid(
        // ceil(NUM_MASKS, threadsPerBlock.x)
        (NUM_MASKS + threadsPerBlock.x - 1) / threadsPerBlock.x,
        // ceil(SCAN_SIZE, threadsPerBlock.y)
        (SCAN_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    //CUPLA_KERNEL_OPTI(maskedSum)
    CUPLA_KERNEL(maskedSum)
    (blocksPerGrid, threadsPerBlock, 0, 0)(d_dataset, d_results, d_maskbuf, NUM_MASKS, SCAN_SIZE);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(results, d_results, sizeof(result_t) * RESULT_SIZE, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < RESULT_SIZE; ++i)
    {
        if (results[i] != DETECTOR_SIZE)
        {
            fprintf(stderr, "Result verification failed at element %d value was %f, expected %d!\n", i, results[i], DETECTOR_SIZE);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_dataset);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_maskbuf);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_results);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(dataset);
    free(maskbuf);
    free(results);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Done\n");

    return 0;
}