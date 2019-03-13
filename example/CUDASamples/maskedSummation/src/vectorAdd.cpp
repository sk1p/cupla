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
#include <cstdlib>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_to_cupla.hpp>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_profiler_api.h>
#endif

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

typedef float result_t;
typedef float pixel_t;
typedef char mask_t;

const static int numMaskBits = sizeof(mask_t) * 8;

int popcount( uint8_t b )
{
     b = b - ((b >> 1) & 0x55);
     b = (b & 0x33) + ((b >> 2) & 0x33);
     return (((b + (b >> 4)) & 0x0F) * 0x01);
}

struct maskedSum
{
    template <typename T_Acc, class PixelType>
    ALPAKA_FN_HOST_ACC void operator()(T_Acc const &acc,
                                       const PixelType *images, PixelType *result, mask_t *masks,
                                       unsigned int numMasks, unsigned int scanSize,
                                       unsigned int detectorSize, unsigned int frameStride) const
    {
        const int maskBase = blockDim.x * blockIdx.x * elemDim.x + threadIdx.x * elemDim.x;
        const int frameBase = blockDim.y * blockIdx.y * elemDim.y + threadIdx.y * elemDim.y;
        const int stopX = (maskBase + elemDim.x < numMasks) ? maskBase + elemDim.x : numMasks;
        const int stopY = (frameBase + elemDim.y < scanSize) ? frameBase + elemDim.y : scanSize;

        for(int f = frameBase; f < stopY; f++) {
            for(int mask = maskBase; mask < stopX; mask++) {
                int resultIdx = mask + frameStride * f;
                //int resultIdx = mask * frameStride + f;

                //printf("mask=%d f=%d stopY=%d frameBase=%d elemDim.y=%d scanSize=%d\n",
                //        mask, f, stopY, frameBase, elemDim.y, scanSize);

                PixelType res = 0;
                for (int p = 0; p < detectorSize / numMaskBits; p++)
                {
                    mask_t maskbyte = masks[(detectorSize / numMaskBits) * mask + p];
                    PixelType r0 = (maskbyte & 0x80) ? images[detectorSize*f + p + 0] : 0;
                    PixelType r1 = (maskbyte & 0x40) ? images[detectorSize*f + p + 1] : 0;
                    PixelType r2 = (maskbyte & 0x20) ? images[detectorSize*f + p + 2] : 0;
                    PixelType r3 = (maskbyte & 0x10) ? images[detectorSize*f + p + 3] : 0;
                    PixelType r4 = (maskbyte & 0x08) ? images[detectorSize*f + p + 4] : 0;
                    PixelType r5 = (maskbyte & 0x04) ? images[detectorSize*f + p + 5] : 0;
                    PixelType r6 = (maskbyte & 0x02) ? images[detectorSize*f + p + 6] : 0;
                    PixelType r7 = (maskbyte & 0x01) ? images[detectorSize*f + p + 7] : 0;
                    res += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
                }
                result[resultIdx] = res;
            }
        }
    }
};

/**
 * Host main routine
 */
int main(int argc, char *argv[])
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    using boost::lexical_cast;
    using boost::bad_lexical_cast;
    std::vector<int> args;
    while (*++argv){
        try{
            args.push_back(lexical_cast<int>(*argv));
        }
        catch( const bad_lexical_cast &){
            args.push_back(0);
        }
    }

    if(args.size() != 4) {
        fprintf(stderr, "usage: ./NAME [detectorSize] [scanSize] [numMasks] [numStreams]\n");
        exit(1);
    }

    int detectorSize = args[0];
    int scanSize = args[1];
    int numMasks = args[2];
    const int numStreams = args[3];

    printf("parameters: detectorSize=%d scanSize=%d numMasks=%d\n",
            detectorSize, scanSize, numMasks);

    size_t datasetSize = scanSize * detectorSize; // pixels per dataset
    size_t resultSize = numMasks * scanSize;      // number of result items
    int masksSize = numMasks * detectorSize / numMaskBits;

    int streamSize = datasetSize / numStreams;
    int resultStreamSize = resultSize / numStreams;

    int frameStride = numMasks;

    if(datasetSize % numStreams != 0 || scanSize % numStreams != 0) {
        fprintf(stderr, "FIXME: datasetSize or scanSize not divisible by numStreams\n");
        exit(1);
    }

    if(detectorSize % numMaskBits != 0) {
        fprintf(stderr, "FIXME: detectorSize not divisible by numMaskBits\n");
        exit(1);
    }

    printf("allocating host memory: %ld bytes\n", sizeof(result_t) * resultSize + sizeof(mask_t) * masksSize
        + sizeof(pixel_t) * datasetSize);

    // init host buffers:
    result_t *results;
    mask_t *maskbuf;
    pixel_t *dataset;

    err = cuplaMallocHost((void**)&results, sizeof(result_t) * resultSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "faild to allocate host memory: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cuplaMallocHost((void**)&maskbuf, sizeof(mask_t) * masksSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "faild to allocate host memory: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cuplaMallocHost((void**)&dataset, sizeof(pixel_t) * datasetSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "faild to allocate host memory: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*
    results =(result_t*)malloc(sizeof(result_t) * resultSize);
    maskbuf = (mask_t*)malloc(sizeof(mask_t) * masksSize);
    dataset = (pixel_t*)malloc(sizeof(pixel_t) * datasetSize);
    */

    printf("allocated host memory\n");

    for (int i = 0; i < resultSize; i++)
    {
        results[i] = 0x42;
    }

    srand(21);
    // init masks
    for (int i = 0; i < masksSize; i++)
    {
        maskbuf[i] = rand() % 0xFF;
    }

    // init source data buffer
    for (size_t i = 0; i < datasetSize; i++)
    {
        dataset[i] = 1;
    }

    printf("allocating device memory\n");

    std::vector<cuplaStream_t> streams;

    for(int i = 0; i < numStreams; i++) {
        cuplaStream_t stream;
        err = cudaStreamCreate(&stream);
        streams.push_back(stream);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "failed to create stream (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    // Allocate the device input vector dataset
    pixel_t *d_dataset = NULL;
    err = cudaMalloc((void **)&d_dataset, sizeof(pixel_t) * datasetSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector dataset (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector maskbuf
    mask_t *d_maskbuf = NULL;
    err = cudaMalloc((void **)&d_maskbuf, sizeof(mask_t) * masksSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector masks (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector results
    result_t *d_results = NULL;
    err = cudaMalloc((void **)&d_results, sizeof(result_t) * resultSize);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector results (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_maskbuf, maskbuf, masksSize * sizeof(mask_t), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector maskbuf from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the CUDA Kernel
    dim3 threadsPerBlock(4, 64);
    //dim3 threadsPerBlock(2, 1);
    //dim3 threadsPerBlock(1, 1);
    //
    //
    if(numMasks < 4) {
        threadsPerBlock.x = numMasks;
    }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    cudaProfilerStart();
#endif
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    dim3 blocksPerGrid(
        // ceil(numMasks, threadsPerBlock.x)
        (numMasks + threadsPerBlock.x - 1) / threadsPerBlock.x,
        // ceil(scanSize, threadsPerBlock.y)
        (scanSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    //CUPLA_KERNEL_OPTI(maskedSum)


    for (int stream = 0; stream < 1; stream++) {
        int offset = stream * streamSize;

        err = cudaMemcpyAsync(
            &d_dataset[offset], &dataset[offset],
            streamSize * sizeof(pixel_t),
            cudaMemcpyHostToDevice,
            streams[stream]
        );

        int resultOffset = stream * resultStreamSize;
        CUPLA_KERNEL_OPTI(maskedSum)
        (blocksPerGrid, threadsPerBlock, 0, streams[stream])(&d_dataset[offset], &d_results[resultOffset], d_maskbuf, numMasks, scanSize / numStreams, detectorSize, frameStride);
        err = cudaGetLastError();


        err = cudaMemcpyAsync(&results[resultOffset], &d_results[resultOffset],
                sizeof(result_t) * resultStreamSize, cudaMemcpyDeviceToHost, streams[stream]);
    }

    for (int stream = 1; stream < numStreams; stream++) {
        int offset = stream * streamSize;

        err = cudaMemcpyAsync(
            &d_dataset[offset], &dataset[offset],
            streamSize * sizeof(pixel_t),
            cudaMemcpyHostToDevice,
            streams[stream]
        );
    }

    for (int stream = 1; stream < numStreams; stream++) {
        int offset = stream * streamSize;
        int resultOffset = stream * resultStreamSize;
        CUPLA_KERNEL_OPTI(maskedSum)
        (blocksPerGrid, threadsPerBlock, 0, streams[stream])(&d_dataset[offset], &d_results[resultOffset], d_maskbuf, numMasks, scanSize / numStreams, detectorSize, frameStride);
        err = cudaGetLastError();


        err = cudaMemcpyAsync(&results[resultOffset], &d_results[resultOffset],
                sizeof(result_t) * resultStreamSize, cudaMemcpyDeviceToHost, streams[stream]);
    }


    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    cudaProfilerStop();
#endif

    std::cout << "Time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms" << std::endl;
    
    std::vector<unsigned int> maskPopcounts;

    for(int i = 0; i < numMasks; i++) {
        unsigned int totalMaskPopcount = 0;
        for(int j = 0; j < detectorSize / numMaskBits; j++) {
            totalMaskPopcount += popcount(maskbuf[(i * (detectorSize / numMaskBits)) + j]);
        }
        maskPopcounts.push_back(totalMaskPopcount);
        fprintf(stderr, "totalMaskPopcount %d: %d\n", i, totalMaskPopcount);
    }


    // Verify that the result vector is correct
    for (int i = 0; i < resultSize; ++i)
    {
#if 1
        // alternative validation with random masks:
        int currMask = i % frameStride;
        int expected = maskPopcounts[currMask];
#else
        int expected = detectorSize;
#endif
        fprintf(stderr, "result element %d = %f\n", i, results[i]);
        if (results[i] != (float)expected)
        {
            fprintf(stderr, "Result verification failed at element %d value was %f, expected %d!\n", i, results[i], expected);
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
    cuplaFreeHost(dataset);
    cuplaFreeHost(maskbuf);
    cuplaFreeHost(results);

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
