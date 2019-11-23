
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <random>

#define WRAP_SIZE 32
#define N 2048
#define M 2048
#define P 2048
#define A_SIZE N *P
#define B_SIZE P *M
#define C_SIZE N *M

cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int n, unsigned int m, unsigned int p);

void cpu__mult_matrix(float *c, const float *a, const float *b, unsigned int n, unsigned int m, unsigned int p);

void fill_rand(float *a, unsigned int size);

bool checkEquals(float *a, float *b, unsigned int n, unsigned int m);

void printMatrix(float* a, unsigned int n, unsigned int m);

int main()
{
    float *a = (float *)calloc(A_SIZE, sizeof(float));
    float *b = (float *)calloc(B_SIZE, sizeof(float));
    float *c = (float *)calloc(C_SIZE, sizeof(float));
    float *d = (float *)calloc(C_SIZE, sizeof(float));

    fill_rand(a, A_SIZE);
    fill_rand(b, B_SIZE);

    std::clock_t start = std::clock();
    cpu__mult_matrix(d, a, b, N, M, P);
    printf("\nCPU's time spent executing: %.9f seconds\n", (std::clock() - start) / (double)CLOCKS_PER_SEC);

    cudaError_t cudaStatus = addWithCuda(c, a, b, N, M, P);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }/*
    printMatrix(d, N, M);
    printf("\n\r --------\n\r");
    printMatrix(c, N, M);*/
    printf("matrixs is equals: %d\n", checkEquals(c, d, N, M));
    return 0;
}

cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int n, unsigned int m, unsigned int p)
{
    float *dev_a = 0, *dev_b = 0, *dev_c = 0;
    dim3 threads(WRAP_SIZE, WRAP_SIZE);
    dim3 grid(N / WRAP_SIZE, M);
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA start event: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA end event: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaEventRecord(start, 0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Cannot record CUDA event: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaEventSynchronize(start);

    cudaStatus = cudaMalloc((void **)&dev_c, C_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&dev_a, A_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&dev_b, B_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy a array from host to device: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    mmkernel<<<grid, threads>>>(dev_c, dev_a, dev_b, n, m, p);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, C_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaEventRecord(stop, 0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaEventSynchronize(stop);

    cudaStatus = cudaEventElapsedTime(&gpuTime, start, stop);
    printf("\nGPU's time spent executing %s: %.9f seconds\n", "kernel", gpuTime / 1000);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

void fill_rand(float *a, unsigned int size)
{
    for (int i = 0; i < size; i++)
        a[i] = (float)rand() / RAND_MAX;
}

void cpu__mult_matrix(float *c, const float *a, const float *b, unsigned int n, unsigned int m, unsigned int p)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < p; k++)
            {
                c[i * m + j] += a[i * p + k] * b[k * m + j];
            }
        }
    }
}

bool checkEquals(float *a, float *b, unsigned int n, unsigned int m)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < m; j++)
        {
            if (a[i * m + j] != b[i * m + j])
            {
                return false;
            }
        }
    }
    return true;
}

void printMatrix(float *a, unsigned int n, unsigned int m)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < m; j++)
        {
            printf("%.3f ", a[i * m + j]);
        }
    }
    printf("\n\r");
}