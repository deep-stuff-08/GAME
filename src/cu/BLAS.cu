#include<stdio.h>
#include"../include/blas.cuh"

__global__ void add(float* a, float* b, float* c, int len) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < len) {
		c[id] = a[id] + b[id];
	}
}

__global__ void sub(float* a, float* b, float* c, int len) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < len) {
		c[id] = a[id] - b[id];
	}
}

__global__ void mul(float* a, float* b, float* c, int len) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < len) {
		c[id] = a[id] * b[id];
	}
}

__global__ void div(float* a, float* b, float* c, int len) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < len) {
		c[id] = a[id] / b[id];
	}
}

void addKernel(float* a, float* b, float* c, int len) {
	add<<<len, 1>>>(a, b, c, len);
	cudaError_t err = cudaDeviceSynchronize();
	if(err != cudaSuccess) {
		printf("%d at %d\n", err, __LINE__);
	}
}

void subKernel(float* a, float* b, float* c, int len) {
	sub<<<len, 1>>>(a, b, c, len);
	cudaError_t err = cudaDeviceSynchronize();
	if(err != cudaSuccess) {
		printf("%d at %d\n", err, __LINE__);
	}
}

void mulKernel(float* a, float* b, float* c, int len) {
	mul<<<len, 1>>>(a, b, c, len);
	cudaError_t err = cudaDeviceSynchronize();
	if(err != cudaSuccess) {
		printf("%d at %d\n", err, __LINE__);
	}
}

void divKernel(float* a, float* b, float* c, int len) {
	div<<<len, 1>>>(a, b, c, len);
	cudaError_t err = cudaDeviceSynchronize();
	if(err != cudaSuccess) {
		printf("%d at %d\n", err, __LINE__);
	}
}
