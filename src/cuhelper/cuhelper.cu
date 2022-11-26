#include"cuhelper.cuh"
#include<stdio.h>

float* createCUDABuffer(size_t s, void* data) {
	cudaError_t err;
	float *cu;
	err = cudaMalloc(&cu, s);
	if(err != cudaSuccess) {
		printf("%d at %d\n", err, __LINE__);
	}

	if(data != NULL) {
		err = cudaMemcpy(cu, data, s, cudaMemcpyHostToDevice);
		if(err != cudaSuccess) {
			printf("%d at %d\n", err, __LINE__);
		}
	}
	return cu;
}

void getCudaData(float* buffer, size_t s, float** data) {
	cudaError_t err = cudaMemcpy(*data, buffer, s, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("%d at %d\n", err, __LINE__);
	}
}