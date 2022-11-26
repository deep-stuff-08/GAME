#include"include/GAME/CUDAContext.h"
#include"cuhelper/cuhelper.cuh"
#include"include/blas.cuh"
#include<iostream>


using namespace std;
using namespace GAME;

cuda_context::cuda_context() : context(){
	
}

memobj cuda_context::createBuffer(size_t size, void* data) {
	memobj m;
	m.size = size;
	m.memory.cumemory = createCUDABuffer(size, data);
	return m;
}

memobj cuda_context::callKernel(memobj a, memobj b, kerneltype kernel) {
	memobj m;
	m.size = a.size;
	m.memory.cumemory = createCUDABuffer(a.size, NULL);
	if(kernel == ADD_KERNEL) {
		addKernel(a.memory.cumemory, b.memory.cumemory, m.memory.cumemory, m.size / sizeof(float));
	} else if(kernel == SUB_KERNEL) {
		subKernel(a.memory.cumemory, b.memory.cumemory, m.memory.cumemory, m.size / sizeof(float));
	} else if(kernel == MUL_KERNEL) {
		mulKernel(a.memory.cumemory, b.memory.cumemory, m.memory.cumemory, m.size / sizeof(float));
	} else if(kernel == DIV_KERNEL) {
		divKernel(a.memory.cumemory, b.memory.cumemory, m.memory.cumemory, m.size / sizeof(float));
	}
	return m;
}

vector<float> cuda_context::getBufferData(memobj m) {
	float *data = (float*)malloc(m.size);
	getCudaData(m.memory.cumemory, m.size, &data);
	vector<float> datastore;
	datastore.insert(datastore.end(), data, data + (m.size / sizeof(float)));
	return datastore;
}

cuda_context::~cuda_context() {
	
}