#include"include/GAME/OpenCLContext.h"
#include"clhelper/clhelper.h"
#include<iostream>

using namespace std;
using namespace GAME;

opencl_context::opencl_context() : context(){
	this->kernelList = new cl_kernel[4];

	cout<<initOpenCLContext(&this->context);
	cout<<compilePrograms(&this->context, { "../src/cl/BLAS.cl" });
	cout<<getKernelFromProgram(&this->context, "add", &this->kernelList[ADD_KERNEL]);
	cout<<getKernelFromProgram(&this->context, "sub", &this->kernelList[SUB_KERNEL]);
	cout<<getKernelFromProgram(&this->context, "mul", &this->kernelList[MUL_KERNEL]);
	cout<<getKernelFromProgram(&this->context, "div", &this->kernelList[DIV_KERNEL]);
}

memobj opencl_context::createBuffer(size_t size, void* data) {
	memobj m;
	cl_int err;
	m.memory.clmemory = clCreateBuffer(this->context.context, CL_MEM_READ_WRITE, size, NULL, &err);
	if(err != CL_SUCCESS) {
		cout<<err<<" at "<<__LINE__<<endl;
	}
	err = clEnqueueWriteBuffer(this->context.cmdQueue, m.memory.clmemory, CL_TRUE, 0, size, data, 0, NULL, NULL);
	if(err != CL_SUCCESS) {
		cout<<err<<" at "<<__LINE__<<endl;
	}
	m.size = size;
	return m;
}

memobj opencl_context::callKernel(memobj a, memobj b, kerneltype kernel) {
	memobj m;
	cl_int err;
	m.memory.clmemory = clCreateBuffer(this->context.context, CL_MEM_READ_WRITE, a.size, NULL, &err);
	m.size = a.size;
	if(err != CL_SUCCESS) {
		cout<<err<<" at "<<__LINE__<<endl;
	}
	size_t globalSize = a.size / sizeof(float);
	int lenght = globalSize;
	cout<<setKernelParameters(this->kernelList[kernel], { param(0, a.memory.clmemory), param(1, b.memory.clmemory), param(2, m.memory.clmemory), param(3, lenght)});
	cout<<runCLKernel(&this->context, this->kernelList[kernel], 1, &globalSize, NULL, {});
	return m;
}

vector<float> opencl_context::getBufferData(memobj m) {
	float *data = (float*)malloc(m.size);
	cl_int err;
	err = clEnqueueReadBuffer(this->context.cmdQueue, m.memory.clmemory, CL_TRUE, 0, m.size, data, 0, NULL, NULL);
	if(err != CL_SUCCESS) {
		cout<<err<<" at "<<__LINE__<<endl;
	}
	vector<float> datastore;
	datastore.insert(datastore.end(), data, data + (m.size / sizeof(float)));
	return datastore;
}

opencl_context::~opencl_context() {
	
}