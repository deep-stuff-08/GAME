#ifndef __OPENCLDL__
#define __OPENCLDL__

#define CL_TARGET_OPENCL_VERSION 120
#include<CL/cl.h>
#include<vector>
#include<iostream>
#include<GL/gl.h>

#define param(p, x) {p, &x, sizeof(x)}

struct clContext_dl {
	cl_context context;
	cl_device_id device;
	cl_command_queue cmdQueue;
	std::vector<cl_kernel> kernels;
};

struct glclKernelParamater_dl {
	int position;
	void *param;
	size_t size;
};

std::string initOpenCLContext(clContext_dl *context, int platformNo = 0, int deviceNo = 0);
std::string compilePrograms(clContext_dl *context, std::vector<std::string> programNames);
std::string getKernelFromProgram(clContext_dl *context, std::string name, cl_kernel *ret);
std::string setKernelParameters(cl_kernel kernel, std::vector<glclKernelParamater_dl> kernelList);
std::string runCLKernel(clContext_dl *context, cl_kernel kernel, cl_uint workDims, size_t *globalSize, size_t *localSize, std::vector<cl_mem> globjects);

#endif