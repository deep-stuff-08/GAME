#pragma once

#include<vector>
#define CL_TARGET_OPENCL_VERSION 120
#include<CL/cl.h>

//Stub
typedef int error_t;

namespace GAME {
	enum GAME_config_platform {
		GAME_DEFAULT,
		GAME_CUDA,
		GAME_OPENCL,
		GAME_OPENMP
	};
	
	enum kerneltype {
		ADD_KERNEL = 0,
		SUB_KERNEL = 1,
		MUL_KERNEL = 2,
		DIV_KERNEL = 3
	};

	struct memobj {
		union {
			cl_mem clmemory;
			float* cumemory;
		} memory;
		size_t size;
	};

	class context {
	public:
		context() {

		}
		virtual memobj createBuffer(size_t sizeLen, void* data) = 0;
		virtual memobj callKernel(memobj a, memobj b, kerneltype kernel) = 0;
		virtual std::vector<float> getBufferData(memobj m) = 0;
		virtual ~context();
	};

	class engine {
	private:
		GAME_config_platform platform;
		context* current_context;
	public:
		engine(GAME_config_platform platform = GAME_DEFAULT);
		memobj createDataObject(std::vector<float> data);
		memobj addVectors(memobj a, memobj b);
		memobj subtractVectors(memobj a, memobj b);
		memobj multiplyVectors(memobj a, memobj b);
		memobj divideVectors(memobj a, memobj b);
		std::vector<float> getDataObject(memobj m);
		~engine();
	};
}