#pragma once

#include"GAME.h"
#include"../../clhelper/clhelper.h"

namespace GAME {
	class opencl_context : public GAME::context {
	public:
		opencl_context();
		memobj createBuffer(size_t size, void* data);
		memobj callKernel(memobj a, memobj b, kerneltype kernel);
		std::vector<float> getBufferData(memobj m);
		~opencl_context();
	private:
		clContext_dl context;
		cl_kernel* kernelList;
	};
}