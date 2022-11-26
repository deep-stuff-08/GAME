#pragma once

#include"GAME.h"

namespace GAME {
	class cuda_context : public GAME::context {
	public:
		cuda_context();
		memobj createBuffer(size_t size, void* data);
		memobj callKernel(memobj a, memobj b, kerneltype kernel);
		std::vector<float> getBufferData(memobj m);
		~cuda_context();
	private:
	};
}