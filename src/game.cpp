#include"include/GAME/GAME.h"
#include"include/GAME/OpenCLContext.h"
#include"include/GAME/CUDAContext.h"

using namespace std;
using namespace GAME;

engine::engine(GAME_config_platform platform) {
	if(platform == GAME_DEFAULT || platform == GAME_OPENCL) {
		this->platform = platform;
		this->current_context = new opencl_context();
	// } else if(platform == GAME_CUDA) {
	// 	this->platform = platform;
	// 	this->current_context = new cuda_context();
	} else {

	}
}

memobj engine::createDataObject(vector<float> data) {
	return this->current_context->createBuffer(sizeof(float) * data.size(), data.data());
}

memobj engine::addVectors(memobj a, memobj b) {
	return this->current_context->callKernel(a, b, ADD_KERNEL);
}

memobj engine::subtractVectors(memobj a, memobj b) {
	return this->current_context->callKernel(a, b, SUB_KERNEL);
}

memobj engine::multiplyVectors(memobj a, memobj b) {
	return this->current_context->callKernel(a, b, MUL_KERNEL);
}

memobj engine::divideVectors(memobj a, memobj b) {
	return this->current_context->callKernel(a, b, DIV_KERNEL);
}

vector<float> engine::getDataObject(memobj m) {
	return this->current_context->getBufferData(m);
}

engine::~engine() {
	delete this->current_context;
}