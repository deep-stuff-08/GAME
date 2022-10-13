#pragma once

#include<complex.h>
#include<vector>

//Stub
typedef int error_t;

namespace GAME {
	enum GAME_config_platform {
		GAME_DEFAULT,
		GAME_CUDA,
		GAME_OPENCL,
		GAME_OPENMP
	};

	class context {
		virtual void needToFindCommonFunctionButNotFoundYet() = 0;
	};

	class engine {
	private:
		GAME_config_platform platform;
		context current_context;
	public:
		engine(GAMEConfigPlatform platform = GAME_DEFAULT);
		error_t fft1D(std::vector<std::complex> input, /*param, */ std::vector<std::complex> &output);
		~engine();
	};
}