#pragma once

float* createCUDABuffer(size_t s, void* data);
void getCudaData(float* buffer, size_t s, float** data);
