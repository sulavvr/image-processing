#pragma once
#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <tuple>

// kernel functions
__global__ void grayscale(float *r, float *g, float *b, float *gray, int height, int width);
__global__ void convolute(float *ch, float *res, int height, int width);

std::vector<float> deviceGrayscale(float *r, float *g, float *b, int height, int width);
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> deviceBlur(float *r, float *g, float *b, int height, int width);
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> deviceBlurSingleStream(float *r, float *g, float *b, int height, int width);
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> deviceBlurMultiStreams(float *r, float *g, float *b, int height, int width);
void checkCudaError(cudaError_t result, char *loc);
void startTime();	// start time
void stopTime(char *type);	// stop time

#endif