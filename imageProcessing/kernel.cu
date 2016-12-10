#include <iostream>
#include "kernel.cuh"
#include "math_functions.h"
#include <vector>
using namespace std;

float elapsed;
cudaEvent_t start, stop;
#define BLOCK_SIZE 32
// kernel function to gray an image
__global__ void grayscale(float *r, float *g, float *b, float *gray, int height, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;	// width index
	int y = blockIdx.y * blockDim.y + threadIdx.y;	// height index

	if ((x < width) && (y < height)) {
		int idx = x * width + y;	// current pixel index

		//Gray = (Max(Red, Green, Blue) + Min(Red, Green, Blue)) / 2
		uint8_t calc = (fmaxf(r[idx], fmaxf(g[idx], b[idx])) + fminf(r[idx], fminf(g[idx], b[idx]))) / 2;

		gray[idx] = calc;
	}
}

// kernel function to blur a single color channel (R || G || B)
__global__ void convolute(float *ch, float *res, int height, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;	// width index
	int y = blockIdx.y * blockDim.y + threadIdx.y;	// height index

	int radius = 8;
	float PI = atanf(1) * 4;
	if ((x < width) && (y < height)) {
		float sum = 0;
		float val = 0;
		int idx = x * width + y;	// current pixel index

		for (int i = y - radius; i < y + radius + 1; i++) {
			for (int j = x - radius; j < x + radius + 1; j++) {
				int h = fminf(height - 1, fmaxf(0, i));
				int w = fminf(width - 1, fmaxf(0, j));
				int dsq = (j - x) * (j - x) + (i - y) * (i - y);
				float wght = expf(-dsq / (2 * radius * radius)) / (PI * 2 * radius * radius);
				
				val += ch[w * width + h] * wght;
				sum += wght;
			}
		}
		res[idx] = round(val / sum);
	}
}

// deviceGrayscale allocates memory in host and device and copies data to and from host and device
// and also calls the appropriate kernel function for turning an image into black and white
vector<float> deviceGrayscale(float *r, float *g, float *b, int height, int width) {
	float *d_r;
	float *d_g;
	float *d_b;
	float *d_gray;

	float *h_gray;
	int size = height * width * sizeof(float);


	// start execution check
	startTime();

	h_gray = new float[size];
	checkCudaError(cudaMalloc((void **)&d_r, size), "cudaMalloc d_r");
	checkCudaError(cudaMalloc((void **)&d_g, size), "cudaMalloc d_g");
	checkCudaError(cudaMalloc((void **)&d_b, size), "cudaMalloc d_b");
	checkCudaError(cudaMalloc((void **)&d_gray, size), "cudaMalloc d_gray");

	checkCudaError(cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice), "cudaMemcpy r to d_r");
	checkCudaError(cudaMemcpy(d_g, g, size, cudaMemcpyHostToDevice), "cudaMemcpy g to d_g");
	checkCudaError(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice), "cudaMemcpy b to d_b");

	int x = (int)(ceilf((float)(height) / BLOCK_SIZE));
	int y = (int)(ceilf((float)(width) / BLOCK_SIZE));

	const dim3 grid_size(x, y);
	const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

	grayscale <<<grid_size, block_size >>> (d_r, d_g, d_b, d_gray, height, width);

	checkCudaError(cudaMemcpy(h_gray, d_gray, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_gray to h_gray");

	checkCudaError(cudaFree(d_r), "cudaFree d_r");
	checkCudaError(cudaFree(d_g), "cudaFree d_g");
	checkCudaError(cudaFree(d_b), "cudaFree d_b");
	checkCudaError(cudaFree(d_gray), "cudaFree d_gray");

	// stop and print execution time
	stopTime("GRAYSCALE");

	vector<float> gray_channel;

	for (int i = 0; i < height * width; i++) {
		gray_channel.push_back(h_gray[i]);
	}

	delete[]h_gray;

	return gray_channel;
}

// deviceBlur allocates memory in host and device and copies data to and from host and device
// and also calls the appropriate kernel function to blur (gaussian) an image
tuple<vector<float>, vector<float>, vector<float>> deviceBlur(float *r, float *g, float *b, int height, int width) {
	float *d_r;
	float *d_g;
	float *d_b;
	float *d_blur;
	
	float *h_blur_r;
	float *h_blur_g;
	float *h_blur_b;

	int size = height * width * sizeof(float);

	h_blur_r = new float[size];
	h_blur_g = new float[size];
	h_blur_b = new float[size];


	checkCudaError(cudaMalloc((void **)&d_r, size), "cudaMalloc d_r");
	checkCudaError(cudaMalloc((void **)&d_g, size), "cudaMalloc d_g");
	checkCudaError(cudaMalloc((void **)&d_b, size), "cudaMalloc d_b");
	checkCudaError(cudaMalloc((void **)&d_blur, size), "cudaMalloc d_blur");

	startTime();

	checkCudaError(cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice), "cudaMemcpy r to d_r");
	checkCudaError(cudaMemcpy(d_g, g, size, cudaMemcpyHostToDevice), "cudaMemcpy g to d_g");
	checkCudaError(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice), "cudaMemcpy b to d_b");

	int x = (int)(ceilf((float)(height) / BLOCK_SIZE));
	int y = (int)(ceilf((float)(width) / BLOCK_SIZE));

	const dim3 grid_size(x, y);
	const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);


	convolute << <grid_size, block_size >> > (d_r, d_blur, height, width);
	checkCudaError(cudaMemcpy(h_blur_r, d_blur, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_blur to h_blur_r");

	convolute << <grid_size, block_size >> > (d_g, d_blur, height, width);
	checkCudaError(cudaMemcpy(h_blur_g, d_blur, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_blur to h_blur_g");

	convolute << <grid_size, block_size >> > (d_b, d_blur, height, width);
	checkCudaError(cudaMemcpy(h_blur_b, d_blur, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_blur to h_blur_b");

	stopTime("BLUR");
	// free allocated memory on device
	checkCudaError(cudaFree(d_r), "cudaFree d_r");
	checkCudaError(cudaFree(d_g), "cudaFree d_g");
	checkCudaError(cudaFree(d_b), "cudaFree d_b");
	checkCudaError(cudaFree(d_blur), "cudaFree d_blur");

	

	vector<float> r_ch, g_ch, b_ch;

	for (int i = 0; i < height * width; i++) {
		r_ch.push_back(h_blur_r[i]);
		g_ch.push_back(h_blur_g[i]);
		b_ch.push_back(h_blur_b[i]);
	}

	// free allocated memory on host
	delete[]h_blur_r;
	delete[]h_blur_g;
	delete[]h_blur_b;
	
	
	return make_tuple(r_ch, g_ch, b_ch);
}

// Single stream (Stream 0)
tuple<vector<float>, vector<float>, vector<float>> deviceBlurSingleStream(float *r, float *g, float *b, int height, int width) {
	float *d_r;
	float *d_g;
	float *d_b;
	float *d_blur;

	float *h_blur_r;
	float *h_blur_g;
	float *h_blur_b;

	int size = height * width * sizeof(float);

	cudaDeviceProp properties;
	int device;
	checkCudaError(cudaGetDevice(&device), "cudaGetDevice");
	checkCudaError(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
	// check if device handles overlaps
	if (!properties.deviceOverlap) {
		cout << "Device cannot handle overlaps\n";
		exit(1);
	}

	cudaStream_t stream;
	checkCudaError(cudaStreamCreate(&stream), "cudaStreamCreate");

	checkCudaError(cudaMallocHost((void **)&h_blur_r, size), "cudaMallocHost h_res_r");
	checkCudaError(cudaMallocHost((void **)&h_blur_g, size), "cudaMallocHost h_res_g");
	checkCudaError(cudaMallocHost((void **)&h_blur_b, size), "cudaMallocHost h_res_b");

	startTime();

	checkCudaError(cudaMalloc((void **)&d_r, size), "cudaMalloc d_r");
	checkCudaError(cudaMalloc((void **)&d_g, size), "cudaMalloc d_g");
	checkCudaError(cudaMalloc((void **)&d_b, size), "cudaMalloc d_b");
	checkCudaError(cudaMalloc((void **)&d_blur, size), "cudaMalloc d_blur");

	checkCudaError(cudaMemcpyAsync(d_r, r, size, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync r to d_r");
	checkCudaError(cudaMemcpyAsync(d_g, g, size, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync g to d_g");
	checkCudaError(cudaMemcpyAsync(d_b, b, size, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync b to d_b");

	int x = (int)(ceilf((float)(height) / BLOCK_SIZE));
	int y = (int)(ceilf((float)(width) / BLOCK_SIZE));

	const dim3 grid_size(x, y);
	const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);


	convolute << <grid_size, block_size, 0, stream >> > (d_r, d_blur, height, width);
	checkCudaError(cudaMemcpyAsync(h_blur_r, d_blur, size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync d_blur to h_blur_r");

	convolute << <grid_size, block_size, 0, stream >> > (d_g, d_blur, height, width);
	checkCudaError(cudaMemcpyAsync(h_blur_g, d_blur, size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync d_blur to h_blur_g");

	checkCudaError(cudaThreadSynchronize(), "cudaThreadSync 1");

	convolute << <grid_size, block_size, 0, stream >> > (d_b, d_blur, height, width);
	checkCudaError(cudaMemcpyAsync(h_blur_b, d_blur, size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync d_blur to h_blur_b");

	checkCudaError(cudaThreadSynchronize(), "cudaThreadSync 2");

	checkCudaError(cudaStreamSynchronize(stream), "cudaStreamnSync 1");

	// free allocated memory on device
	checkCudaError(cudaFree(d_r), "cudaFree d_r");
	checkCudaError(cudaFree(d_g), "cudaFree d_g");
	checkCudaError(cudaFree(d_b), "cudaFree d_b");
	checkCudaError(cudaFree(d_blur), "cudaFree d_blur");

	checkCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy");
	stopTime("BLUR - SINGLE STREAM");

	vector<float> r_ch, g_ch, b_ch;

	for (int i = 0; i < height * width; i++) {
		r_ch.push_back(h_blur_r[i]);
		g_ch.push_back(h_blur_g[i]);
		b_ch.push_back(h_blur_b[i]);
	}

	// free page-locked memory
	checkCudaError(cudaFreeHost(h_blur_r), "cudaFreeHost h_blur_r");
	checkCudaError(cudaFreeHost(h_blur_g), "cudaFreeHost h_blur_g");
	checkCudaError(cudaFreeHost(h_blur_b), "cudaFreeHost h_blur_b");


	return make_tuple(r_ch, g_ch, b_ch);
}

// Multiple streams
tuple<vector<float>, vector<float>, vector<float>> deviceBlurMultiStreams(float *r, float *g, float *b, int height, int width) {
	float *d_r;
	float *d_g;
	float *d_b;
	float *d_blur;

	float *h_blur_r;
	float *h_blur_g;
	float *h_blur_b;

	const int num_streams = 2;
	int size = height * width;
	int stream_size = size / num_streams;
	int stream_bytes = stream_size * sizeof(float);
	int bytes = size * sizeof(float);

	cudaDeviceProp properties;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&properties, device);
	// check if device handles overlaps
	if (!properties.deviceOverlap) {
		cout << "Device cannot handle overlaps\n";
		exit(1);
	}

	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++) {
		checkCudaError(cudaStreamCreate(&streams[i]), "stream create");
	}

	// allocate page-locked memory
	checkCudaError(cudaMallocHost((void **)&h_blur_r, bytes), "cuda malloc host h_blur_r");
	checkCudaError(cudaMallocHost((void **)&h_blur_g, bytes), "cuda malloc host h_blur_g");
	checkCudaError(cudaMallocHost((void **)&h_blur_b, bytes), "cuda malloc host h_blur_b");

	startTime();

	checkCudaError(cudaMalloc((void **)&d_r, bytes), "cuda malloc dev d_r");
	checkCudaError(cudaMalloc((void **)&d_g, bytes), "cuda malloc dev d_g");
	checkCudaError(cudaMalloc((void **)&d_b, bytes), "cuda malloc dev d_b");
	checkCudaError(cudaMalloc((void **)&d_blur, bytes), "cuda malloc dev d_blur");

	int x = (int)(ceilf((float)(height) / BLOCK_SIZE));
	int y = (int)(ceilf((float)(width) / BLOCK_SIZE));

	const dim3 grid_size(x, y);
	const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

	for (int i = 0; i < num_streams; i++) {
		int offset = i * stream_size;

		checkCudaError(cudaMemcpyAsync(&d_r[offset], &r[offset], stream_bytes, cudaMemcpyHostToDevice, streams[i]), "cuda memcpy d_r");
		checkCudaError(cudaMemcpyAsync(&d_g[offset], &g[offset], stream_bytes, cudaMemcpyHostToDevice, streams[i]), "cuda memcpy d_g");
		checkCudaError(cudaMemcpyAsync(&d_b[offset], &b[offset], stream_bytes, cudaMemcpyHostToDevice, streams[i]), "cuda memcpy d_b");

		convolute << <grid_size, block_size, 0, streams[i] >> > (&d_r[offset], d_blur, height, width);
		checkCudaError(cudaMemcpyAsync(&h_blur_r[offset], d_blur, stream_bytes, cudaMemcpyDeviceToHost, streams[i]), "cuda memcpy h_blur_r");

		convolute << <grid_size, block_size, 0, streams[i] >> > (&d_g[offset], d_blur, height, width);
		checkCudaError(cudaMemcpyAsync(&h_blur_g[offset], d_blur, stream_bytes, cudaMemcpyDeviceToHost, streams[i]), "cuda memcpy h_blur_g");

		convolute << <grid_size, block_size, 0, streams[i] >> > (&d_b[offset], d_blur, height, width);
		checkCudaError(cudaMemcpyAsync(&h_blur_b[offset], d_blur, stream_bytes, cudaMemcpyDeviceToHost, streams[i]), "cuda memcpy h_blur_b");

		//checkCudaError(cudaStreamSynchronize(streams[i]), "cuda stream sync");
	}

	checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSync");

	// free allocated memory on device
	checkCudaError(cudaFree(d_r), "cudaFree d_r");
	checkCudaError(cudaFree(d_g), "cudaFree d_g");
	checkCudaError(cudaFree(d_b), "cudaFree d_b");
	checkCudaError(cudaFree(d_blur), "cudaFree d_blur");

	for (int i = 0; i < num_streams; i++) {
		checkCudaError(cudaStreamDestroy(streams[i]), "cudastreamdestroy");
	}

	stopTime("BLUR - MULTI STREAM");

	vector<float> r_ch, g_ch, b_ch;

	for (int i = 0; i < height * width; i++) {
		r_ch.push_back(h_blur_r[i]);
		g_ch.push_back(h_blur_g[i]);
		b_ch.push_back(h_blur_b[i]);
	}

	// free page-locked memory
	checkCudaError(cudaFreeHost(h_blur_r), "cudafreehost h_res_r");
	checkCudaError(cudaFreeHost(h_blur_g), "cudafreehost h_res_g");
	checkCudaError(cudaFreeHost(h_blur_b), "cudafreehost h_res_b");


	return make_tuple(r_ch, g_ch, b_ch);
}

// print error and exit if result is not cudaSuccess
void checkCudaError(cudaError_t result, char *loc) {
	if (result != cudaSuccess) {
		cout << "ERROR at " << loc << " - " << cudaGetErrorString(result) << endl;
		exit(1);
	}
}

void startTime() {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void stopTime(char *type) {
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cout << "GPU Conversion Time (" << type << "): " << elapsed << " ms.\n";
}