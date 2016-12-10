#include <iostream>
#include "imageProcessing.h"
#include "cuda_runtime.h"
#include <stdio.h>

using namespace std;

int main(int argc, const char** argv) {
	ImageProcessing img;

	// read image
	img.readRawImage("2400x2400.raw", 2400, 2400);

	// ***********************CPU (HOST)*******************************

	cout << "Start CPU image filter... \n";
	// convert image to grayscale on CPU
	img.convertToGrayscale("grayscale_CPU.raw");

	img.blurImage("blur_CPU.raw");
	cout << "End CPU image filter\n\n";

	// ***********************GPU (DEVICE)*******************************
	
	cout << "Starting GPU image filter... \n";
	// convert image to grayscale on GPU
	img.runGrayscaleOnDevice("grayscale_GPU.raw");
	img.runBlurOnDevice("blur_GPU.raw");
	img.runBlurOnDeviceSingleStream("blur_GPU_ss.raw");
	img.runBlurOnDeviceMultiStreams("blur_GPU_ms.raw");
	cout << "GPU image filter end.\n\n";

	cout << "Image processing complete!" << endl;

	return 0;
}