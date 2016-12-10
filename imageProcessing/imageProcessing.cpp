#include <iostream>
#include "imageProcessing.h"
#include "kernel.cuh"
#include <cmath>
#include <chrono>
#include <tuple>

using namespace std;
using namespace std::chrono;

ImageProcessing::ImageProcessing() {
	
}

void ImageProcessing::readRawImage(char *filename, int height, int width) {

	cout << "Reading image... \n";
	// open file
	this->r_img.open(filename, ifstream::in | ifstream::binary);

	if (!this->r_img.is_open()) {
		cerr << "File couldn't be open!\n";
		exit;
	}

	for (int i = 0; i < height * width; i++) {
		this->r.push_back(this->r_img.get());
		this->g.push_back(this->r_img.get());
		this->b.push_back(this->r_img.get());
	}

	this->r_img.close();
	
	this->height = height;
	this->width = width;

	cout << "Image read, stored RGB channels and dimensions. \n\n";
}

void ImageProcessing::writeRawImage(char *filename, vector<float> &r, vector<float> &g, vector<float> &b) {
	cout << "Writing RGB channels to file... \n";
	this->w_img.open(filename, ofstream::out | ofstream::binary);

	if (!this->w_img.is_open()) {
		cerr << "File couldn't be open!\n";
		exit;
	}

	for (int i = 0; i < this->height * this->width; i++) {
		this->w_img.put((int)r[i]);
		this->w_img.put((int)g[i]);
		this->w_img.put((int)b[i]);
	}

	this->w_img.close();
	cout << "Image file written successfully. \n\n";
}

void ImageProcessing::convertToGrayscale(char *filename) {
	vector<float> gray;

	// start
	high_resolution_clock::time_point start = high_resolution_clock::now();
	
	for (int i = 0; i < height * width; i++) {
		//Gray = (Max(Red, Green, Blue) + Min(Red, Green, Blue)) / 2
		uint8_t calc = (fmax(this->r[i], fmax(this->g[i], this->b[i])) + fmin(this->r[i], fmin(this->g[i], this->b[i]))) / 2;
		gray.push_back(calc);
	}
	// end
	high_resolution_clock::time_point stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start).count();	// time difference

	cout << "CPU Conversion Time (GRAYSCALE): " << duration << " ms.\n";

	this->writeRawImage(filename, gray, gray, gray);
}

void ImageProcessing::blurImage(char *filename) {
	// start
	high_resolution_clock::time_point start = high_resolution_clock::now();

	vector<float> n_r = this->convolute(this->r, this->height, this->width);
	vector<float> n_g = this->convolute(this->g, this->height, this->width);
	vector<float> n_b = this->convolute(this->b, this->height, this->width);

	// end
	high_resolution_clock::time_point stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start).count();	// time difference

	cout << "CPU Conversion Time (BLUR): " << duration << " ms.\n";

	this->writeRawImage(filename, n_r, n_g, n_b);
}

vector<float> ImageProcessing::convolute(vector<float> &channel, int height, int width) {
	int radius = 8;
	vector<float> c = channel;

	float PI = atan(1) * 4;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float sum = 0;
			float val = 0;
			for (int iy = i - radius; iy < i + radius + 1; iy++) {
				for (int ix = j - radius; ix < j + radius + 1; ix++) {
					float h = fmin(width - 1, fmax(0, ix));
					float w = fmin(height - 1, fmax(0, iy));
					float dsq = (ix - j) * (ix - j) + (iy - i) * (iy - i);
					float wght = exp(-dsq / (2 * radius * radius)) / (PI * 2 * radius * radius);
					val += channel[w * width + h] * wght;
					sum += wght;
				}
			}
			c[i * width + j] = round(val / sum);
		}
	}

	return c;
}

void ImageProcessing::runGrayscaleOnDevice(char *filename) {
	int size = this->height * this->width;
	float *r = new float[size];
	float *g = new float[size];
	float *b = new float[size];

	for (int i = 0; i < size; i++) {
		r[i] = this->r.at(i);
		g[i] = this->g.at(i);
		b[i] = this->b.at(i);
	}
	
	vector<float> gray = deviceGrayscale(r, g, b, this->height, this->width);
	
	delete []r;
	delete []g;
	delete []b;

	this->writeRawImage(filename, gray, gray, gray);

}

void ImageProcessing::runBlurOnDevice(char *filename) {
	int size = this->height * this->width;
	float *r = new float[size];
	float *g = new float[size];
	float *b = new float[size];

	for (int i = 0; i < size; i++) {
		r[i] = this->r.at(i);
		g[i] = this->g.at(i);
		b[i] = this->b.at(i);
	}

	tuple<vector<float>, vector<float>, vector<float>> res = deviceBlur(r, g, b, this->height, this->width);

	delete[]r;
	delete[]g;
	delete[]b;

	this->writeRawImage(filename, get<0>(res), get<1>(res), get<2>(res));
}

void ImageProcessing::runBlurOnDeviceSingleStream(char *filename) {
	int size = this->height * this->width;
	float *r = new float[size];
	float *g = new float[size];
	float *b = new float[size];

	for (int i = 0; i < size; i++) {
		r[i] = this->r.at(i);
		g[i] = this->g.at(i);
		b[i] = this->b.at(i);
	}

	tuple<vector<float>, vector<float>, vector<float>> res = deviceBlurSingleStream(r, g, b, this->height, this->width);

	delete[]r;
	delete[]g;
	delete[]b;

	this->writeRawImage(filename, get<0>(res), get<1>(res), get<2>(res));
}

void ImageProcessing::runBlurOnDeviceMultiStreams(char *filename) {
	int size = this->height * this->width;
	float *r = new float[size];
	float *g = new float[size];
	float *b = new float[size];

	for (int i = 0; i < size; i++) {
		r[i] = this->r.at(i);
		g[i] = this->g.at(i);
		b[i] = this->b.at(i);
	}

	tuple<vector<float>, vector<float>, vector<float>> res = deviceBlurMultiStreams(r, g, b, this->height, this->width);

	delete[]r;
	delete[]g;
	delete[]b;

	this->writeRawImage(filename, get<0>(res), get<1>(res), get<2>(res));
}

ImageProcessing::~ImageProcessing() {
} 