#pragma once
#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H

#include <vector>
#include <fstream>
using namespace std;

class ImageProcessing {
public:
	ImageProcessing();
	~ImageProcessing();
	// read and write Photoshop raw image
	void readRawImage(char *filename, int height, int width);
	void writeRawImage(char *filename, vector<float> &r, vector<float> &g, vector<float> &b);

	// host image processing
	void convertToGrayscale(char *filename);
	void blurImage(char *filename);
	vector<float> convolute(vector<float> &channel, int height, int width);

	// device image processing
	void runGrayscaleOnDevice(char *filename);
	void runBlurOnDevice(char *filename);
	void runBlurOnDeviceSingleStream(char *filename);
	void runBlurOnDeviceMultiStreams(char *filename);
	
private:
	ifstream r_img;		// read image
	ofstream w_img;		// write image

	vector<float> r;	// image red channel
	vector<float> g;	// image green channel
	vector<float> b;	// image blue channel

	int height;			// image height
	int width;			// image width
};

#endif
