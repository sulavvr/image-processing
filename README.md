# CUDA Image Processing

- (cudacpp<sup>1</sup>) CUDA C++ Image Processing Project for High Performance Computing
- (pycuda<sup>2</sup>) PyCUDA Image Processing Project for High Performance Computing

## Image filters implemented
### Grayscale
- Formula:

> gray = (Max(R, G, B) + Min(R, G, B)) / 2

### Gaussian Blur
- Gaussian Function:

> G(x, y) = (1 / 2 * PI * sigma²) * exp <sup>(-(x² + y²))/ 2 * sigma²</sup>

## Notes
- Implementation using C++11 & CUDA 8.0 (cudacpp<sup>1</sup>)
- Implementation using PyCUDA & CUDA 8.0 (pycuda<sup>2</sup>)
- Windows 10 | Visual Studio 2015 (cudacpp<sup>1</sup>])
- Used Photoshop RAW image (*.raw) for reading the RGB channels
- Host code implemented for CUDA C++ only
- Device code implemented for PyCUDA & CUDA C++
- Contains default stream (stream 0), non-default stream and multi-streams codes for Gaussian Blur for CUDA C++ implementation
- **Multi-streams implementation not working fully (image contains black lines for number of stream - 1)**
- Depth-First multi-streams implemented, Breadth-First faster.
- Same kernel function used for PyCUDA & CUDA C++

## Reference
- [Gaussian Blur](http://www.pixelstech.net/article/1353768112-Gaussian-Blur-Algorithm)
- [Grayscale Algorithms](http://www.tannerhelland.com/3643/grayscale-image-algorithm-vb6/)

1: cudacpp branch

2: pycuda branch
