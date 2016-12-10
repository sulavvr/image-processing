import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

def grayscale():
	print('Start memory allocations, memory copy & GPU filtering...')
	# start timer
	start.record()
	# allocate memory on the device with the number of bytes
	d_r = cuda.mem_alloc(h_r.nbytes)
	d_g = cuda.mem_alloc(h_g.nbytes)
	d_b = cuda.mem_alloc(h_b.nbytes)
	d_gray = cuda.mem_alloc(h_r.nbytes)

	# copy host to device
	cuda.memcpy_htod(d_r, h_r)
	cuda.memcpy_htod(d_g, h_g)
	cuda.memcpy_htod(d_b, h_b)

	# define the kernel function (GRAYSCALE)
	mod = SourceModule("""
		__global__ void grayscale(float *r, float *g, float *b, float *gray, int height, int width) {
			int x = blockIdx.x * blockDim.x + threadIdx.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;

			if ((x < height) && (y < width)) {
				int idx = x * width + y;
				int calc = (fmaxf(r[idx], fmaxf(g[idx], b[idx])) + fminf(r[idx], fminf(g[idx], b[idx]))) / 2;
				gray[idx] = calc;
			}
		}
	""")

	# calculate number of blocks in grid based on the image height and width
	ix = int(round(height / BLOCK_SIZE))
	iy = int(round(width / BLOCK_SIZE))
	# get function name from SourceModule
	func = mod.get_function('grayscale')
	# pass arguments to function along with block and grid size
	func(d_r, d_g, d_b, d_gray, height, width, block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=(ix, iy, 1))

	# stop timer
	stop.record()
	# synchronize events
	stop.synchronize()
	print('Finish processing image.\n')

	time_taken = start.time_till(stop)
	print('GPU Conversion Time (GRAYSCALE) ', time_taken, ' ms.\n')
	# create numpy array similar to h_r array, but empty
	h_gray = numpy.empty_like(h_r)
	# copy device to host
	cuda.memcpy_dtoh(h_gray, d_gray)

	# convert values to unsigned int 8 bit (0 - 255)
	h_gray = numpy.uint8(h_gray)
	# numpy repeat, copies the array 3 times for R G B channels
	h_gray = numpy.repeat(h_gray, 3)

	print('Writing channels to file...')
	# write to GPU_grayscale.raw
	h_gray.tofile('grayscale_GPU.raw')

	print('Image file written successfully.\n')

def blur():
	print('Start memory allocations, memory copy & GPU filtering...')
	# start timer
	start.record()
	# allocate memory on the device with the number of bytes
	d_r = cuda.mem_alloc(h_r.nbytes)
	d_g = cuda.mem_alloc(h_g.nbytes)
	d_b = cuda.mem_alloc(h_b.nbytes)
	d_blur_r = cuda.mem_alloc(h_r.nbytes)
	d_blur_g = cuda.mem_alloc(h_g.nbytes)
	d_blur_b = cuda.mem_alloc(h_b.nbytes)

	# copy host to device
	cuda.memcpy_htod(d_r, h_r)
	cuda.memcpy_htod(d_g, h_g)
	cuda.memcpy_htod(d_b, h_b)

	# define the kernel function (GRAYSCALE)
	mod = SourceModule("""
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
	""")

	# calculate number of blocks in grid based on the image height and width
	ix = int(round(height / BLOCK_SIZE))
	iy = int(round(width / BLOCK_SIZE))
	# get function name from SourceModule
	func = mod.get_function('convolute')
	# pass arguments to function along with block and grid size
	func(d_r, d_blur_r, height, width, block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=(ix, iy, 1))
	func(d_g, d_blur_g, height, width, block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=(ix, iy, 1))
	func(d_b, d_blur_b, height, width, block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=(ix, iy, 1))

	# stop timer
	stop.record()
	# synchronize events
	stop.synchronize()
	print('Finish processing image.\n')

	time_taken = start.time_till(stop)
	print('GPU Conversion Time (BLUR) ', time_taken, ' ms.\n')
	# create numpy array similar to h_r array, but empty
	h_blur_r = numpy.empty_like(h_r)
	h_blur_g = numpy.empty_like(h_r)
	h_blur_b = numpy.empty_like(h_r)
	# copy device to host
	cuda.memcpy_dtoh(h_blur_r, d_blur_r)
	cuda.memcpy_dtoh(h_blur_g, d_blur_g)
	cuda.memcpy_dtoh(h_blur_b, d_blur_b)

	h_blur_r = numpy.uint8(h_blur_r)
	h_blur_g = numpy.uint8(h_blur_g)
	h_blur_b = numpy.uint8(h_blur_b)

	print('Writing channels to file...')
	# write to GPU_grayscale.raw
	# numpy.tofile('grayscale_GPU.raw', (h_blur_r, h_blur_g, h_blur_b))
	with open('blur_GPU.raw', 'wb') as w:
		for r in range(total_px):
			w.write(h_blur_r[r])
			w.write(h_blur_g[r])
			w.write(h_blur_b[r])
	print('Image file written successfully.\n')

BLOCK_SIZE = 32
filename = '600x600.raw'
# define height and width
height = numpy.intc(600)
width = numpy.intc(600)
total_px = height * width	# get total pixels
r_ch, g_ch, b_ch = [], [], []

# display all numpy values
# numpy.set_printoptions(threshold=numpy.inf)
# 
# open file for reading in binary mode
img = open(filename, 'rb') 

print('Reading file, storing channels from pixels....')
# get numpy array as uint8 type (0-255)
data = numpy.fromfile(img, dtype=numpy.uint8) 

# assign appropriate channels to appropriate lists
i = 0
while (i < data.shape[0] - 2):
	r_ch.append(data[i])
	g_ch.append(data[i+1])
	b_ch.append(data[i+2])
	i += 3

print('RGB channels stored succesfully.\n')

# define start and end events
start = cuda.Event()
stop = cuda.Event()

# covert to numpy array as float32 to pass to kernel function
h_r = numpy.array(r_ch, dtype=numpy.float32)
h_g = numpy.array(g_ch, dtype=numpy.float32)
h_b = numpy.array(b_ch, dtype=numpy.float32)

grayscale()
blur()
