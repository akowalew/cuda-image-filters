///////////////////////////////////////////////////////////////////////////////
// filters.cu
// 
// Contains implementation of `filters` library
///////////////////////////////////////////////////////////////////////////////

#include "filters.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

namespace filters {
	
namespace {

//! Maximum size of the squared kernel
const auto KSizeMax = 64;

//! Fixed size constant buffer for convolution filter kernels
__constant__ float c_kernel[KSizeMax * KSizeMax];

// Number of threads in both X and Y dimensions in the block
const auto K = 32;

} // namespace

void check_fail(cudaError_t result, 
	char const *const func, const char *const file, int const line)
{
	fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", 
		file, line, result, cudaGetErrorString(result), func);
	exit(EXIT_FAILURE);
}

__host__
void init()
{
	check_errors(cudaSetDevice(0));
}

__host__
void cleanup()
{
	check_errors(cudaDeviceReset());
}

std::pair<uchar* /*d_img*/, size_t /*d_pitch*/> 
	create_image(size_t cols, size_t rows)
{
	uchar* d_img;
	size_t d_pitch;
	check_errors(cudaMallocPitch(&d_img, &d_pitch, 
		cols * sizeof(uchar), rows));

	return {d_img, d_pitch};
}

void free_image(uchar* d_img)
{
	check_errors(cudaFree(d_img));
}

void set_image(uchar* d_dst, size_t d_dpitch, 
	const uchar* src, size_t spitch, 
	size_t cols, size_t rows)
{
	const auto width = (cols * sizeof(uchar));
	const auto height = rows;
	check_errors(cudaMemcpy2D(d_dst, d_dpitch, src, spitch, 
		width, height, cudaMemcpyHostToDevice));
}

void get_image(uchar* dst, size_t dpitch,
	const uchar* d_src, size_t d_spitch,
	size_t cols, size_t rows)
{
	const auto width = (cols * sizeof(uchar));
	const auto height = rows;
	check_errors(cudaMemcpy2D(dst, dpitch, d_src, d_spitch, 
		width, height, cudaMemcpyDeviceToHost));
}

__host__
void set_kernel(const float* kernel, size_t ksize)
{
	// Ensure proper size of the kernel
	assert(ksize <= KSizeMax);

	// Copy data from host kernel to constant memory
	check_errors(cudaMemcpyToSymbol(c_kernel, kernel, 
		ksize * ksize * sizeof(float)));	
}

__host__
cv::Mat filter2d(const cv::Mat& src, const cv::Mat& kernel)
{
	auto dst = cv::Mat(src.rows, src.cols, src.type());

	filter2d(src, kernel, dst);

	return dst;
}

__host__
void filter2d(const cv::Mat& src, const cv::Mat& kernel, cv::Mat& dst)
{
	// Ensure, that images have equal size
	assert(src.rows == dst.rows);
	assert(src.cols == dst.cols);
	const auto cols = src.cols;
	const auto rows = src.rows;

	// Ensure, that image is laid without spaces between
	assert(src.isContinuous() && dst.isContinuous());
	const auto spitch = cols * sizeof(uchar);
	const auto dpitch = cols * sizeof(uchar);

	// Ensure proper type of images
	assert(src.type() == CV_8UC1 && dst.type() == CV_8UC1);
	const auto src_data = (const uchar*) src.data;
	const auto dst_data = (uchar*) dst.data;

	// Ensure, that kernel is squared
	assert(kernel.rows == kernel.cols);
	const auto ksize = kernel.rows;

	// Ensure proper type of kernel
	assert(kernel.type() == CV_32F);
	assert(kernel.isContinuous());
	const auto kernel_data = (const float*) kernel.data;

	// Invoke low-level filtering method
	filter2d(src_data, spitch, cols, rows,
		kernel_data, ksize,
		dst_data, dpitch);
}

__host__
void filter2d(
	const uchar* src, size_t spitch, size_t cols, size_t rows,
	const float* kernel, size_t ksize,
	uchar* dst, size_t dpitch)
{
	// Allocate memories
	uchar* d_src; size_t d_spitch;
	std::tie(d_src, d_spitch) = create_image(cols, rows);
	uchar* d_dst; size_t d_dpitch;
	std::tie(d_dst, d_dpitch) = create_image(cols, rows);

	// Copy input data
	set_image(d_src, d_spitch, src, spitch, cols, rows);
	set_kernel(kernel, ksize);

	// Launch filtering CUDA kernel
	filter2d_launch(d_src, d_spitch, cols, rows,
		ksize,
		d_dst, d_dpitch);

	// Wait for kernel launch to be done
	check_errors(cudaDeviceSynchronize());

	// Copy output data
	get_image(dst, dpitch, d_dst, d_dpitch, cols, rows);

	// Free memories
	free_image(d_src);
	free_image(d_dst);
}

__host__
void filter2d_launch(
	const uchar* d_src, size_t d_spitch, size_t cols, size_t rows,
	size_t ksize,
	uchar* d_dst, size_t d_dpitch)
{
	// Let use as much threads in block as possible
	const auto dim_block = dim3(K, K);

	// Use as much KxK blocks as needed for this image
	const auto dim_grid = dim3((cols+K-1)/K, (rows+K-1)/K);

	// Invoke algorithm 
	filter2d_kernel<<<dim_grid, dim_block>>>(
		d_src, d_spitch, cols, rows,
		ksize,
		d_dst, d_dpitch);

	// Check errors in kernel invocation
	check_errors(cudaGetLastError());
}

__global__
void filter2d_kernel(
	const uchar* src, size_t spitch, size_t cols, size_t rows,
	/*const float* kernel, */size_t ksize,
	uchar* dst, size_t dpitch)
{
	// We need shared memory buffer to cache pixels from image
	// According to current block + surrounding half'es of kernel
	__shared__ uchar s_buffer[K + KSizeMax][K + KSizeMax];

	const int i = (blockIdx.y*K + threadIdx.y);
	const int j = (blockIdx.x*K + threadIdx.x);
	const int half_ksize = (ksize / 2);

	for(int m = threadIdx.y, y = (i-half_ksize); m < (K+ksize); m += K, y += K)
	{
		for(int n = threadIdx.x, x = (j-half_ksize); n < (K+ksize); n += K, x += K)
		{
			if((x < 0 || x > cols)
				|| (y < 0 || y > rows))
			{
				s_buffer[m][n] = 0;
				continue;
			}

			s_buffer[m][n] = src[y*spitch + x];
		}
	}

	__syncthreads();

	if((i > (rows-half_ksize)) || (i < half_ksize) 
		|| (j > (cols-half_ksize)) || (j < half_ksize))
	{
		// If we are outside filter kernel range, do not calculate nor write anything
		// aka BORDER_NONE
		return;
	}

	// Calculate partial sums with each element of the kernel
	float sum = 0.0;
	for(int m = 0, y = threadIdx.y; m < ksize; ++m, ++y)
	{
		for(int n = 0, x = threadIdx.x; n < ksize; ++n, ++x)
		{
			sum += s_buffer[y][x] * c_kernel[m*ksize + n];
		}
	}

	// Store final sum in the destination image
	dst[i*dpitch + j] = sum;
}

} // namespace filters