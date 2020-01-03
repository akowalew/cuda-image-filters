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

//! Fixed size constant buffer for convolution filter kernels
__constant__ float c_kernel[1024];

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
	uchar* d_src;
	size_t d_spitch;
	check_errors(cudaMallocPitch(&d_src, &d_spitch, 
		cols * sizeof(uchar), rows));

	uchar* d_dst;
	size_t d_dpitch;
	check_errors(cudaMallocPitch(&d_dst, &d_dpitch, 
		cols * sizeof(uchar), rows));

	// Copy input data
	check_errors(cudaMemcpy2D(d_src, d_spitch, src, spitch, 
		cols * sizeof(uchar), rows, cudaMemcpyHostToDevice));
	set_kernel(kernel, ksize);

	// Launch filtering CUDA kernel
	filter2d_launch(d_src, d_spitch, cols, rows,
		ksize,
		d_dst, d_dpitch);

	// Wait for kernel launch to be done
	check_errors(cudaDeviceSynchronize());

	// Copy output data
	check_errors(cudaMemcpy2D(dst, dpitch, d_dst, d_dpitch, 
		cols * sizeof(uchar), rows, cudaMemcpyDeviceToHost));

	// Free memories
	check_errors(cudaFree(d_dst));
	check_errors(cudaFree(d_src));
}

__host__
void set_kernel(const float* kernel, size_t ksize)
{
	// Copy data from host kernel to constant memory
	check_errors(cudaMemcpyToSymbol(c_kernel, kernel, 
		ksize * ksize * sizeof(float)));	
}

__host__
void filter2d_launch(
	const uchar* d_src, size_t d_spitch, size_t cols, size_t rows,
	size_t ksize,
	uchar* d_dst, size_t d_dpitch)
{
	// Let use as much threads in block as possible
	const auto K = 32;
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
	const auto i = blockIdx.y * blockDim.y + threadIdx.y;
	const auto j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > rows || j > cols)
	{
		// If we are out of image size, return
		return;
	}

	if(i > rows - ksize || j > cols - ksize)
	{
		// If we are outside filter kernel range, do nothing
		return;
	}

	// Calculate partial sums with each element of the kernel
	auto sum = 0.0f;
	for(size_t m = 0; m < ksize; ++m)
	{
		for(size_t n = 0; n < ksize; ++n)
		{
			sum += src[(i+m)*spitch + (j+n)] * c_kernel[m*ksize + n];
		}
	}

	// Store final sum in the destination image
	const auto half_ksize = (ksize/2);
	dst[(i+half_ksize)*dpitch + (j+half_ksize)] = sum;
}

} // namespace filters