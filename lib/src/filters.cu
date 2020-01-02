///////////////////////////////////////////////////////////////////////////////
// filters.cu
// 
// Contains implementation of `filters` library
///////////////////////////////////////////////////////////////////////////////

#include "filters.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

namespace filters {
	
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

	float* d_kernel;
	check_errors(cudaMalloc(&d_kernel, ksize * ksize * sizeof(float)));

	// Copy input data
	check_errors(cudaMemcpy2D(d_src, d_spitch, src, spitch, 
		cols * sizeof(uchar), rows, cudaMemcpyHostToDevice));
	check_errors(cudaMemcpy(d_kernel, kernel, 
		ksize * ksize * sizeof(float), cudaMemcpyHostToDevice));

	// Launch filtering CUDA kernel
	filter2d_launch(d_src, d_spitch, cols, rows,
		d_kernel, ksize,
		d_dst, d_dpitch);

	// Wait for kernel launch to be done
	check_errors(cudaDeviceSynchronize());

	// Copy output data
	check_errors(cudaMemcpy2D(dst, dpitch, d_dst, d_dpitch, 
		cols * sizeof(uchar), rows, cudaMemcpyDeviceToHost));

	// Free memories
	check_errors(cudaFree(d_kernel));
	check_errors(cudaFree(d_dst));
	check_errors(cudaFree(d_src));
}

__host__
void filter2d_launch(
	const uchar* d_src, size_t d_spitch, size_t cols, size_t rows,
	const float* d_kernel, size_t ksize,
	uchar* d_dst, size_t d_dpitch)
{
	// Invoke algorithm 
	filter2d_kernel<<<1, 1>>>(d_src, d_spitch, cols, rows,
		d_kernel, ksize,
		d_dst, d_dpitch);

	// Check errors in kernel invocation
	check_errors(cudaGetLastError());
}

__global__
void filter2d_kernel(
	const uchar* src, size_t spitch, size_t cols, size_t rows,
	const float* kernel, size_t ksize,
	uchar* dst, size_t dpitch)
{
	const auto half_ksize = (ksize/2);

	for(size_t i = 0; i < rows - 2*half_ksize; ++i)
	{
		for(size_t j = 0; j < cols - 2*half_ksize; ++j)
		{
			auto sum = 0.0f;

			for(size_t m = 0; m < ksize; ++m)
			{
				for(size_t n = 0; n < ksize; ++n)
				{
					sum += src[(i+m)*spitch + (j+n)] * kernel[m*ksize + n];
				}
			}

			dst[(i+half_ksize)*dpitch + (j+half_ksize)] = sum;
		}
	}
}

} // namespace filters