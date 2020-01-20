///////////////////////////////////////////////////////////////////////////////
// filters.cu
// 
// Contains implementation of `filters` library
///////////////////////////////////////////////////////////////////////////////

#include "filters.hpp"

#include <cassert>
#include <cstdlib>

#include <opencv2/imgproc.hpp>

#include "filters_errors.hpp"

namespace filters {
	
namespace {

//
// Private globals
//

//! Maximum size of the squared kernel
const auto KSizeMax = 64;

//! Fixed size constant buffer for convolution filter kernels
__constant__ float c_kernel[KSizeMax * KSizeMax];

//! Number of threads in both X and Y dimensions in the block
const auto K = 32;

//! Whether to print logs to stdout or not
bool g_verbose = false;

//! Number of cuda device to use
unsigned g_devnum = 0;

//
// Private functions
//

/**
 * @brief Retrieves attribute of the device
 * @details 
 * 
 * @param attr attribute to get
 * @return attribute value
 */
int get_attribute(cudaDeviceAttr attr)
{
	int value;
	check_errors(cudaDeviceGetAttribute(&value, attr, g_devnum));

	return value;
}

/**
 * @brief Gets from environment, whether verbosity should be enabled
 * @details 
 * @return verbosity status
 */
bool obtain_verbosity()
{
	const auto verbose = std::getenv("VERBOSE");
	if(verbose == nullptr || (std::strcmp(verbose, "0") == 0))
	{
		return false;
	}

	return true;
}

/**
 * @brief Gets from environment number of device to select
 * @details 
 * @return number of device to select
 */
int obtain_devnum()
{
	const auto devnum = std::getenv("DEVNUM");
	if(devnum == nullptr)
	{
		return 0;
	}

	return atoi(devnum);
}

/**
 * @brief Prints device attributes
 * @details 
 */
void print_attributes()
{
	printf("Device attributes:\n");
	printf(" cudaDevAttrMaxThreadsPerBlock=%d\n", get_attribute(cudaDevAttrMaxThreadsPerBlock));
	printf(" cudaDevAttrMaxBlockDimX=%d\n", get_attribute(cudaDevAttrMaxBlockDimX));
	printf(" cudaDevAttrMaxBlockDimY=%d\n", get_attribute(cudaDevAttrMaxBlockDimY));
	printf(" cudaDevAttrMaxBlockDimZ=%d\n", get_attribute(cudaDevAttrMaxBlockDimZ));
	printf(" cudaDevAttrMaxGridDimX=%d\n", get_attribute(cudaDevAttrMaxGridDimX));
	printf(" cudaDevAttrMaxGridDimY=%d\n", get_attribute(cudaDevAttrMaxGridDimY));
	printf(" cudaDevAttrMaxGridDimZ=%d\n", get_attribute(cudaDevAttrMaxGridDimZ));
	printf(" cudaDevAttrMaxSharedMemoryPerBlock=%d\n", get_attribute(cudaDevAttrMaxSharedMemoryPerBlock));
	printf(" cudaDevAttrTotalConstantMemory=%d\n", get_attribute(cudaDevAttrTotalConstantMemory));
	printf(" cudaDevAttrWarpSize=%d\n", get_attribute(cudaDevAttrWarpSize));
	printf(" cudaDevAttrClockRate=%d\n", get_attribute(cudaDevAttrClockRate));
	printf(" cudaDevAttrMultiProcessorCount=%d\n", get_attribute(cudaDevAttrMultiProcessorCount));
	printf(" cudaDevAttrMemoryClockRate=%d\n", get_attribute(cudaDevAttrMemoryClockRate));
	printf(" cudaDevAttrL2CacheSize=%d\n", get_attribute(cudaDevAttrL2CacheSize));
	printf(" cudaDevAttrMaxThreadsPerMultiProcessor=%d\n", get_attribute(cudaDevAttrMaxThreadsPerMultiProcessor));
	printf(" cudaDevAttrComputeCapabilityMajor=%d\n", get_attribute(cudaDevAttrComputeCapabilityMajor));
	printf(" cudaDevAttrComputeCapabilityMinor=%d\n", get_attribute(cudaDevAttrComputeCapabilityMinor));
	printf(" cudaDevAttrMaxSharedMemoryPerMultiprocessor=%d\n", get_attribute(cudaDevAttrMaxSharedMemoryPerMultiprocessor));
	printf(" cudaDevAttrMaxRegistersPerMultiprocessor=%d\n", get_attribute(cudaDevAttrMaxRegistersPerMultiprocessor));
}

} // namespace

__host__
void init()
{
	// Obtain environment variables
	g_verbose = obtain_verbosity();
	g_devnum = obtain_devnum();

	// If verbose, print device attributes
	if(g_verbose)
	{
		print_attributes();
	}

	// Select proper device
	check_errors(cudaSetDevice(g_devnum));
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
	const uchar* d_src, size_t d_spitch, 
	size_t cols, size_t rows, size_t ksize,
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
	const uchar* src, size_t spitch, 
	size_t cols, size_t rows, size_t ksize,
	uchar* dst, size_t dpitch)
{
	// We need shared memory buffer to cache pixels from image.
	// In general, in every pixel of the block we must provide access to 
	// surrounding halfes of the kernel (which at the end gives full kernel size)
	constexpr auto BufferSizeMax = (K + KSizeMax);

	// Note that we are declaring buffer using 2D notation, instead of 1D notation
	// This is because, after benchmarking, iterating over rows of fixed size
	// works faster than iterating over dynamic size array.
	__shared__ uchar s_buffer[BufferSizeMax][BufferSizeMax];

	// Cache source image into shared memory
	// Each thread has to fetch every K-th element starting from that thread's
	// position inside the block. We are incrementing with K, because buffer
	// must contain also surrounding kernel elements.
	const auto buffer_size = (K + ksize);
	for(int m = threadIdx.y; m < buffer_size; m += K)
	{
		for(int n = threadIdx.x; n < buffer_size; n += K)
		{
			// Note that we are not caching result of ksize/2
			// Benchmark showed, that current variant is better
			const int y = (m + blockIdx.y*K - ksize/2);
			const int x = (n + blockIdx.x*K - ksize/2);

			// If we are out of bound of the image, assume that buffer is zero
			if(x < 0 || y < 0 || x > cols || y > rows)
			{
				s_buffer[m][n] = 0;
				continue;
			}

			// Store copy of source image pixel into the buffer
			s_buffer[m][n] = src[y*spitch + x];
		}
	}

	// Wait until all threads has done caching
	__syncthreads();

	// Perform convolution on shared memory buffer
	const int i = (blockIdx.y*K + threadIdx.y);
	const int j = (blockIdx.x*K + threadIdx.x);
	const int half_ksize = (ksize / 2);

	// Check, if we are at the image border	
	if((i > (rows - half_ksize)) || (i < half_ksize) 
		|| (j > (cols - half_ksize)) || (j < half_ksize))
	{
		// Do not calculate nor write anything (aka BORDER_NONE)
		return;
	}

	// Calculate partial sums of buffer pixels with kernel's elements
	// Note that we are iterating only over kernel using pointer (benchmarks)
	auto sum = 0.0f;
	auto kernel = c_kernel;
	for(int m = 0; m < ksize; ++m)
	{
		for(int n = 0; n < ksize; ++n)
		{
			const auto y = (threadIdx.y + m);
			const auto x = (threadIdx.x + n);

			const auto buffer_v = s_buffer[y][x];
			const auto kernel_v = *(kernel++);

			sum += (buffer_v * kernel_v);
		}
	}

	// Store final sum in the destination image
	dst[i*dpitch + j] = sum;
}

} // namespace filters