///////////////////////////////////////////////////////////////////////////////
// filters.hpp
// 
// Contains declaration of API for `filters` library
///////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>

#include <opencv2/core.hpp>

namespace filters {

void check_fail(cudaError_t result, 
	char const *const func, const char *const file, int const line);

#define check_errors(val) ((val == 0) ? void(0) : filters::check_fail((val), #val, __FILE__, __LINE__))

/**
 * @brief Initializes `filters` library
 * @details 
 */
__host__
void init();

/**
 * @brief Deinitializes `filters` library
 * @details 
 */
__host__
void cleanup();

/**
 * @brief Performs filtering on 2D image with specified filter kernel - Host version
 * @details 
 * 
 * @param src source image
 * @param kernel filtering kernel
 * 
 * @return destination image after filtration
 */
__host__
cv::Mat filter2d(const cv::Mat& src, const cv::Mat& kernel);

/**
 * @brief Performs filtering on 2D image with specified filter kernel - Host version
 * @details 
 * 
 * @param src source image
 * @param kernel filtering kernel
 * @param dst destination image after filtration
 */
__host__
void filter2d(const cv::Mat& src, const cv::Mat& kernel, cv::Mat& dst);

/**
 * @brief Performs filtering on 2D image with specified filter kernel - Host version
 * @details
 * 
 * @param src source image (host)
 * @param spitch pitch of source image
 * @param cols images cols 
 * @param rows images rows
 * @param kernel filtering square kernel (host)
 * @param ksize size of kernel
 * @param dst destination image
 * @param dpitch pitch of destination image
 */
__host__
void filter2d(
	const uchar* src, size_t spitch, size_t cols, size_t rows,
	const float* kernel, size_t ksize,
	uchar* dst, size_t dpitch);

/**
 * @brief Performs filtering on 2D image with specified filter kernel - CUDA kernel launcher
 * @details
 * 
 * @param d_src source image (device)
 * @param d_spitch pitch of source image
 * @param cols images cols 
 * @param rows images rows
 * @param kernel filtering square kernel (device)
 * @param ksize size of kernel
 * @param d_dst destination image (device)
 * @param d_dpitch pitch of destination image
 */
__host__
void filter2d_launch(
	const uchar* d_src, size_t d_spitch, size_t cols, size_t rows,
	const float* d_kernel, size_t ksize,
	uchar* d_dst, size_t d_dpitch);

/**
 * @brief Performs filtering on 2D image with specified filter kernel - CUDA kernel
 * @details
 * 
 * @param src source image
 * @param spitch pitch of source image
 * @param cols images cols 
 * @param rows images rows
 * @param kernel filtering square kernel
 * @param ksize size of kernel
 * @param dst destination image
 * @param dpitch pitch of destination image
 */
__global__
void filter2d_kernel(
	const uchar* src, size_t spitch, size_t cols, size_t rows,
	const float* kernel, size_t ksize,
	uchar* dst, size_t dpitch);

} // namespace filters