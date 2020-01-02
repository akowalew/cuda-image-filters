///////////////////////////////////////////////////////////////////////////////
// filters.hpp
// 
// Contains declaration of API for `filters` library
///////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>

#include <opencv2/core.hpp>

namespace filters {

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
 * @param width images width 
 * @param height images height
 * @param kernel filtering square kernel
 * @param ksize size of kernel
 * @param dst destination image
 */
__host__
void filter2d(
	const void* src, size_t width, size_t height,
	const void* kernel, size_t ksize,
	void* dst);

/**
 * @brief Performs filtering on 2D image with specified filter kernel - CUDA kernel
 * @details
 * 
 * @param src source image
 * @param spitch pitch of source image
 * @param width images width 
 * @param height images height
 * @param kernel filtering square kernel
 * @param ksize size of kernel
 * @param dst destination image
 * @param dpitch pitch of destination image
 */
__global__
void filter2d_kernel(
	const void* src, size_t spitch, size_t width, size_t height,
	const void* kernel, size_t ksize,
	void* dst, size_t dpitch);

} // namespace filters