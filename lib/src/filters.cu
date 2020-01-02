///////////////////////////////////////////////////////////////////////////////
// filters.cu
// 
// Contains implementation of `filters` library
///////////////////////////////////////////////////////////////////////////////

#include "filters.hpp"

#include <opencv2/imgproc.hpp>

namespace filters {
	
__host__
void init()
{

}

__host__
void cleanup()
{

}

__host__
cv::Mat filter2d(const cv::Mat& src, const cv::Mat& kernel)
{
	auto dst = cv::Mat(src.rows, src.cols, src.type());

	const auto ddepth = -1;
	cv::filter2D(src, dst, ddepth, kernel);

	return dst;
}

} // namespace filters