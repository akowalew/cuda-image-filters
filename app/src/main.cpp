///////////////////////////////////////////////////////////////////////////////
// main.cpp
// 
// Implements main application routine
///////////////////////////////////////////////////////////////////////////////

#include <cstdio>

#include <stdexcept>

#include <opencv2/highgui.hpp>

#include "filters.hpp"

void init()
{
	printf("*** Initializing\n");
	
	filters::init();	
}

cv::Mat load_image(const char* path)
{
	printf("*** Loading image\n");

	return cv::imread(path, cv::IMREAD_GRAYSCALE);
}

cv::Mat filter_image(const cv::Mat& image)
{
	printf("*** Filtering\n");

	// Mean-blur 3x3 kernel
	static float kernel_data[] {
		1.0/9, 1.0/9, 1.0/9,
		1.0/9, 1.0/9, 1.0/9,
		1.0/9, 1.0/9, 1.0/9
	};  

	const auto kernel = cv::Mat(3, 3, CV_32F, kernel_data);
	return filters::filter2d(image, kernel);
}

void save_image(const char* path, const cv::Mat& image)
{
	printf("*** Saving image\n");

	if(!cv::imwrite(path, image))
	{
		throw std::runtime_error("Failed to save image to file: " 
			+ std::string(path));
	}
}

void cleanup()
{
	printf("*** Cleaning up\n");

	filters::cleanup();
}

/**
 * @brief Main program routine
 * @details 
 * @return 
 */
int main(int argc, char** argv)
{
	if(argc != 3)
	{
		printf("Usage: ./filter-image <src_file> <dst_file>\n");
		return -1;
	}

	const auto src_file = argv[1];
	const auto dst_file = argv[2];

	init();

	const auto src_image = load_image(src_file);
	const auto dst_image = filter_image(src_image);
	save_image(dst_file, dst_image);

	cleanup();

	return 0;
}