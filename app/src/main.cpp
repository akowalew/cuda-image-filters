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

	return cv::imread(cv::String(path), cv::IMREAD_GRAYSCALE);
}

cv::Mat gen_kernel(size_t ksize)
{
	printf("*** Generating kernel\n");

	// Create Mean-Blur square kernel
	auto kernel = cv::Mat(ksize, ksize, CV_32F);
	const auto sq_ksize = (ksize*ksize);
	const auto kvalue = (1.0f / sq_ksize);

	// Fill kernel with same values
	const auto kbegin = (float*)kernel.data;
	const auto kend = (kbegin + sq_ksize);
	std::fill(kbegin, kend, kvalue);

	return kernel;
}

cv::Mat filter_image(const cv::Mat& image, const cv::Mat& kernel)
{
	printf("*** Filtering\n");

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
	if(argc != 4)
	{
		printf("Usage: ./filter-image <src_file> <dst_file> <kernel_size>\n");
		return -1;
	}

	const auto src_file = argv[1];
	const auto dst_file = argv[2];
	const auto kernel_size = std::atoi(argv[3]);

	init();

	const auto src_image = load_image(src_file);
	const auto kernel = gen_kernel(kernel_size);
	const auto dst_image = filter_image(src_image, kernel);
	save_image(dst_file, dst_image);

	cleanup();

	return 0;
}