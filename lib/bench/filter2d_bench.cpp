///////////////////////////////////////////////////////////////////////////////
// filter2d_bench.cpp
// 
// Contains implementation of benchmarks for filter2d-family functions
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include "filters.hpp"

/**
 * @brief Benchmarks performance of 2D filtering with all housekeeping stuff
 * @details 
 * 
 * @param state benchmark state
 * @param width image width 
 * @param height image height
 * @param ksize kernel size
 */			
void filter2d_total(benchmark::State& state, int width, int height, int ksize)
{
	filters::init();
	auto img = cv::Mat(height, width, CV_8UC1);
	auto dst = cv::Mat(height, width, CV_8UC1);
	auto kernel = cv::Mat(ksize, ksize, CV_32F);

	for(auto _ : state)
	{
		filters::filter2d(img, kernel, dst);
	}

	filters::cleanup();
}

BENCHMARK_CAPTURE(filter2d_total, 320x240x3, 320, 240, 3)
	->UseRealTime();
