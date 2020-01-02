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
 * @param cols image cols 
 * @param rows image rows
 * @param ksize kernel size
 */			
void filter2d(benchmark::State& state, int cols, int rows, int ksize)
{
	filters::init();
	auto img = cv::Mat(rows, cols, CV_8UC1);
	auto dst = cv::Mat(rows, cols, CV_8UC1);
	auto kernel = cv::Mat(ksize, ksize, CV_32F);

	for(auto _ : state)
	{
		filters::filter2d(img, kernel, dst);
	}

	filters::cleanup();
}

BENCHMARK_CAPTURE(filter2d, 320x240x3, 320, 240, 3)
	->UseRealTime();
BENCHMARK_CAPTURE(filter2d, 640x480x3, 640, 480, 3)
	->UseRealTime();
BENCHMARK_CAPTURE(filter2d, 1024x768x3, 1024, 768, 3)
	->UseRealTime();
BENCHMARK_CAPTURE(filter2d, 320x240x13, 320, 240, 13)
	->UseRealTime();
BENCHMARK_CAPTURE(filter2d, 640x480x13, 640, 480, 13)
	->UseRealTime();
BENCHMARK_CAPTURE(filter2d, 1024x768x13, 1024, 768, 13)
	->UseRealTime();

/**
 * @brief Benchmarks performance of 2D filtering direct by CUDA kernel launcher
 * @details 
 * 
 * @param state benchmark state
 * @param cols image cols
 * @param rows image rows
 * @param ksize kernel size
 */
void filter2d_launch(benchmark::State& state, int cols, int rows, int ksize)
{
	filters::init();

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

	// Create event stamps for time measuring
	cudaEvent_t start, stop;
	check_errors(cudaEventCreate(&start));
	check_errors(cudaEventCreate(&stop));

	// Perform benchmarking
	for(auto _ : state)
	{
		// Start kernel time measuring
		check_errors(cudaEventRecord(start, 0));

		// Invoke algorithm 
		filters::filter2d_launch(d_src, d_spitch, cols, rows,
			d_kernel, ksize,
			d_dst, d_dpitch);

		// Stop kernel time measuring
		check_errors(cudaEventRecord(stop, 0));
		check_errors(cudaEventSynchronize(stop));

		// Get the time spent in kernel
		float elapsed_time_ms;
		check_errors(cudaEventElapsedTime(&elapsed_time_ms, start, stop));

		// Wait for kernel to be fully finished
		check_errors(cudaDeviceSynchronize());

		// Manually set measured iteration time
		const auto elapsed_time_s = (elapsed_time_ms * 1000);
		state.SetIterationTime(elapsed_time_s);
	}

	// Destroy event stamps
	check_errors(cudaEventDestroy(start));
	check_errors(cudaEventDestroy(stop));
	
	// Free memory
	check_errors(cudaFree(d_kernel));
	check_errors(cudaFree(d_dst));
	check_errors(cudaFree(d_src));

	filters::cleanup();
}

BENCHMARK_CAPTURE(filter2d_launch, 320x240x3, 320, 240, 3)
	->UseRealTime();
BENCHMARK_CAPTURE(filter2d_launch, 640x480x3, 640, 480, 3)
	->UseRealTime();
BENCHMARK_CAPTURE(filter2d_launch, 1024x768x3, 1024, 768, 3)
	->UseRealTime();
BENCHMARK_CAPTURE(filter2d_launch, 320x240x13, 320, 240, 13)
	->UseRealTime();
BENCHMARK_CAPTURE(filter2d_launch, 640x480x13, 640, 480, 13)
	->UseRealTime();
BENCHMARK_CAPTURE(filter2d_launch, 1024x768x13, 1024, 768, 13)
	->UseRealTime();