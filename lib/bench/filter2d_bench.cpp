///////////////////////////////////////////////////////////////////////////////
// filter2d_bench.cpp
// 
// Contains implementation of benchmarks for filter2d-family functions
///////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include <opencv2/imgproc.hpp>

#include "filters.hpp"
#include "filters_errors.hpp"

/**
 * @brief Helper function to generate arguments for filter-related benchmarks
 */
void filter_arguments(benchmark::internal::Benchmark* b)
{
	const auto ksizes = {3, 9, 17, 33, 55};
	const auto resolutions = {
		std::pair{320, 240},
		std::pair{640, 480},
		std::pair{1280, 720},
		std::pair{1920, 1080},
		std::pair{3840, 2160}
	};

	// Add each combination of resolution and ksize into benchmark cases
	for(const auto [cols, rows] : resolutions)
	{
		for(const auto ksize : ksizes)
		{
			b->Args({cols, rows, ksize});
		}
	}
}

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
	auto src = cv::Mat(rows, cols, CV_8UC1);
	auto dst = cv::Mat(rows, cols, CV_8UC1);
	auto kernel = cv::Mat(ksize, ksize, CV_32F);

	for(auto _ : state)
	{
		filters::filter2d(src, kernel, dst);
	}

	filters::cleanup();
}

BENCHMARK(filter2d)
	->UseRealTime()
	->Apply(filter_arguments)
	;

/**
 * @brief Benchmarks performance of 2D filtering direct by CUDA kernel launcher
 * @details 
 * 
 * @param state benchmark state
 * @param cols image cols
 * @param rows image rows
 * @param ksize kernel size
 */
void filter2d_launch(benchmark::State& state)
{
	const auto cols = (size_t) state.range(0);
	const auto rows = (size_t) state.range(1);
	const auto ksize = (size_t) state.range(2);

	filters::init();

	// Allocate memories
	uchar* d_src; size_t d_spitch;
	std::tie(d_src, d_spitch) = filters::create_image(cols, rows);
	uchar* d_dst; size_t d_dpitch;
	std::tie(d_dst, d_dpitch) = filters::create_image(cols, rows);

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
		filters::filter2d_launch(d_src, d_spitch, 
			cols, rows, ksize,
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
		const auto elapsed_time_s = (elapsed_time_ms / 1000);
		state.SetIterationTime(elapsed_time_s);
	}

	// Destroy event stamps
	check_errors(cudaEventDestroy(start));
	check_errors(cudaEventDestroy(stop));
	
	// Free memory
	filters::free_image(d_dst);
	filters::free_image(d_src);

	filters::cleanup();
}

BENCHMARK(filter2d_launch)
	->UseRealTime()
	->UseManualTime()
	->Apply(filter_arguments);

/**
 * @brief Benchmarks performance of 2D filtering with OpenCV implementation
 * @details 
 * 
 * @param state benchmark state
 * @param cols image cols 
 * @param rows image rows
 * @param ksize kernel size
 */			
void cv_filter2d(benchmark::State& state, int cols, int rows, int ksize)
{
	const auto src = cv::Mat(rows, cols, CV_8UC1);
	auto dst = cv::Mat(rows, cols, CV_8UC1);
	const auto kernel = cv::Mat(ksize, ksize, CV_32F);

	const auto ddepth = -1; // Keep depth as in source
	for(auto _ : state)
	{
		cv::filter2D(src, dst, ddepth, kernel);
	}
}

BENCHMARK(cv_filter2d)
	->UseRealTime()
	->Apply(filter_arguments)
	;