###############################################################################
# CMakeLists.txt
# 
# Defines build structure for the `filters` library
###############################################################################

###############################################################################
# Library definition 
###############################################################################

add_library(filters
	src/filters.cu
	src/filters_errors.cu
)

target_link_libraries(filters
	PUBLIC
		${OpenCV_LIBRARIES}
)

target_include_directories(filters
	PUBLIC 
		include
)

set_target_properties(filters
	PROPERTIES
		CXX_STANDARD 17
		CXX_EXTENSIONS OFF
		CUDA_STANDARD 14
		CUDA_EXTENSIONS OFF
)

target_compile_options(filters
	PRIVATE
		# Enable all warnings on CXX
		$<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>
)

###############################################################################
# Unit tests
###############################################################################

if(BUILD_TESTING)
	add_executable(filters-test
		test/test_main.cpp
	)

	target_compile_options(filters-test
		PRIVATE
			-Wall -Wextra -pedantic
	)

	set_target_properties(filters-test
		PROPERTIES
			CXX_STANDARD 17
			CXX_EXTENSIONS OFF
	)

	target_link_libraries(filters-test
		PRIVATE
			filters

            doctest::doctest
	)
endif()

###############################################################################
# Benchmarks
###############################################################################

if(BUILD_BENCHMARKING)
	add_executable(filters-bench
		bench/bench_main.cpp
		bench/filter2d_bench.cpp
	)

	target_compile_options(filters-bench
		PRIVATE
			-Wall -Wextra -pedantic
	)

	set_target_properties(filters-bench
		PROPERTIES
			CXX_STANDARD 17
			CXX_EXTENSIONS OFF
	)

	target_link_libraries(filters-bench
		PRIVATE
			filters

            ${benchmark_LIBRARIES}
			benchmark
			pthread
	)
endif()