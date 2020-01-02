###############################################################################
# CMakeLists.txt
# 
# Defines build structure for the project
###############################################################################

cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(cuda-image-filters VERSION 0.1.0 LANGUAGES CXX CUDA)

###############################################################################
# Project options 
###############################################################################

option(BUILD_CONAN "Whether to use Conan package manager or not" ON)

option(BUILD_TESTING "Whether to build unit tests or not" ON)
option(BUILD_BENCHMARKING "Whether to build benchmarks or not" ON)

option(BUILD_LIB "Whether to build `filters` library or not" ON)
option(BUILD_APP "Whether to build `filter-image` application or not" ON)

###############################################################################
# Project setup
###############################################################################

if(BUILD_TESTING)
    # If unit-tests needed, enable CMake testing suite
	enable_testing()
endif()

# If some CXX or CUDA standard is needed for a target, require it  
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

###############################################################################
# Conan integration 
###############################################################################

if(BUILD_CONAN)
    # Include outputs of conan installation
    find_file(CONANBUILDINFO_FILE conanbuildinfo.cmake HINTS ${CMAKE_BINARY_DIR})
    include(${CONANBUILDINFO_FILE})

    # Perform basic Conan setup:
    #  - read installed dependencies
    #  - configure output directories (bin, lib)
    #  - adjust compiler flags
    conan_basic_setup()
endif()

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

###############################################################################
# External packages
###############################################################################

# We need OpenCV solely for image reading-writing and image API
find_package(OpenCV REQUIRED
    COMPONENTS
    	opencv_core
    	opencv_imgproc
    	opencv_videoio
    	opencv_highgui
    	opencv_imgcodecs
)

if(BUILD_BENCHMARKING)
    # We need this library only when benchmarking
    find_package(benchmark REQUIRED)
endif()

if(BUILD_TESTING)
	# We need this framework only when testing
	find_package(doctest REQUIRED)
endif()

###############################################################################
# Subdirectories
###############################################################################

if(BUILD_LIB)
	add_subdirectory(lib)
endif()

if(BUILD_APP)
	add_subdirectory(app)
endif()

