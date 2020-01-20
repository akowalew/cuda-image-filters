# cuda-image-filters

The aim of this studies project was to write image filtering algorithms for CUDA platform. There is only one implemented algorithm in the code: a 2D convolution filter. It is written using naive method, by calculating convolution with 2D kernel in every destination pixel, but it takes advantage of some CUDA features like shared memory, constant memory, data tailing. There are some restrictions though, in order to simplify the project:

- Both source and destination image are represented as 2D, 8-bit, `unsigned char`, grayscale images
- Convolution kernel is represented as 2D, 32-bit, `float`, square matrix. 
- During filtering, no action takes place at image borders, so basically they are zeroed.

There are two main targets defined in the project: `filters` library and `filter-image` application, which uses the first. Image filtering algorithms are benchmarked and may be compared to reference, 3rd party, version. All of the host-side code is written in C++17 (CUDA-related in C++14), build system is defined using CMake, 3rd party dependencies are handled using Conan package manager, benchmarks are implemented using Google Benchmark library. Project uses also OpenCV library, but only for reading/writing images from/to hard-disk and for managing host-side matrices. 

## Requirements

In order to build and use this code, you may need following stuff:

- C++17 compliant compiler (tested on GCC 7.3.0 and GCC 9.1.0),
- Latest CMake (tested 3.16.2),
- Nvidia CUDA Toolkit (tested v10.1) with compatible CUDA graphics card,
- Conan package manager (tested 1.21) - if you would like to use it

Third party libraries are handled using Conan. If you wouldn't like to use it, you may also need to install following libraries:

- OpenCV, tested on v4.1.0, for image I/O and host-side matrix handling,
- Google Benchmark, v1.4.1, for benchmarking of filtering code
- doctest, v2.3.4, for unit testing

## Building 

In order to build the project, clone the repository first, and then type in the terminal:

```sh
# Go to project directory
cd cuda-image-filters/

# Create build directory and go to it
mkdir build/
cd build/
```

Now, if you would like to handle dependencies using Conan, type also:

```sh
# Configure conan remotes
conan config install ../conan

# Install dependencies with conan
conan install ../ --build=missing --setting compiler.libcxx=libstdc++
```

Note that we are specyfing old ABI in libcxx variable, because that way may only OpenCV and Google Benchmark work.

Now configure CMake and compile everything:

```sh
# Configure CMake
cmake ../ -DCMAKE_BUILD_TYPE=<Debug/Release/MinSizeRel> \
	-DBUILD_CONAN=<ON/OFF> \
	-DBUILD_TESTING=<ON/OFF> \
	-DBUILD_BENCHMARKING=<ON/OFF> \
	-DBUILD_APP=<ON/OFF> \
	-DBUILD_LIB=<ON/OFF> \
	-DBUILD_VERBOSE=<ON/OFF> \

# Compile everything
make -j${nproc}
```

## Running `filter-image` app

In order to run `filter-image` you have to type:

```sh
# Filter some source image file, results store in some destination file
./bin/filter-image <src_image> <dst_image> <kernel_size>
```

where `src_image` is path to source image file, `dst_image` is path to destination image file and `kernel_size` is size of the square-type convolution filter. Now, to see the results you can watch them 

## Running `filter` library benchmarks

In order to run benchmarks of `filters` library you have to type:

```sh
./bin/filters_bench
```

Benchmarks are written using Google Benchmark, so you can use its option directly:

```sh
./bin/filters_bench --benchmark_repetitions=25 --benchmark_filter=filter2d_launch/4096x2160x9
```

Note that in order to benchmark algorithms properly, you should have compiled your project in `Release` mode. You should also disable energy saving mode in your computer, and, as possible, exclude your CUDA GPU from rendering tasks (e.g. use integrated graphics in display manager).

## Authors:

- [akowalew](http://github.com/akowalew)