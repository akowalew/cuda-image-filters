///////////////////////////////////////////////////////////////////////////////
// filters_errors.cu
// 
// Contains implementation of error handling routines for  `filters` library
///////////////////////////////////////////////////////////////////////////////

#include "filters_errors.hpp"

#include <cstdio>

namespace filters {

void check_fail(cudaError_t result, 
	char const *const func, const char *const file, int const line)
{
	fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", 
		file, line, result, cudaGetErrorString(result), func);
	exit(EXIT_FAILURE);
}

} // namespace filters