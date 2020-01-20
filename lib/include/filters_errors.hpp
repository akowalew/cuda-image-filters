///////////////////////////////////////////////////////////////////////////////
// filters_error.hpp
// 
// Contains declaration of errors handling routines in `filters` library
///////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>

namespace filters {

void check_fail(cudaError_t result, 
	char const *const func, const char *const file, int const line);

#define check_errors(val) ((val == 0) ? void(0) : filters::check_fail((val), #val, __FILE__, __LINE__))

} // namespace filters