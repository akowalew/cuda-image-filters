###############################################################################
# CMakeLists.txt
# 
# Defines build structure for the `filter-image` application
###############################################################################

###############################################################################
# Library definition 
###############################################################################

add_executable(filter-image
	src/main.cpp
)

target_include_directories(filter-image
	PRIVATE
		include
)

target_link_libraries(filter-image
	PRIVATE
		${OpenCV_LIBRARIES}
		
		filters
)

set_target_properties(filter-image
	PROPERTIES
		CXX_STANDARD 17
		CXX_EXTENSIONS OFF
)

target_compile_options(filter-image
	PRIVATE
		-Wall -Wextra -pedantic
)
