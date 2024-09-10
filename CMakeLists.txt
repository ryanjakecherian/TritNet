## to build and run this project:
# $ cmake ..
# $ make run

## NOTE IF THIS CMAKE BUILD SYSTEM IS BEING ANNOYING, JUST RUN THIS:
# nvcc --expt-relaxed-constexpr -I./cutlass/include src/test.cu -o exe


cmake_minimum_required(VERSION 3.10)
project(MyBNNProject LANGUAGES CXX CUDA)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/bintern/dependencies/include) #this is the line we change if we want to swap to terntern/mma



include_directories(${CMAKE_SOURCE_DIR}/cutlass/include)
include_directories(${CMAKE_SOURCE_DIR}/cutlass)

# Source files
set(SOURCES
${CMAKE_SOURCE_DIR}/src/bintern/dependencies/src/matrix.cu
${CMAKE_SOURCE_DIR}/src/bintern/dependencies/src/weights.cpp 
${CMAKE_SOURCE_DIR}/src/bintern/dependencies/src/activations.cpp

${CMAKE_SOURCE_DIR}/src/bintern/TritNet/fwd.cu
${CMAKE_SOURCE_DIR}/src/bintern/TritNet/forward_pass.cu
${CMAKE_SOURCE_DIR}/src/bintern/TritNet/random_init.cpp
${CMAKE_SOURCE_DIR}/src/bintern/TritNet/constructors.cpp

${CMAKE_SOURCE_DIR}/src/main.cpp

)

# Add executable
add_executable(test ${SOURCES})

# Find CUDA package
find_package(CUDA REQUIRED)

# Set NVCC flags
set_target_properties(test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80" #"60;61;70;75;80"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")  # Debug symbols and no optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")   # CUDA Debug

# Define a custom target that runs the executable
add_custom_target(run
    COMMAND $<TARGET_FILE:test>
    DEPENDS test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running the executable"
)

# Make the 'run' target depend on the 'test' target
add_dependencies(run test)