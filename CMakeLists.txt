## to build and run this project:
# $ cmake ..
# $ make run

cmake_minimum_required(VERSION 3.10)
project(tritnet LANGUAGES CXX CUDA)

#
                                # External dependencies (CUTLASS)
                                # include(ExternalProject)
                                # ExternalProject_Add(
                                #     cutlass
                                #     GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
                                #     GIT_TAG main

                                #     CMAKE_ARGS -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON -DCUTLASS_NVCC_ARCHS=80a
                                #     CTEST_UPDATE_COMMAND ""
                                #     INSTALL_COMMAND ""
                                # )
                                # ExternalProject_Get_Property(cutlass source_dir binary_dir)
#

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/bintern/dependencies/include) #this is the line we change if we want to swap to terntern/mma


# Source files
set(SOURCES
# ${CMAKE_SOURCE_DIR}/src/bintern/dependencies/src/matrix.cu
# ${CMAKE_SOURCE_DIR}/src/bintern/dependencies/src/weights.cpp 
# ${CMAKE_SOURCE_DIR}/src/bintern/dependencies/src/activations.cpp

${CMAKE_SOURCE_DIR}/src/bintern/TritNet/prop_kernel.cu
${CMAKE_SOURCE_DIR}/src/bintern/TritNet/propagate.cu
${CMAKE_SOURCE_DIR}/src/bintern/TritNet/forward_pass.cu
${CMAKE_SOURCE_DIR}/src/bintern/TritNet/init.cpp
${CMAKE_SOURCE_DIR}/src/bintern/TritNet/constructors.cpp

${CMAKE_SOURCE_DIR}/src/main.cpp
)

# Add executable
add_executable(exe ${SOURCES})

#
                                # Add dependencies
                                # add_dependencies(exe cutlass)
#

# Find CUDA package
find_package(CUDA REQUIRED)

# Set NVCC flags
set_target_properties(exe PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80" #"60;61;70;75;80"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")  # Debug symbols and no optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")   # CUDA Debug

# Define a custom target that runs the executable
add_custom_target(run
    COMMAND $<TARGET_FILE:exe>
    DEPENDS exe
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running the executable"
)

# Make the 'run' target depend on the 'exe' target
add_dependencies(run exe)