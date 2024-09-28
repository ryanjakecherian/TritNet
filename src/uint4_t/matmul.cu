#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

#include <iostream>

//
// START handy host functions
//

#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                             \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "       \
                      << cudaGetErrorString(error) << std::endl;                \
            exit(1);                                                            \
        }                                                                       \
    }

void print_matrix(const int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

int pack_4bit_integers(int low, int high) {
    return (high << 4) | (low & 0xF); // Pack two 4-bit integers into one byte
}

//
// ENDD handy host functions
//



//
// START GEMM SETUP
//
    using ElementInputA = cutlass::int4b_t;  // 4-bit signed integer type for A matrix
    using ElementInputB = cutlass::int4b_t;  // 4-bit signed integer type for B matrix
    using ElementOutput = int32_t;           // 32-bit integer for output

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using AccumulatorType = int32_t;
    using EpilogueComputeType = AccumulatorType;
    using MmaOp = cutlass::arch::OpClassTensorOp;
    using SM_arch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using ShapeTB = cutlass::gemm::GemmShape<128, 256, 64>;         // <- threadblock tile M = 128, N = 256, K = 64
    // This code section describes tile size a warp will compute
    using ShapeWarp = cutlass::gemm::GemmShape<64, 64, 64>;                  // <- warp tile M = 64, N = 64, K = 64 
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;                      // <- MMA Op tile M = 8, N = 8, K = 16


    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // This code section describes the epilogue part of the kernel
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    AccumulatorType,                                // <- data type of accumulator
    EpilogueComputeType>;  // <- data type for alpha/beta in linear combination function

    // Number of pipelines you want to use
    constexpr int NumStages = 2;



    // define Gemm template
    using Gemm = cutlass::gemm::device::Gemm< ElementInputA,LayoutInputA,   ElementInputB,LayoutInputB,   ElementOutput,LayoutOutput,   AccumulatorType,
                                              MmaOp, SM_arch,
                                              ShapeTB,ShapeWarp,ShapeMMAOp,
                                              EpilogueOp,
                                              SwizzleThreadBlock,
                                              NumStages
                                              >;

//
// END GEMM SETUP
//



//
// MAIN LOOP
//

int main() {

    int M = 5120;  // Rows of A and C
    int N = 4096;  // Columns of B and C
    int K = 4096;  // Columns of A and rows of B
    

    // Create matrices in host memory
    int8_t* h_A = new int8_t[M * K/2]; // Each int8_t holds two 4-bit integers
    int8_t* h_B = new int8_t[K * N/2]; // Each int8_t holds two 4-bit integers
    int32_t* h_C = new int32_t[M * N];

    // Initialize matrices with random data (fill with small 4-bit integers)
    for (int i = 0; i < M * K / 2; ++i) {
        int low = rand() % 16 - 8;  // Random 4-bit integer [-8, 7]
        int high = rand() % 16 - 8; // Another random 4-bit integer [-8, 7]
        h_A[i] = pack_4bit_integers(low, high);
    }

    for (int i = 0; i < K * N / 2; ++i) {
        int low = rand() % 16 - 8;  // Random 4-bit integer [-8, 7]
        int high = rand() % 16 - 8; // Another random 4-bit integer [-8, 7]
        h_B[i] = pack_4bit_integers(low, high);
    }

    // Allocate device memory
    cutlass::int4b_t *d_A, *d_B;
    int32_t *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(int8_t) * M * K / 2));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(int8_t) * K * N / 2));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(int32_t) * M * N));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(int8_t) * M * K / 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(int8_t) * K * N / 2, cudaMemcpyHostToDevice));




    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(M,N,K);

    // Initialize alpha and beta for dot product computation
    EpilogueComputeType alpha = EpilogueComputeType(1);
    EpilogueComputeType beta = EpilogueComputeType(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Set up GEMM parameters
    typename Gemm::Arguments arguments{ problem_size,  // Gemm dimensions
                                        {d_A, K / 2},   // Tensor A
                                        {d_B, N / 2},   // Tensor B
                                        {d_C, N},   // Tensor C
                                        {d_C, N},   // Output tensor
                                        {alpha, beta},    // Alpha, Beta scalars for GEMM operation
                                        split_k_slices
                                      };


    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate the GEMM kernel
    Gemm gemm_op;

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = gemm_op();
    CUTLASS_CHECK(status);





    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeof(int32_t) * M * N, cudaMemcpyDeviceToHost));

    // Print a portion of the result matrix
    std::cout << "Matrix C (result of A * B):" << std::endl;
    print_matrix(h_C, M, N);

    // Cleanup
    delete[] (h_A);
    delete[] (h_B);
    delete[] (h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
