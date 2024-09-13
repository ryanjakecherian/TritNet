#include "TritNet.hpp"

#include <cuda.h>
#include <iostream>
#include <chrono> //for timing

extern __constant__ int d_n;

void TritNet::forward_pass(activations &input_batch){
    activations_list[0] = &input_batch;
    
    batch_samples = input_batch.A.rows;
    cudaMemcpyToSymbol(d_n, &batch_samples, sizeof(int));

    // Get the starting timepoint
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i<=depth; i++){
        // std::cout<<i<<std::endl; //debug
        propagate_layer(i);
    }

    // Get the ending timepoint
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    // Print the duration in milliseconds
    std::cout << "Forward pass CUDA execution time: " << duration.count() << "ms" << std::endl << std::endl;
    
    return;
}