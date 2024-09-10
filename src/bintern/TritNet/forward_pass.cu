#include "TritNet.hpp"

#include <cuda.h>
#include <iostream>


extern __constant__ int d_n;

void TritNet::forward_pass(activations &input_batch){
    activations_list[0] = &input_batch;
    
    batch_samples = input_batch.A.rows;
    cudaMemcpyToSymbol(d_n, &batch_samples, sizeof(int));

    for(int i = 0; i<=depth; i++){
        // std::cout<<i<<std::endl; //debug
        fwd(i);
    }
    
    return;
}