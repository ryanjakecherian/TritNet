#include "TritNet.hpp"

#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

//potentially could store the whole A n B arrays in const mem
extern __constant__ int d_n;                // n is A.rows
extern __constant__ int d_m;                       // m is A.cols = B.rows
extern __constant__ int d_p;                       // p is B.cols
extern __constant__ int d_WORD_SIZE;        // word_size is probably most efficient at 32 bits. 
                                            // ALSO: is it worth putting word size in const memory? or should i just do compile-time replacement using #define?


//toy activation function - to be replaced by actual function. this could even be placed within the kernel itself, without the need for passing the tid parameter into another function!!
__device__ bool sigma(int &a){
	if (a < 0) {return true;}
	return 0;
}

template<typename T> // lets use int
__global__ void FORWARD(T* d_A, T* d_W_plus, T* d_W_neg, T* d_c){

    //the if statement is kinda pointless since, so far, our kernel launches are only launching enough blocks & threads to cover the whole matrix. so itll never go outside this region. so if statement is currently pointless.
	if ((blockIdx.x + (d_p/d_WORD_SIZE)*blockIdx.y) < (d_n*d_p/d_WORD_SIZE)) {

        int thread_result = 0; //waste of an instruction... but i need this such that there is a variable which can be updated within the for loop below...

        //old indexing:
        // int A_idx = (blockIdx.x*d_WORD_SIZE/d_p)*d_m;                       //(blockIdx.x/d_p)*d_WORD_SIZE+threadIdx.x)*d_m 
        // int W_idx = (blockIdx.x*d_WORD_SIZE + threadIdx.x)%d_p;             //blockIdx.x%d_p

        //new indexing:
        int A_idx = blockIdx.y*d_m;
        int W_idx = blockIdx.x*d_WORD_SIZE + threadIdx.x;

		for(int k=0; k<d_m; k++){
            // bitwise &, popcount, accumulate.
			thread_result += __popc( d_A[A_idx] ^ d_W_plus[W_idx] ) - __popc( d_A[A_idx] ^ d_W_neg[W_idx] );
            A_idx = A_idx + 1;
            W_idx = W_idx + d_p;
        }

        //activation
        if(sigma(thread_result)){thread_result = (1<<(d_WORD_SIZE-1-threadIdx.x));} else{thread_result = 0;}     //left-most bit as MSB: 2^(word_size-threadIdx.x). left-most bit as LSB: (2^threadIdx.x).
        
        //compression (effectively reduction)
        //now we can replace the line below with a reduction algorithm:
		atomicOr(&(d_c[blockIdx.x + (d_p/d_WORD_SIZE)*blockIdx.y]),  thread_result);                                   //apparently atomicOr only works on 32bits - 4 byte ints.
	}

}

void TritNet::propagate_layer(int i){ //pytorch and tensorflow convention is Y = X W because it is more efficient to store each individual image as a row, than as a column.
    //is there potential to optimise away all this pointer dereferencing?

    int m = activations_list[i]->A.cols;
    int p = weights_list[i]->W_plus.cols;
    
    if (p%WORD_SIZE != 0) {
        std::cout<<p<<std::endl;
        throw std::invalid_argument("Matrix class needs zero-padding functionality");
    }

    if (m != weights_list[i]->W_plus.rows) {
    throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    activations_list[i+1] = new activations(batch_samples, p/WORD_SIZE); //&input_batch.A->size, hidden_layers[i]
    


    cudaMemcpyToSymbol(d_m, &m, sizeof(int));
    cudaMemcpyToSymbol(d_p, &p, sizeof(int));
    
    int *d_A, *d_W_plus, *d_W_neg, *d_c;
    
    if (cudaSuccess!= cudaMalloc(&d_A, activations_list[i]->A.byte_size) ) {throw std::runtime_error("CUDA memory allocation failed");};
    if (cudaSuccess!= cudaMalloc(&d_W_plus, weights_list[i]->W_plus.byte_size) ) {throw std::runtime_error("CUDA memory allocation failed");};
    if (cudaSuccess!= cudaMalloc(&d_W_neg , weights_list[i]->W_plus.byte_size) ) {throw std::runtime_error("CUDA memory allocation failed");}; //because w_plus and w_neg have the same size
    if (cudaSuccess!= cudaMalloc(&d_c, activations_list[i+1]->A.byte_size) ) {throw std::runtime_error("CUDA memory allocation failed");};

    if (cudaSuccess!= cudaMemcpy(d_A, activations_list[i]->A.head_flat, activations_list[i]->A.byte_size, cudaMemcpyHostToDevice) ) {throw std::runtime_error("CUDA memcpy failed");};
    if (cudaSuccess!= cudaMemcpy(d_W_plus, weights_list[i]->W_plus.head_flat, weights_list[i]->W_plus.byte_size, cudaMemcpyHostToDevice) ) {throw std::runtime_error("CUDA memcpy failed");};
    if (cudaSuccess!= cudaMemcpy(d_W_neg , weights_list[i]->W_neg .head_flat, weights_list[i]->W_plus.byte_size, cudaMemcpyHostToDevice) ) {throw std::runtime_error("CUDA memcpy failed");}; //because w_plus and w_neg have the same size
    if (cudaSuccess!= cudaMemset(d_c, 0, activations_list[i+1]->A.byte_size) ) {throw std::runtime_error("CUDA memcpy failed");};



    dim3 blockDim = WORD_SIZE;                 
    dim3 gridDim(p/WORD_SIZE, batch_samples);       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1

    FORWARD<int><<<gridDim,blockDim>>>(d_A, d_W_plus, d_W_neg, d_c);
    

    //free W's and A earlier in case that increases speed of the cudamemcpy
    cudaFree(d_W_plus);
    cudaFree(d_W_neg);
    cudaFree(d_A);
    cudaMemcpy(activations_list[i+1]->A.head_flat, d_c, activations_list[i+1]->A.byte_size, cudaMemcpyDeviceToHost);
    cudaFree(d_c);

    return;
}