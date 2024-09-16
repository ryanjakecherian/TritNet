#include "TritNet.hpp"

#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

template<typename T>
void TritNet<T>::propagate_layer(int i){ //pytorch and tensorflow convention is Y = X W because it is more efficient to store each individual image as a row, than as a column.
    //is there potential to optimise away all this pointer dereferencing?

    T *d_A, *d_W, *d_O;

    if (cudaSuccess!= cudaMalloc(&d_A, n*A_bytesizes[i]) ) {throw std::runtime_error("CUDA memory allocation failed");};
    if (cudaSuccess!= cudaMemcpy(d_A, A_list[i], n*A_bytesizes[i], cudaMemcpyHostToDevice) ) {throw std::runtime_error("CUDA memcpy failed");};
    if (cudaSuccess!= cudaMalloc(&d_W, W_bytesizes[i]) ) {throw std::runtime_error("CUDA memory allocation failed");};
    if (cudaSuccess!= cudaMemcpy(d_W, W_list[i], W_bytesizes[i], cudaMemcpyHostToDevice) ) {throw std::runtime_error("CUDA memcpy failed");};
    
    if (cudaSuccess!= cudaMalloc(&d_O, n*A_bytesizes[i+1]) ) {throw std::runtime_error("CUDA memory allocation failed");};         
    if (cudaSuccess!= cudaMemset(d_O, 0, n*A_bytesizes[i+1]) ) {throw std::runtime_error("CUDA memset failed");};
    dim3 blockDim = WORD_SIZE;                 
    dim3 gridDim(layers[i+1], n);       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1

    propagate<T><<<gridDim,blockDim>>>(d_A, d_W, d_O);
    

    //free W's and A earlier in case that increases speed of the cudamemcpy
    cudaFree(d_W);
    cudaFree(d_A);
    A_list[i+1] = new T[n*layers[i+1]];
    cudaMemcpy(A_list[i+1], d_O, n*A_bytesizes[i+1], cudaMemcpyDeviceToHost);
    cudaFree(d_O);

    return;
}