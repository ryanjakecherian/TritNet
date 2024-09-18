#include "TritNet.hpp"

#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

template<typename T>
void TritNet<T>::propagate_layer(int i, T *&d_A, T *&d_W, T *&d_O){ //pytorch and tensorflow convention is Y = X W because it is more efficient to store each individual image as a row, than as a column.
    
    if (cudaSuccess!= cudaMemcpy(d_A, A_list[i], n*A_bytesizes[i]+1, cudaMemcpyHostToDevice) ) {throw std::runtime_error("CUDA memcpy failed");};
    if (cudaSuccess!= cudaMemcpy(d_W, W_list[i], W_bytesizes[i], cudaMemcpyHostToDevice) ) {throw std::runtime_error("CUDA memcpy failed");};
    
    if (cudaSuccess!= cudaMemset(d_O, 0, n*A_bytesizes[i+1]+1) ) {throw std::runtime_error("CUDA memset failed");};
    
    dim3 blockDim = WORD_SIZE;                 
    dim3 gridDim(layers[i+1], n);       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1 
    propagate<T><<<gridDim,blockDim>>>(d_A, d_W, d_O);

    A_list[i+1] = new T[n*layers[i+1]+1](); //this auto-initialises all to zero. FOR SOME REASON, IF I DONT INITIALISE THE LAST ELEMENT (be that thru either initialising the whole array, or just the last element), THE FIRST TIME DOING THE FORWARD PASS FAILS TO TRANSFER THE LAST ELEMENT.
    // A_list[i+1][n*layers[i+1]] = 0;  //for really large layers, it might be faster just to init the last element. this bug is super annoying.
    if (cudaSuccess != cudaMemcpy(A_list[i+1], d_O, n*A_bytesizes[i+1]+1, cudaMemcpyDeviceToHost) ) {throw std::runtime_error("CUDA memcpy failed");};

    return;
}