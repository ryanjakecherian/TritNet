#include "TritNet.hpp"

#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <cassert> //debug

using namespace std;

extern __constant__ int d_WORD_SIZE;
extern __constant__ int d_n;
extern __device__ int d_m;
extern __device__ int d_p;


//toy activation function - to be replaced by actual function. this could even be placed within the kernel itself, without the need for passing the tid parameter into another function!!
__device__ bool sigma(int &a){
	if (a < 0) {return true;}
	return 0;
}

template<typename T> // lets use int
__global__ void propagate(T* d_A, T* d_W, T* d_c){
    
    d_p = d_W[0];   //should i put d_p and d_m in shared memory?        // also note that d_W[0] is of type T, so it is getting recast to int, since d_p is of type int.
    
    int thread_result = 0; //waste of an instruction... but i need this such that there is a variable which can be updated within the for loop below...

    //new indexing:
    int A_idx = blockIdx.y*d_m;                                 assert(A_idx < 3);
    int W_idx = 1 + blockIdx.x*d_WORD_SIZE + threadIdx.x;       assert(W_idx < 901);
    int W_neg_offset = d_m*d_p;                                 assert((W_idx + W_neg_offset) < 1801);

    for(int k=0; k<d_m; k++){
        // bitwise &, popcount, accumulate.
        thread_result += __popc( d_A[A_idx] ^ d_W[W_idx] ) - __popc( d_A[A_idx] ^ d_W[W_neg_offset + W_idx] );  //i.e. \sigma { (A^W_pos) - (A^W_neg) }
        A_idx = A_idx + 1;
        W_idx = W_idx + d_p;
    }

    //activation
    if(sigma(thread_result)){thread_result = (1<<(d_WORD_SIZE-1-threadIdx.x));} else{thread_result = 0;}     //left-most bit as MSB: 2^(word_size-threadIdx.x). left-most bit as LSB: (2^threadIdx.x).
    
    //compression (effectively reduction)
    //now we can replace the line below with a reduction algorithm (potentially by storing thread results in shared memory):
    atomicOr(&(d_c[blockIdx.x + (d_p/d_WORD_SIZE)*blockIdx.y]),  thread_result);
   

    (blockIdx.x == d_n-1 && blockIdx.y == d_p/d_WORD_SIZE) ? d_m = d_p/d_WORD_SIZE : d_m = d_m; //only update d_m if its the last block

}


//this instantiation is just to prevent linker errors due to this being an implementation file for a templated class - https://stackoverflow.com/a/495056/23298718
template __global__ void propagate<uint32_t>(uint32_t* d_A, uint32_t* d_W, uint32_t* d_O);