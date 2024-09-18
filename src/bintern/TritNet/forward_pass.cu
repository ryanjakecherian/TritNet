#include "TritNet.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono> //for timing

__constant__ int d_WORD_SIZE  = WORD_SIZE;
__constant__ int d_n;
__device__ int d_m;
__device__ int d_p;

template <typename T>
void TritNet<T>::forward_pass(int batch_samples, T* &input_batch, bool multistream){

    if ((multistream == true) && depth > 0) {                      //minimum depth for multistream is 1: LOADING inputs 2 weights & calcs 1 activation, and DELOADING calcs 2 activations & outputs 1 result. These 2 phases alone sum to 3 layers, i.e. a depth of 1.
        auto start = std::chrono::high_resolution_clock::now();

        A_list[0] = input_batch;
        n = batch_samples;
        cudaMemcpyToSymbol(d_n, &n, sizeof(int));

        // INITIALISE DUAL-BUFFER MULTI-STREAM SYSTEM:
            // Device pointers and block dimension                
                T *d_b1_A, *d_b1_W, *d_b2_A, *d_b2_W;
                dim3 blockDim = WORD_SIZE;   


            // allocate buffers (of max required size) 
                if (cudaSuccess!= cudaMalloc(&d_b1_A, n*A_max +1) ) {throw std::runtime_error("CUDA memory allocation failed");}; 
                if (cudaSuccess!= cudaMalloc(&d_b2_A, n*A_max +1) ) {throw std::runtime_error("CUDA memory allocation failed");};
                if (cudaSuccess!= cudaMalloc(&d_b1_W, W_max) ) {throw std::runtime_error("CUDA memory allocation failed");}; 
                if (cudaSuccess!= cudaMalloc(&d_b2_W, W_max) ) {throw std::runtime_error("CUDA memory allocation failed");};
                //somehow its faster to do this with one stream instead of two?? do some more testing.
                

            // H2D: A[0]
                if (cudaSuccess!= cudaMemcpy(d_b2_A, A_list[0], n*A_bytesizes[0]+1, cudaMemcpyHostToDevice) ) {throw std::runtime_error("CUDA memcpy failed");};
            

            // create 2 streams
                cudaStream_t TransferStream;
                cudaStream_t ProcessStream;
                cudaStreamCreate(&TransferStream);
                cudaStreamCreate(&ProcessStream);



        // LOADING: phase 1
            // H2D: W[0] and p      //S1
                if (cudaSuccess!= cudaMemcpyAsync(d_b1_W, W_list[0], W_bytesizes[0], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};

            // synchronise
                cudaStreamSynchronize(TransferStream);


        // LOADING: phase 2
            // H2D: W[1] and p      //S1
                if (cudaSuccess!= cudaMemcpyAsync(d_b2_W, W_list[1], W_bytesizes[1], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};
                
            // <<<prop>>>(0)        //S2
                if (cudaSuccess!= cudaMemsetAsync(d_b1_A, 0, n*A_bytesizes[1]+1, ProcessStream) ) {throw std::runtime_error("CUDA memset failed");};
                dim3 gridDim(layers[0+1], n);       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1    where p is the #cols of the next layer 
                propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b2_A, d_b1_W, d_b1_A);
                
            // synchronise
                cudaStreamSynchronize(TransferStream);
                cudaStreamSynchronize(ProcessStream);



        //consider putting BOTH the propagate kernel and D2H memset in stream 2!! and just the H2D memset in stream 1. This might avoid bank conflicts if both stream1 and 2 are accessing the activation...
        // CYCLE:
            for(int i = 1; i<depth; i++){
                
                if (i%2 == 1){

                    // H2D: W[i+1] and p    //S1
                        if (cudaSuccess!= cudaMemcpyAsync(d_b1_W, W_list[i+1], W_bytesizes[i+1], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};

                    // D2H: A[i]            //S1
                        A_list[i] = new T[n*layers[i]+1]();     //this auto-initialises all to zero. FOR SOME REASON, IF I DONT INITIALISE THE LAST ELEMENT (be that thru either initialising the whole array, or just the last element), THE FIRST TIME DOING THE FORWARD PASS FAILS TO TRANSFER THE LAST ELEMENT.
                        cudaMemcpyAsync(A_list[i], d_b1_A, n*A_bytesizes[i]+1, cudaMemcpyDeviceToHost, TransferStream);

                    // <<<prop>>>(i)        //S2
                        if (cudaSuccess!= cudaMemset(d_b2_A, 0, n*A_bytesizes[i+1]+1) ) {throw std::runtime_error("CUDA memset failed");};
                        gridDim.x = layers[i+1];       //p is the #cols of W[i] (which is the #cols of A[i+1])
                        propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b1_A, d_b2_W, d_b2_A);

                    //synchronise
                        cudaStreamSynchronize(TransferStream);
                        cudaStreamSynchronize(ProcessStream);

                    // std::cout<<i<<std::endl; //debug

                }

                else{
                    // H2D: W[i+1] and p    //S1
                        if (cudaSuccess!= cudaMemcpyAsync(d_b2_W, W_list[i+1], W_bytesizes[i+1], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};
                    
                    // D2H: A[i]            //S1
                        A_list[i] = new T[n*layers[i]+1]();
                        cudaMemcpyAsync(A_list[i], d_b2_A, n*A_bytesizes[i]+1, cudaMemcpyDeviceToHost, TransferStream);
    
                    // <<<prop>>>(i)        //S2
                        if (cudaSuccess!= cudaMemset(d_b1_A, 0, n*A_bytesizes[i+1]+1) ) {throw std::runtime_error("CUDA memset failed");};
                        gridDim.x = layers[i+1];       //p is the #cols of W[i] (which is the #cols of A[i+1])
                        propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b2_A, d_b1_W, d_b1_A);

                    //synchronise
                        cudaStreamSynchronize(TransferStream);
                        cudaStreamSynchronize(ProcessStream);
                    
                    // std::cout<<i<<std::endl; //debug
                }
                // std::cout<<i<<std::endl; //debug
            }



        // DELOADING
        if ( (depth)%2 == 1) {

            // DELOADING: phase 1
                // D2H: A[depth]        //S1
                    A_list[depth] = new T[n*layers[depth]+1]();
                    if (cudaSuccess!=cudaMemcpyAsync(A_list[depth], d_b1_A, n*A_bytesizes[depth]+1, cudaMemcpyDeviceToHost, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};

                // <<<prop>>>(depth)    //S2
                    if (cudaSuccess!= cudaMemsetAsync(d_b2_A, 0, n*A_bytesizes[depth+1]+1, ProcessStream) ) {throw std::runtime_error("CUDA memset failed");};
                    gridDim.x = layers[depth+1];       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1    where p is the #cols of the next layer 
                    propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b1_A, d_b2_W, d_b2_A);
                    
                // synchronise
                    cudaStreamSynchronize(TransferStream);
                    cudaStreamSynchronize(ProcessStream);



            // DELOADING: phase 2
                // D2H: A[depth+1]      //S1
                    A_list[depth+1] = new T[n*layers[depth+1]+1]();
                    cudaMemcpyAsync(A_list[depth+1], d_b2_A, n*A_bytesizes[depth+1]+1, cudaMemcpyDeviceToHost, TransferStream);
                
                // synchronise
                    cudaStreamSynchronize(TransferStream);
        }

        else{

            // DELOADING: phase 1
                // D2H: A[depth]        //S1
                    A_list[depth] = new T[n*layers[depth]+1]();
                    cudaMemcpyAsync(A_list[depth], d_b2_A, n*A_bytesizes[depth]+1, cudaMemcpyDeviceToHost, TransferStream);

                // <<<prop>>>(depth)    //S2
                    if (cudaSuccess!= cudaMemset(d_b1_A, 0, n*A_bytesizes[depth+1]+1) ) {throw std::runtime_error("CUDA memset failed");};
                    gridDim.x = layers[depth+1];       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1    where p is the #cols of the next layer 
                    propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b2_A, d_b1_W, d_b1_A);
                    
                // synchronise
                    cudaStreamSynchronize(TransferStream);
                    cudaStreamSynchronize(ProcessStream);



            // DELOADING: phase 2
                // D2H: A[depth+1]      //S1
                    A_list[depth+1] = new T[n*layers[depth+1]+1]();
                    cudaMemcpyAsync(A_list[depth+1], d_b1_A, n*A_bytesizes[depth+1]+1, cudaMemcpyDeviceToHost, TransferStream);

                // synchronise
                    cudaStreamSynchronize(TransferStream);

        }

        // DECOMISSIONING
            // Free all buffers:
                cudaFreeAsync(d_b1_A, TransferStream);
                cudaFreeAsync(d_b1_W, ProcessStream);
                cudaFreeAsync(d_b2_A, TransferStream);
                cudaFreeAsync(d_b2_W, ProcessStream);


        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Forward pass CUDA execution time: " << duration.count() << "ms" << std::endl << std::endl;
        

        cudaStreamSynchronize(TransferStream);
        cudaStreamSynchronize(ProcessStream);
        cudaStreamDestroy(TransferStream);
        cudaStreamDestroy(ProcessStream);

    }


    else{
        if (multistream == 1) {std::cout<<"The network is too shallow to work with a dual-buffer system. Running standard single stream system..."<<std::endl;}

        auto start = std::chrono::high_resolution_clock::now();

        A_list[0] = input_batch;
        n = batch_samples;
        cudaMemcpyToSymbol(d_n, &n, sizeof(int));
        
        T *d_A, *d_W, *d_O;
        

        if (cudaSuccess!= cudaMalloc(&d_A, n*A_max+1) ) {throw std::runtime_error("CUDA memory allocation failed");};     //this is where malloc error coming from on second run 
        if (cudaSuccess!= cudaMalloc(&d_W, W_max) ) {throw std::runtime_error("CUDA memory allocation failed");};
        if (cudaSuccess!= cudaMalloc(&d_O, n*A_max+1) ) {throw std::runtime_error("CUDA memory allocation failed");};
        
        for(int i = 0; i<=depth; i++){
            // std::cout<<i<<std::endl; //debug
            propagate_layer(i, d_A, d_W, d_O);
        }

        cudaFree(d_W);
        cudaFree(d_A);
        cudaFree(d_O);
        d_W = nullptr;
        d_A = nullptr;
        d_O = nullptr;


        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Forward pass CUDA execution time: " << duration.count() << "ms" << std::endl << std::endl;
        
        return;
    } 

    return;

}



// // code comparison needed for initialisation of buffer:
// which is faster:

//             // create 2 streams
//                 cudaStream_t TransferStream;
//                 cudaStream_t ProcessStream;
//                 cudaStreamCreate(&TransferStream);
//                 cudaStreamCreate(&ProcessStream);
            

//             // allocate buffers (of max required size)
//                 if (cudaSuccess!= cudaMallocAsync(&d_b1_A, n*A_max, TransferStream) ) {throw std::runtime_error("CUDA memory allocation failed");}; 
//                 if (cudaSuccess!= cudaMallocAsync(&d_b2_A, n*A_max, TransferStream) ) {throw std::runtime_error("CUDA memory allocation failed");};
//                 if (cudaSuccess!= cudaMallocAsync(&d_b1_W, W_max, ProcessStream) ) {throw std::runtime_error("CUDA memory allocation failed");}; 
//                 if (cudaSuccess!= cudaMallocAsync(&d_b2_W, W_max, ProcessStream) ) {throw std::runtime_error("CUDA memory allocation failed");};


//             // H2D: A[0]
//                 if (cudaSuccess!= cudaMemcpyAsync(d_b2_A, &input_batch, n*A_bytesizes[0], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memset failed");};
            

//             // synchronise
//                 cudaStreamSynchronize(TransferStream);
//                 cudaStreamSynchronize(ProcessStream);

//             // more work...

// or:


//             // allocate buffers (of max required size) 
//                 if (cudaSuccess!= cudaMalloc(&d_b1_A, n*A_max) ) {throw std::runtime_error("CUDA memory allocation failed");}; 
//                 if (cudaSuccess!= cudaMalloc(&d_b2_A, n*A_max) ) {throw std::runtime_error("CUDA memory allocation failed");};
//                 if (cudaSuccess!= cudaMalloc(&d_b1_W, W_max) ) {throw std::runtime_error("CUDA memory allocation failed");}; 
//                 if (cudaSuccess!= cudaMalloc(&d_b2_W, W_max) ) {throw std::runtime_error("CUDA memory allocation failed");};
//                 //somehow its faster to do this with one stream instead of two?? do some more testing.
                

//             // H2D: A[0]
//                 if (cudaSuccess!= cudaMemcpy(d_b2_A, &input_batch, n*A_bytesizes[0], cudaMemcpyHostToDevice) ) {throw std::runtime_error("CUDA memset failed");};
            

//             // create 2 streams
//                 cudaStream_t TransferStream;
//                 cudaStream_t ProcessStream;
//                 cudaStreamCreate(&TransferStream);
//                 cudaStreamCreate(&ProcessStream);
            
//             // more work...
