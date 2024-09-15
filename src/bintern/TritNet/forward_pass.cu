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
void TritNet<T>::forward_pass(int batch_samples, T* &input_batch){
    if (depth > 2) {                                               //minimum depth is 3: loading and DELOADING costs 4 layers, so in order to have any benefit we need at least 1 more layer, i.e. 5 layers, or, a depth of 3.
        auto start = std::chrono::high_resolution_clock::now();

        A_list[0] = input_batch;  
        n = batch_samples;
        int m = layers[0];

        // INITIALISE DUAL-BUFFER MULTI-STREAM SYSTEM:
            // Device constants and pointers
                cudaMemcpyToSymbol(d_n, &n, sizeof(int));
                cudaMemcpyToSymbol(d_m, &m, sizeof(int));
                
                T *d_b1_A, *d_b1_W, *d_b2_A, *d_b2_W;     //need to change weights and activations to contain a matrix of the "same size", but then double it to hold two matrices: W_pos, W_neg.
                dim3 blockDim = WORD_SIZE;   


            // create 2 streams
                cudaStream_t TransferStream;
                cudaStream_t ProcessStream;
                cudaStreamCreate(&TransferStream);
                cudaStreamCreate(&ProcessStream);
            

            // allocate buffers (of max required size)
                if (cudaSuccess!= cudaMallocAsync(&d_b1_A, n*A_max, TransferStream) ) {throw std::runtime_error("CUDA memory allocation failed");}; 
                if (cudaSuccess!= cudaMallocAsync(&d_b2_A, n*A_max, TransferStream) ) {throw std::runtime_error("CUDA memory allocation failed");};
                if (cudaSuccess!= cudaMallocAsync(&d_b1_W, W_max, ProcessStream) ) {throw std::runtime_error("CUDA memory allocation failed");}; 
                if (cudaSuccess!= cudaMallocAsync(&d_b2_W, W_max, ProcessStream) ) {throw std::runtime_error("CUDA memory allocation failed");};


            // H2D: A[0]
                if (cudaSuccess!= cudaMemcpyAsync(d_b2_A, &input_batch, n*A_bytesizes[0], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};
            

            // synchronise
                cudaStreamSynchronize(TransferStream);
                cudaStreamSynchronize(ProcessStream);





        // LOADING: phase 1
            // H2D: W[0] and p      //S1
                if (cudaSuccess!= cudaMemcpyAsync(d_b1_W, W_list[0], W_bytesizes[0], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};

            // synchronise
                cudaStreamSynchronize(TransferStream);


        // LOADING: phase 2
            // H2D: W[1] and p      //S1
                if (cudaSuccess!= cudaMemcpyAsync(d_b2_W, W_list[1], W_bytesizes[1], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};

            // <<<prop>>>(0)        //S2
                if (cudaSuccess!= cudaMemset(d_b1_A, 0, n*A_bytesizes[1]) ) {throw std::runtime_error("CUDA memcpy failed");};
                dim3 gridDim(layers[0+1], n);       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1    where p is the #cols of the next layer 
                propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b2_A, d_b1_W, d_b1_A);
                
            // synchronise
                cudaStreamSynchronize(TransferStream);
                cudaStreamSynchronize(ProcessStream);




        //consider putting BOTH the propagate kernel and D2H memcpy in stream 2!! and just the H2D memcpy in stream 1. This might avoid bank conflicts if both stream1 and 2 are accessing the activation...
        // CYCLE:
            for(int i = 1; i<depth; i++){
                
                if (i%2 == 1){

                    // H2D: W[i+1] and p    //S1
                        if (cudaSuccess!= cudaMemcpyAsync(d_b1_W, W_list[i+1], W_bytesizes[i+1], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};

                    // D2H: A[i]            //S1
                        A_list[i] = new T[n*layers[i]];
                        cudaMemcpyAsync(A_list[i], d_b1_A, n*A_bytesizes[i], cudaMemcpyDeviceToHost, TransferStream);

                    // <<<prop>>>(i)        //S2
                        if (cudaSuccess!= cudaMemset(d_b2_A, 0, n*A_bytesizes[i+1]) ) {throw std::runtime_error("CUDA memcpy failed");};
                        gridDim.x = layers[i+1]/WORD_SIZE;       //p is the #cols of W[i] (which is the #cols of A[i+1])
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
                        A_list[i] = new T[n*layers[i]];
                        cudaMemcpyAsync(A_list[i], d_b2_A, n*A_bytesizes[i], cudaMemcpyDeviceToHost, TransferStream);
    
                    // <<<prop>>>(i)        //S2
                        if (cudaSuccess!= cudaMemset(d_b1_A, 0, n*A_bytesizes[i+1]) ) {throw std::runtime_error("CUDA memcpy failed");};
                        gridDim.x = layers[i+1]/WORD_SIZE;       //p is the #cols of W[i] (which is the #cols of A[i+1])
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
                    A_list[depth] = new T[n*layers[depth]];
                    cudaMemcpyAsync(A_list[depth], d_b1_A, n*A_bytesizes[depth], cudaMemcpyDeviceToHost, TransferStream);

                // <<<prop>>>(depth)    //S2
                    if (cudaSuccess!= cudaMemset(d_b2_A, 0, n*A_bytesizes[depth+1]) ) {throw std::runtime_error("CUDA memcpy failed");};
                        dim3 gridDim(layers[0+1], batch_samples);       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1    where p is the #cols of the next layer 
                    propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b1_A, d_b2_W, d_b2_A);
                    
                // synchronise
                    cudaStreamSynchronize(TransferStream);
                    cudaStreamSynchronize(ProcessStream);



            // DELOADING: phase 2
                // D2H: A[depth+1]      //S1
                    A_list[depth+1] = new T[n*layers[depth+1]];
                    cudaMemcpyAsync(A_list[depth+1], d_b2_A, n*A_bytesizes[depth+1], cudaMemcpyDeviceToHost, TransferStream);


        }

        else{

            // DELOADING: phase 1
                // D2H: A[depth]        //S1
                    A_list[depth] = new T[n*layers[depth]];
                    cudaMemcpyAsync(A_list[depth], d_b2_A, n*A_bytesizes[depth], cudaMemcpyDeviceToHost, TransferStream);

                // <<<prop>>>(depth)    //S2
                    if (cudaSuccess!= cudaMemset(d_b1_A, 0, n*A_bytesizes[depth+1]) ) {throw std::runtime_error("CUDA memcpy failed");};
                        dim3 gridDim(layers[0+1], batch_samples);       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1    where p is the #cols of the next layer 
                    propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b2_A, d_b1_W, d_b1_A);
                    
                // synchronise
                    cudaStreamSynchronize(TransferStream);
                    cudaStreamSynchronize(ProcessStream);



            // DELOADING: phase 2
                // D2H: A[depth+1]      //S1
                    A_list[depth+1] = new T[n*layers[depth+1]];
                    cudaMemcpyAsync(A_list[depth+1], d_b1_A, n*A_bytesizes[depth+1], cudaMemcpyDeviceToHost, TransferStream);


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
        auto start = std::chrono::high_resolution_clock::now();

        A_list[0] = input_batch;
        n = batch_samples;
        int m = layers[0];
        cudaMemcpyToSymbol(d_n, &n, sizeof(int));
        cudaMemcpyToSymbol(d_m, &m, sizeof(int));
        
        for(int i = 0; i<=depth; i++){
            // std::cout<<i<<std::endl; //debug
            propagate_layer(i);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Forward pass CUDA execution time: " << duration.count() << "ms" << std::endl << std::endl;
        
        return;
    } 

    return;

}


template <typename T>
void TritNet<T>::forward_pass_debug(int batch_samples, T* &input_batch, bool multistream){
    if (depth > 2) {                                               //minimum depth is 3: loading and DELOADING costs 4 layers, so in order to have any benefit we need at least 1 more layer, i.e. 5 layers, or, a depth of 3.
        }

    if (multistream == true) {
        auto start = std::chrono::high_resolution_clock::now();

        A_list[0] = input_batch;  
        n = batch_samples;
        int m = layers[0];

        // INITIALISE DUAL-BUFFER MULTI-STREAM SYSTEM:
            // Device constants and pointers
                cudaMemcpyToSymbol(d_n, &n, sizeof(int));
                cudaMemcpyToSymbol(d_m, &m, sizeof(int));
                
                T *d_b1_A, *d_b1_W, *d_b2_A, *d_b2_W;     //need to change weights and activations to contain a matrix of the "same size", but then double it to hold two matrices: W_pos, W_neg.
                dim3 blockDim = WORD_SIZE;   


            // create 2 streams
                cudaStream_t TransferStream;
                cudaStream_t ProcessStream;
                cudaStreamCreate(&TransferStream);
                cudaStreamCreate(&ProcessStream);
            

            // allocate buffers (of max required size)
                if (cudaSuccess!= cudaMallocAsync(&d_b1_A, n*A_max, TransferStream) ) {throw std::runtime_error("CUDA memory allocation failed");}; 
                if (cudaSuccess!= cudaMallocAsync(&d_b2_A, n*A_max, TransferStream) ) {throw std::runtime_error("CUDA memory allocation failed");};
                if (cudaSuccess!= cudaMallocAsync(&d_b1_W, W_max, ProcessStream) ) {throw std::runtime_error("CUDA memory allocation failed");}; 
                if (cudaSuccess!= cudaMallocAsync(&d_b2_W, W_max, ProcessStream) ) {throw std::runtime_error("CUDA memory allocation failed");};


            // H2D: A[0]
                if (cudaSuccess!= cudaMemcpyAsync(d_b2_A, &input_batch, n*A_bytesizes[0], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};
            

            // synchronise
                cudaStreamSynchronize(TransferStream);
                cudaStreamSynchronize(ProcessStream);





        // LOADING: phase 1
            // H2D: W[0] and p      //S1
                if (cudaSuccess!= cudaMemcpyAsync(d_b1_W, W_list[0], W_bytesizes[0], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};

            // synchronise
                cudaStreamSynchronize(TransferStream);


        // LOADING: phase 2
            // H2D: W[1] and p      //S1
                if (cudaSuccess!= cudaMemcpyAsync(d_b2_W, W_list[1], W_bytesizes[1], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};

            // <<<prop>>>(0)        //S2
                if (cudaSuccess!= cudaMemset(d_b1_A, 0, n*A_bytesizes[1]) ) {throw std::runtime_error("CUDA memcpy failed");};
                dim3 gridDim(layers[0+1], n);       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1    where p is the #cols of the next layer 
                propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b2_A, d_b1_W, d_b1_A);
                
            // synchronise
                cudaStreamSynchronize(TransferStream);
                cudaStreamSynchronize(ProcessStream);




        //consider putting BOTH the propagate kernel and D2H memcpy in stream 2!! and just the H2D memcpy in stream 1. This might avoid bank conflicts if both stream1 and 2 are accessing the activation...
        // CYCLE:
            for(int i = 1; i<depth; i++){
                
                if (i%2 == 1){

                    // H2D: W[i+1] and p    //S1
                        if (cudaSuccess!= cudaMemcpyAsync(d_b1_W, W_list[i+1], W_bytesizes[i+1], cudaMemcpyHostToDevice, TransferStream) ) {throw std::runtime_error("CUDA memcpy failed");};

                    // D2H: A[i]            //S1
                        A_list[i] = new T[n*layers[i]];
                        cudaMemcpyAsync(A_list[i], d_b1_A, n*A_bytesizes[i], cudaMemcpyDeviceToHost, TransferStream);

                    // <<<prop>>>(i)        //S2
                        if (cudaSuccess!= cudaMemset(d_b2_A, 0, n*A_bytesizes[i+1]) ) {throw std::runtime_error("CUDA memcpy failed");};
                        gridDim.x = layers[i+1]/WORD_SIZE;       //p is the #cols of W[i] (which is the #cols of A[i+1])
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
                        A_list[i] = new T[n*layers[i]];
                        cudaMemcpyAsync(A_list[i], d_b2_A, n*A_bytesizes[i], cudaMemcpyDeviceToHost, TransferStream);
    
                    // <<<prop>>>(i)        //S2
                        if (cudaSuccess!= cudaMemset(d_b1_A, 0, n*A_bytesizes[i+1]) ) {throw std::runtime_error("CUDA memcpy failed");};
                        gridDim.x = layers[i+1]/WORD_SIZE;       //p is the #cols of W[i] (which is the #cols of A[i+1])
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
                    A_list[depth] = new T[n*layers[depth]];
                    cudaMemcpyAsync(A_list[depth], d_b1_A, n*A_bytesizes[depth], cudaMemcpyDeviceToHost, TransferStream);

                // <<<prop>>>(depth)    //S2
                    if (cudaSuccess!= cudaMemset(d_b2_A, 0, n*A_bytesizes[depth+1]) ) {throw std::runtime_error("CUDA memcpy failed");};
                        dim3 gridDim(layers[0+1], batch_samples);       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1    where p is the #cols of the next layer 
                    propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b1_A, d_b2_W, d_b2_A);
                    
                // synchronise
                    cudaStreamSynchronize(TransferStream);
                    cudaStreamSynchronize(ProcessStream);



            // DELOADING: phase 2
                // D2H: A[depth+1]      //S1
                    A_list[depth+1] = new T[n*layers[depth+1]];
                    cudaMemcpyAsync(A_list[depth+1], d_b2_A, n*A_bytesizes[depth+1], cudaMemcpyDeviceToHost, TransferStream);


        }

        else{

            // DELOADING: phase 1
                // D2H: A[depth]        //S1
                    A_list[depth] = new T[n*layers[depth]];
                    cudaMemcpyAsync(A_list[depth], d_b2_A, n*A_bytesizes[depth], cudaMemcpyDeviceToHost, TransferStream);

                // <<<prop>>>(depth)    //S2
                    if (cudaSuccess!= cudaMemset(d_b1_A, 0, n*A_bytesizes[depth+1]) ) {throw std::runtime_error("CUDA memcpy failed");};
                        dim3 gridDim(layers[0+1], batch_samples);       //i.e. blockIdx.y goes from 0 -> n-1. blockIdx.x goes from 0 -> p/word_size -1    where p is the #cols of the next layer 
                    propagate<T><<<gridDim,blockDim,0,ProcessStream>>>(d_b2_A, d_b1_W, d_b1_A);
                    
                // synchronise
                    cudaStreamSynchronize(TransferStream);
                    cudaStreamSynchronize(ProcessStream);



            // DELOADING: phase 2
                // D2H: A[depth+1]      //S1
                    A_list[depth+1] = new T[n*layers[depth+1]];
                    cudaMemcpyAsync(A_list[depth+1], d_b1_A, n*A_bytesizes[depth+1], cudaMemcpyDeviceToHost, TransferStream);


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
        auto start = std::chrono::high_resolution_clock::now();

        A_list[0] = input_batch;
        n = batch_samples;
        int m = layers[0];
        cudaMemcpyToSymbol(d_n, &n, sizeof(int));
        cudaMemcpyToSymbol(d_m, &m, sizeof(int));
        
        for(int i = 0; i<=depth; i++){
            // std::cout<<i<<std::endl; //debug
            propagate_layer(i);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Forward pass CUDA execution time: " << duration.count() << "ms" << std::endl << std::endl;
        
        return;
    } 

    return;

}

