#include "TritNet.hpp"

#include <cuda.h>
#include <iostream>
#include <chrono> //for timing

extern __constant__ int d_n;

void TritNet::forward_pass(activations &input_batch){
    auto start = std::chrono::high_resolution_clock::now();



    // activations_list[0] = &input_batch;   ==> UPDATE THIS
    
    batch_samples = input_batch.A.rows;  //==> update this too

    A_full[input_batch.A.rows*A_size_divbatchsize];

    cudaMemcpyToSymbol(d_n, &batch_samples, sizeof(int));




    // OK SO HERE I NEED TO PUT IN THE DUAL BUFFER SYSTEM
    // HAVE ODD AND EVEN ONES
    // THREE CUDA STREAMS... WAIT SO HOW MANY BUFFERS THEN?? I THINK 2 IS SITLL PERFECT - ONE FOR DATA TRANSFER BOTH WAYS, JUST NEED TO WAIT FOR TWO EVENTS (H2D AND D2H) TO FINISH.

    
    for(int i = 0; i<=depth; i++){
    // if odd
        // D2H (buffer1)

        // propagate (buffer2)

        // H2D (buffer1)


        // sync - where does the wait for event command go?

    //if even
        // D2H (buffer2)

        // propagate (buffer1)

        // H2D (buffer2)



        // sync - where does the wait for event command go?
    //





    
        // std::cout<<i<<std::endl; //debug
        propagate_layer(i);
    }




    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Forward pass CUDA execution time: " << duration.count() << "ms" << std::endl << std::endl;
    
    return;
}


// cudaMemcpyAsync(d_buffer1, h_matrix_chunk + i * chunk_size, 
//                 chunk_size * sizeof(float), cudaMemcpyHostToDevice, stream1);

// // Wait for transfer into d_buffer2 from the previous iteration to complete
// cudaEventRecord(event, stream1);
// cudaStreamWaitEvent(stream2, event, 0);

// // Use d_buffer2 for computation (process the chunk transferred in the previous iteration)
// process_chunk_kernel<<<grid, block, 0, stream2>>>(d_buffer2);