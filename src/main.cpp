#include "TritNet.hpp"

#include <cuda_runtime.h>
#include <iostream> //for debug

int main() {

 //========= GPU SELECT ==================
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaSetDevice(0);  // Select the first GPU (index 0)
    }
 //=======================================


 //========== MAIN PROGRAM ===============
    int N = 32*24;
    //params
    int input_dim = N;                  //                              NOTE: MUST BE MULTIPLE OF THE WORD_SIZE (zero-padding functionality not implemented yet)
    int output_dim = N;                 //                              NOTE: MUST BE MULTIPLE OF THE WORD_SIZE (zero-padding functionality not implemented yet)
    int depth = 0;                      //number of hidden layers.
    int hidden_layers[depth]={};        //dimensions of hidden layers.  NOTE: MUST BE MULTIPLES OF THE WORD_SIZE (zero-padding functionality not implemented yet)


    int batch_size = N;                 //number of samples per batch.


    // Set-up network
    TritNet network = TritNet<uint32_t>(input_dim, output_dim, depth, hidden_layers);


    // #########################################################################################
    // ## Random initialisation

        //weights init
            network.random_init();
        //

        //input init
            uint32_t* input_batch = new uint32_t[1 + (batch_size*input_dim/WORD_SIZE)];
            input_batch[0] = network.layers[0];
            std::cout<<"Initialising input activations batch to random values..."<<std::endl;
            randomise<uint32_t>(input_batch+1, batch_size*input_dim/WORD_SIZE);
            // print<uint32_t>(input_batch+1, batch_size, input_dim/WORD_SIZE);
            std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;
        //

    // ## Manual initialisation - useful for debug. with python 
        // IMPORTANT: THERE IS NO SIZE ASSERTION HERE. MAKE SURE INPUT IS CORRECT.
         
        // //weights init
        //     uint32_t python_weights[65536] = {};    //fill this out from python


        //     network.copy_init(python_weights);

        //  //if need to go back to <int> template of tritnet, can just do a "reinterpret_cast<int*>" on the data from python before inputting it to copy_init()
         
         
        //  //input init
        //      uint32_t original[batch_size*input_dim/WORD_SIZE] = {}; //fill this out from python

        //      std::cout<<"Initialising input activations batch to given list..."<<std::endl;
        //      uint32_t* input_batch = new uint32_t[1+batch_size*input_dim/WORD_SIZE];
        //      input_batch[0] = network.layers[0];
        //      copy_array<uint32_t>(input_batch+1, batch_size*input_dim/WORD_SIZE, original, 0);
        //      print<uint32_t>(input_batch+1, batch_size,input_dim/WORD_SIZE);
        //      std::cout<<std::endl<<"Inputs initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;
      
    // #########################################################################################


    for (int i = 0; i<3; i++){
        network.forward_pass(batch_size, input_batch, 0);   // 0 = single-stream, 1 = dual-buffer multi-stream.
        // network.print_layer(1);
        // std::cout<<network.A_list[1][32768]<<std::endl;
        network.clear();

        network.forward_pass(batch_size, input_batch, 1);   // 0 = single-stream, 1 = dual-buffer multi-stream.
        // network.print_layer(1);
        // std::cout<<network.A_list[1][32768]<<std::endl;
        network.clear();

    }
        
    //functions yet to be defined:
    
    // network.loss_calc(){
    //     loss = true_output - output_batch;
    // }

    // network.backprop(){
    //     ...
    // }

    return 0;
}
