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

    //params
    int input_dim = 8;                  //                              NOTE: MUST BE MULTIPLE OF THE WORD_SIZE (zero-padding functionality not implemented yet)
    int output_dim = 4;                 //                              NOTE: MUST BE MULTIPLE OF THE WORD_SIZE (zero-padding functionality not implemented yet)
    int depth = 2;                      //number of hidden layers.
    int hidden_layers[depth]={8, 12};        //dimensions of hidden layers.  NOTE: MUST BE MULTIPLES OF THE WORD_SIZE (zero-padding functionality not implemented yet)


    int batch_size = 300;                 //number of samples per batch.


    // Set-up network
    TritNet network = TritNet<uint32_t>(input_dim, output_dim, depth, hidden_layers);


    // #########################################################################################
    // ## Random initialisation

        // //weights init
        //     network.random_init();
        // //

        // //input init
        //     uint32_t* input_batch = new uint32_t[1 + (batch_size*input_dim/WORD_SIZE)];
        //     input_batch[0] = network.layers[0];
        //     std::cout<<"Initialising input activations batch to random values..."<<std::endl;
        //     randomise<uint32_t>(input_batch+1, batch_size*input_dim/WORD_SIZE);
        //     print<uint32_t>(input_batch+1, batch_size, input_dim/WORD_SIZE);
        //     std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;
        // //

    // ## Manual initialisation - useful for debug. with python 
        // IMPORTANT: THERE IS NO SIZE ASSERTION HERE. MAKE SURE INPUT IS CORRECT.
         
        // //weights init

            uint32_t python_weights[104] = {4, 0, 8, 0, 0, 12, 8, 2, 0, 6, 0, 4, 14, 0, 0, 2, 1, 3, 5, 2, 7, 0, 0, 9, 14, 1, 3, 11, 1, 15, 0, 4, 1, 12, 9, 9, 0, 14, 0, 11, 4, 5, 4, 8, 8, 2, 9, 0, 0, 15, 4, 11, 2, 8, 9, 3, 0, 1, 6, 6, 11, 0, 11, 0, 9, 0, 2, 7, 3, 9, 4, 9, 14, 0, 8, 0, 5, 7, 0, 8, 2, 1, 2, 0, 8, 0, 0, 9, 2, 0, 5, 0, 1, 10, 0, 11, 7, 6, 9, 2, 1, 5, 2, 5};    //fill this out from python


            network.copy_init(python_weights);

        //  //if need to go back to <int> template of tritnet, can just do a "reinterpret_cast<int*>" on the data from python before inputting it to copy_init()
         
         
        //  //input init
             uint32_t original[batch_size*input_dim/WORD_SIZE] = {6, 5, 14, 7, 14, 12, 2, 1, 12, 7, 4, 8, 5, 9, 5, 1, 11, 2, 0, 13, 1, 6, 14, 8, 8, 12, 12, 7, 1, 1, 12, 1, 9, 4, 14, 10, 8, 12, 9, 4, 3, 10, 15, 4, 4, 0, 1, 7, 3, 14, 10, 1, 8, 9, 4, 10, 6, 2, 14, 1, 4, 10, 6, 6, 1, 8, 13, 2, 13, 8, 15, 11, 14, 9, 8, 3, 15, 1, 13, 6, 14, 7, 7, 1, 0, 5, 15, 6, 12, 10, 3, 15, 8, 7, 4, 3, 8, 5, 14, 15, 10, 7, 4, 5, 0, 0, 5, 10, 4, 8, 13, 2, 3, 2, 2, 14, 15, 5, 2, 11, 3, 9, 13, 15, 7, 4, 5, 8, 0, 14, 3, 5, 7, 0, 15, 9, 10, 14, 13, 12, 1, 2, 14, 6, 2, 2, 3, 14, 6, 11, 13, 9, 10, 3, 6, 7, 0, 0, 13, 8, 14, 0, 5, 3, 6, 1, 15, 8, 9, 14, 2, 11, 6, 10, 14, 0, 5, 5, 1, 11, 0, 9, 8, 1, 1, 2, 1, 14, 6, 1, 10, 1, 4, 14, 3, 14, 4, 7, 9, 8, 13, 0, 9, 13, 6, 4, 6, 2, 6, 9, 13, 10, 12, 3, 10, 12, 3, 14, 3, 1, 9, 12, 7, 5, 2, 5, 12, 13, 12, 10, 10, 5, 5, 15, 3, 15, 6, 4, 4, 11, 4, 15, 6, 7, 0, 12, 0, 9, 9, 9, 6, 15, 7, 15, 14, 3, 4, 7, 7, 8, 6, 5, 7, 0, 2, 5, 3, 11, 13, 10, 15, 12, 12, 10, 1, 1, 14, 12, 4, 8, 12, 13, 9, 14, 8, 15, 13, 0, 5, 1, 5, 15, 9, 1, 12, 11, 15, 2, 4, 10, 13, 8, 8, 12, 8, 13, 6, 0, 15, 0, 12, 15, 6, 14, 0, 14, 12, 1, 0, 12, 5, 10, 3, 14, 10, 4, 11, 9, 3, 2, 12, 0, 5, 9, 14, 14, 3, 9, 2, 4, 3, 3, 9, 14, 9, 14, 13, 0, 15, 13, 9, 5, 5, 6, 6, 12, 11, 7, 4, 12, 1, 10, 12, 9, 2, 4, 7, 9, 9, 12, 5, 1, 2, 13, 6, 10, 11, 15, 1, 2, 8, 5, 12, 1, 13, 4, 0, 14, 0, 14, 10, 1, 3, 14, 13, 8, 1, 0, 14, 13, 14, 3, 11, 11, 2, 9, 4, 14, 3, 11, 6, 5, 13, 1, 0, 14, 11, 7, 7, 4, 15, 14, 0, 12, 5, 13, 8, 8, 10, 4, 9, 9, 11, 13, 7, 13, 8, 13, 15, 13, 10, 5, 1, 6, 1, 15, 6, 7, 14, 2, 11, 12, 7, 0, 0, 10, 12, 0, 11, 11, 14, 10, 5, 8, 12, 7, 11, 14, 8, 0, 7, 3, 14, 0, 1, 14, 1, 2, 5, 6, 14, 14, 0, 9, 3, 8, 9, 0, 2, 0, 15, 5, 12, 13, 9, 2, 14, 10, 2, 3, 13, 9, 5, 12, 15, 10, 2, 5, 2, 12, 12, 0, 0, 7, 6, 10, 10, 10, 14, 6, 11, 7, 9, 7, 1, 0, 10, 15, 12, 1, 12, 14, 2, 12, 15, 8, 0, 14, 12, 4, 15, 14, 11, 10, 0, 8, 14, 15, 11, 0, 6, 5, 6, 15, 3, 10, 10, 10, 2, 12, 14, 2, 10, 7, 3, 10, 4, 11, 9, 15, 0, 0, 8, 6, 11, 2, 12, 11, 6, 4, 2, 7, 5, 4, 9, 5, 6, 12, 4, 6, 10, 0, 13, 11, 5, 0, 3, 12, 7, 7}; //fill this out from python

             std::cout<<"Initialising input activations batch to given list..."<<std::endl;
             uint32_t* input_batch = new uint32_t[1+batch_size*input_dim/WORD_SIZE];
             input_batch[0] = network.layers[0];
             copy_array<uint32_t>(input_batch+1, batch_size*input_dim/WORD_SIZE, original, 0);
             print<uint32_t>(input_batch+1, batch_size,input_dim/WORD_SIZE);
             std::cout<<std::endl<<"Inputs initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;
      
    // #########################################################################################


    for (int i = 0; i<2; i++){
        network.forward_pass(batch_size, input_batch, 0);   // 0 = single-stream, 1 = dual-buffer multi-stream.
        network.print_layer(depth+1);
        network.clear();

        network.forward_pass(batch_size, input_batch, 1);   // 0 = single-stream, 1 = dual-buffer multi-stream.
        network.print_layer(depth+1);
        network.clear();

    }
    
    //     network.forward_pass(batch_size, input_batch, 1);

    //         //print first layer activations for comparison with python
    //         std::cout<<"First layer:"<<std::endl;
    //         print(network.A_list[1]+1, batch_size,network.layers[1]);
    //         std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;

    //         std::cout<<"Second layer:"<<std::endl;
    //         print(network.A_list[2]+1, batch_size,network.layers[1]);
    //         std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;


    //         //print final layer activations matrix
    //         std::cout<<"Output activations:"<<std::endl;
    //         print(network.A_list[depth+1]+1, batch_size,network.layers[depth+1]);
    //         std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;

    //         for (int i=1; i<depth+2; i++){
    //             delete[] network.A_list[i];
    //         }

    // }
        
    //functions yet to be defined:
    
    // network.loss_calc(){
    //     loss = true_output - output_batch;
    // }

    // network.backprop(){
    //     ...
    // }

    return 0;
}
