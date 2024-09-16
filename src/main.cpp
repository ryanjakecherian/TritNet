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
    int input_dim = 4;                  //                              NOTE: MUST BE MULTIPLE OF THE WORD_SIZE (zero-padding functionality not implemented yet)
    int output_dim = 4;                 //                              NOTE: MUST BE MULTIPLE OF THE WORD_SIZE (zero-padding functionality not implemented yet)
    int depth = 1;                      //number of hidden layers.
    int hidden_layers[depth]={4};        //dimensions of hidden layers.  NOTE: MUST BE MULTIPLES OF THE WORD_SIZE (zero-padding functionality not implemented yet)
                                        //1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024

    int batch_size = 4;                 //number of samples per batch.


    // Set-up network
    TritNet network = TritNet<uint32_t>(input_dim, output_dim, depth, hidden_layers);


    // #########################################################################################
    // // ## Random initialisation

    //     //weights init
    //         network.random_init();
    //     //

    //     //input init
    //         uint32_t* input_batch = new uint32_t[batch_size*input_dim/WORD_SIZE];
    //         std::cout<<"Initialising input activations batch to random values..."<<std::endl;
    //         randomise<uint32_t>(input_batch, batch_size*input_dim/WORD_SIZE);
    //         print<uint32_t>(input_batch, batch_size, input_dim/WORD_SIZE);
    //         std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;
    //     //

    // ## Manual initialisation - useful for debug. with python 
         // IMPORTANT: THERE IS NO SIZE ASSERTION HERE. MAKE SURE INPUT IS CORRECT.
         
         //weights init

            uint32_t python_weights[32] = {0, 0, 2, 2, 2, 1, 1, 0, 3, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 3, 3, 2}; //fill this out from python


             network.copy_init(python_weights);
             //if need to go back to <int> template of tritnet, can just do a "reinterpret_cast<int*>" on the data from python before inputting it to copy_init()
         
         
         //input init
             uint32_t original[batch_size*input_dim/WORD_SIZE] = {2, 3, 1, 3, 0, 3, 0, 2}; //fill this out from python

             std::cout<<"Initialising input activations batch to given list..."<<std::endl;
             uint32_t* input_batch = new uint32_t[batch_size*input_dim/WORD_SIZE];
             copy_array<uint32_t>(input_batch, batch_size*input_dim/WORD_SIZE, original, 0);
             print<uint32_t>(input_batch, batch_size,input_dim/WORD_SIZE);
             std::cout<<std::endl<<"Inputs initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;
      
    // #########################################################################################


    //forward pass
        // network.forward_pass(batch_size, input_batch);  


    //forward pass debug mode:
        network.forward_pass(batch_size, input_batch, 0);

            //print first layer activations for comparison with python
            std::cout<<"First layer:"<<std::endl;
            print(network.A_list[1], batch_size,network.layers[1]);
            std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;


            //print final layer activations matrix
            std::cout<<"Output activations:"<<std::endl;
            print(network.A_list[depth+1], batch_size,network.layers[depth+1]);
            std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;


        network.forward_pass(batch_size, input_batch, 1);

            //print first layer activations for comparison with python
            std::cout<<"First layer:"<<std::endl;
            print(network.A_list[1], batch_size,network.layers[1]);
            std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;


            //print final layer activations matrix
            std::cout<<"Output activations:"<<std::endl;
            print(network.A_list[depth+1], batch_size,network.layers[depth+1]);
            std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;


        network.forward_pass(batch_size, input_batch, 1);

            //print first layer activations for comparison with python
            std::cout<<"First layer:"<<std::endl;
            print(network.A_list[1], batch_size,network.layers[1]);
            std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;


            //print final layer activations matrix
            std::cout<<"Output activations:"<<std::endl;
            print(network.A_list[depth+1], batch_size,network.layers[depth+1]);
            std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;


        network.forward_pass(batch_size, input_batch, 0);

            //print first layer activations for comparison with python
            std::cout<<"First layer:"<<std::endl;
            print(network.A_list[1], batch_size,network.layers[1]);
            std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;


            //print final layer activations matrix
            std::cout<<"Output activations:"<<std::endl;
            print(network.A_list[depth+1], batch_size,network.layers[depth+1]);
            std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;   


    //functions yet to be defined:
    
    // network.loss_calc(){
    //     loss = true_output - output_batch;
    // }

    // network.backprop(){
    //     ...
    // }

    return 0;
}
