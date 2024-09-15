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
    int input_dim = 32;                  //                              NOTE: MUST BE MULTIPLE OF THE WORD_SIZE (zero-padding functionality not implemented yet)
    int output_dim = 32;                 //                              NOTE: MUST BE MULTIPLE OF THE WORD_SIZE (zero-padding functionality not implemented yet)
    int depth = 6;                      //number of hidden layers.
    int hidden_layers[depth]={64, 64, 64, 64, 64, 64};        //dimensions of hidden layers.  NOTE: MUST BE MULTIPLES OF THE WORD_SIZE (zero-padding functionality not implemented yet)
                                        //1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024

    int batch_size = 4;                 //number of samples per batch.


    // Set-up network
    TritNet network = TritNet<int>(input_dim, output_dim, depth, hidden_layers);


    // // #########################################################################################
    // // ## Random initialisation

    //     //weights init
    //         network.random_init();
    //     //

    //     //input init
    //         int* input_batch = new int[batch_size*input_dim/WORD_SIZE];
    //         std::cout<<"Initialising input activations batch to random values..."<<std::endl;
    //         randomise<int>(input_batch, batch_size, input_dim/WORD_SIZE);
    //         print<int>(input_batch, batch_size, input_dim/WORD_SIZE);
    //         std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;
    //     //

    // ## Manual initialisation - useful for debug. with python 
         // IMPORTANT: WHILST I HAVE IMPLEMENTED SIZE CHECKING FOR ARRAY_INIT(), THERE IS NO DIMENSION (ROWS, COLS) CHECKER, AS THE INPUT IS A FLAT LIST.
         //
         //weights init

            int python_weights[] = {}; //fill this out from python

             std::cout<<"Initialising weight matrices from list provided..."<<std::endl<<std::endl;
             network.copy_init(python_weights);
             std::cout<<"Weights initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;
         
         
         //input init
             int original[batch_size*input_dim/WORD_SIZE] = {}; //fill this out from python

             std::cout<<"Initialising input activations batch to given list..."<<std::endl;
             int* input_batch = new int[batch_size*input_dim/WORD_SIZE];
             copy_array<int>(input_batch, batch_size*input_dim/WORD_SIZE, original, 0);
             print<int>(input_batch, batch_size*input_dim/WORD_SIZE);
             std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;
      
    // #########################################################################################


    //forward pass
        // network.forward_pass(batch_size, input_batch);  


    //forward pass debug mode:
        network.forward_pass_debug(batch_size, input_batch, 1);
        
        network.forward_pass_debug(batch_size, input_batch, 0);

        network.forward_pass_debug(batch_size, input_batch, 1);

        network.forward_pass_debug(batch_size, input_batch, 0);




    //print final layer activations matrix
    std::cout<<"Output activations:"<<std::endl;
    print(network.A_list[depth+1], batch_size*network.layers[depth+1]);
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
