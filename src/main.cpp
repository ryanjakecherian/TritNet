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
    int depth = 3;                      //number of hidden layers.
    int hidden_layers[depth]={4, 2, 2};        //dimensions of hidden layers.  NOTE: MUST BE MULTIPLES OF THE WORD_SIZE (zero-padding functionality not implemented yet)
                                        //1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024

    int batch_size = 4;                 //number of samples per batch.


    // Set-up network
    TritNet network = TritNet<int>(input_dim, output_dim, depth, hidden_layers);


    // #########################################################################################
    // ## Random initialisation

        //weights init
            network.random_init();
        //

        //input init
            int* input_batch = new int[batch_size*input_dim/WORD_SIZE];
            std::cout<<"Initialising input activations batch to random values..."<<std::endl;
            randomise<int>(input_batch, batch_size, input_dim/WORD_SIZE);
            print<int>(input_batch, batch_size, input_dim/WORD_SIZE);
            std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;
        //

    // ## Manual initialisation - useful for debug. with python 
         // IMPORTANT: WHILST I HAVE IMPLEMENTED SIZE CHECKING FOR ARRAY_INIT(), THERE IS NO DIMENSION (ROWS, COLS) CHECKER, AS THE INPUT IS A FLAT LIST.
         //
         //weights init
            //  std::cout<<"Initialising weight matrices from lists provided..."<<std::endl<<std::endl;
            //  
            //  std::cout<<"W_{"<<n<<"}"<<std::endl;
            //    network.weights_list[0]->W_plus.array_init( {7, 1, 0 ,8, 3, 6, 15, 10} );
            //    network.weights_list[0]->W_neg.array_init( {5, 15, 4, 5, 13, 4, 11, 14} );
            //  std::cout<<std::endl;
            //
            //  std::cout<<"W_{"<<n<<"}"<<std::endl;
            //    network.weights_list[1]->W_plus.array_init( {1, 0, 15, 4 ,5, 5, 14, 3} );
            //    network.weights_list[1]->W_neg.array_init( {0, 9, 15, 1, 10, 1, 3, 6} );
            //  std::cout<<std::endl;
            //
            //  std::cout<<"Weights initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;
         //
         //
         //input init
            //  activations input_batch = activations(batch_size, input_dim/WORD_SIZE);
            //  std::cout<<"Initialising input activations batch to given list..."<<std::endl;
            //  input_batch.A.array_init( {15, 15, 5, 6} );  
            //
            //  std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;
      
    // #########################################################################################


    //forward pass
    network.forward_pass(batch_size, input_batch);  


    // //debug
    // network.activations_list[depth]->A.print();


    //print final layer activations matrix
    std::cout<<"Output activations:"<<std::endl;
    print(network.A_list[depth+1], batch_size, network.layers[depth+1]);
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
