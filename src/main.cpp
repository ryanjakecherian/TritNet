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
    int input_dim = 4;
    int output_dim = 4;
    int depth = 1;                      //number of hidden layers.
    int hidden_layers[1]={8};           //dimensions of hidden layers.

    int batch_size = 4;                 //number of samples per batch.

    //set-up network
    TritNet network = TritNet(input_dim, output_dim, depth, hidden_layers); // constructor constructs all the weight matrices for each layer. Here, there will be 3 matrices as there are 4 layers in total (2 hidden).
    network.random_init();

    //random input
    activations input_batch = activations(batch_size, input_dim/WORD_SIZE);
    std::cout<<"Initialising input activations batch to random values..."<<std::endl;
    input_batch.A.random_init();
    std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;

    //forward pass
    network.forward_pass(input_batch);  // This is where the segfault is coming from
    
    //print output
    std::cout<<"Output activations:"<<std::endl;
    network.activations_list[depth+1]->A.print(); //print final layer activations matrix
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
