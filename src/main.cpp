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

    int input_dim = 2;
    int output_dim = 2;
    int depth = 1;                      //number of hidden layers.
    int hidden_layers[1]={4};           //dimensions of hidden layers.

    TritNet network = TritNet(input_dim, output_dim, depth, hidden_layers); 
    // constructor constructs all the weight matrices for each layer. Here, there will be 3 matrices as there are 4 layers in total (2 hidden).
    network.random_init();

    activations input_batch = activations(4, input_dim/WORD_SIZE);
    input_batch.A(1,1) = 1;
    input_batch.A(2,1) = 0;
    input_batch.A(3,1) = 1;
    input_batch.A(4,1) = 1;
    input_batch.A.print();  
    

    network.forward_pass(input_batch);  // This is where the segfault is coming from
    network.activations_list[depth+1]->A.print(); //print final layer activations matrix


    //functions yet to be defined:
    
    // network.loss_calc(){
    //     loss = true_output - output_batch;
    // }

    // network.backprop(){
    //     ...
    // }

    return 0;
}
