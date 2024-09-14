#include "TritNet.hpp"

#include <iostream>

TritNet::TritNet() {

}

TritNet::TritNet(int input_dim, int output_dim, int depth, int* hidden_layers):depth(depth){    // Constructs the weight matrices for each layer, with appropriate (compressed) dimensions.
    
    layers[depth+2];
    W_ptrs[depth+1];                         // is there need for pointers? can simply accumulate W_size and use that as an index.
    W_sizes[depth+1];
    A_ptrs[depth+2];                         // is there need for pointers?
    //A_sizes[i] is just batch_size*layers[i]   -- no need to calculate.


    layers[0] = input_dim;
    W_size = 0;
    A_size_divbatchsize = 0;        //multiply this by batch_samples and you will get true A_size

    for(int i = 0; i<depth; i++) {

        layers[i+1] = hidden_layers[i];
        W_sizes[i] = layers[i]*layers[i+1];                 //NO SIZEOF(INT) YET ??
        
        W_size += W_sizes[i];
        A_size_divbatchsize += layers[i];
        
    }

    layers[depth+1] = output_dim;
    W_sizes[depth] = layers[depth]*layers[depth+1];         //NO SIZEOF(INT) YET ??
    W_size += W_sizes[depth];
    A_size_divbatchsize += layers[depth];

    A_size_divbatchsize += layers[depth+1];



    W_full[W_size];
    // A_full[batch_samples*A_size_divbatchsize];      //this shouldnt be called until the forward_pass(). because the array is static, not dynamic, so cannot be reallocated.


    //set up ptrs
    W_size = 0;
    for(int i=0; i<depth+1; i++){
        W_ptrs[i] = &(W_full[W_size]);      //   \sigma_0^i { W_sizes[i] }
        W_size += W_sizes[i];
    }

}