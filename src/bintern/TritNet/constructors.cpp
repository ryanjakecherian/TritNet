#include "TritNet.hpp"

#include <iostream>

TritNet::TritNet() {

}

TritNet::TritNet(int input_dim, int output_dim, int depth, int* hidden_layers):depth(depth){    // Constructs the weight matrices for each layer, with appropriate (compressed) dimensions.

    this->weights_list = new weights*[depth+1];             //can i avoid using the new keyword here?
    this->activations_list = new activations*[depth+2];
    
    weights_list[0] = new weights(input_dim/WORD_SIZE, hidden_layers[0]);
    for(int i = 1; i<depth; i++) {
        weights_list[i] = new weights(hidden_layers[i-1]/WORD_SIZE, hidden_layers[i]);
    } 
    weights_list[depth] = new weights(hidden_layers[depth-1]/WORD_SIZE, output_dim);
}


// BELOW IS A STATIC VERSION OF THE CONSTRUCTOR.
// This doesnt work because the pointers become dangling, because all the weights were declared locally, and then go out of scope when the constructor finishes, and thus are deleted.

    // TritNet::TritNet(int input_dim, int output_dim, int depth, int* hidden_layers):depth(depth){

    //     weights* weights_list[depth];             //pointer to an array of pointers to matrices, of size depth+2
    //     activations* activations_list[depth+1];

    //     bintern_weights a(input_dim, hidden_layers[0]);
    //     weights_list[0] = &a;  //can i avoid using the new keyword here?
    //     for(int i = 1; depth-1; i++) {
    //         bintern_weights a(hidden_layers[i], hidden_layers[i+1]);
    //         weights_list[i] = &a;
    //     } 
    //     bintern_weights a(hidden_layers[depth-1], output_dim);
    //     weights_list[depth] = &a;

    // }

//


//NEED TO DEFINE A DESTRUCTOR WHICH DESTRUCTS ALL THESE FKN WEIGHTS AND ACTIVATIONS