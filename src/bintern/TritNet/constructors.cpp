#include "TritNet.hpp"

#include <iostream>

template<typename T>
TritNet<T>::TritNet() {

}

template<typename T>
TritNet<T>::TritNet(int input_dim, int output_dim, int depth, int* hidden_layers):depth(depth){    // Constructs the weight matrices for each layer, with appropriate (compressed) dimensions.

    if (depth < 1) {
        // throw std::invalid_argument("Network construction requires at least one hidden layer. Zero-depth network not implemented. Zero-depth is not compatible with the dual-buffer MatMul, but would work with the single streaming MatMul, providing a constructor was built to handle it.");
        
        layers = new int[depth+2];
        W_list = new T*[depth+1];
        A_list = new T*[depth+2];
        W_bytesizes = new int[depth+1];
        A_bytesizes = new int[depth+2];
        W_max = 0;
        A_max = 0;

        if (input_dim%WORD_SIZE != 0) {std::cout<<input_dim<<std::endl;   throw std::invalid_argument("Matrix class needs zero-padding functionality");    }
        layers[0] = input_dim/WORD_SIZE;
        A_bytesizes[0] = sizeof(T)*layers[0];
        if (A_bytesizes[0]>A_max) {A_max = A_bytesizes[0];};
        W_list[0] = new T[(2 * input_dim/WORD_SIZE * output_dim)+1];
        W_list[0][0] = output_dim;
        W_bytesizes[0] = sizeof(T) * (2 * input_dim/WORD_SIZE * output_dim +1);
        if (W_bytesizes[0]>W_max) {W_max = W_bytesizes[0];};

        if (output_dim%WORD_SIZE != 0) {std::cout<<output_dim<<std::endl;   throw std::invalid_argument("Matrix class needs zero-padding functionality");    }
        layers[depth+1] = output_dim/WORD_SIZE;
        A_bytesizes[depth+1] = sizeof(T)*layers[depth+1];
        if (A_bytesizes[depth+1]>A_max) {A_max = A_bytesizes[depth+1];};


    }
    else{
        layers = new int[depth+2];
        W_list = new T*[depth+1];
        A_list = new T*[depth+2];
        W_bytesizes = new int[depth+1];
        A_bytesizes = new int[depth+2];
        W_max = 0;
        A_max = 0;

        

        if (input_dim%WORD_SIZE != 0) {std::cout<<input_dim<<std::endl;   throw std::invalid_argument("Matrix class needs zero-padding functionality");    }
        layers[0] = input_dim/WORD_SIZE;                    //note that layers is compressed
        A_bytesizes[0] = sizeof(T)*layers[0];
        if (A_bytesizes[0]>A_max) {A_max = A_bytesizes[0];};
        W_list[0] = new T[(2 * input_dim/WORD_SIZE * hidden_layers[0])+1];      //only n (#rows) is compressed
        W_list[0][0] = hidden_layers[0];                        //note the #COLS of W are the original layer dimensions (NOT compressed). note that since hidden layers is an array of INTS, the first element of each weight will be CAST from int to whatever type T the network is.
        W_bytesizes[0] = sizeof(T) * (2 * input_dim/WORD_SIZE * hidden_layers[0] +1);
        if (W_bytesizes[0]>W_max) {W_max = W_bytesizes[0];};
        

        for(int i = 1; i<depth; i++) {
            if (hidden_layers[i-1]%WORD_SIZE != 0) {std::cout<<hidden_layers[i-1]<<std::endl;   throw std::invalid_argument("Matrix class needs zero-padding functionality");    }
            layers[i] = hidden_layers[i-1]/WORD_SIZE;
            A_bytesizes[i] = sizeof(T)*layers[i];
            if (A_bytesizes[i]>A_max) {A_max = A_bytesizes[i];};
            W_list[i] = new T[2 * hidden_layers[i-1]/WORD_SIZE * hidden_layers[i]];
            W_list[i][0] = hidden_layers[i];
            W_bytesizes[i] = sizeof(T) * (2 * hidden_layers[i-1]/WORD_SIZE * hidden_layers[i]+1);
            if (W_bytesizes[i]>W_max) {W_max = W_bytesizes[i];};
        } 
        

        if (hidden_layers[depth-1]%WORD_SIZE != 0) {std::cout<<hidden_layers[depth-1]<<std::endl;   throw std::invalid_argument("Matrix class needs zero-padding functionality");    }
        layers[depth] = hidden_layers[depth-1]/WORD_SIZE;
        A_bytesizes[depth] = sizeof(T)*layers[depth];
        if (A_bytesizes[depth]>A_max) {A_max = A_bytesizes[depth];};
        W_list[depth] = new T[2 * hidden_layers[depth-1]/WORD_SIZE * output_dim];
        W_list[depth][0] = output_dim;
        W_bytesizes[depth] = sizeof(T) * (2 * hidden_layers[depth-1]/WORD_SIZE * output_dim+1);
        if (W_bytesizes[depth]>W_max) {W_max = W_bytesizes[depth];};
        


        if (output_dim%WORD_SIZE != 0) {std::cout<<output_dim<<std::endl;   throw std::invalid_argument("Matrix class needs zero-padding functionality");    }
        layers[depth+1] = output_dim/WORD_SIZE;
        A_bytesizes[depth+1] = sizeof(T)*layers[depth+1];
        if (A_bytesizes[depth+1]>A_max) {A_max = A_bytesizes[depth+1];};
    
    }
    //TO DO: MAKE THIS PINNED (PAGE-LOCKED) MEMORY.

    
    // DEBUG:
    // std::cout<<"The largest layer is of size: "<<A_max<<"*batch_size bytes"<<std::endl;
    // std::cout<<"The largest weights array is of size: "<<W_max<<" bytes"<<std::endl;
}


//NEED TO DEFINE A DESTRUCTOR WHICH DESTRUCTS ALL THESE FKN WEIGHTS AND ACTIVATIONS