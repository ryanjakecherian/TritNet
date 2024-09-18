#include "TritNet.hpp"

#include <iostream>
#include <random>
#include <ctime>
#include <initializer_list>
#include <cstdint>

template<typename T>
void TritNet<T>::clear(){
    
    for (int i=1; i<depth+2; i++){
        delete[] A_list[i];
    }

    std::cout<<"Cleared all activations (except for input batch)."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;
}

template<typename T>

void TritNet<T>::print_layer(int i){
    if (n==0) {std::runtime_error("Batch size not provided. Provide an input and run the forward pass.");} 
    else {
        std::cout<<"Layer #"<< i+1 <<" activations:"<<std::endl;        //(shifted to account for zero-indexing - just more intuitive this way)
        print(A_list[i]+1, n, layers[i]);
        std::cout<<std::endl<<std::endl<<"_____________________"<<std::endl;
    };
}