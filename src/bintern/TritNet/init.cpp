#include "TritNet.hpp"

#include <iostream>
#include <random>
#include <ctime>
#include <initializer_list>
#include <cstdint>




template<typename T>
void randomise(T* arr, int n, int m){
    int word_size = WORD_SIZE;                                                  //to bypass compiler warnings
    int32_t upper = (WORD_SIZE < 32) ? ((1 << word_size) - 1) : UINT32_MAX;

    static std::default_random_engine engine(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<int>distribution(0, upper);

    // std::cout<<"(";
    for(int i = 0;i<n*m;i++){
        arr[i] = distribution(engine);
        // std::cout<<arr[i];

        // if((i+1)%m==0) {std::cout<<"; ";} else {std::cout<<", ";}
        // std::cout<<" ";
    }
    // std::cout<<")"<<std::endl;

    std::cout<<"Matrix initialised to random values from 0 to "<< upper << ":"<<std::endl;
}

template<typename T>
void print(T* arr, int n, int m){
    std::cout<<"( ";
    for(int i = 0;i<n*m;i++){
        std::cout<<arr[i];
        if((i+1)%m==0) {std::cout<<"; ";} else {std::cout<<", ";}
    }
    std::cout<<")"<<std::endl;
}

//where for weights, n = layers[i] m = layers[i+1]
//  for activations, n = batch_size, m = layers[i]

//this instantiation is just to prevent linker errors due to this being an implementation file for a templated class - https://stackoverflow.com/a/495056/23298718
template void print<int>(int* arr, int n, int m);



template<typename T>
void TritNet<T>::random_init(){

    // THIS IS THE MORE OBJECT ORIENTED INITIALISATION METHOD:

        std::cout<<"Initialising weight matrices to random values..."<<std::endl<<std::endl;
    
        for(int n=0;n<=depth;n++){
            std::cout<<"W_{"<<n<<"}"<<std::endl;

            std::cout<<"W_pos"<<std::endl;
            randomise<T>(W_list[n]+1,layers[n],layers[n+1]);
            // print<T>(W_list[n]+1,layers[n],layers[n+1]);

            std::cout<<"W_neg"<<std::endl;
            randomise<T>(W_list[n]+1+(layers[n]*layers[n+1]),layers[n],layers[n+1]);
            // print<T>(W_list[n]+1+(layers[n]*layers[n+1]),layers[n],layers[n+1]);

            std::cout<<std::endl;
        }
        std::cout<<"Weights initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;

    //

    return;
}

template<typename T>
void TritNet<T>::array_init(int* head){

    // THIS IS THE MORE OBJECT ORIENTED INITIALISATION METHOD:

        std::cout<<"Initialising weight matrices from lists provided..."<<std::endl<<std::endl;

        for(int n=0;n<=depth;n++){
            std::cout<<"W_{"<<n<<"}"<<std::endl;
            // (weights_list[n])->W_plus.array_init(head[n][0]);
            // (weights_list[n])->W_neg.array_init(head[n][1]);
            std::cout<<std::endl;
        }

        std::cout<<"Weights initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;

    //
}