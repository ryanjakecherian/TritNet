#include "TritNet.hpp"

#include <iostream>
#include <random>
#include <ctime>
#include <initializer_list>
#include <cstdint>


//NOTE:
//  for weights, n = layers[i] m = layers[i+1]
//  for activations, n = batch_size, m = layers[i]


template<typename T>
void randomise(T* arr, int size){
    // int word_size = WORD_SIZE;                                                  //to bypass compiler warnings
    int32_t upper = (WORD_SIZE < 32) ? ((1 << WORD_SIZE) - 1) : UINT32_MAX;

    static std::default_random_engine engine(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<int>distribution(0, upper);

    // std::cout<<"(";
    for(int i = 0;i<size;i++){
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


template<typename T>
void copy_array(T* copy, int size, T* original, int idx){

    for(int i=0; i<size; i++){
        copy[i] = original[idx+i];        
    }
}


//these instantiations are just to prevent linker errors due to this being an implementation file for a templated class - https://stackoverflow.com/a/495056/23298718
template void randomise<uint32_t>(uint32_t* arr, int size);
template void print<uint32_t>(uint32_t* arr, int n, int m);
template void copy_array<uint32_t>(uint32_t* copy, int size, uint32_t* original, int idx);


template<typename T>
void TritNet<T>::random_init(){

    // THIS IS THE MORE OBJECT ORIENTED INITIALISATION METHOD:

        std::cout<<"Initialising weight matrices to random values..."<<std::endl<<std::endl;
    
        for(int n=0;n<=depth;n++){
            std::cout<<"W_{"<<n<<"}"<<std::endl;

            std::cout<<"W_pos"<<std::endl;
            randomise<T>(W_list[n]+1,layers[n]*layers[n+1]*WORD_SIZE);        //hifted by 1 since W[n][0] always holds its column dimension!
            print<T>(W_list[n]+1,layers[n],layers[n+1]*WORD_SIZE);

            std::cout<<"W_neg"<<std::endl;
            randomise<T>(W_list[n]+1+(layers[n]*layers[n+1]*WORD_SIZE),layers[n]*layers[n+1]*WORD_SIZE);
            print<T>(W_list[n]+1+(layers[n]*layers[n+1]*WORD_SIZE),layers[n],layers[n+1]*WORD_SIZE);

            std::cout<<std::endl;
        }
        std::cout<<"Weights initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;

    //

    return;
}

template<typename T>
void TritNet<T>::copy_init(T* original){

    std::cout<<"Initialising weight matrices from lists provided..."<<std::endl<<std::endl;
    int idx = 0;
    for(int n=0;n<=depth;n++){

        std::cout<<"W_{"<<n<<"}"<<std::endl;
        
        std::cout<<"W_pos"<<std::endl;
        copy_array<T>(W_list[n]+1, layers[n]*layers[n+1]*WORD_SIZE, original, idx);           //shifted by 1 since W[n][0] always holds its column dimension! 
        // print<T>(W_list[n]+1,layers[n],layers[n+1]*WORD_SIZE);

        idx +=  layers[n]*layers[n+1]*WORD_SIZE;

        std::cout<<"W_neg"<<std::endl;
        copy_array<T>(W_list[n]+1+(layers[n]*layers[n+1]*WORD_SIZE), layers[n]*layers[n+1]*WORD_SIZE, original, idx);
        // print<T>(W_list[n]+1+(layers[n]*layers[n+1]*WORD_SIZE),layers[n],layers[n+1]*WORD_SIZE);
        
        std::cout<<std::endl;


        idx +=  layers[n]*layers[n+1]*WORD_SIZE;
    }

    std::cout<<"Weights initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;

}