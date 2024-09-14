#include "TritNet.hpp"

#include <iostream>
#include <random>
#include <ctime>

void TritNet::random_init(){

    std::cout<<"Initialising weight matrices to random values..."<<std::endl<<std::endl;
    
    std::default_random_engine engine(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<int>distribution(0, (1<<WORD_SIZE) - 1);

    int W_idx = 0;
    for(int i=0; i<depth+1; i++ ){

        std::cout<<"W"<<i<<":"<<std::endl<<"(";

        for(int n=0; n < W_sizes[i]; n++){
            
            W_full[W_idx+n] = distribution(engine);               
            // *(W_ptrs[i] + n) = random_int;           //this is less readable, requires storing an array of pointers, and may not be much faster
            std::cout<<W_full[W_idx+n]<<",";
            //need to implement the rows and columns part lol.
        }

        std::cout<<")"<<std::endl<<std::endl;
        W_idx += W_sizes[i];    //DEBUG - just check this works out correctly.
        
    }

    std::cout<<"Weights initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;
   

    return;
}

void TritNet::array_init(int* input){

    std::cout<<"Initialising weight matrices to values from list provided..."<<std::endl<<std::endl;

    int W_idx = 0;
    for(int i=0; i<depth+1; i++ ){

        std::cout<<"W"<<i<<":"<<std::endl<<"(";

        for(int n=0; n < W_sizes[i]; n++){
            
            W_full[W_idx+n] = input[W_idx+n];               
            // *(W_ptrs[i] + n) = random_int;           //this requires storing an array of pointers, and may not be much faster?
            std::cout<<" "<<W_full[W_idx+n]<<",";
            //need to implement the rows and columns part of the printing mechanism lol.
        }

        std::cout<<")"<<std::endl<<std::endl;
        W_idx += W_sizes[i];    //DEBUG - just check this works out correctly.
        
    }

    std::cout<<"Weights initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;
   

    return;
}