#include "TritNet.hpp"

#include <iostream>
#include <random>
#include <ctime>

void TritNet::random_init(){

    std::default_random_engine engine(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<int>distribution(0, 1<<WORD_SIZE - 1);

    for(int n=0;n<=depth;n++){
        for(int i=0; i<weights_list[n]->W_plus.size; i++){

            (weights_list[n])->W_plus.head_flat[i] = distribution(engine);
            (weights_list[n])->W_neg.head_flat[i] = distribution(engine);
            
        }
    }

    std::cout<<"Weight matrices initialised to random values:"<<std::endl;
    for(int n=0;n<=depth;n++){
        
        weights_list[n]->W_plus.print();
        weights_list[n]->W_neg.print();
        
    }

    return;
}