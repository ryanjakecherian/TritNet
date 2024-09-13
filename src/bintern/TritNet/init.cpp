#include "TritNet.hpp"

#include <iostream>
#include <random>
#include <ctime>

void TritNet::random_init(){

    // THIS IS THE MORE OBJECT ORIENTED INITIALISATION METHOD:

        std::cout<<"Initialising weight matrices to random values..."<<std::endl<<std::endl;
        for(int n=0;n<=depth;n++){
            std::cout<<"W_{"<<n<<"}"<<std::endl;
            (weights_list[n])->W_plus.random_init();
            (weights_list[n])->W_neg.random_init();
            std::cout<<std::endl;
        }
        std::cout<<"Weights initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;

    //

    // THIS IS THE MORE EFFICIENT RANDOMISATION METHOD:

        // std::default_random_engine engine(static_cast<unsigned int>(std::time(nullptr)));
        // std::uniform_int_distribution<int>distribution(0, (1<<WORD_SIZE) - 1);

        // for(int n=0;n<=depth;n++){
        //     for(int i=0; i<weights_list[n]->W_plus.size; i++){

        //         (weights_list[n])->W_plus.head_flat[i] = distribution(engine);
        //         (weights_list[n])->W_neg.head_flat[i] = distribution(engine);
                
        //     }
        // }

        // std::cout<<"Weight matrices initialised to random values:"<<std::endl;
        // for(int n=0;n<=depth;n++){
            
        //     weights_list[n]->W_plus.print();
        //     weights_list[n]->W_neg.print();
            
        // }

    //

    return;
}

// void TritNet::array_init(int*** head){

//     // THIS IS THE MORE OBJECT ORIENTED INITIALISATION METHOD:

//         std::cout<<"Initialising weight matrices from lists provided..."<<std::endl<<std::endl;

//         for(int n=0;n<=depth;n++){
//             std::cout<<"W_{"<<n<<"}"<<std::endl;
//             (weights_list[n])->W_plus.array_init(head[n][0]);
//             (weights_list[n])->W_neg.array_init(head[n][1]);
//             std::cout<<std::endl;
//         }

//         std::cout<<"Weights initialised."<<std::endl<<std::endl<<std::endl<<"_____________________"<<std::endl;

//     //
// }