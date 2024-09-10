#ifndef TRITNET_HPP
#define TRITNET_HPP

#include "weights.hpp"
#include "activations.hpp"

class TritNet{
    public:
        //methods
        TritNet();

        TritNet(int input_dim, int output_dim, int depth,int* hidden_layers);
        
        void random_init();

        void forward_pass(activations& input_batch);
        void fwd(int i);  //specify in makefile whether to include bintern, bintern_mma, terntern, or terntern_mma implementation files


        matrix<int> loss_calc(matrix<int> output_batch, matrix<int> true_outputs);


        void backward_pass();
        void bp(int i); //specify in makefile whether to include bintern, bintern_mma, terntern, or terntern_mma implementation files

        //fields
        int depth;  //# of hidden layers
        weights** weights_list;
        activations** activations_list;
        int batch_samples;

};




#endif