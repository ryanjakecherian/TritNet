#ifndef TRITNET_HPP
#define TRITNET_HPP

#undef WORD_SIZE
#define WORD_SIZE 2

#include <cuda_runtime.h>

template<typename T>
class TritNet{
    public:
        //methods
        TritNet();

        TritNet(int input_dim, int output_dim, int depth,int* hidden_layers);   // Constructs the weight matrices for each layer, with appropriate (compressed) dimensions.
        
        void random_init();
        void array_init(int* weight_init);

        void forward_pass(int batch_samples, T* &input_batch);
        void propagate_layer(int i);  //specify in makefile whether to include bintern, bintern_mma, terntern, or terntern_mma implementation files


        int* loss_calc(int* output_batch, int* true_outputs);


        void backward_pass();
        void bp(int i); //specify in makefile whether to include bintern, bintern_mma, terntern, or terntern_mma implementation files

        //fields
        int depth;  //# of hidden layers
        int n; //batch_samples
        int* layers;

        T **W_list, ** A_list;
        int* W_bytesizes, *A_bytesizes;
        int W_max, A_max;
};


// Kernel declaration
template<typename T> // lets use int
__global__ void propagate(T* d_A, T* d_W, T* d_c);

//handy functions
template<typename T>
void randomise(T* arr, int n, int m);
template<typename T>
void print(T* arr, int n, int m);

//this instantiation is just to prevent linker errors due to this being an implementation file for a templated class - https://stackoverflow.com/a/495056/23298718
template class TritNet<int>;

#endif