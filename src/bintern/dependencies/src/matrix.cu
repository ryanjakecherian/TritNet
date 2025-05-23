#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <random>

#include "matrix.hpp"

__constant__ int d_WORD_SIZE = WORD_SIZE;   // word_size is probably most efficient at 32 bits. 
__constant__ int d_n;
__constant__ int d_m;                       // m is A.cols = B.rows
__constant__ int d_p;                       // p is B.cols

template<typename T>
__global__ void MatMul(T* d_a, T* d_b, T* d_c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    //NOTE: tid is zero indexed.

    if (tid < (d_n*d_p)) { // < n*p because less than, not less than or equal to!!

        d_c[tid] = 0;
        for(int k=0; k<d_m; ++k){

            d_c[tid] += d_a[(tid/d_p)*d_m + k] * d_b[(tid%d_p) + k*d_p];
            // ^above translates to: d_c[(tid)*c.cols+(k-1)] += d_a[(tid-1)*a.cols + k] * d_b[(tid*b.cols + k];
            // basically each thread outputs one element of the output matrix. 

            
            //need to figure out a way to truly parallelise this completely, and also to maximise cache usage not only by spacing each thread and incrementing, but also by using shared memory for each col/row. 
        }
    }
};



// constructors
template<typename T>
matrix<T>::matrix():rows(1), cols(1), size(1){
    byte_size = size*sizeof(T);
    head_flat = new T[1];
    // std::cout<<"default constructor called"<<std::endl; //for debug 
};

template<typename T>
matrix<T>::matrix(int n, int m):rows(n), cols(m), size(n*m){
    byte_size = size*sizeof(T);
    head_flat = new T[size];
    // std::cout<<"normal constructor called"<<std::endl; //for debug 
};

template<typename T>
matrix<T>::matrix(matrix<T>& A, matrix<T>& B){
    
    if (A.cols != B.rows) {
    throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    
    head_flat = new T[(A.rows)*(B.cols)];
    rows = A.rows;
    cols = B.cols;
    size = rows*cols; 
    byte_size = size*sizeof(T);

    cudaMemcpyToSymbol(d_n, &A.rows, sizeof(int));
    cudaMemcpyToSymbol(d_m, &A.cols, sizeof(int));
    cudaMemcpyToSymbol(d_p, &B.cols, sizeof(int));

    //cout << "(" << this->rows << "," << this->cols << ")" << endl;        //for debug

    T *d_a, *d_b, *d_c;
    
    if (cudaSuccess!= cudaMalloc(&d_a, A.byte_size) ) {throw std::runtime_error("CUDA memory allocation failed");};
    if (cudaSuccess!= cudaMalloc(&d_b, B.byte_size) ) {throw std::runtime_error("CUDA memory allocation failed");};
    if (cudaSuccess!= cudaMalloc(&d_c, byte_size) ) {throw std::runtime_error("CUDA memory allocation failed");};

    if (cudaSuccess!= cudaMemcpy(d_a, A.head_flat, A.byte_size, cudaMemcpyHostToDevice) ) {};
    if (cudaSuccess!= cudaMemcpy(d_b, B.head_flat, B.byte_size, cudaMemcpyHostToDevice) ) {};
    // if (cudaSuccess!= cudaMemcpy(d_c, this->head_flat, (this->rows)*(this->cols)*sizeof(T), cudaMemcpyHostToDevice) ) {};

    dim3 blockDim = 1024; //NEED TO FIGURE OUT THE OPTIMUM LAUNCH CONFIG.
    dim3 gridDim = 64;
    MatMul<<<gridDim,blockDim>>>(d_a,d_b,d_c);

    cudaMemcpy(this->head_flat, d_c, byte_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
//------------------

//copy constructor
template<typename T>
matrix<T>::matrix(matrix<T>& A){
    cols = A.cols;
    rows = A.rows;
    size = A.size;
    byte_size = A.byte_size;
    head_flat = new T[(A.rows)*(A.cols)];
    std::copy(A.head_flat, A.head_flat + size, head_flat);
};
//-----------------

// destructor
template<typename T>
matrix<T>::~matrix(){
    std::cout<<"matrix being destructed"<<std::endl;            //debugging purposes
    delete[] head_flat;
};
//----------------


// operator overloads
    //access operator()
    template<typename T>                                  
    T& matrix<T>::operator()(int i, int j) {
        int idx = (i-1)*cols+(j-1);     //the each element of a single row of the array is stored contiguously, such that the next row of elements is strided by the row length (number of columns).
        if((i>this->rows) | (j>this->cols)){                                                          //? do i need to OR this into the boolean expression: (idx > rows*cols) 
            throw std::runtime_error("Out of bounds access");                                         //! NEED TO ADD/CHANGE ERROR/EXCEPTION CATCHING TO ONE WHICH WORKS WITH CUDA.
        }
        else{return this->head_flat[idx];}                             
    }
        
                
    //multiply operator*
    template<typename T>
    matrix<T>& matrix<T>::operator*(matrix<T> &B){
        matrix<T>* C = new matrix(*this,B);
        return (*C);
    }

    //assignment operators=
    template<typename T>
    matrix<T>& matrix<T>::operator=(const matrix<T>& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }

        delete[] head_flat; // Clean up existing resources

        rows = other.rows;
        cols = other.cols;
        size = other.size;
        byte_size = other.byte_size;
        head_flat = new T[size];
        std::copy(other.head_flat, other.head_flat + size, head_flat);

        return *this;
    }

    template<typename T>
    matrix<T>& matrix<T>::operator=(matrix<T>&& other) noexcept {
        if (this == &other) {
            return *this; // Handle self-assignment
        }

        delete[] head_flat; // Clean up existing resources

        rows = other.rows;
        cols = other.cols;
        size = other.cols;
        byte_size = other.byte_size;
        head_flat = other.head_flat;
        other.head_flat = nullptr;

        return *this;
    }
//-------------


// handy functions
template<typename T>
void matrix<T>::print(){
    std::cout<<"( ";
    for(int i = 0;i<size;i++){
        std::cout<<head_flat[i];
        if((i+1)%cols==0){std::cout<<";";}
        std::cout<<" ";
    }
    std::cout<<")"<<std::endl;
}

template<typename T>
void matrix<T>::random_init(){
    static std::default_random_engine engine(static_cast<unsigned int>(std::time(nullptr)));  //the static keyword isnt particularly efficient here, as this instruction will be read each time random_init() is called, and then a decision made based on whether its been read before. There is probably a better alternative, but this part is not time critical. Also, this is fine for initialising an individual matrix. For initialising a group, its better to do the more optimal function (see random_init.cpp [line 22])
    std::uniform_int_distribution<int>distribution(0, (1<<WORD_SIZE) - 1);

    std::cout<< "Debug"<<std::endl;
    
    for(int i=0; i<size; i++){

        head_flat[i] = distribution(engine);
        
    }

    std::cout<<"Matrix initialised to random values from 0 to "<< (1<<WORD_SIZE) - 1 << ":"<<std::endl;
    print();

}
//--------------


template<typename T>
void matrix<T>::array_init(std::initializer_list<T> head){
    
    if(head.size() != size){
        throw std::runtime_error("Initialiser list does not match array size. Check the list you gave me!");
    }

    int i=0;
    for (int value : head) {

        head_flat[i] = value;
        i++;
    }

    std::cout<<"Matrix initialised to values from list provided:"<<std::endl;
    print();

}

//this instantiation is just to prevent linker errors due to this being an implementation file for a templated class - https://stackoverflow.com/a/495056/23298718
template class matrix<int>;