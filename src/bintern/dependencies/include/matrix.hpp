#ifndef MATRIX_HPP
#define MATRIX_HPP

#undef WORD_SIZE
#define WORD_SIZE 4

template<typename T> // lets use int
struct matrix{
     // fields
        int rows=1;
        int cols=1;
        int size = rows*cols;
        int bit_size = size*sizeof(T);
        T* head_flat;   //head (pointer to first element) of flattened array
    //-------------

    // constructors
        matrix();
        
        matrix(int n, int m);

        matrix(matrix<T>& A, matrix<T>& B);
    //------------------

    //copy constructor
        matrix(matrix<T>& A);
    //-----------------

    // destructor
        ~matrix();
    //----------------


    // operator overloads
        //access operator()                                  
        T& operator()(int i, int j);
        
                
        //multiply operator*
        matrix<T>& operator*(matrix<T> &B);

        //assignment operators=
        matrix<T>& operator=(const matrix<T>& other);

        matrix<T>& operator=(matrix<T>&& other) noexcept;
    //-------------


    // handy functions
        void print();
        void random_init();
    //--------------
};

// #include "matrix.cu"


#endif