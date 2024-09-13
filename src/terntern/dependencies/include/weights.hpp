#ifndef WEIGHTS_HPP
#define WEIGHTS_HPP

#include "matrix.hpp"

struct weights{                             
    weights(const int& n, const int& m); // the first parameter is being passed by value. This is unfortunately the most efficient option so far, because the first input is always some calculated value (input_dim/WORD_SIZE), and thus passing by reference would first require saving the result of this calculation, which is even slower than pass by value. Also the compiler might be able to optimise this away, so...
    ~weights();

    matrix<int> W0, W1;
};

#endif