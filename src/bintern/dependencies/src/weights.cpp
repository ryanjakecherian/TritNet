#include "weights.hpp"

#include <iostream>


weights::~weights() = default;

weights::weights(const int& n, const int& m):W_plus(n,m), W_neg(n,m){
    // the first parameter has been passed by value. This is unfortunately the most efficient option so far, because the first input is always some calculated value (input_dim/WORD_SIZE), and thus passing by reference would first require saving the result of this calculation, which is even slower than pass by value. Also the compiler might be able to optimise this away, so...
    
    // size, rows and cols of W_plus and neg are the same, but are calculated twice. this should be optimised away.
}