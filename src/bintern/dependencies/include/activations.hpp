#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "matrix.hpp"

struct activations{
    matrix<int> A;

    activations(const int& n, const int& m); //const in order to allow the use of lvalues (e.g. literals such as '5')
    activations(activations& original); //would const make this more optimised?
    ~activations();
    activations&  operator=(const activations& other);
    activations&  operator=(activations&& other) noexcept;
};


#endif