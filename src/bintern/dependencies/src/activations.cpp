#include "activations.hpp"

//constructors
activations::activations(const int& n, const int& m):A(n,m){
}


//copy constructor
    activations::activations(activations& original):A(original.A){ //would putting this as const make it more optimised?
    };
//-----------------

// destructor
    activations::~activations() = default;
//----------------

//assignment operators=
    activations& activations::operator=(const activations& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }

        A = other.A;

        return *this;
    }

    activations& activations::operator=(activations&& other) noexcept {
        if (this == &other) {
            return *this; // Handle self-assignment
        }

        A = other.A;

        return *this;
    }
//-------------