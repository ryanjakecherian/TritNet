# running tritnet forward pass (for users)

```
cd TritNet/src
cmake ..
make
make run
```
Configuration is set in the cmakelists.txt via selecting the desired `src` and `include` folders.
However this is unecessary as we are only interested in the `bintern` folder.





# main repo structure (for devs)
The `bintern` folder implements the binary activations ternary weights matrix multiply algorithm in `TritNet.pdf`. 
The `terntern` folder is in development to implement the ternary-ternary matrix multiply algorithm in `TritNet.pdf`.

Below we show the `bintern` dependency structure for a visual understanding.

```
.
├── README.md
|
├── CMakeLists.txt
├── build
|
├── include
│   └── TritNet.hpp
|
└── src
    ├── main.cpp
    |
    └── bintern
        |
        ├── TritNet
        |   ├── constructors.cpp
        |   ├── forward_pass.cu
        |   ├── propagate.cu
        |   └── init.cpp
        |    
        └── dependencies
            ├── include
            │   ├── activations.hpp
            │   ├── matrix.hpp
            │   └── weights.hpp
            |
            └── src
                ├── activations.cpp
                ├── matrix.cu
                └── weights.cpp
    

48 directories, 75 files

built with:
tree -a
```
