# tritnet
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