# generate {-1,0,1} random matrices, with appropriate sizes for each layer
# generate {-1,0,1} random matrix, with appropriate size for input batch
import random
import time

WORD_SIZE = 32

# activation
def sigma(A):
    for i in range(len(A)):

        if (A[i] >= 0) :
            A[i] = 1

        else:
            A[i] = -1
          
    return A

# Matmul
def MatMul(A, B, N, M, P):
    C = [0] * (N * P)
    
    for i in range(N):
        for j in range(P):
            
            C[i * P + j] = sum(A[i * M + k] * B[k * P + j] for k in range(M))
    
    return C

# convert weights => (W_pos - W_neg)
def convW_BT(weights):
    W_pos = [0] * len(weights)
    W_neg = [0] * len(weights)
    for n in range(len(weights)):
        W_pos[n] = [0] * (layers[n]*layers[n+1])
        W_neg[n] = [0] * (layers[n]*layers[n+1])
        for i in range (layers[n]*layers[n+1]):
            if (weights[n][i] == -1) :
                W_pos[n][i] = 0
                W_neg[n][i] = 1
            
            if (weights[n][i] == 1) :
                W_pos[n][i] = 1
                W_neg[n][i] = 0

    return W_pos, W_neg
    
# convert A => A_bin 
def convA_BT(input):
    A_bin = [0] * batch_size*layers[0]
    for i in range (batch_size*layers[0]):
        if (input[i] == -1) :
            A_bin[i] = 0

    return A_bin
    
# horizontal compression
def comp_hori(arr, n, m, WORD_SIZE):
    arr_c = [0]*(n*m//WORD_SIZE)

    for i in range(n*m//WORD_SIZE):
        
        for j in range (WORD_SIZE):
            arr_c[i] = arr_c[i] | (arr[(i*WORD_SIZE)+j] << (WORD_SIZE-1-j))


    return arr_c

# vertical compression
def comp_vert(arr,n,m,WORD_SIZE):
    arr_c = [0]*(n*m//WORD_SIZE)

    for i in range(n*m//WORD_SIZE):
        
        for j in range (WORD_SIZE):
            index = ((i//m)*WORD_SIZE*m + i%m) + j*m
            assert index < len(arr), f"Index out of bounds: {index} for array length {len(arr)}"
            
            arr_c[i] = arr_c[i] | (arr[( (i//m)*WORD_SIZE*m + i%m) + j*m] << (WORD_SIZE-1-j))


    return arr_c

# NET PARAMS
########################

input_dim = 32                  
output_dim = 32
depth = 6
hidden_layers = [64, 64, 64, 64, 64, 64]       #layers must be divisable by word_size
batch_size = 4

########################


# SETUP
########################
layers = [input_dim] + hidden_layers + [output_dim]


weights = [0] * (len(layers)-1)
for n in range(len(layers)-1):
    weights[n] = [0] * layers[n]*layers[n+1]
    for i in range (layers[n]*layers[n+1]):
       weights[n][i] = random.choice([-1, 0, 1])


input =[0] * batch_size*layers[0]
for i in range (batch_size*layers[0]):
    input[i] = random.choice([-1, 1])


# FORWARD PASS
#########################

# Record the start time
start_time = time.time()

output = MatMul(input, weights[0], batch_size, layers[0], layers[1])
output = sigma(output)

for i in range(1, len(layers)-1):
    output = MatMul(output, weights[i], batch_size, layers[i], layers[i+1])
    output = sigma(output)

print(output)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Forward pass Python execution time: {elapsed_time:.6f} seconds")
#########################





#########################
# CUDA 
# for n in range(len(new_list)):
#     for i in range(len(W)):
#         new_list[n]->W1[i] = not(not( (old_list->W[i]) >> 1 ))  #maps {-1,0,1} -> {1,0,0}
#         new_list[n]->W0[i] = not(old_list->W[i])                #only evaluates true if input is zero


# converting the weights
W_pos, W_neg = convW_BT(weights)

cuda_weights_size = 0
for n in range(len(weights)):
    cuda_weights_size += 2*layers[n]*layers[n+1]
    W_pos[n] = comp_vert(W_pos[n],layers[n],layers[n+1],WORD_SIZE)
    W_neg[n] = comp_vert(W_neg[n],layers[n],layers[n+1],WORD_SIZE)

print(cuda_weights_size//WORD_SIZE)
cuda_weights = [0] * (cuda_weights_size//WORD_SIZE)
for n in range(len(weights)):
    cuda_weights += [W_pos[n]] + [W_neg[n]]


# converting the input
A_bin = convA_BT(input)
A_bin = comp_hori(A_bin,batch_size,layers[0],WORD_SIZE)








# these can be fed into CUDA
print(cuda_weights)
print(A_bin)

########################










# get error with these inputs:
# input_dim = 8                  
# output_dim = 2
# depth = 6
# hidden_layers = [32, 32, 32, 32, 32, 32]       #layers must be divisable by word_size
# batch_size = 4
