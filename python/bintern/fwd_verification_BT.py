# generate {-1,0,1} random matrices, with appropriate sizes for each layer
# generate {-1,0,1} random matrix, with appropriate size for input batch
import random
import time
import sys

# Redirect print output to a file
sys.stdout = open("python_output.txt", "w")

WORD_SIZE = 32

# activation
def sigma(A):
    for i in range(len(A)):

        if (A[i] >= 0) :
            A[i] = 1

        else:
            A[i] = -1
          
    return A


    # Apply the sigma function using PyTorch tensor operations
    result = torch.where(tensor >= 0, torch.tensor(1, dtype=tensor.dtype), torch.tensor(-1, dtype=tensor.dtype))
    return result

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
    A_bin = [0] * len(input)
    for i in range (len(input)):
        if (input[i] == -1) :
            A_bin[i] = 1

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

input_dim = 1024              
output_dim = 1024
depth = 0
hidden_layers = []       #layers must be divisable by word_size
batch_size = 30

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

# print first layer output for comparison with tritnet
first_layer = convA_BT(output)
first_layer = comp_hori(first_layer,batch_size,layers[1],WORD_SIZE)
print("first layer output for comparison with tritnet:")
print(first_layer)

for i in range(1, len(layers)-1):
    output = MatMul(output, weights[i], batch_size, layers[i], layers[i+1])
    output = sigma(output)

print("final output:")
print(output)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Forward pass Python execution time: {elapsed_time:.6f} seconds")

print("output in cuda format:")
output = convA_BT(output)
output = comp_hori(output,batch_size,layers[depth+1],WORD_SIZE)
print(output)
#########################





#########################
# CUDA 

## converting the weights
W_pos, W_neg = convW_BT(weights)

### compressing the weights
cuda_weights_size = 0
for n in range(len(weights)):
    cuda_weights_size += 2*layers[n]*layers[n+1]
    W_pos[n] = comp_vert(W_pos[n],layers[n],layers[n+1],WORD_SIZE)
    W_neg[n] = comp_vert(W_neg[n],layers[n],layers[n+1],WORD_SIZE)

### getting the size of the total list (necessary for cuda code)
print("size of total cuda weights array: ")
print(cuda_weights_size//WORD_SIZE)

### concatenating all into one list
cuda_weights = [0]
for n in range(len(weights)):
    cuda_weights += W_pos[n] + W_neg[n]
cuda_weights.pop(0)




## converting the input
A_bin = convA_BT(input)
A_bin = comp_hori(A_bin,batch_size,layers[0],WORD_SIZE)








## these can be fed into CUDA
print("cuda weights:")
print(cuda_weights)
print("cuda input:")
print(A_bin)



## this is potential code for mapping to TernTern format (from W => W0*W1)
# for n in range(len(new_list)):
#     for i in range(len(W)):
#         new_list[n]->W1[i] = not(not( (old_list->W[i]) >> 1 ))  #maps {-1,0,1} -> {1,0,0}
#         new_list[n]->W0[i] = not(old_list->W[i])                #only evaluates true if input is zero



########################

# Close the file when done
sys.stdout.close()

# Optional: Restore print to the console (if needed after the redirection)
sys.stdout = sys.__stdout__








# get error with these inputs:
# input_dim = 8                  
# output_dim = 2
# depth = 6
# hidden_layers = [32, 32, 32, 32, 32, 32]       #layers must be divisable by word_size
# batch_size = 4
