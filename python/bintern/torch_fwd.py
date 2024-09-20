import random
import time
import sys
import torch




##############################################################
# DEVICE CHECK
print(f"PyTorch CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current GPU Device ID: {torch.cuda.current_device()}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# WIPE FILES
with open('init_for_cuda.txt', 'w') as file:
        pass  # Opening in 'w' mode wipes the file

with open('python_results.txt', 'w') as file:
        pass  # Opening in 'w' mode wipes the file




##############################################################
# NET FUNCTIONS

#pytorch activation
# def sigma_torch(tensor):
#     return torch.where(tensor >= 0, torch.ones_like(tensor), torch.full_like(tensor, -1))

def sigma_torch(tensor):
    return torch.sign(tensor)  # Similar to your custom function but optimized

# convert weights => (W_pos - W_neg)
def convW_BT(weights):
    W_pos = [0] * len(weights)
    W_neg = [0] * len(weights)
    for n in range(len(weights)):
        W_pos[n] = [0] * (layers[n]*layers[n+1])
        W_neg[n] = [0] * (layers[n]*layers[n+1])
        for i in range (layers[n]):
            for j in range (layers[n+1]):

                if (weights[n][i,j] == -1) :
                    W_pos[n][i*layers[n+1]+j] = 0
                    W_neg[n][i*layers[n+1]+j] = 1
                
                if (weights[n][i,j] == 1) :
                    W_pos[n][i*layers[n+1]+j] = 1
                    W_neg[n][i*layers[n+1]+j] = 0

    return W_pos, W_neg
    
# convert A => A_bin 
def convA_BT(input):
    A_bin = [0]*(input.shape[0]*input.shape[1])
    for i in range (input.shape[0]):
        for j in range (input.shape[1]):
            if (input[i,j] == -1) :
                A_bin[i*input.shape[1]+j] = 1

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

## old functions:
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



for z in range(4):
    ############################################################
    # NET PARAMS


    WORD_SIZE = 32
    N = 32*4

    input_dim = N              
    output_dim = N
    depth = 0
    hidden_layers = []       #layers must be divisable by word_size      #1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024
    batch_size = N




    ############################################################
    # SETUP

    layers = [input_dim] + hidden_layers + [output_dim]

    weights = [0] * (len(layers)-1)
    weights_GPU = [0] * (len(layers)-1)

    for n in range(len(layers)-1):
        weights[n] = torch.zeros(layers[n]*layers[n+1], dtype=torch.float16)
        for i in range (layers[n]*layers[n+1]):
            weights[n][i] = random.choice([-1, 0, 1])
        
        weights[n] = weights[n].reshape(layers[n], layers[n+1])

    activations = [0] * (len(layers))

    input = torch.zeros(batch_size*layers[0], dtype=torch.float16)
    for i in range (batch_size*layers[0]):
        input[i] = random.choice([-1, 1])

    input=input.reshape(batch_size, layers[0])

    activations[0] = input



    ############################################################
    # FORWARD PASS
    # Redirect print output to a file
    

    start_time = time.time()


    input = activations[0].to(device)
    weights_GPU[0] = weights[0].to(device)

    output = torch.matmul(input, weights_GPU[0])
    output = sigma_torch(output)
    activations[1] = output.to('cpu')

    for i in range(1, len(layers)-1):
        weights_GPU[i] = weights[i].to(device)
        output = torch.matmul(output,weights_GPU[i])
        output = sigma_torch(output)
        activations[i+1] = output.to('cpu')

    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Iteration #{z}: forward pass PyTorch execution time: {elapsed_time:.6f} seconds")

    print('\n')
    print('\n')
    print("___________")



# sys.stdout = open("python_results.txt", "a")
# print("torch output:")
# print(activations[depth+1])

# print("output in cuda format:")
# output = convA_BT(output)
# output = comp_hori(output,batch_size,layers[depth+1],WORD_SIZE)
# print(output)

# print('\n')
# print('\n')
# print("___________")

# sys.stdout.close()
# sys.stdout = sys.__stdout__





# ############################################################
# # CUDA 
# # Redirect print output to a file
# sys.stdout = open("init_for_cuda.txt", "a")

# ## 1. converting the weights
# W_pos, W_neg = convW_BT(weights)

# ## 2. compressing the weights
# cuda_weights_size = 0
# for n in range(len(weights)):
#     cuda_weights_size += 2*layers[n]*layers[n+1]
#     W_pos[n] = comp_vert(W_pos[n],layers[n],layers[n+1],WORD_SIZE)
#     W_neg[n] = comp_vert(W_neg[n],layers[n],layers[n+1],WORD_SIZE)

# ## 3. getting the size of the total list (necessary for cuda code)
# print("size of total cuda weights array: ")
# print(cuda_weights_size//WORD_SIZE)

# ## 4. concatenating all into one list
# cuda_weights = [0]
# for n in range(len(weights)):
#     cuda_weights += W_pos[n] + W_neg[n]
# cuda_weights.pop(0)


# ## 5. converting & compressing the input
# A_bin = convA_BT(input)
# A_bin = comp_hori(A_bin,batch_size,layers[0],WORD_SIZE)




# ## 6. this output can be fed into CUDA
# print("cuda weights:")
# print(cuda_weights)
# print("cuda input:")
# print(A_bin)







# ## this is potential code for mapping to TernTern format (from W => W0*W1)
# # for n in range(len(new_list)):
# #     for i in range(len(W)):
# #         new_list[n]->W1[i] = not(not( (old_list->W[i]) >> 1 ))  #maps {-1,0,1} -> {1,0,0}
# #         new_list[n]->W0[i] = not(old_list->W[i])                #only evaluates true if input is zero


# # Close the file when done and restore print to the console (if needed after the redirection)
# sys.stdout.close()
# sys.stdout = sys.__stdout__

# ########################









# # get error with these inputs:
# # input_dim = 8                  
# # output_dim = 2
# # depth = 6
# # hidden_layers = [32, 32, 32, 32, 32, 32]       #layers must be divisable by word_size
# # batch_size = 4
