# Just a script where you can input the random integers that the c++ file generates, and verify using basic integer MatMul that the c++ network's operations (bunch of popc, summations and xor's) outputs the correct answer

import time #for timing

word_size = 3

N_input = [3, 6, 4, 7, 4, 0, 2, 0, 3, 1, 7, 4]


N1 = 4
M1 = 3 #9/word_size
P1 = 12

N_weights1_pos = [5, 3, 3, 5, 1, 7, 1, 3, 1, 2, 7, 4, 5, 2, 6, 4, 3, 6, 6, 0, 6, 6, 7, 3, 1, 4, 5, 5, 3, 1, 2, 3, 3, 6, 7, 7]
N_weights1_neg = [6, 7, 4, 2, 6, 6, 1, 7, 0, 0, 4, 2, 4, 5, 2, 0, 4, 3, 4, 0, 6, 6, 1, 3, 7, 1, 5, 2, 0, 4, 5, 1, 1, 0, 2, 6]


N2 = 4
M2 = 4 #12/word_size
P2 = 18

N_weights2_pos = [4, 4, 7, 1, 7, 1, 4, 4, 3, 2, 0, 5, 4, 2, 4, 4, 5, 2, 5, 2, 5, 6, 0, 4, 6, 3, 5, 2, 1, 5, 2, 7, 7, 4, 6, 4, 7, 5, 5, 5, 0, 5, 5, 1, 7, 3, 6, 3, 0, 6, 5, 1, 3, 2, 0, 2, 3, 2, 7, 1, 4, 2, 5, 4, 7, 3, 7, 0, 6, 0, 1, 2]
N_weights2_neg = [0, 0, 6, 7, 6, 0, 0, 1, 7, 0, 7, 5, 3, 2, 1, 7, 3, 0, 2, 6, 1, 7, 3, 7, 2, 2, 5, 1, 0, 0, 5, 3, 0, 1, 7, 1, 1, 1, 2, 0, 0, 7, 2, 1, 5, 3, 2, 1, 0, 1, 5, 6, 1, 6, 5, 6, 3, 5, 6, 7, 3, 0, 6, 1, 5, 0, 6, 1, 7, 2, 4, 0]


N3 = 4
M3 = 6 #18/word_size
P3 = 6

N_weights3_pos = [2, 0, 6, 5, 6, 0, 4, 2, 3, 0, 7, 2, 7, 7, 5, 5, 0, 3, 7, 2, 0, 5, 4, 5, 3, 5, 0, 3, 1, 5, 0, 3, 4, 2, 2, 7]
N_weights3_neg = [6, 7, 7, 3, 5, 2, 0, 7, 7, 3, 3, 1, 7, 3, 1, 7, 5, 6, 5, 0, 2, 0, 0, 3, 0, 2, 6, 0, 6, 4, 4, 1, 3, 4, 4, 4]


N_output = [0, 5, 1, 4, 7, 4, 4, 7]






# Decompression
def expand_horizontal(input):

    output = [0] * (len(input)*word_size)

    for i in range(len(input)):

        rem = input[i]
        
        for n in range(word_size):
            
            output[i*word_size + ((word_size-1)-n)] = rem % 2
            rem = rem // 2

    return output

def expand_vertical(input,P):

    output = [0] * (len(input)*word_size)

    for i in range(len(input)):

        rem = input[i]
        
        for n in range(word_size):
            
            output[P*word_size*(i//P)+(i%P) + P*((word_size-1)-n)] = rem % 2
            rem = rem // 2

    return output


def convert_0_1(input):
    for i in range(len(input)):
        if input[i] == 1 :
            input[i] = -1
        else:
            input[i] = 1

    return input
    

# Matmul
def MatMul(A, B, N, M, P):
    C = [0] * (N * P)
    
    for i in range(N):
        for j in range(P):
            
            C[i * P + j] = sum(A[i * M + k] * B[k * P + j] for k in range(M))
    
    return C

# activation
def sigma(A):
    for i in range(len(A)):

        if (A[i] >= 0) :
            A[i] = 1

        else:
            A[i] = -1
          
    return A




#expansion
T_weights1_pos = expand_vertical(N_weights1_pos,P1)
T_weights1_neg = expand_vertical(N_weights1_neg,P1)

T_weights2_pos = expand_vertical(N_weights2_pos,P2)
T_weights2_neg = expand_vertical(N_weights2_neg,P2)

T_weights3_pos = expand_vertical(N_weights3_pos,P3)
T_weights3_neg = expand_vertical(N_weights3_neg,P3)

T_input = expand_horizontal(N_input)
T_input = convert_0_1(T_input)

# Element-wise subtraction for weights
T_weights1 = [a - b for a, b in zip(T_weights1_pos, T_weights1_neg)]
T_weights2 = [a - b for a, b in zip(T_weights2_pos, T_weights2_neg)]
T_weights3 = [a - b for a, b in zip(T_weights3_pos, T_weights3_neg)]




# Record the start time
start_time = time.time()



#layer 1
T_output = MatMul(T_input, T_weights1,N1,M1*word_size,P1)
T_output = sigma(T_output)
# print(f"{T_output}")    #So this output is FULLY correct!!???

# #layer 1 compare with cuda
# A = expand_horizontal(N_layer1)
# A = convert_0_1(A)
# print(A)

#layer 2
T_output = MatMul(T_output,T_weights2,N2,M2*word_size,P2)
T_output = sigma(T_output)

#layer 3
T_output = MatMul(T_output,T_weights3,N3,M3*word_size,P3)
T_output = sigma(T_output)

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time in seconds
elapsed_time = end_time - start_time
print(f"Forward pass Python execution time: {elapsed_time:.6f} seconds")


#compare with cuda
print(f"________________")
print(f"{T_output}")
print(f"{expand_horizontal(N_output)}")



