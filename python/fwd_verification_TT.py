# generate {-1,0,1} random matrices, with appropriate sizes for each layer
# generate {-1,0,1} random matrix, with appropriate size for input batch



#########################
# PYTHON
# for each layer
    # do the matmul
    # do the sigma
#
# spit out answer
#########################



#########################
# CUDA 
for n in range(len(new_list)):
    for i in range(len(W)):
        new_list[n]->W1[i] = not(not( (old_list->W[i]) >> 1 ))  #maps {-1,0,1} -> {1,0,0}
        new_list[n]->W0[i] = not(old_list->W[i])                #only evaluates true if input is zero


# compress W0 and W1

# this can be fed into CUDA
########################
