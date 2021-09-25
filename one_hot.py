import numpy as np



#One hot encoding for the state of the automaton
def one_hot_encode(x,size, num_labels):

    ret = np.zeros(size,dtype = np.float)
    if size%num_labels !=0:
        return ret
    else:
        block_size = int(size/num_labels)

        i = 0
        k = 0
        while(i<size):
            if k == x:
                ret[i:i+block_size] = 1.0
            k+=1
            i = i+block_size
        return ret


if __name__ == '__main__':

    r1 = one_hot_encode(0,32,4)
    print(r1)

    r1 = one_hot_encode(1,32,4)
    print(r1)
    r2 = one_hot_encode(2,32,4)
    print(r2)
    r3 = one_hot_encode(3,32,4)
    print(r3)
