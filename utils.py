import numpy as np
from typing import List

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def colors2reward_ldlf(colors:list) -> str: 
    """ 
        Convert color list to  Linear Dynamic Logic on finite traces
    """
    reward_ldlf = "<"
    for i,c in enumerate(colors):
        reward_ldlf+="!"+c+"*; "+c
        if i<len(colors)-1:
            reward_ldlf+="; "
        elif i==len(colors)-1:
            reward_ldlf+=">end"
    return reward_ldlf

# One hot encoding for the state of the automaton
def one_hot_encode(x,size, num_labels):

    ret = np.zeros(size,dtype = np.float32)
    if size%num_labels == 0:
        block_size = int(size/num_labels)
        ret[int(x)*block_size:(int(x)+1)*block_size]=1.0
    return ret

def merge_lists(l1:List, l2:List):
    # Merge the 2 lists: [(l1[0], l2[0]),(l1[1], l2[1]),...]
    # merged_list = [tuple for tuple in zip(l1,l2)]
    # Unpack the tuple in a flat list
    return [item for t in zip(l1,l2) for sublist in t for item in sublist]