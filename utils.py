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

def one_hot_encode(x,size, num_labels):
    """
        One hot encoding for the state of the automaton
    """
    ret = np.zeros(size,dtype = np.float32)
    if size%num_labels == 0:
        block_size = int(size/num_labels)
        ret[int(x)*block_size:(int(x)+1)*block_size]=1.0
    return ret

def merge_lists(l1:List, l2:List) -> List:
    """ 
        Merge 2 lists as: [(l1[0], l2[0]),(l1[1], l2[1]),...]
    """    
    # merged_list = [tuple for tuple in zip(l1,l2)]
    # Unpack the tuple in a flat list
    if type(l1[0]) == list:
        return [item for t in zip(l1,l2) for sublist in t for item in sublist]
    else:
        return [item for t in zip(l1,l2) for item in t]

def color_sequence(num_colors:int) -> List[str]:
    if num_colors == 1:
        return ['blue']
    elif num_colors == 2:
        return ['blue','green']
    elif num_colors == 3:
        return ['blue','red','green']
    elif num_colors == 4:
        return ['blue','red','yellow','green']
    else:
        raise AttributeError('Map with ', num_colors,' colors not supported by default. Specify a path for a map file.')

    