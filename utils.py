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