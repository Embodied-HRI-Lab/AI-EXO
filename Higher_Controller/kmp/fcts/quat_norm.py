import numpy as np
    
def quat_norm(q = None): 
    # calculate the norm of a quaternion
    a = q.s
    b = q.v
    qnorm = norm(np.array([a,np.transpose(b)]))
    return qnorm  