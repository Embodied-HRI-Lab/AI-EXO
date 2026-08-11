import numpy as np
    
def quat_exp(w = None): 
    # transform a 3-D angular velociy to 4-D quaternion
    tmp = norm(w)
    if tmp > 1e-12:
        q_w = np.array([[np.cos(tmp)],[np.sin(tmp) * w / tmp]])
    else:
        q_w = np.array([[1],[0],[0],[0]])
    
    dq = q_w / norm(q_w)
    q.s = dq(1)
    q.v = dq(np.arange(2,4+1))
    return q
