import numpy as np
    
def quat_log(q1 = None,q2 = None): 
    #-------------------------------------------------------------------------
# Calculates logarithm of orientation difference between quaternions
# Copyright (C) Fares J. Abu-Dakka  2013
    
    q2c = quat_conjugate(q2)
    q = quat_mult(q1,q2c)
    tmp = quat_norm(q)
    q.s = q.s / tmp
    q.v = q.v / tmp
    #   if q.s < 0
    #     q.s = -q.s;
    #     q.v = -q.v;
    #   end
    
    if norm(q.v) > 1e-12:
        log_q = np.arccos(q.s) * q.v / norm(q.v)
    else:
        log_q = np.array([[0],[0],[0]])
    
    return log_q
    
    return log_q