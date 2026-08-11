import numpy as np
    
def quat_mult(q1 = None,q2 = None): 
    #-------------------------------------------------------------------------
# Quaternion multiplication
# Copyright (C) Fares J. Abu-Dakka  2013
    
    q.s = q1.s * q2.s - np.transpose(q1.v) * q2.v
    q.v = q1.s * q2.v + q2.s * q1.v + np.array([[q1.v(2) * q2.v(3) - q1.v(3) * q2.v(2)],[q1.v(3) * q2.v(1) - q1.v(1) * q2.v(3)],[q1.v(1) * q2.v(2) - q1.v(2) * q2.v(1)]])
    return q 