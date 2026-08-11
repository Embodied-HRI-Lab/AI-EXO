def quat_conjugate(q=None): 
    #-------------------------------------------------------------------------
    # Quaternion conjugation
    # Copyright (C) Fares J. Abu-Dakka  2013
    
    qc.s = q.s
    qc.v = - q.v
    return qc
