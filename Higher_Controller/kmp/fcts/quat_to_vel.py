import numpy as np
    
def quat_to_vel(q = None,dt = None,tau = None): 
    # calculate angular vel/acc using quaternions
    
    N = tau / dt
    for i in np.arange(1,N+1).reshape(-1):
        qq[:,i] = np.array([[q(i).s],[q(i).v]])
    
    # Calculate derivatives
    for j in np.arange(1,4+1).reshape(-1):
        dqq[j,:] = gradient(qq(j,:)) / dt
    
    # Calculate omega and domega
    for i in np.arange(1,N+1).reshape(-1):
        dq.s = dqq(1,i)
        for j in np.arange(1,3+1).reshape(-1):
            dq.v[j,1] = dqq(j + 1,i)
        omega_q = quat_mult(dq,quat_conjugate(q(i)))
        omega[:,i] = 2 * omega_q.v
    
    for j in np.arange(1,3+1).reshape(-1):
        domega[j,:] = gradient(omega(j,:)) / dt
    
    return omega, domega  