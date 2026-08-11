import numpy as np


def generate_orientation_data(q1 = None,q2 = None,tau = None,dt = None): 
    #-------------------------------------------------------------------------
    # Generate quaternion data, angular velocities and accelerations for DMP
    # learning
    # Copyright (C) Fares J. Abu-Dakka  2013
    
    N = np.round(tau / dt)
    t = np.linspace(0,tau,N)
    s = struct('s',cell(1),'v',cell(1))
    q = np.matlib.repmat(s,100,1)
    qq = np.zeros((4,N))
    dqq = np.zeros((4,N))
    omega = np.zeros((3,N))
    domega = np.zeros((3,N))
    # Generate spline data from q1 to q2
    a = minimum_jerk_spline(q1.s,0,0,q2.s,0,0,tau)
    for i in np.arange(1,N+1).reshape(-1):
        q(i).s = minimum_jerk(t(i),a)
    
    for j in np.arange(1,3+1).reshape(-1):
        a = minimum_jerk_spline(q1.v(j),0,0,q2.v(j),0,0,tau)
        for i in np.arange(1,N+1).reshape(-1):
            q[i].v[j,1] = minimum_jerk(t(i),a)
    
    # Normalize quaternions
    for i in np.arange(1,N+1).reshape(-1):
        tmp = quat_norm(q(i))
        q(i).s = q(i).s / tmp
        q(i).v = q(i).v / tmp
        qq[:,i] = np.array([[q(i).s],[q(i).v]])
    
    # Calculate derivatives
    for j in np.arange(1,4+1).reshape(-1):
        dqq[j,:] = gradient(qq(j,:),t)
    
    # Calculate omega and domega
    for i in np.arange(1,N+1).reshape(-1):
        dq.s = dqq(1,i)
        for j in np.arange(1,3+1).reshape(-1):
            dq.v[j,1] = dqq(j + 1,i)
        omega_q = quat_mult(dq,quat_conjugate(q(i)))
        omega[:,i] = 2 * omega_q.v
    
    for j in np.arange(1,3+1).reshape(-1):
        domega[j,:] = gradient(omega(j,:),t)
    
    omega[:,1] = np.array([[0],[0],[0]])
    omega[:,N] = np.array([[0],[0],[0]])
    domega[:,1] = np.array([[0],[0],[0]])
    domega[:,N] = np.array([[0],[0],[0]])
    return q,omega,domega,t
    
    return q,omega,domega,t