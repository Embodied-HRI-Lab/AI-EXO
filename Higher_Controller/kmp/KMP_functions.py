import numpy as np 


def kmp_insertPoint(data=None,num=None,via_time=None,via_point=None,via_var=None):   
    newData = data  
    newNum = num   
    dataExit = 0   
    # print("newNum", newNum)   
    # print("via time t:", via_time)    
    # print("via_point:", via_point)      
    for i in range(newNum):   
        if np.abs(newData['t'][i][0] - via_time) < 0.0005:
            dataExit = 1 
            replaceNum = i   
            # print("replaceNum :", replaceNum)  
            break   
    
    if dataExit:  
        newData['t'][replaceNum] = via_time  
        newData['mu'][replaceNum] = via_point  
        newData['sigma'][replaceNum] = via_var  
    else: 
        newNum = newNum + 1 
        # newData['t'] = np.append(newData['t'], via_time, axis=0)
        # newData['mu'] = np.append(newData['mu'], via_point, axis=0) 
        # newData['sigma'] = np.append(newData['sigma'], via_var, axis=0)
        newData['t'] = np.insert(newData['t'], newNum-1, values=via_time, axis=0)
        newData['mu'] = np.insert(newData['mu'], newNum-1, values=via_point, axis=0) 
        newData['sigma'] = np.insert(newData['sigma'], newNum-1, values=via_var, axis=0) 
    
    return newData, newNum  


def kernel_matrix(ta=None, tb=None, kh=None, dim=None, output_dim=None):    
    ## scalar kernel function   
    kernelMatrix = np.eye((output_dim))    

    kt_t = np.exp(- kh * (ta - tb) * (ta - tb))   
    
    return kt_t * kernelMatrix    

    
def kernel_extend(ta=None, tb=None, h=None, dim=None): 
    # this file is used to generate a kernel value
    # this kernel considers the 'dim-' pos and 'dim-' vel
    
    ## callculate different kinds of kernel
    dt = 0.001   
    tadt = ta + dt  
    tbdt = tb + dt   
    
    kt_t = np.exp(- h * (ta - tb) * (ta - tb))   
    
    kt_dt_temp = np.exp(- h * (ta - tbdt) * (ta - tbdt))  
    
    kt_dt = (kt_dt_temp - kt_t)/dt
    
    kdt_t_temp = np.exp(- h * (tadt - tb) * (tadt - tb))
    
    kdt_t = (kdt_t_temp - kt_t)/dt  
    
    kdt_dt_temp = np.exp(- h * (tadt - tbdt) * (tadt - tbdt))
    
    kdt_dt = (kdt_dt_temp - kt_dt_temp - kdt_t_temp + kt_t)/dt/dt
    
    kernelMatrix = np.zeros((2 * dim, 2 * dim))    
    
    for i in range(dim):  
        kernelMatrix[i,i] = kt_t  
        kernelMatrix[i,i + dim] = kt_dt   
        kernelMatrix[i + dim,i] = kdt_t  
        kernelMatrix[i + dim,i + dim] = kdt_dt  
    
    return kernelMatrix  


def kmp_estimateMatrix_mean(sampleData=None,N=None,kh=None,lamda=None,dim=None,output_dim=None):   
    # Based on paper IJRR 
    # calculate: inv(K+lamda*Sigma) 
    # this function is written for 'dim-' pos and 'dim-' vel  
    
    D = 2 * dim    
        
    kc = np.zeros((D * N, D * N))     
    for i in range(N):     
        for j in range(N):     
            # kc[i * D : (i+1) * D, j * D : (j +1) * D] = kernel_extend(sampleData['t'][i], sampleData['t'][j], kh, dim)
            kc[i * D : (i+1) * D, j * D : (j +1) * D] = kernel_matrix(sampleData['t'][i], sampleData['t'][j], kh, dim, output_dim) 
            if(i == j):     
                C_temp = sampleData['sigma'][i]    
                # kc[i * D : (i+1) * D, j * D : (j+1) * D] = kc[i * D : (i+1) * D, j * D : (j+1) * D] + lamda * C_temp
                kc[i * D : (i+1) * D, j * D : (j+1) * D] += lamda * C_temp
    
    # print("kc: ", kc.shape)     
    Kinv = np.linalg.inv(kc)        
    return Kinv     


def kmp_estimateMatrix_mean_var(sampleData=None, N=None, kh=None, lamda_1=None, lamda_2=None, dim=None, output_dim=None):  
    # calculate: inv(K+lamda*Sigma)      
    # this function is written for 'dim-' pos and 'dim-' vel      
    D = output_dim * dim         
    kc_1 = np.zeros((D * N, D * N))          
    kc_2 = np.zeros((D * N, D * N))            
    for i in range(N):      
        for j in range(N):      
            # kc_1[i * D : (i+1) * D, j * D : (j +1) * D] = kernel_extend(sampleData['t'][i], sampleData['t'][j], kh, dim) 
            kc_1[i * D : (i+1) * D, j * D : (j +1) * D] = kernel_matrix(sampleData['t'][i], sampleData['t'][j], kh, dim, output_dim)      
            kc_2[i * D : (i+1) * D, j * D : (j +1) * D] = kc_1[i * D : (i+1) * D, j * D : (j +1) * D]   
            # print("kernel matrix :", kernel_matrix(sampleData['t'][i], sampleData['t'][j], kh, dim, output_dim))  
            
            if(i == j):   
                C_temp = sampleData['sigma'][i]         
                kc_1[i * D : (i+1)* D, j * D : (j+1) * D] += lamda_1 * C_temp  
                kc_2[i * D : (i+1)* D, j * D : (j+1) * D] += lamda_2 * C_temp   
                
    Kinv_1 = np.linalg.inv(kc_1)     
    Kinv_2 = np.linalg.inv(kc_2)     
    # Kinv_1 = np.linalg.inv(kc_1)        
    # Kinv_2 = Kinv_1             
    return Kinv_1, Kinv_2         


def kmp_pred_mean(t=None,sampleData=None,N=None,kh=None,Kinv=None,dim=None,output_dim=None):   
    # mean: k*inv(K+lamda*Sigma)*Y    
    D = 2 * dim      
    k = np.zeros((D, D * N))        
    Y = np.zeros((D * N, 1))        
    for i in range(N): 
        k[:D, i * D : (i+1) * D] = kernel_extend(t, sampleData['t'][i], kh, dim) 
        for h in range(D):    
            # print(np.array(sampleData['mu']).shape)   
            Y[i * D + h, 0] = np.array(sampleData['mu'])[i, h]   
    
    Mu = k.dot(Kinv).dot(Y)       
    return Mu     


def kmp_pred_mean_var(t=None, sampleData=None, N=None, kh=None, Kinv1=None, Kinv2=None, lambda_2=None, dim=None, output_dim=None):    
    #  mean: k*inv(K+lambda_1*Sigma)*Y and ...
    #  covariance: num/lambda_2*[k(s,s)-k*inv(K+lambda_2*Sigma)*k']
    # k0 = kernel_extend(ta=t, tb=t, h=kh, dim=dim)    
    
    D = output_dim * dim     

    k = np.zeros((D, D * N))       
    Y = np.zeros((D * N, 1))     
    k0 = kernel_matrix(t, t, kh, dim, output_dim)   
    for i in range(N):   
        # k[:D, i * D : (i+1) * D] = kernel_extend(t, sampleData['t'][i], kh, dim)   
        k[:D, i * D : (i+1) * D] = kernel_matrix(t, sampleData['t'][i], kh, dim, output_dim)       
        for h in range(D):     
            Y[i * D + h, 0] = np.array(sampleData['mu'])[i, h]        
    
    Mu = k.dot(Kinv1).dot(Y)    
    Sigma=N/lambda_2 * (k0 - k.dot(Kinv2).dot(k.T))     
    return Mu, Sigma   


def generate_orientation_data(q1=None, q2=None, tau=None, dt=None): 
    #-------------------------------------------------------------------------
    # Generate quaternion data, angular velocities and accelerations for DMP
    # learning
    # Copyright (C) Fares J. Abu-Dakka  2013  
    
    N = int(np.round(tau / dt))      
    t = np.linspace(0, tau, N)      
    
    q = np.ones((N, 4))     
    dq = np.zeros((N, 4))    
    qq = np.zeros((N, 4))    
    dqq = np.zeros((N, 4))    
    omega = np.zeros((N, 3))    
    domega = np.zeros((N, 3))      
    
    # # Generate spline data from q1 to q2   
    
    # for j in range(3): 
    #     a = minimum_jerk_spline(q1[j+1], 0, 0, q2[j+1], 0, 0, tau)  
    #     for i in range(N):  
    #         q[i].v[j, 1] = minimum_jerk(t[i], a)   
    
    for i in range(N):   
        for j in range(4):   
            a = minimum_jerk_spline(q1[j], 0, 0, q2[j], 0, 0, tau)    
            q[i, j] = minimum_jerk(t[i], a)[0]    
            # print("a :", a)  
        # print("q[i, j]", q[i, j])  
        q[i, :] = quat_conjugate(q[i, :])   
        qq[i, :] = quat_normalize(q[i, :])    
    
    # # Normalize quaternions  
    # for i in range(N):   
    #     # tmp = quat_norm(q(i))    
    #     # q(i).s = q(i).s / tmp    
    #     # q(i).v = q(i).v / tmp   
    #     qq[i, :] = quat_normalize(q[i, :])     
    #     # qq[i, :] = np.array([[q(i).s], [q(i).v]])    
    
    # Calculate derivatives  
    # for i in range(N): 
    for j in range(4):    
        # print("qq :", qq[i, j]) 　
        # print("time :", t[i])  
        dqq[:, j] = np.gradient(qq[:, j], t.T)       
    
    # Calculate omega and domega  
    for i in range(N):  
        for j in range(4):    
            dq[i, j] = dqq[i, j]      
        omega_q = quat_mult(dq[i, :], quat_conjugate(q[i, :]))   
        omega[i, :] = 2 * omega_q[1:]  
    
    for j in range(3):   
        domega[:, j] = np.gradient(omega[:, j], t.T)   
    
    omega[0, :] = np.array([0, 0, 0])    
    omega[N-1, :] = np.array([0, 0, 0])   
    domega[0, :] = np.array([0, 0, 0])   
    domega[N-1, :] = np.array([0, 0, 0])   
    
    return q, omega, domega, t   


def minimum_jerk_spline(x0=None,dx0=None,ddx0=None,x1=None,dx1=None,ddx1=None,T=None): 
    a = np.zeros(6)  
    a[0] = x0   
    a[1] = dx0   
    a[2] = ddx0/2  
    a[3] = (20 * x1 - 20 * x0 - (8 * dx1 + 12 * dx0) * T - (3 * ddx0 - ddx1) * T * T) / (2 * T * T * T)
    a[4] = (30 * x0 - 30 * x1 + (14 * dx1 + 16 * dx0) * T + (3 * ddx0 - 2 * ddx1) * T * T) / (2 * T * T * T * T)
    a[5] = (12 * x1 - 12 * x0 - (6 * dx1 + 6 * dx0) * T - (ddx0 - ddx1) * T * T) / (2 * T * T * T * T * T)  
    return a  


def minimum_jerk(t=None, a=None):   
    t2 = t * t  
    t3 = t2 * t  
    t4 = t2 * t2   
    t5 = t3 * t2   
    pos = a[5] * t5 + a[4] * t4 + a[3] * t3 + a[2] * t2 + a[1] * t + a[0]    
    vel = 5 * a[5] * t4 + 4 * a[4] * t3 + 3 * a[3] * t2 + 2 * a[2] * t + a[1]    
    acc = 20 * a[5] * t3 + 12 * a[4] * t2 + 6 * a[3] * t + 2 * a[2]    
    return pos, vel, acc    


def quat_conjugate(q=None):  
    qc = q 
    for i in range(1, 4):   
        qc[i] = -1 * q[i]    
    # qc.s = q.s   
    # qc.v = - q.v   
    return qc   


def quat_exp(w = None):   
    # transform a 3-D angular velociy to 4-D quaternion
    tmp = np.linalg.norm(w)    
    if tmp > 1e-12:    
        q_w = np.array([[np.cos(tmp)],[np.sin(tmp) * w / tmp]])     
    else:  
        q_w = np.array([[1],[0],[0],[0]])   
    
    dq = q_w/np.linalg.norm(q_w)   
    
    q.s = dq[0]    
    q.v = dq(np.arange(1,4))   
    return q  


def trans_angVel(qDes=None, wDes=None, dt=None, q0=None): 
    # transform desired angular velocity into the Euclidean space
    q_new = quat_mult(quat_exp(wDes / 2 * dt), qDes)  
    
    zeta1 = quat_log(qDes, q0)  
    zeta2 = quat_log(q_new,q0)  
    localVel = (zeta2 - zeta1) / dt
    return localVel   


def quat_to_vel(q=None, dt=None, tau=None): 
    # calculate angular vel/acc using quaternions
    
    N = tau/dt  
    dq = np.zeros((N, 4))  
    qq = np.zeros((N, 4))   
    dqq = np.zeros((N, 4))   
    omega = np.zeros((N, 3))   
    domega = np.zeros((N, 3))    
    for i in range(N):    
        qq[i, :] = q    
    
    # Calculate derivatives  
    for j in range(4):    
        dqq[:, j] = np.gradient(qq[:, j])/dt  
    
    # Calculate omega and domega
    for i in range(N):    
        for j in range(4): 
            dq[i, j] = dqq[i, j]        
            # for j in range(3):    
            #     dq.v[j,1] = dqq(j + 1, i)     
        omega_q = quat_mult(dq[i, j], quat_conjugate(q[i, :]))   
        omega[i, :] = 2 * omega_q[1:]   
    
    for j in range(3):  
        domega[:, j] = np.gradient(omega[:, j])/dt     
    
    return omega, domega  


def quat_log(q1=None,q2=None): 
    #-------------------------------------------------------------------------
    # Calculates logarithm of orientation difference between quaternions
    # Copyright (C) Fares J. Abu-Dakka  2013
    
    q2c = quat_conjugate(q2)  
    q = quat_mult(q1, q2c)  
    # tmp = quat_norm(q)  
    # q.s = q.s/tmp  
    # q.v = q.v/tmp  
    #   if q.s < 0
    #     q.s = -q.s;
    #     q.v = -q.v;
    #   end
    q = quat_normalize(q)   
    if np.linalg.norm(q[1:]) > 1e-12:  
        log_q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])    
    else:  
        log_q = np.array([0, 0, 0])   
    
    return log_q


def quat_mult(q1=None, q2=None):  
    #-------------------------------------------------------------------------
    # Quaternion multiplication  
    # Copyright (C) Fares J. Abu-Dakka  2013  
    q = np.zeros(4)
    q[0] = q1[0] * q2[0] - np.transpose(q1[1:]).dot(q2[1:])    
    q[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(np.transpose(q1[1:]), q2[1:])
    
    # np.array([[q1.v[1] * q2.v[2] - q1.v[2] * q2.v[1]], [q1.v[2] * q2.v[0] - q1.v[0] * q2.v[2]], [q1.v[0] * q2.v[1] - q1.v[1] * q2.v[0]]])
    return q  


def quat_norm(q=None):  
    # calculate the norm of a quaternion
    # a = q.s
    # b = q.v
    # qnorm = np.linalg.norm(np.array([a, np.transpose(b)]))  
    qnorm = np.linalg.norm(q)  
    return qnorm   


def quat_normalize(q=None):   
    return q/np.linalg.norm(q)    