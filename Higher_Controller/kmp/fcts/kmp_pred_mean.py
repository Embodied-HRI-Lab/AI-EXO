import numpy as np
    
def kmp_pred_mean(t = None,sampleData = None,N = None,kh = None,Kinv = None,dim = None): 
    # mean: k*inv(K+lamda*Sigma)*Y
    
    D = 2 * dim
    for i in np.arange(1,N+1).reshape(-1):
        k[np.arange[1,D+1],np.arange[[i - 1] * D + 1,i * D+1]] = kernel_extend(t,sampleData(i).t,kh,dim)
        for h in np.arange(1,D+1).reshape(-1):
            Y[[i - 1] * D + h,1] = sampleData(i).mu(h)
    
    Mu = k * Kinv * Y
    
    return Mu
    
    return Mu