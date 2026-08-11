import numpy as np
    
def kmp_estimateMatrix_mean(sampleData = None,N = None,kh = None,lamda = None,dim = None): 
    # calculate: inv(K+lamda*Sigma)
# this function is written for 'dim-' pos and 'dim-' vel
    
    D = 2 * dim
    for i in np.arange(1,N+1).reshape(-1):
        for j in np.arange(1,N+1).reshape(-1):
            kc[np.arange[[i - 1] * D + 1,i * D+1],np.arange[[j - 1] * D + 1,j * D+1]] = kernel_extend(sampleData(i).t,sampleData(j).t,kh,dim)
            if i == j:
                C_temp = sampleData(i).sigma
                kc[np.arange[[i - 1] * D + 1,i * D+1],np.arange[[j - 1] * D + 1,j * D+1]] = kc(np.arange((i - 1) * D + 1,i * D+1),np.arange((j - 1) * D + 1,j * D+1)) + lamda * C_temp
    
    Kinv = inv(kc)
    
    return Kinv
    
    return Kinv