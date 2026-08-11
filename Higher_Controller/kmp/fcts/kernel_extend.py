import numpy as np
    
    
def kernel_extend(ta = None,tb = None,h = None,dim = None): 
    # this file is used to generate a kernel value
    # this kernel considers the 'dim-' pos and 'dim-' vel
    
    ## callculate different kinds of kernel
    dt = 0.001 
    tadt = ta + dt 
    tbdt = tb + dt 
    kt_t = np.exp(- h * (ta - tb) * (ta - tb))  
    
    kt_dt_temp = np.exp(- h * (ta - tbdt) * (ta - tbdt))
    
    kt_dt = (kt_dt_temp - kt_t) / dt
    
    kdt_t_temp = np.exp(- h * (tadt - tb) * (tadt - tb))
    
    kdt_t = (kdt_t_temp - kt_t) / dt
    
    kdt_dt_temp = np.exp(- h * (tadt - tbdt) * (tadt - tbdt))
    
    kdt_dt = (kdt_dt_temp - kt_dt_temp - kdt_t_temp + kt_t) / dt / dt
    
    kernelMatrix = np.zeros((2 * dim,2 * dim))

    for i in range(dim): 
        kernelMatrix[i,i] = kt_t
        kernelMatrix[i,i + dim] = kt_dt
        kernelMatrix[i + dim,i] = kdt_t
        kernelMatrix[i + dim,i + dim] = kdt_dt
    
    return kernelMatrix