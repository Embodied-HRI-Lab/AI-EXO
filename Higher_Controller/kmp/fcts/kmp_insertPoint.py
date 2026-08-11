import numpy as np
    
def kmp_insertPoint(data = None,num = None,via_time = None,via_point = None,via_var = None): 
    # insert data format:[time px py ... vx vy ...]';
    
    newData = data
    newNum = num
    dataExit = 0
    for i in np.arange(1,newNum+1).reshape(-1):
        if np.abs(newData(i).t - via_time) < 0.0005:
            dataExit = 1
            replaceNum = i
            break
    
    if dataExit:
        newData(replaceNum).t = via_time
        newData(replaceNum).mu = via_point
        newData(replaceNum).sigma = via_var
    else:
        newNum = newNum + 1
        newData(newNum).t = via_time
        newData(newNum).mu = via_point
        newData(newNum).sigma = via_var
    
    return newData, newNum  