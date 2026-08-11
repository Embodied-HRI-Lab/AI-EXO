import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)  
  
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat  
from GMRbasedGP.utils.gmr import Gmr, plot_gmm
from KMP_functions import *  
from robInfLibpy.plot_pred import *  
import seaborn as sns 
import copy as cp  

# Load data
file_name = 'data/' 

letter = 'B' 
datapath = file_name + '2Dletters/'   
data = loadmat(datapath + '%s.mat' % letter)   

demos_pos = [d['pos'][0][0].T for d in data['demos'][0]]   
demos_vel = [d['vel'][0][0].T for d in data['demos'][0]]    
# print("demos_pos :", demos_pos.shape)     
# print("demos_vel :", demos_vel.shape)     
# demos = np.vstack((demos_pos, demos_vel))     
demos = demos_pos  

# Parameters
nb_data = demos[0].shape[0]   
print("nb_data :", demos[0].shape)     
nb_data_sup = 0    
nb_samples = 5   
dt = 0.01   
demodura = dt * nb_data    

# model parameter 
input_dim = 1   
output_dim = 4    
in_idx = [0]   
out_idx = [1, 2, 3, 4]     
nb_states = 6    

dim = 2   

# Create time data
demos_t = [np.arange(demos[i].shape[0])[:, None] for i in range(nb_samples)]
# print("demos_t :", demos_t) 

# Stack time and position data
demos_tx = [np.hstack([demos_t[i]*dt, demos_pos[i], demos_vel[i]]) for i in range(nb_samples)]
print("demos_tx :", np.array(demos_tx).shape) 

# Stack demos
demos_np = demos_tx[0] 
print("demos_np :", demos_np.shape) 
for i in range(1, nb_samples): 
    demos_np = np.vstack([demos_np, demos_tx[i]])
print("demos_np :", demos_np.shape)  

X = demos_np[:, 0][:, None]  
Y = demos_np[:, 1:]  
print('X shape: ', X.shape, 'Y shape: ', Y.shape)   

# Test data  
Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]  

# GMM + GMR 
gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim+output_dim, in_idx=in_idx, out_idx=out_idx)  
gmr_model.init_params_kbins(demos_np.T, nb_samples=nb_samples)  
gmr_model.gmm_em(demos_np.T)   

# GMR
mu_gmr = [] 
sigma_gmr = []  
sigma_gmr_1 = []   
sigma_gmr_2 = []   
for i in range(Xt.shape[0]):
    mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])  
    mu_gmr.append(mu_gmr_tmp)  
    sigma_gmr.append(sigma_gmr_tmp)  
    sigma_gmr_1.append(sigma_gmr_tmp[0, 0])  
    sigma_gmr_2.append(sigma_gmr_tmp[1, 1])  

sigma_gmr_1_diag = np.diag(np.array(sigma_gmr_1))  
sigma_gmr_2_diag = np.diag(np.array(sigma_gmr_2))  
mu_gmr = np.array(mu_gmr)  
sigma_gmr = np.array(sigma_gmr)  

print("Xt :", Xt.shape)  
print("mu_gmr :", mu_gmr.shape)   
print("sigma_gmr :", sigma_gmr.shape)   

# plot_GMM_raw_data(font_name=letter, nb_samples=nb_samples, nb_data=nb_data, Y=Y, gmr_model=gmr_model)   

# plot_mean_var(font_name=letter, nb_samples=nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr)   

# /////////////////////////////////////////////////////////////////////////
refTraj = {}   
refTraj['t'] = Xt   
refTraj['mu'] = mu_gmr   
refTraj['sigma'] = sigma_gmr   

ori_refTraj = {}
ori_refTraj['t'] = cp.deepcopy(Xt)   
ori_refTraj['mu'] = cp.deepcopy(mu_gmr)    
ori_refTraj['sigma'] = cp.deepcopy(sigma_gmr)    

#  KMP parameters
dt = 0.01   
len = int(demodura/dt)    
lamda_1 = 1    
lamda_2 = 60     
kh = 6     

# newRef=refTraj    
# newLen=len    

# ////////////// via points //////////////
# viaFlag = np.array([1, 1, 1, 1])  
# viaNum = 4  

# via_time = np.zeros(viaNum)   
# via_point = np.zeros((viaNum, 4))   
# via_time[0] = dt   
# via_point[0,:] = np.array([8, 10, -50, 0])   
# via_time[1] = 0.25    
# via_point[1,:] = np.array([-1, 6, -25, -40])   
# via_time[2] = 1.2   
# via_point[2,:] = np.array([8, -4, 30, 10])     
# via_time[3] = 2   
# via_point[3, :] = np.array([-3, 1, -10, 3])  

# via_var = 1E-6 * np.eye(viaNum, dtype='float')   
# via_var(3,3) = 1000   
# via_var(4,4) = 1000   

# //// B 
viaFlag = np.array([1, 1, 1])  # determine which via-points are used
viaNum = 3  
via_time = np.zeros(viaNum)   
via_point = np.zeros((viaNum, 4))   

via_time[0] = dt
via_point[0, :] = np.array([-12, -12, 0, 0])  # format:[2D-pos 2D-vel]
via_time[1] = 1
via_point[1, :] = np.array([0, -1, 0, 0])     
via_time[2] = 1.99
via_point[2, :] = np.array([-14, -8, 0, 0])  

via_var = 1E-6 * np.eye(4)  
via_var[2, 2] = 1000  
via_var[3, 3] = 1000   

# //// F
# viaFlag = np.array([1, 1])  
# viaNum = 2 
# via_time = np.zeros(viaNum)   
# via_point = np.zeros((viaNum, 4))   
   
# via_time[0] = dt    
# via_point[0, :] = np.array([0.0, 0.0, 0, 0])    
# via_time[1] = 1.  
# via_point[1, :] = np.array([-8, -10, 0, 0])    

# via_var=1E-6 * np.eye(4)
# via_var[2, 2] = 1000 
# via_var[3, 3] = 1000    

# update reference trajectory using desired points
newRef = refTraj     
newLen = int(len)      

# insert points   
# print("newLen :", newLen)    
# print("via_point_before:", newRef['mu'].shape)     
for viaIndex in range(viaNum):     
    print("viaIndex :", viaIndex)     
    if (viaFlag[viaIndex]==1):        
        newRef, newLen = kmp_insertPoint(newRef, newLen, via_time[viaIndex], via_point[viaIndex, :], via_var)   

# print("newNum", newLen)  
# print("via_point_after:", newRef['mu'].shape)   

# Prediction using kmp   
# Kinv = kmp_estimateMatrix_mean(newRef, newLen, kh, lamda, dim) 
Kinv_1, Kinv_2 = kmp_estimateMatrix_mean_var(newRef, newLen, kh, lamda_1, lamda_2, dim)  
print("Kinv_1 :", Kinv_1.shape)   

uncertainLen = 0.0 * len   
totalLen = int(len + uncertainLen)     
  
# kmpPredTraj
kmpPredTraj = {}
new_time_t = np.zeros((totalLen, 1))   
new_mu_t = np.zeros((totalLen, 4))   
new_sigma_t = np.zeros((totalLen, 4, 4))    

# t, mu, sigma 
for index in range(totalLen):   
    t = index * dt   
    # mu = kmp_pred_mean(t, newRef, newLen, kh, Kinv, dim)  
    mu, sigma = kmp_pred_mean_var(t, newRef, newLen, kh, Kinv_1, Kinv_2, lamda_2, dim)  
    new_time_t[index, 0] = t 
    new_mu_t[index, :] = mu.T 
    new_sigma_t[index, :, :] = sigma  

kmpPredTraj['t'] = new_time_t      
kmpPredTraj['mu'] = new_mu_t          
kmpPredTraj['sigma'] = new_sigma_t      

# gmr = np.zeros((4, newLen))   
# kmp = np.zeros((4, newLen))   
# sigma_kmp = np.zeros((4, 4, newLen)) 
# for i in range(newLen):    
#     gmr[:, i] = refTraj['mu'][i]    
       
#     kmp[:, i] = kmpPredTraj['mu'][i]       
#     sigma_kmp[:, :, i] = kmpPredTraj['sigma'][i]   

# print("sigma_kmp :", sigma_kmp)    
plot_via_points(
    font_name=letter, nb_posterior_samples=viaNum, via_points=via_point, 
    mu_gmr=ori_refTraj['mu'], pred_gmr=kmpPredTraj['mu'], 
    sigma_gmr=ori_refTraj['sigma'], sigma_kmp=kmpPredTraj['sigma']
)  

# plot_mean_var(
#     font_name=letter, 
#     nb_samples=nb_samples, 
#     nb_data=nb_data, 
#     Xt=kmpPredTraj['t'], 
#     Y=Y, 
#     mu_gmr=kmpPredTraj['mu'], 
#     sigma_gmr=kmpPredTraj['sigma']
# )   