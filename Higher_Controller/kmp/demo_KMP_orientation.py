import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)
# print(sys.path)  
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat  
from GMRbasedGP.utils.gmr import Gmr, plot_gmm
from KMP_functions import *    
from robInfLibpy.plot_pred import *  
import seaborn as sns 
import copy as cp  

tau = 10   
dt = 0.01  
len = int(np.round(tau / dt))  
tau1 = 0.4 * tau     
tau2 = tau - tau1        
N1 = int(tau1/dt)      
N2 = len - N1     
demosAll = []      
demoNum = 5     
demo = np.zeros((len, 8))       

# generate demonstration data    
for num in range(demoNum):   
    a = np.array([1, 1.2, 1, 1.5])   
    b = np.array([2, 3.4, 1.5, 4]) + np.array([0.5, -0.1, 1, -0.2]) * num/5 * 0.5  
    c = np.array([4, 3, 2.5, 3]) + np.array([0.5, -0.0, -0.1, -1.5]) * num/5 * 0.2
    
    q1_ini = quat_normalize(a)    
    q2_mid = quat_normalize(b)    
    q3_end = quat_normalize(c)    

    q1 = q1_ini         
    # q1.s = q1_ini[0]   
    # q1.v = np.transpose(q1_ini[np.arange(1, 4)]) 

    q2 = q2_mid    
    # q2.s = q2_mid[0]   
    # q2.v = np.transpose(q2_mid[np.arange(1, 4)])  

    q3 = q3_end      
    # q3.s = q3_end[0]  
    # q3.v = np.transpose(q3_end[np.arange(1, 4)])   
    
    # print("q1_ :", q1_ini, q2_mid) 
    q, omega, domega, t = generate_orientation_data(q1, q2, tau1, dt) 
    print("q :", q)  
    for i in range(N1):  
        demo[i, 0] = i * dt   
        demo[i, 1:5] = q[i, :]      
        demo[i, 5:8] = omega[i, :]      
    
    q, omega, domega, t = generate_orientation_data(q2, q3, tau2, dt)  
    
    for i in range(N2):  
        demo[i + N1, 0] = i * dt + N1 * dt
        demo[i + N1, 1:5] = q[i, :]  
        demo[i + N1, 5:8] = omega[i, :]      
    
    demosAll.append(cp.deepcopy(demo))  

#Project demos into Euclidean space and model new trajectories using GMM/GMR
Data = [] 
zeta = np.zeros((len, 8))    
for i in range(demoNum):    
    for j in range(len):    
        time = demosAll[i][j, 0]    
        qtemp = demosAll[i][j, 1:5]    
        # qtemp.s = demosAll(2,qindex)    
        # qtemp.v = demosAll(np.arange(3,5+1),qindex)   

        zeta[j, 0] = time   
        zeta[j, 1:4] = quat_log(qtemp, q1)  
    
    for h in range(3): 
        zeta[:, 4 + h] = np.gradient(zeta[:, h + 1])/dt 

    Data.append(zeta)    
    
print("data shape :", np.array(Data).shape)   

# model parameter 
input_dim = 1   
output_dim = 7    
in_idx = [0]   
out_idx = [1, 2, 3, 4, 5, 6]     
nb_states = 7    
nb_samples = demoNum  

# # Create time data
# demos_t = [np.arange(demos[i].shape[0])[:, None] for i in range(nb_samples)]
# # print("demos_t :", demos_t) 

# # Stack time and position data
# demos_tx = [np.hstack([demos_t[i]*dt, demos_pos[i], demos_vel[i]]) for i in range(nb_samples)]
# print("demos_tx :", np.array(demos_tx).shape) 

demos_tx = demosAll 

# Stack demos
demos_np = demos_tx[0]  
for i in range(1, nb_samples):   
    demos_np = np.vstack([demos_np, demos_tx[i]])   
print("demos_np :", demos_np.shape)   

X = demos_np[:, 0][:, None]  
Y = demos_np[:, 1:]  
print('X shape:', X.shape, 'Y shape:', Y.shape)  

nb_data = len 
nb_data_sup = 0 

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

# plot_GMM_raw_data(font_name='orientation', nb_samples=nb_samples, nb_data=nb_data, Y=Y, gmr_model=gmr_model)   
plot_raw_data(font_name='orientation', nb_samples=nb_samples, nb_dim=output_dim, nb_data=nb_data, X=Xt, Y=Y)  
# plot_mean_var(font_name='orientation', nb_samples=nb_samples, nb_data=nb_data, Xt=Xt, Y=Y, mu_gmr=mu_gmr, sigma_gmr=sigma_gmr)   

# model.nbStates = 5
# model.nbVar = 7
# model.dt = 0.01
# nbData = 1000

# model = init_GMM_timeBased(Data_,model)
# model = EM_GMM(Data_,model)
# DataOut,SigmaOut = GMR(model,np.array([np.arange(1,nbData+1)]) * model.dt,1,np.arange(2,model.nbVar+1))
# for i in np.arange(1,nbData+1).reshape(-1):  
#     quatRef(i).t = i * model.dt  
#     quatRef(i).mu = DataOut(:,i)  
#     quatRef(i).sigma = SigmaOut(:,:,i)  

# Set kmp parameters
lamda = 1  
kh = 0.01  
dim = 3  

# Set desired quaternion and angular velocity
viaFlag = np.array([1, 1, 1])   

# viaFlag=[0 0 0]; # reproduction

via_time[1] = 0

qDes_temp[:,1] = q1_ini
qDes1.s = qDes_temp(1,1)
qDes1.v = qDes_temp(np.arange(2,4+1),1)

dqDes[:,1] = np.transpose(np.array([0.0,0.0,0.0]))

via_time[2] = 4.0
des_temp2 = np.array([0.2,0.6,0.4,0.8])
qDes_temp[:,2] = quatnormalize(des_temp2)
qDes2.s = qDes_temp(1,2)
qDes2.v = qDes_temp(np.arange(2,4+1),2)
dqDes[:,2] = np.transpose(np.array([0.1,0.1,0.0]))
via_time[3] = 10.0
des_temp3 = np.array([0.6,0.4,0.6,0.4])
qDes_temp[:,3] = quatnormalize(des_temp3)
qDes3.s = qDes_temp(1,3)
qDes3.v = qDes_temp(np.arange(2,4+1),3)
dqDes[:,3] = np.transpose(np.array([0,0.3,0.3]))  

# Transform desired points into Euclidean space
via_point[np.arange[1,3+1],1] = quat_log(qDes1,q1)  
via_point[np.arange[4,6+1],1] = trans_angVel(qDes1,dqDes(:,1),dt,q1)
via_point[np.arange[1,3+1],2] = quat_log(qDes2,q1)
via_point[np.arange[4,6+1],2] = trans_angVel(qDes2,dqDes(:,2),dt,q1)
via_point[np.arange[1,3+1],3] = quat_log(qDes3,q1)
via_point[np.arange[4,6+1],3] = trans_angVel(qDes3,dqDes(:,3),dt,q1)
via_var = 1e-10 * np.eye(6)   
# via_var(4,4)=1000;via_var(5,5)=1000;via_var(6,6)=1000;

# Update the reference trajectory using transformed desired points
interval = 5

refTraj = {}   
# refTraj['t'] = Xt   
# refTraj['mu'] = mu_gmr   
# refTraj['sigma'] = sigma_gmr   

num = np.round(len_/interval) + 1  
for i in range(num):  
    if i == 1:
        index = 1
    else:
        index = (i - 1) * interval 
    sampleData(i).t = quatRef(index).t
    sampleData(i).mu = quatRef(index).mu
    sampleData(i).sigma = quatRef(index).sigma

for i in range(3):   
    if viaFlag[i]:   
        sampleData, num = kmp_insertPoint(sampleData, num, via_time[i], via_point[:,i], via_var)

## KMP prediction
Kinv = kmp_estimateMatrix_mean(sampleData, num, kh, lamda, dim)  
for index in range(len): 
    t = index * dt  
    mu = kmp_pred_mean(t, sampleData, num, kh, Kinv, dim) 
    kmpTraj[1,index] = t    
    kmpTraj[np.arange[2,7+1], index] = mu

## Project predicted trajectory from Euclidean space into quaternion space
for i in range(len):  
    qnew = quat_exp(kmpTraj(np.arange(2, 4+1),i))
    trajAdaQuat[i] = quat_mult(qnew,q1)
    trajAda[1,i] = kmpTraj(1,i)
    trajAda[2,i] = trajAdaQuat(i).s 
    trajAda[np.arange[3,5+1],i] = trajAdaQuat(i).v

adaOmega, adaDomega = quat_to_vel(trajAdaQuat, dt, tau)   


# ## Show demonstrations
# figure
# set(gcf,'Position',np.array([695,943,1350,425]))
# subplot(1,2,1)
# hold('on')
# plt.plot(demosAll(1,np.arange(1,end()+1,1)),demosAll(np.arange(2,5+1),np.arange(1,end()+1,1)),'.')
# plt.xlabel('t [s]','interpreter','tex')
# plt.ylabel('  ','interpreter','tex')
# plt.ylim(np.array([0,1]))
# set(gca,'xtick',np.array([0,5,10]))
# set(gca,'ytick',np.array([0,0.5,1]))
# set(gca,'FontSize',18)
# grid('on')
# box('on')
# set(gca,'gridlinestyle','--')
# ax = gca
# ax.GridAlpha = 0.3  
# plt.legend(np.array(['$q_s$','$q_x$','$q_y$','$q_z$']),'interpreter','latex','Orientation','horizontal','FontSize',20)
# subplot(1,2,2)
# hold('on')
# plt.plot(demosAll(1,np.arange(1,end()+1,1)),demosAll(np.arange(6,8+1),np.arange(1,end()+1,1)),'.')
# plt.xlabel('t [s]','interpreter','tex')
# plt.ylabel('  [rad/s]','interpreter','tex')
# plt.ylim(np.array([- 0.5,0.5]))
# set(gca,'xtick',np.array([0,5,10]))
# set(gca,'ytick',np.array([- 0.5,0,0.5]))
# set(gca,'FontSize',18)
# grid('on')
# box('on')
# set(gca,'gridlinestyle','--')
# ax = gca
# ax.GridAlpha = 0.3
# plt.legend(np.array(['$\omega_x$','$\omega_y$','$\omega_z$']),'interpreter','latex','Orientation','horizontal','FontSize',20)
# ## Show kmp predictions
# figure
# set(gcf,'Position',np.array([690,384,1357,425]))
# subplot(1,2,1)
# plt.plot(trajAda(1,:),trajAda(np.arange(2,5+1),:),'linewidth',3.0)
# for plotIndex in np.arange(1,4+1).reshape(-1):
#     for viaIndex in np.arange(1,3+1).reshape(-1):
#         if viaFlag(viaIndex) == 1:
#             hold('on')
#             plt.plot(via_time(viaIndex),qDes_temp(plotIndex,viaIndex),'o','color',mycolors.nr,'markersize',8,'linewidth',1.5)

# plt.xlabel('t [s]','interpreter','tex')
# plt.ylabel('  ','interpreter','tex')
# plt.ylim(np.array([0,1]))
# set(gca,'xtick',np.array([0,5,10]))
# set(gca,'ytick',np.array([0,0.5,1]))
# set(gca,'FontSize',18)
# grid('on')
# set(gca,'gridlinestyle','--')
# ax = gca
# ax.GridAlpha = 0.3
# plt.legend(np.array(['$q_s$','$q_x$','$q_y$','$q_z$']),'interpreter','latex','Orientation','horizontal','FontSize',20)
# subplot(1,2,2)
# plt.plot(trajAda(1,:),adaOmega(np.arange(1,3+1),:),'linewidth',3.0)
# for plotIndex in np.arange(1,3+1).reshape(-1):
#     for viaIndex in np.arange(1,3+1).reshape(-1):
#         if viaFlag(viaIndex) == 1:
#             hold('on')
#             plt.plot(via_time(viaIndex),dqDes(plotIndex,viaIndex),'o','color',mycolors.nr,'markersize',8,'linewidth',1.5)

# plt.xlabel('t [s]','interpreter','tex')
# plt.ylabel('  [rad/s]','interpreter','tex')
# plt.ylim(np.array([- 0.5,0.5]))
# set(gca,'xtick',np.array([0,5,10]))
# set(gca,'ytick',np.array([- 0.5,0,0.5]))
# set(gca,'FontSize',18)
# grid('on')
# set(gca,'gridlinestyle','--')
# ax = gca
# ax.GridAlpha = 0.3  
# plt.legend(np.array(['$\omega_x$','$\omega_y$','$\omega_z$']),'interpreter','latex','Orientation','horizontal','FontSize',20)