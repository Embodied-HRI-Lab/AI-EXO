#-----------demo_KMP_uncertainty----------
# This file provide a simple demo of using kmp, where both the trajectory covariance
# and uncertainty are predicted.
#
# This code is written by Dr. Yanlong Huang

# @InProceedings{silverio2019uncertainty,
#   Title = {Uncertainty-Aware Imitation Learning using Kernelized Movement Primitives},
#   Author = {Silv\'erio, J. and Huang, Y. and Abu-Dakka, Fares J and Rozo, L. and  Caldwell, D. G.},
#   Booktitle = {Proc. {IEEE/RSJ} International Conference on Intelligent Robots and Systems ({IROS})},
#   Year = {2019, to appear},
# }
##
import os
import numpy as np
import matplotlib.pyplot as plt

clear
close_('all')
myColors
addpath('../fcts/')
## Extract position and velocity from demos
scipy.io.loadmat('../2Dletters/F.mat')

demoNum = 5

demo_dt = 0.01

demoLen = demos[0].pos.shape[2-1]

demo_dura = demoLen * demo_dt

dim = 2   

totalNum = 0
for i in np.arange(1,demoNum+1).reshape(-1):
    for j in np.arange(1,demoLen+1).reshape(-1):
        totalNum = totalNum + 1
        Data_[1,totalNum] = j * demo_dt
        Data_[np.arange[2,dim + 1+1],totalNum] = demos[i].pos(np.arange(1,dim+1),j)
    lowIndex = (i - 1) * demoLen + 1
    upIndex = i * demoLen
    for k in np.arange(1,dim+1).reshape(-1):
        Data_[dim + 1 + k,np.arange[lowIndex,upIndex+1]] = gradient(Data_(1 + k,np.arange(lowIndex,upIndex+1))) / demo_dt

## Extract the reference trajectory
model.nbStates = 8  

model.nbVar = 1 + 2 * dim  

model.dt = 0.005
 
nbData = demo_dura / model.dt

model = init_GMM_timeBased(Data_,model)
model = EM_GMM(Data_,model)
DataOut,SigmaOut = GMR(model,np.array([np.arange(1,nbData+1)]) * model.dt,1,np.arange(2,model.nbVar+1))

for i in np.arange(1,nbData+1).reshape(-1):
    refTraj(i).t = i * model.dt
    refTraj(i).mu = DataOut(:,i)
    refTraj(i).sigma = SigmaOut(:,:,i)

## Set kmp parameters
dt = 0.005
len_ = demo_dura/dt  
lamda = 1  
lamdac = 60

kh = 6  

## Set desired points
viaFlag = np.array([1,1])

viaNum = 2
via_time[1] = dt
via_point[:,1] = np.transpose(np.array([- 4,- 12,0,0]))

via_time[2] = 2
via_point[:,2] = np.transpose(np.array([8,0,0,0]))
via_var = 1e-06 * np.eye(4)

via_var[3,3] = 1000
via_var[4,4] = 1000

## Update the reference trajectory using desired points
newRef = refTraj
newLen = len_
for viaIndex in np.arange(1,viaNum+1).reshape(-1):
    if viaFlag(viaIndex) == 1:
        newRef,newLen = kmp_insertPoint(newRef,newLen,via_time(viaIndex),via_point(:,viaIndex),via_var)

## Prediction mean and variance INSIDE AND OUTSIDE the training region using kmp
Kinv1,Kinv2 = kmp_estimateMatrix_mean_var(newRef,newLen,kh,lamda,lamdac,dim)
uncertainLen = 0.8 * len_

totalLen = len_ + uncertainLen
for index in np.arange(1,totalLen+1).reshape(-1):  
    t = index * dt
    mu,sigma = kmp_pred_mean_var(t,newRef,newLen,kh,Kinv1,Kinv2,lamdac,dim)
    kmpPredTraj(index).t = t
    kmpPredTraj(index).mu = mu
    kmpPredTraj(index).sigma = sigma

for i in np.arange(1,totalLen+1).reshape(-1):
    kmp[:,i] = kmpPredTraj(i).mu
    for h in np.arange(1,2 * dim+1).reshape(-1):
        kmpVar[h,i] = kmpPredTraj(i).sigma(h,h)
        kmpVar[h,i] = np.sqrt(kmpVar(h,i))
    SigmaOut_kmp[:,:,i] = kmpPredTraj(i).sigma

## Show kmp predictions (mean and covariance/uncertainty)
value = np.array([0.5,0,0.5])
curveValue = mycolors.o
plt.figure('units','normalized','outerposition',np.array([0,0,1,1]))
set(gcf,'Position',np.array([0.1597,0.1311,0.6733,0.2561]))
## plot px-py within the training region
subplot(1,3,1)
plotGMM(kmp(np.arange(1,2+1),np.arange(1,len_+1)),SigmaOut_kmp(np.arange(1,2+1),np.arange(1,2+1),np.arange(1,len_+1)),curveValue,0.03)
hold('on')
plt.plot(kmp(1,np.arange(1,len_+1)),kmp(2,np.arange(1,len_+1)),'color',curveValue,'linewidth',2)
hold('on')
plt.plot(Data_(2,:),Data_(3,:),'.','markersize',5,'color','k')

for viaIndex in np.arange(1,viaNum+1).reshape(-1):
    if viaFlag(viaIndex) == 1:
        plt.plot(via_point(1,viaIndex),via_point(2,viaIndex),'o','color',value,'markersize',12,'linewidth',1.5)

box('on')
plt.xlim(np.array([- 15,15]))
plt.ylim(np.array([- 15,15]))
plt.xlabel('${x}$ [cm]','interpreter','latex')
plt.ylabel('${y}$ [cm]','interpreter','latex')
set(gca,'xtick',np.array([- 10,0,10]))
set(gca,'ytick',np.array([- 10,0,10]))
set(gca,'FontSize',17)
grid('on')
set(gca,'gridlinestyle','--')
ax = gca
ax.GridAlpha = 0.3
## plot t-px and t-py
for plotIndex in np.arange(1,2+1).reshape(-1):
    subplot(1,3,1 + plotIndex)
    # plot original data
    hold('on')
    plt.plot(Data_(1,np.arange(1,end()+1,1)),Data_(1 + plotIndex,np.arange(1,end()+1,1)),'.','markersize',5,'color','k')
    # plot kmp prediction INSIDE the training region
    shadowT = np.arange(dt,dt * len_+dt,dt)
    shadow_time = np.array([shadowT,fliplr(shadowT)])  
    shadowUpKmp = kmp(plotIndex,np.arange(1,len_+1)) + kmpVar(plotIndex,np.arange(1,len_+1))
    shadowLowKmp = kmp(plotIndex,np.arange(1,len_+1)) - kmpVar(plotIndex,np.arange(1,len_+1))
    shadow_kmp = np.array([shadowUpKmp,fliplr(shadowLowKmp)])
    fill(shadow_time,np.transpose(shadow_kmp),curveValue,'facealpha',0.3,'edgecolor','none')
    hold('on')
    plt.plot(shadowT,kmp(plotIndex,np.arange(1,len_+1)),'color',curveValue,'linewidth',2.0)
    # plot kmp prediction OUTSIDE the training region
    shadowT = np.arange(dt * (len_ + 1),dt * totalLen+dt,dt)
    shadow_time = np.array([shadowT,fliplr(shadowT)])
    shadowUpKmp = kmp(plotIndex,np.arange(len_ + 1,totalLen+1)) + kmpVar(plotIndex,np.arange(len_ + 1,totalLen+1))
    shadowLowKmp = kmp(plotIndex,np.arange(len_ + 1,totalLen+1)) - kmpVar(plotIndex,np.arange(len_ + 1,totalLen+1))
    shadow_kmp = np.array([shadowUpKmp,fliplr(shadowLowKmp)])
    fill(shadow_time,np.transpose(shadow_kmp),mycolors.y,'facealpha',0.3,'edgecolor','none')
    hold('on')
    plt.plot(shadowT,kmp(plotIndex,np.arange(len_ + 1,totalLen+1)),'color',mycolors.y,'linewidth',2.0)
    # plot desired points
    for viaIndex in np.arange(1,viaNum+1).reshape(-1):
        if viaFlag(viaIndex) == 1:
            plt.plot(via_time(viaIndex),via_point(plotIndex,viaIndex),'o','color',value,'markersize',12,'linewidth',1.5)
    box('on')
    plt.xlim(np.array([0,totalLen * dt]))
    if plotIndex == 1 or plotIndex == 2:
        if plotIndex == 1:
            plt.ylabel('${x}$ [cm]','interpreter','latex')
        if plotIndex == 2:
            plt.ylabel('${y}$ [cm]','interpreter','latex')
        plt.ylim(np.array([- 15,15]))
        set(gca,'ytick',np.array([- 10,0,10]))
    plt.xlabel('$t$ [s]','interpreter','latex')
    set(gca,'xtick',np.array([0,1,2]))
    set(gca,'FontSize',17)
    grid('on')
    set(gca,'gridlinestyle','--')
    ax = gca
    ax.GridAlpha = 0.3
