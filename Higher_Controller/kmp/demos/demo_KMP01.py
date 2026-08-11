#-----------function demo_KMP01----------
# This file provide a simple demo of using kmp, where trajectory adaptations towards
# various desired points in terms of positions and velocities are studied.
#
# This code is written by Dr. Yanlong Huang

# @InProceedings{Huang19IJRR,
#   Title = {Kernelized Movement Primitives},
#   Author = {Huang, Y. and Rozo, L. and Silv\'erio, J. and Caldwell, D. G.},
#   Booktitle = {International Journal of Robotics Research},
#   Year= {2019},
#   pages= {833--852}
# }

# @InProceedings{Huang19ICRA_1,
#   Title = {Non-parametric Imitation Learning of Robot Motor Skills},
#   Author = {Huang, Y. and Rozo, L. and Silv\'erio, J. and Caldwell, D. G.},
#   Booktitle = {Proc. {International Conference on Robotics and Automation ({ICRA})},
#   Year = {2019},
#   Address = {Montreal, Canada},
#   Month = {May},
#   pages = {5266--5272}
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
scipy.io.loadmat('../2Dletters/G.mat')
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
len_ = demo_dura / dt
lamda = 1

kh = 6
## Set desired points
viaFlag = np.array([1,1,1,1])

viaNum = 4
via_time[1] = dt
via_point[:,1] = np.transpose(np.array([8,10,- 50,0]))
via_time[2] = 0.25
via_point[:,2] = np.transpose(np.array([- 1,6,- 25,- 40]))
via_time[3] = 1.2
via_point[:,3] = np.transpose(np.array([8,- 4,30,10]))
via_time[4] = 2
via_point[:,4] = np.transpose(np.array([- 3,1,- 10,3]))
via_var = 1e-06 * np.eye(4)

# via_var(3,3)=1000;via_var(4,4)=1000; # low adaptation precision for velocity

## Update the reference trajectory using desired points
newRef = refTraj
newLen = len_
for viaIndex in np.arange(1,viaNum+1).reshape(-1):
    if viaFlag(viaIndex) == 1:
        newRef,newLen = kmp_insertPoint(newRef,newLen,via_time(viaIndex),via_point(:,viaIndex),via_var)

## Prediction using kmp
Kinv = kmp_estimateMatrix_mean(newRef,newLen,kh,lamda,dim)
for index in np.arange(1,len_+1).reshape(-1):
    t = index * dt
    mu = kmp_pred_mean(t,newRef,newLen,kh,Kinv,dim)
    kmpPredTraj(index).t = index * dt
    kmpPredTraj(index).mu = mu

for i in np.arange(1,len_+1).reshape(-1):
    gmr[:,i] = refTraj(i).mu
    kmp[:,i] = kmpPredTraj(i).mu

## Show demonstrations and the corresponding reference trajectory
figure
set(gcf,'position',np.array([468,875,1914,401]))
## show demonstrations
for i in np.arange(1,demoNum+1).reshape(-1):
    subplot(2,3,1)
    hold('on')
    plt.plot(np.array([np.arange(demo_dt,demo_dura+demo_dt,demo_dt)]),Data_(2,np.arange((i - 1) * demoLen + 1,i * demoLen+1)),'linewidth',3,'color',mycolors.g)
    box('on')
    plt.ylabel('x [cm]','interpreter','tex')
    set(gca,'xtick',np.array([0,1,2]))
    set(gca,'ytick',np.array([- 10,0,10]))
    set(gca,'FontSize',12)
    grid('on')
    set(gca,'gridlinestyle','--')
    ax = gca
    ax.GridAlpha = 0.3
    subplot(2,3,4)
    hold('on')
    plt.plot(np.array([np.arange(demo_dt,demo_dura+demo_dt,demo_dt)]),Data_(3,np.arange((i - 1) * demoLen + 1,i * demoLen+1)),'linewidth',3,'color',mycolors.g)
    box('on')
    plt.xlabel('t [s]','interpreter','tex')
    plt.ylabel('y [cm]','interpreter','tex')
    set(gca,'xtick',np.array([0,1,2]))
    set(gca,'ytick',np.array([- 10,0,10]))
    set(gca,'FontSize',12)
    grid('on')
    set(gca,'gridlinestyle','--')
    ax = gca
    ax.GridAlpha = 0.3

## show GMM
subplot(2,3,np.array([2,5]))
for i in np.arange(1,demoNum+1).reshape(-1):
    hold('on')
    plt.plot(Data_(2,np.arange((i - 1) * demoLen + 1,i * demoLen+1)),Data_(3,np.arange((i - 1) * demoLen + 1,i * demoLen+1)),'linewidth',3,'color',mycolors.g)

hold('on')
for i in np.arange(1,demoLen * demoNum+demoLen,demoLen).reshape(-1):
    hold('on')
    plt.plot(Data_(2,i),Data_(3,i),'*','markersize',12,'color',mycolors.g)

for i in np.arange(demoLen,demoLen * demoNum+demoLen,demoLen).reshape(-1):
    hold('on')
    plt.plot(Data_(2,i),Data_(3,i),'+','markersize',12,'color',mycolors.g)

hold('on')
plotGMM(model.Mu(np.arange(2,3+1),:),model.Sigma(np.arange(2,3+1),np.arange(2,3+1),:),np.array([0.8,0,0]),0.5)
box('on')
grid('on')
set(gca,'gridlinestyle','--')
plt.xlabel('x [cm]','interpreter','tex')
plt.ylabel('y [cm]','interpreter','tex')
set(gca,'xtick',np.array([- 10,0,10]))
set(gca,'ytick',np.array([- 10,0,10]))
set(gca,'FontSize',12)
ax = gca
ax.GridAlpha = 0.3
## show reference trajectory
subplot(2,3,np.array([3,6]))
hold('on')
plotGMM(DataOut(np.arange(1,2+1),:),SigmaOut(np.arange(1,2+1),np.arange(1,2+1),:),mycolors.g,0.025)
hold('on')
plt.plot(DataOut(1,:),DataOut(2,:),'color',mycolors.g,'linewidth',3.0)
hold('on')
plt.plot(DataOut(1,1),DataOut(2,1),'*','markersize',15,'color',mycolors.g)
hold('on')
plt.plot(DataOut(1,end()),DataOut(2,end()),'+','markersize',15,'color',mycolors.g)
box('on')
plt.xlim(np.array([- 10.5,10.5]))
plt.xlabel('x [cm]','interpreter','tex')
plt.ylabel('y [cm]','interpreter','tex')
set(gca,'xtick',np.array([- 10,0,10]))
set(gca,'ytick',np.array([- 10,0,10]))
set(gca,'FontSize',12)
grid('on')
set(gca,'gridlinestyle','--')
ax = gca
ax.GridAlpha = 0.3
## Show kmp predictions
value = np.array([0.5,0,0.5])
curveValue = mycolors.o
plt.figure('units','normalized','outerposition',np.array([0,0,1,1]))
set(gcf,'Position',np.array([0.0465,0.1794,0.9535,0.2394]))
## plot px-py
subplot(1,5,1)
hold('on')
plt.plot(gmr(1,:),gmr(2,:),'--','color',mycolors.gy,'linewidth',1.5)
hold('on')
plt.plot(kmp(1,:),kmp(2,:),'color',curveValue,'linewidth',2)
hold('on')
plt.plot(gmr(1,1),gmr(2,1),'*','markersize',12,'color',mycolors.gy)
hold('on')
plt.plot(gmr(1,end()),gmr(2,end()),'+','markersize',12,'color',mycolors.gy)
for viaIndex in np.arange(1,viaNum+1).reshape(-1):
    if viaFlag(viaIndex) == 1:
        plt.plot(via_point(1,viaIndex),via_point(2,viaIndex),'o','color',value,'markersize',12,'linewidth',1.5)

box('on')
plt.xlim(np.array([- 12,12]))
plt.ylim(np.array([- 12,12]))
plt.xlabel('${x}$ [cm]','interpreter','latex')
plt.ylabel('${y}$ [cm]','interpreter','latex')
set(gca,'xtick',np.array([- 12,0,12]))
set(gca,'ytick',np.array([- 12,0,12]))
set(gca,'FontSize',17)
grid('on')
set(gca,'gridlinestyle','--')
ax = gca
ax.GridAlpha = 0.3
## plot t-px, t-py, t-vx and t-vy
for plotIndex in np.arange(1,4+1).reshape(-1):
    subplot(1,5,1 + plotIndex)
    hold('on')
    plt.plot(np.array([np.arange(dt,dt * len_+dt,dt)]),gmr(plotIndex,:),'--','color',mycolors.gy,'linewidth',1.5)
    hold('on')
    plt.plot(np.array([np.arange(dt,dt * len_+dt,dt)]),kmp(plotIndex,:),'color',curveValue,'linewidth',2.0)
    for viaIndex in np.arange(1,viaNum+1).reshape(-1):
        if viaFlag(viaIndex) == 1:
            plt.plot(via_time(viaIndex),via_point(plotIndex,viaIndex),'o','color',value,'markersize',12,'linewidth',1.5)
    box('on')
    plt.xlim(np.array([0,len_ * dt]))
    if plotIndex == 1 or plotIndex == 2:
        if plotIndex == 1:
            plt.ylabel('${x}$ [cm]','interpreter','latex')
        if plotIndex == 2:
            plt.ylabel('${y}$ [cm]','interpreter','latex')
        plt.ylim(np.array([- 12,12]))
        set(gca,'ytick',np.array([- 12,0,12]))
    if plotIndex == 3 or plotIndex == 4:
        if plotIndex == 3:
            plt.ylabel('$\dot{x}$ [cm/s]','interpreter','latex')
        if plotIndex == 4:
            plt.ylabel('$\dot{y}$ [cm/s]','interpreter','latex')
        plt.ylim(np.array([- 80,80]))
        set(gca,'ytick',np.array([- 80,0,80]))
    plt.xlabel('$t$ [s]','interpreter','latex')
    set(gca,'xtick',np.array([0,1,2]))
    set(gca,'FontSize',17)
    grid('on')
    set(gca,'gridlinestyle','--')
    ax = gca
    ax.GridAlpha = 0.3
