#!/usr/bin/python3.8
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/standard')
sys.path.append('/scratch/ppcode/standard/palm_std')
import os
import pickle
import numpy as np
import palm_data_ext
from palm_data_ext import *
import funcs_pr
from funcs_pr import *
import matplotlib.pyplot as plt


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'sigma_test_1'
jobDir = prjDir + '/' + jobName
file_in = jobDir + "/OUTPUT/" + jobName + '_xz.000.nc'
tSeq, xSeq, ySeq, zSeq = getDataInfo_palm(file_in, 'p_xz')
tSeq, xSeq, ySeq, zSeq, varSeq = sec_data_palm(jobDir, jobName, '_xz', ['.000'], 'p_xz', (0,21), (0,64), (0,1), (0,32))



xx, zz = np.zeros((zSeq.size,xSeq.size)), np.zeros((zSeq.size,xSeq.size))
XX, ZZ = np.meshgrid(xSeq, zSeq)

H_av = 1000
wa = 5.0
wn = 0.0495 # 0.0262, 6.545e-3
eta = wa * np.sin(wn * xSeq)

xx = XX
for i in range(xSeq.size):
    zz[:,i] = ZZ[:,i] / 1000 * (H_av - eta[i]) + eta[i]
    

# single plot
tInd = 20
vMin, vMax, vDelta = (-1.5, 0.0, 0.3)
vMin, vMax, vDelta = (-12, 8, 2)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
fig, axs = plt.subplots(figsize=(10,4.8), constrained_layout=False)
v_ = varSeq[tInd,1:,0,:]
#v_ -= v_.mean()
#v_[np.where(v_ < vMin)] = vMin
#v_[np.where(v_ > vMax)] = vMax
CS = axs.contourf(xx[1:], zz[1:], v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
axs.text(0.7, 1.01, 't = ' + str(np.round(tSeq[tInd]-tSeq[0],0)) + 's', transform=axs.transAxes, fontsize=16)
cbar.ax.set_ylabel(r"$\mathrm{p}$" + r'($\mathrm{kg \cdot m^{-1} \cdot s^{-2}}$)', fontsize=16)
cbar.ax.tick_params(labelsize=20)
plt.xlim([0,1000])
plt.ylim([-5,400])
plt.xticks([0,200,400,600,800,1000], [0,200,400,600,800,1000], fontsize=16)
plt.yticks([0,100,200,300,400], [0,100,200,300,400], fontsize=16)
plt.ylabel('z (m)', fontsize=16)
plt.xlabel('x (m)', fontsize=16)
plt.title('')
#saveName = "%.4d" % tInd + '.png'
#saveDir = '/scratch/prjdata/sigma_imp/1by1/mg_nowave_gs20/velo_contour_Ny'
#plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()


# single plot zoom-in
rho = 1.225
Ug = 10.755
tInd = 20
vMin, vMax, vDelta = (-0.08, 0.04, 0.02)
#vMin, vMax, vDelta = (-0.01, -0.004, 0.001)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
fig, axs = plt.subplots(figsize=(10,4.8), constrained_layout=False)
v_ = varSeq[tInd,1:,0,:]/rho/Ug**2
#v_ -= v_.mean()
v_[np.where(v_ < vMin)] = vMin
v_[np.where(v_ > vMax)] = vMax
CS = axs.contourf(xx[1:], zz[1:], v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
axs.text(0.7, 1.01, 't = ' + str(np.round(tSeq[tInd]-tSeq[0],0)) + 's', transform=axs.transAxes, fontsize=16)
cbar.ax.set_ylabel(r"$\mathrm{\frac{p}{\rho U_g^2}}$", fontsize=16)
cbar.ax.tick_params(labelsize=20)
plt.xlim([252,504])
plt.ylim([-5,40])
plt.xticks([252,315,378,441,504], [2.0,2.5,3.0,3.5,4.0], fontsize=16)
plt.yticks([0,10,20,30,40], [0.0,0.5,1.0,1.5,2.0], fontsize=16)
plt.ylabel(r'$\mathrm{2\pi z / \lambda}$', fontsize=16)
plt.xlabel(r'$\mathrm{x / \lambda}$', fontsize=16)
plt.title('')
#saveName = "%.4d" % tInd + '.png'
#saveDir = '/scratch/prjdata/sigma_imp/1by1/mg_nowave_gs20/velo_contour_Ny'
#plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()



# animation
vMin, vMax, vDelta = (-1.5, 0.0, 0.3)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
for tInd in range(0,20,1):
    fig, axs = plt.subplots(figsize=(10,4.8), constrained_layout=False)
    v_ = varSeq[tInd,1:,0,:]
    #v_ -= v_.mean()
    #v_[np.where(v_ < vMin)] = vMin
    #v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(xx[1:], zz[1:], v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    axs.text(0.7, 1.01, 't = ' + str(np.round(tSeq[tInd]-tSeq[0],0)) + 's', transform=axs.transAxes, fontsize=16)
    cbar.ax.set_ylabel(r"$\mathrm{p}$" + r'($\mathrm{kg \cdot m^{-1} \cdot s^{-2}}$)', fontsize=16)
    cbar.ax.tick_params(labelsize=20)
    plt.xlim([0,1000])
    plt.ylim([-5,400])
    plt.xticks([0,200,400,600,800,1000], [0,200,400,600,800,1000], fontsize=16)
    plt.yticks([0,100,200,300,400], [0,100,200,300,400], fontsize=16)
    plt.ylabel('z (m)', fontsize=16)
    plt.xlabel('x (m)', fontsize=16)
    plt.title('')
#    saveName = "%.4d" % tInd + '.png'
#    saveDir = '/scratch/prjdata/sigma_imp/1by1/mg_nowave_gs20/velo_contour_Ny'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()