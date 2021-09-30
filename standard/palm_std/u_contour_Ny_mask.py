#!/usr/bin/python3.8
import os
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/standard')
sys.path.append('/scratch/ppcode/standard/palm_std')
import numpy as np
import palm_data_ext
from palm_data_ext import *
import matplotlib.pyplot as plt


jobName  = 'sigma_test_1'
jobDir = '/scratch/palmdata/JOBS/' + jobName
ppDir = '/scratch/palmdata/pp/' + jobName
tSeq, xSeq, ySeq, zSeq = getInfo_palm(jobDir, jobName, 'M02', '.000', 'u')
tSeq, xSeq, ySeq, zSeq, varSeq = mask_data_palm(jobDir, jobName, 'M02', ['.000'], 'u', (0,120), (0,xSeq.size), (0,1), (0,zSeq.size))
varSeq[:,0,:,:] = 0.0

xx, zz = np.zeros((zSeq.size,xSeq.size)), np.zeros((zSeq.size,xSeq.size))
XX, ZZ = np.meshgrid(xSeq, zSeq)

H_av = 1000
wa = 5.0
wn = 0.0262 # 0.0262, 6.545e-3
eta = wa * np.sin(wn * xSeq)

xx = XX
for i in range(xSeq.size):
    zz[:,i] = ZZ[:,i] / 1000 * (H_av - eta[i]) + eta[i]


## subtract averaged velocity at various heights
#for zInd in range(zSeq.size):
#    tmp = np.copy(varSeq[:,zInd,:,:])
#    mean_ = np.mean(tmp)
#    varSeq[:,zInd,:,:] -= mean_
#print(np.min(varSeq),np.max(varSeq)) # find min and max


# plot
vMin, vMax, vDelta = (0, 10, 1.0)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
for tInd in range(0,100,1):
    fig, axs = plt.subplots(figsize=(8,3.8), constrained_layout=False)
    v_ = varSeq[tInd,:,0,:]
    #v_ -= v_.mean()
#    v_[np.where(v_ < vMin)] = vMin
#    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(xx, zz, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
#    axs.text(0.8, 1.01, 't = ' + str(np.round(tSeq[tInd]-tSeq[0],2)) + 's', transform=axs.transAxes, fontsize=16)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=16)
    #plt.xlim([0,2560])
    plt.ylim([-5,100])
    plt.ylabel('z (m)')
    plt.xlabel('x (m)')
    plt.title('')
#    saveName = "%.4d" % tInd + '.png'
#    saveDir = '/scratch/prjdata/sigma_imp/1by1/mg_nowave_gs20/velo_contour_Ny'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()