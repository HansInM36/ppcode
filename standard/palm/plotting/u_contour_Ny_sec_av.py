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
jobName  = 'sigma_ref'
jobDir = prjDir + '/' + jobName
file_in = jobDir + "/OUTPUT/" + jobName + '_av_xz.000.nc'

tSeq, xSeq, ySeq, zSeq = getDataInfo_palm(file_in, 'u_xz')
tSeq, xSeq, ySeq, zSeq, varSeq = sec_data_palm(jobDir, jobName, '_av_xz', ['.000'], 'u_xz', (0,21), (0,64), (0,1), (0,32))
#varSeq[:,0,:,:] = 0.0

xx, zz = np.zeros((zSeq.size,xSeq.size)), np.zeros((zSeq.size,xSeq.size))
XX, ZZ = np.meshgrid(xSeq, zSeq)

H_av = 1000
wa = 0.0
wn = 0.0491 # 0.0262, 6.545e-3
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
vMin, vMax, vDelta = (4.0, 12.0, 1.0)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
for tInd in range(0,10,1):
    fig, axs = plt.subplots(figsize=(10,4.8))
    v_ = varSeq[tInd,:,0,:]
    #v_ -= v_.mean()
#    v_[np.where(v_ < vMin)] = vMin
#    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(xx, zz, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    axs.text(0.7, 1.01, 't = ' + str(np.round(tSeq[tInd]-tSeq[0],0)) + 's', transform=axs.transAxes, fontsize=16)
    cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=16)
    cbar.ax.tick_params(labelsize=20)
    plt.xlim([0,1000])
    plt.ylim([-5,400])
#    plt.xticks([0,200,400,600,800,1000], [0,200,400,600,800,1000], fontsize=16)
#    plt.yticks([0,100,200,300,400], [0,100,200,300,400], fontsize=16)
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