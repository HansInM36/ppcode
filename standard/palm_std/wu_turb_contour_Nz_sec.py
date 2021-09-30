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
jobName  = 'sigma_test'
jobDir = prjDir + '/' + jobName
file_in = jobDir + "/OUTPUT/" + jobName + '_xy.001.nc'
tSeq, xSeq, ySeq, zSeq = getDataInfo_palm(file_in, 'u_xy')
tSeq, xSeq, ySeq, zSeq, uSeq = sec_data_palm(jobDir, jobName, '_xy', ['.001'], 'u_xy', (0,600), (0,64), (0,64), (0,11))
tSeq, xSeq, ySeq, zSeq, wSeq = sec_data_palm(jobDir, jobName, '_xy', ['.001'], 'w_xy', (0,600), (0,64), (0,64), (0,11))
tSeq, xSeq, ySeq, zSeq, u_avSeq = sec_data_palm(jobDir, jobName, '_av_xy', ['.001'], 'u_xy', (0,1), (0,64), (0,64), (0,32))
tSeq, xSeq, ySeq, zSeq, w_avSeq = sec_data_palm(jobDir, jobName, '_av_xy', ['.001'], 'w_xy', (0,1), (0,64), (0,64), (0,32))
tSeq, xSeq, ySeq, zSeq, ustarSeq = sec_data_palm(jobDir, jobName, '_xy', ['.001'], 'us*_xy', (0,600), (0,64), (0,64), (0,11))
ustar = ustarSeq.mean()


zInd = 3
vMin, vMax, vDelta = (-4.0, 4.0, 2.0)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)
for tInd in range(500,600,1):
    fig, axs = plt.subplots(figsize=(10,10))
    v_ = (wSeq[tInd,zInd,:,:] - w_avSeq[0,zInd,:,:]) * (uSeq[tInd,zInd,:,:] - u_avSeq[0,zInd,:,:])/np.power(ustar,2)
    #v_ -= v_.mean()
    v_[np.where(v_ < vMin)] = vMin
    v_[np.where(v_ > vMax)] = vMax
    CS = axs.contourf(xSeq, ySeq, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
    cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    axs.text(0.7, 1.01, 't = ' + str(np.round(tSeq[tInd]-tSeq[0],0)) + 's', transform=axs.transAxes, fontsize=16)
    cbar.ax.set_ylabel(r"$\mathrm{w'u'/u^2_*}$", fontsize=16)
    cbar.ax.tick_params(labelsize=20)
    plt.xlim([0,1000])
    plt.ylim([0,1000])
    plt.xticks([0,200,400,600,800,1000], [0,200,400,600,800,1000], fontsize=16)
    plt.yticks([0,200,400,600,800,1000], [0,200,400,600,800,1000], fontsize=16)
    plt.ylabel('y (m)', fontsize=16)
    plt.xlabel('x (m)', fontsize=16)
    plt.title('')
    saveName = "%.4d" % tInd + '.png'
    saveDir = '/scratch/projects/sigma_test/animation/wu_turb_contour_Nz_H15_test'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()