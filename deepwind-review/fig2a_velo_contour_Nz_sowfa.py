#!/usr/bin/python3.8
import os
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/standard')
sys.path.append('/scratch/ppcode/standard/palm_std')
sys.path.append('/scratch/ppcode/standard/sowfa_std')
sys.path.append('/scratch/ppcode/sowfa/src')
import numpy as np
import palm_data_ext
from palm_data_ext import *
import sowfa_data_ext_L2
from sowfa_data_ext_L2 import *
import matplotlib.pyplot as plt


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs10'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq, xSeq, ySeq, H, varSeq = getSliceData_Nz_sowfa(ppDir, 'Nz2', 'U', 0, ((0,0,0),30), (0,30), (680,1880,240), (680,1880,240))


# transform coors of prbg
prbg0 = np.vstack((780 + 20*np.arange(0,51), np.array([1280 for i in range(51)]), np.array([0 for i in range(51)])))
prbg0 = funcs.trs(prbg0.T, (1280,1280,0), -30); prbg0[:,0] += 1280; prbg0[:,1] += 1280;
prbg1 = np.vstack((np.array([1280 for i in range(51)]), 780 + 20*np.arange(0,51), np.array([0 for i in range(51)])))
prbg1 = funcs.trs(prbg1.T, (1280,1280,0), -30); prbg1[:,0] += 1280; prbg1[:,1] += 1280;

vMin, vMax, vDelta = (-2, 2, 0.4)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

### plot
tInd = 25
fig, axs = plt.subplots(figsize=(6,6), constrained_layout=False)
x_ = xSeq
y_ = ySeq
v_ = varSeq[tInd]
v_ -= v_.mean()
v_[np.where(v_ < vMin)] = vMin
v_[np.where(v_ > vMax)] = vMax
CS = axs.contourf(x_, y_, v_, cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
plt.scatter(prbg0[:,0], prbg0[:,1], 1, marker='o', color='k')
plt.scatter(prbg1[:,0], prbg1[:,1], 1, marker='o', color='k')
cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
#axs.text(0.8, 1.01, 't = ' + str(np.round(tSeq[tInd]-tSeq[0],2)) + 's', transform=axs.transAxes, fontsize=20)
cbar.ax.set_ylabel(r"$\mathrm{u'}$" + ' (m/s)', fontsize=20)
cbar.ax.tick_params(labelsize=20)
plt.xlim([680,1880])
plt.ylim([680,1880])
plt.ylabel('y (km)', fontsize=20)
plt.xlabel('x (km)', fontsize=20)
plt.xticks([800,1000,1200,1400,1600,1800], [0.8,1.0,1.2,1.4,1.6,1.8], fontsize=20)
plt.yticks([800,1000,1200,1400,1600,1800], [0.8,1.0,1.2,1.4,1.6,1.8], fontsize=20)
plt.title('')
saveName = 'fig2a_velo_contour_Nz_sowfa.png'
saveDir  = '/scratch/projects/deepwind/photo/review'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()

