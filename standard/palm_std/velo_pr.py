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


prjDir = '/scratch/palmdata/JOBS'


jobName_0  = 'sigma_ref'
jobDir_0 = prjDir + '/' + jobName_0
tSeq_0, zSeq_0, uSeq_0 = velo_pr_palm(jobDir_0, jobName_0, ['.000'], 'u')
tSeq_0, zSeq_0, vSeq_0 = velo_pr_palm(jobDir_0, jobName_0, ['.000'], 'v')

jobName_1  = 'sigma_test'
jobDir_1 = prjDir + '/' + jobName_1
tSeq_1, zSeq_1, uSeq_1 = velo_pr_palm(jobDir_1, jobName_1, ['.000'], 'u')
tSeq_1, zSeq_1, vSeq_1 = velo_pr_palm(jobDir_1, jobName_1, ['.000'], 'v')



""" u profile of stationary flow (last time step) """
fig, ax = plt.subplots(figsize=(4.5,6))
tInd = -1
plt.plot(uSeq_0[tInd], zSeq_0/zSeq_0[-1], label='Case 0 u', linewidth=1.0, marker='', linestyle='-', color='k')
plt.plot(vSeq_0[tInd], zSeq_0/zSeq_0[-1], label='Case 0 v', linewidth=1.0, marker='', linestyle='--', color='k')
plt.plot(uSeq_1[tInd], zSeq_1/zSeq_1[-1], label='Case 1 u', linewidth=1.0, marker='', linestyle='-', color='r')
plt.plot(vSeq_1[tInd], zSeq_1/zSeq_1[-1], label='Case 1 v', linewidth=1.0, marker='', linestyle='--', color='r')
plt.xlabel("horizontal time-average velocity (m/s)", fontsize=16)
plt.ylabel('z/H', fontsize=16)
xaxis_min = -6
xaxis_max = 12.0
xaxis_d = 2.0
yaxis_min = 0.0
yaxis_max = 1.0
yaxis_d = 0.1
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=16)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=16)
plt.legend(bbox_to_anchor=(0.2,0.86), loc=6, borderaxespad=0, fontsize=16) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'velo' + '_pr.png'
saveDir = '/scratch/prjdata/sigma_imp/1by1/mg_nowave_gs20'
#plt.savefig(saveDir + '/' + saveName)
plt.show()
plt.close()