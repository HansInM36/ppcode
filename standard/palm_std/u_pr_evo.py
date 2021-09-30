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
tSeq, zSeq, uSeq = velo_pr_palm(jobDir, jobName, ['.000'], 'u')
tplotSeq, varplotSeq = palm_pr_evo((0.0,24000.0,2400.0), tSeq, zSeq, uSeq)


fig, ax = plt.subplots(figsize=(4,6))
colors = plt.cm.jet(np.linspace(0,1,tplotSeq.size))
for i in range(tplotSeq.size):
    plt.plot(varplotSeq[i], zSeq, label='t = ' + str(int(tplotSeq[i])) + 's', linewidth=1.0, color=colors[i])
#plt.axhline(y=102, ls='--', c='black')
plt.xlabel(r'$\overline{u}$ (m/s)')
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 14
xaxis_d = 2
yaxis_min = 0
yaxis_max = 800.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(0.12,0.66), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveDir = '/scratch/prjdata/sigma_test'
saveName = 'u_pr_evo_No0.png'
#if not os.path.exists(saveDir):
#    os.makedirs(saveDir)
#plt.savefig(saveDir + '/' +saveName)
plt.show()
plt.close()