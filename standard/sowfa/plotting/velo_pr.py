#!/usr/bin/python3.8
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/standard')
sys.path.append('/scratch/ppcode/standard/sowfa/dataExt_L2')
import os
import pickle
import numpy as np
import sowfa_data_ext_L2
from sowfa_data_ext_L2 import *
import funcs_pr
from funcs_pr import *
import matplotlib.pyplot as plt


prjName = 'examples'
jobName = 'NBL'
jobDir = '/scratch/sowfadata/JOBS/' + prjName + '/' + jobName
tSeq, zSeq, uSeq =  velo_pr_sowfa(jobDir, ((0,0,0),30.0), 0)


""" u profile of stationary flow (sowfa vs palm) """
fig, ax = plt.subplots(figsize=(6,6))

plt.plot(uSeq[-1], zSeq, label='sowfa', linewidth=1.0, linestyle='-', color='k')
plt.xlabel(r"$\overline{\mathrm{u}}$ (m/s)", fontsize=20)
plt.ylabel('z (m)', fontsize=20)
xaxis_min = 0
xaxis_max = 12.0
xaxis_d = 2.0
yaxis_min = 0.0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=20)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=20)
plt.legend(bbox_to_anchor=(0.05,0.86), loc=6, borderaxespad=0, fontsize=20) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
#saveName = 'fig1a_velo_pr.png'
#plt.savefig('/scratch/projects/deepwind/photo/review' + '/' + saveName)
plt.show()
plt.close()
