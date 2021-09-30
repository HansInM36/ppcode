#!/usr/bin/python3.8
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/standard')
sys.path.append('/scratch/ppcode/standard/palm_std')
sys.path.append('/scratch/ppcode/standard/sowfa_std')
import os
import pickle
import numpy as np
from netCDF4 import Dataset
import palm_data_ext
from palm_data_ext import *
import sowfa_data_ext_L2
from sowfa_data_ext_L2 import *
import funcs_pr
from funcs_pr import *
import matplotlib.pyplot as plt


prjName = 'deepwind'
jobName_1 = 'gs10_refined'
dir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, zSeq_1, uSeq_1 =  velo_pr_sowfa(dir_1, ((0,0,0),30.0), 0)
tSeq_1, zSeq_1, vSeq_1 =  velo_pr_sowfa(dir_1, ((0,0,0),30.0), 1)
t_seq_1, uSeq_1 = velo_pr_ave((3600,151200,151200,1e6), tSeq_1, tSeq_1[-1]-tSeq_1[-2], zSeq_1.size, uSeq_1)
t_seq_1, vSeq_1 = velo_pr_ave((3600,151200,151200,1e6), tSeq_1, tSeq_1[-1]-tSeq_1[-2], zSeq_1.size, vSeq_1)

prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName_4  = 'deepwind_gs5_main'
dir_4 = prjDir + '/' + jobName_4
tSeq_4, zSeq_4, uSeq_4 = velo_pr_palm(dir_4, jobName_4, ['.000','.001'], 'u')
tSeq_4, zSeq_4, vSeq_4 = velo_pr_palm(dir_4, jobName_4, ['.000','.001'], 'v')


""" wind direction profile of stationary flow (sowfa vs palm) """
fig, ax = plt.subplots(figsize=(6,6))

wdSeq_1 = 270 - np.arctan(vSeq_1[-1] / uSeq_1[-1]) * 180/np.pi
wdSeq_4 = 270 - np.arctan(vSeq_4[-1][1:] / uSeq_4[-1][1:]) * 180/np.pi

plt.plot(wdSeq_1, zSeq_1, label='sowfa', linewidth=1.0, linestyle='-', color='k')
plt.plot(wdSeq_4, zSeq_4[1:], label='palm', linewidth=1.0, linestyle='--', color='k')
# plt.axhline(y=hubH, ls='--', c='black')
# plt.axhline(y=dampH, ls=':', c='black')
plt.xlabel(r"wind direction ($\degree$)", fontsize=20)
plt.ylabel('z (m)', fontsize=20)
xaxis_min = 260
xaxis_max = 290
xaxis_d = 10
yaxis_min = 0
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
saveName = 'fig1c_wd_pr.png'
plt.savefig('/scratch/projects/deepwind/photo/review' + '/' + saveName)
plt.show()
plt.close()