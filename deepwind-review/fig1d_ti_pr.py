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
tSeq_1, zSeq_1, TIuSeq_1, TIvSeq_1, TIwSeq_1 = TI_pr_sowfa(dir_1, ((0,0,0),30.0))
t_seq_1, TIuSeq_1, TIvSeq_1, TIwSeq_1 = TI_pr_ave((3600,151200,151200,1e6), tSeq_1, tSeq_1[-1]-tSeq_1[-2], zSeq_1.size, TIuSeq_1, TIvSeq_1, TIwSeq_1)


prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName_4  = 'deepwind_gs5_main'
dir_4 = prjDir + '/' + jobName_4
tSeq_4, zSeq_4, TIuSeq_4, TIvSeq_4, TIwSeq_4 = TI_pr_palm(dir_4, jobName_4, ['.001'])
TIuSeq_4, TIvSeq_4, TIwSeq_4 = TIuSeq_4[-1], TIvSeq_4[-1], TIwSeq_4[-1]


fig, ax = plt.subplots(figsize=(6,6))
fltw = 1
plt.plot(funcs.flt_seq(TIuSeq_1[-1],fltw), zSeq_1, label=r'$TI_x$-sowfa', linewidth=1.0, linestyle='-', color='r')
plt.plot(funcs.flt_seq(TIvSeq_1[-1],fltw), zSeq_1, label=r'$TI_y$-sowfa', linewidth=1.0, linestyle='-', color='b')
plt.plot(funcs.flt_seq(TIwSeq_1[-1,::3],fltw), zSeq_1[::3], label=r'$TI_z$-sowfa', linewidth=1.0, linestyle='-', color='g')

plt.plot(TIuSeq_4, zSeq_4, label=r'$TI_x$-palm', linewidth=1.0, linestyle='--', color='r')
plt.plot(TIvSeq_4, zSeq_4, label=r'$TI_y$-palm', linewidth=1.0, linestyle='--', color='b')
plt.plot(TIwSeq_4, zSeq_4, label=r'$TI_z$-palm', linewidth=1.0, linestyle='--', color='g')

plt.xlabel('TI (%)', fontsize=20)
plt.ylabel('z (m)', fontsize=20)
xaxis_min = 0
xaxis_max = 12
xaxis_d = 2
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=20)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=20)
plt.legend(bbox_to_anchor=(0.4,0.64), loc=6, borderaxespad=0, fontsize=20) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'fig1d_ti_pr.png'
plt.savefig('/scratch/projects/deepwind/photo/review' + '/' + saveName)
plt.show()
plt.close()