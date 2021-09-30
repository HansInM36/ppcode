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
from scipy.interpolate import interp1d
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


""" dimensionless u gradient profile of stationary flow (sowfa vs palm) """
startH = 5.000
topH = 205.0
zNum_ = 21
kappa = 0.4
uStar_1 = kappa / np.log(zSeq_1[0]/0.001) * np.power(uSeq_1[-1][0]**2 + vSeq_1[-1][0]**2,0.5)
uStar_4 = kappa / np.log(zSeq_4[1]/0.001) * np.power(uSeq_4[-1][1]**2 + vSeq_4[-1][1]**2,0.5)

fig, ax = plt.subplots(figsize=(6,6))
z_ = np.linspace(startH,topH,zNum_)
dz = (topH - startH) / (zNum_-1)
# sowfa
zero = np.zeros(1)
v_1 = np.concatenate((zero, uSeq_1[-1]))
z_1 = np.concatenate((zero, zSeq_1))
f_1 = interp1d(z_1, v_1, kind='linear', fill_value='extrapolate')
v_1 = funcs.calc_deriv_1st_FD(dz, f_1(z_))
v_1 = v_1 * kappa * z_ / uStar_1
# palm
v_4 = uSeq_4[-1]
z_4 = zSeq_4
f_4 = interp1d(z_4, v_4, kind='linear', fill_value='extrapolate')
v_4 = funcs.calc_deriv_1st_FD(dz, f_4(z_))
v_4 = v_4 * kappa * z_ / uStar_4

plt.plot(v_1, z_, label='sowfa', linewidth=1.0, linestyle='-', color='k')
plt.plot(v_4, z_, label='palm', linewidth=1.0, linestyle='--', color='k')
plt.xlabel(r"$\mathrm{\phi_m}$", fontsize=20)
plt.ylabel('z (m)', fontsize=20)
xaxis_min = -3
xaxis_max = 5
xaxis_d = 2
yaxis_min = 0
yaxis_max = 200.0
yaxis_d = 20.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=20)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=20)
plt.legend(bbox_to_anchor=(0.05,0.86), loc=6, borderaxespad=0, fontsize=20) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'fig1b_phim_pr.png'
plt.savefig('/scratch/projects/deepwind/photo/review' + '/' + saveName)
plt.show()
plt.close()