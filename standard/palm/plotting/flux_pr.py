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
tSeq_0, zSeq_0, wutotSeq_0 = velo_pr_palm(jobDir_0, jobName_0, ['.000'], 'wu')
#tSeq_0, zSeq_0, wursvSeq_0 = velo_pr_palm(jobDir_0, jobName_0, ['.000'], 'w*u*')
#tSeq_0, zSeq_0, wusgsSeq_0 = velo_pr_palm(jobDir_0, jobName_0, ['.000'], 'w"u"')
#tSeq_0, zSeq_0, wvtotSeq_0 = velo_pr_palm(jobDir_0, jobName_0, ['.000'], 'wv')
#ustar_0 = np.power(np.power(wutotSeq_0[-1,0],2) + np.power(wvtotSeq_0[-1,0],2),0.25)


jobName_1  = 'sigma_test'
jobDir_1 = prjDir + '/' + jobName_1
tSeq_1, zSeq_1, wutotSeq_1 = velo_pr_palm(jobDir_1, jobName_1, ['.000'], 'wu')
#tSeq_1, zSeq_1, wvtotSeq_1 = velo_pr_palm(jobDir_1, jobName_1, ['.000'], 'wv')
#ustar_1 = np.power(np.power(wutotSeq_1[-1,0],2) + np.power(wvtotSeq_1[-1,0],2),0.25)


""" u profile of stationary flow (last time step) """
fig, ax = plt.subplots(figsize=(4.5,6))
tInd = -1
plt.plot(wutotSeq_0[tInd], zSeq_0/zSeq_0[-1], label='ref-tot', linewidth=1.0, marker='', linestyle='-', color='k')
#plt.plot(wursvSeq_0[tInd], zSeq_0/zSeq_0[-1], label='rsv', linewidth=1.0, marker='', linestyle='--', color='r')
#plt.plot(wusgsSeq_0[tInd], zSeq_0/zSeq_0[-1], label='sgs', linewidth=1.0, marker='', linestyle=':', color='b')
#plt.plot(wursvSeq_0[tInd]+wusgsSeq_0[tInd], zSeq_0/zSeq_0[-1], label='rsv+sgs', linewidth=1.0, marker='o', linestyle='', color='k')
plt.plot(wutotSeq_1[tInd], zSeq_1/zSeq_1[-1], label='test-tot', linewidth=1.0, marker='', linestyle='-', color='r')
plt.xlabel(r"momentum flux $(\mathrm{m^2/s^2})$", fontsize=12)
plt.ylabel('z/H', fontsize=12)
xaxis_min = -0.1
xaxis_max = 0.02
xaxis_d = 0.02
yaxis_min = 0.0
yaxis_max = 1.0
yaxis_d = 0.1
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
plt.legend(bbox_to_anchor=(0.05,0.86), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'flux' + '_pr.png'
saveDir = '/scratch/prjdata/sigma_imp'
#plt.savefig(saveDir + '/' + saveName)
plt.show()
plt.close()