import os
import sys
sys.path.append('/scratch/ppcode/standard')
sys.path.append('/scratch/ppcode/standard/palm_std')
sys.path.append('/scratch/ppcode/standard/sowfa_std')
import imp
import palm_data_ext
from palm_data_ext import *
import sowfa_data_ext_L2
from sowfa_data_ext_L2 import *
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

segNum = 4096

prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, uSeq_0, coors_0 = PSD_data_sowfa(ppDir_0, 'prbg0', ((0,0,0),30.0), 'U', 0)
segNum = int(0.4*tSeq_0.size)
f_seq_0, PSD_u_seq_0_100 = PSD_sowfa((144000.0, 146400.0, 0.1), tSeq_0, 4, xSeq_0.size, ySeq_0.size, uSeq_0, segNum)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_1 = 'gs10'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, xSeq_1, ySeq_1, zSeq_1, uSeq_1, coors_1 = PSD_data_sowfa(ppDir_1, 'prbg0', ((0,0,0),30.0), 'U', 0)
segNum = int(0.4*tSeq_1.size)
f_seq_1, PSD_u_seq_1_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_1, 4, xSeq_1.size, ySeq_1.size, uSeq_1, segNum)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_2 = 'gs20'
ppDir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_2
tSeq_2, xSeq_2, ySeq_2, zSeq_2, uSeq_2, coors_2 = PSD_data_sowfa(ppDir_2, 'prbg0', ((0,0,0),30.0), 'U', 0)
segNum = int(0.4*tSeq_2.size)
f_seq_2, PSD_u_seq_2_100 = PSD_sowfa((432000.0, 434400, 0.1), tSeq_2, 4, xSeq_2.size, ySeq_2.size, uSeq_2, segNum)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_3 = 'gs40'
ppDir_3 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_3
tSeq_3, xSeq_3, ySeq_3, zSeq_3, uSeq_3, coors_3 = PSD_data_sowfa(ppDir_3, 'prbg0', ((0,0,0),30.0), 'U', 0)
segNum = int(0.4*tSeq_3.size)
f_seq_3, PSD_u_seq_3_100 = PSD_sowfa((144000.0, 146400, 0.1), tSeq_3, 4, xSeq_3.size, ySeq_3.size, uSeq_3, segNum)


prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName  = 'deepwind_gs5_main'
dir_4 = prjDir + '/' + jobName
tSeq_4, xSeq_4, ySeq_4, zSeq_4, uSeq_4 = PSD_data_palm(dir_4, jobName, 'M03', ['.001'], 'u')
f_seq_4, PSD_u_seq_4_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_4, 4, xSeq_4.size, ySeq_4.size, uSeq_4, segNum)


prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName  = 'deepwind_gs10_main'
dir_5 = prjDir + '/' + jobName
tSeq_5, xSeq_5, ySeq_5, zSeq_5, uSeq_5 = PSD_data_palm(dir_5, jobName, 'M03', ['.006','.007'], 'u')
f_seq_5, PSD_u_seq_5_100 = PSD_palm((75800.0, 77800.0, 0.1), tSeq_5, 4, xSeq_5.size, ySeq_5.size, uSeq_5, segNum)


prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName  = 'deepwind_gs20_main'
dir_6 = prjDir + '/' + jobName
tSeq_6, xSeq_6, ySeq_6, zSeq_6, uSeq_6 = PSD_data_palm(dir_6, jobName, 'M03', ['.001'], 'u')
f_seq_6, PSD_u_seq_6_100 = PSD_palm((3600.0, 10800.0, 0.5), tSeq_6, 4, xSeq_6.size, ySeq_6.size, uSeq_6, segNum)


prjDir = '/scratch/palmdata/JOBS/Deepwind'
jobName  = 'deepwind_gs40_main'
dir_7 = prjDir + '/' + jobName
tSeq_7, xSeq_7, ySeq_7, zSeq_7, uSeq_7 = PSD_data_palm(dir_7, jobName, 'M03', ['.004'], 'u')
f_seq_7, PSD_u_seq_7_100 = PSD_palm((25200.0, 27600.0, 0.1), tSeq_7, 2, xSeq_7.size, ySeq_7.size, uSeq_7, segNum)



def calc_uStar_sowfa(xNum,yNum,zSeq,uSeq,zMO_ind=0,kappa=0.4,z0=0.001):
    zMO = zSeq[zMO_ind]
    pInd_start = xNum*yNum*zMO_ind
    pInd_end = xNum*yNum*(zMO_ind+1)
    uMO = np.mean(uSeq[pInd_start:pInd_end])
    uStar = kappa * uMO / np.log(zMO/z0)
    return uStar
def calc_uz_sowfa(xNum,yNum,zInd,uSeq):
    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)
    uz = np.mean(uSeq[pInd_start:pInd_end])
    return uz
def calc_uStar_palm(zSeq,uSeq,zMO_ind=0,kappa=0.4,z0=0.001):
    zMO = zSeq[zMO_ind]
    uMO = np.mean(uSeq[:,zMO_ind,:,:])
    uStar = kappa * uMO / np.log(zMO/z0)
    return uStar
def calc_uz_palm(zInd,uSeq):
    uz = np.mean(uSeq[:,zInd,:,:])
    return uz


""" Su cmp gs """

fig, ax = plt.subplots(figsize=(6,6))
plt.loglog(f_seq_0, PSD_u_seq_0_100, label='sowfa-gs5', linewidth=1.0, linestyle='-', color=colors_sowfa[0])
plt.loglog(f_seq_1, PSD_u_seq_1_100, label='sowfa-gs10', linewidth=1.0, linestyle='-', color=colors_sowfa[1])
plt.loglog(f_seq_2, PSD_u_seq_2_100, label='sowfa-gs20', linewidth=1.0, linestyle='-', color=colors_sowfa[2])
plt.loglog(f_seq_3, PSD_u_seq_3_100, label='sowfa-gs40', linewidth=1.0, linestyle='-', color=colors_sowfa[3])
plt.loglog(f_seq_4, PSD_u_seq_4_100, label='palm-gs5', linewidth=1.0, linestyle='-', color=colors_palm[0])
plt.loglog(f_seq_4, PSD_u_seq_4_100, label='palm-gs10', linewidth=1.0, linestyle='-', color=colors_palm[1])
plt.loglog(f_seq_6, PSD_u_seq_6_100, label='palm-gs20', linewidth=1.0, linestyle='-', color=colors_palm[2])
plt.loglog(f_seq_7, PSD_u_seq_7_100, label='palm-gs40', linewidth=1.0, linestyle='-', color=colors_palm[3])


# plot -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e0*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')

plt.xlabel('f (1/s)', fontsize=20)
plt.ylabel(r'$\mathrm{S_u}$' + ' (' + r'$\mathrm{m^2/s}$' + ')', fontsize=20)
xaxis_min = 1e-3
xaxis_max = 1 # f_seq.max()
yaxis_min = 1e-12
yaxis_max = 1e4
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(bbox_to_anchor=(0.05,0.32), loc=6, borderaxespad=0, fontsize=16) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
# plt.legend(bbox_to_anchor=(1.02,0.5), loc=6, borderaxespad=0)
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'fig3_PSD_cmp_gs.png'
saveDir  = '/scratch/projects/deepwind/photo/review'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()