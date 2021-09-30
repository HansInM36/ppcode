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
import matplotlib.pyplot as plt


""" SOWFA """
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'

jobName = 'gs10_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName
tSeq_0, zSeq_0, rsvSeq_0, sgsSeq_0, totSeq_0 = TKE_sowfa(ppDir_0, ((0,0,0),30), 0)
rsvSeq_0 = TKE_av_sowfa(rsvSeq_0, tSeq_0, zSeq_0.size, (3600.0,151200.0))
sgsSeq_0 = TKE_av_sowfa(sgsSeq_0, tSeq_0, zSeq_0.size, (3600.0,151200.0))
totSeq_0 = TKE_av_sowfa(totSeq_0, tSeq_0, zSeq_0.size, (3600.0,151200.0))


""" PALM """
prjDir = '/scratch/palmdata/JOBS/Deepwind'

jobName  = 'deepwind_gs5'
dir = prjDir + '/' + jobName
tSeq_4, zSeq_4, rsvSeq_4, sgsSeq_4, totSeq_4 = TKE_palm(dir, jobName, ['.010','.011'])
rsvSeq_4 = rsvSeq_4[-1]
sgsSeq_4 = sgsSeq_4[-1]
totSeq_4 = totSeq_4[-1]

""" TKE group plot """
zi = 700
fig = plt.figure()
fig.set_figwidth(6)
fig.set_figheight(6)
rNum, cNum = (1,2)
axs = fig.subplots(nrows=rNum, ncols=cNum)

axs[0].plot(rsvSeq_0[0::3], zSeq_0[0::3]/zi, label='sowfa-rsv', marker='', markersize=1, linestyle='--', linewidth=1.0, color='r')
axs[0].plot(sgsSeq_0[0::3], zSeq_0[0::3]/zi, label='sowfa-sgs', marker='', markersize=1, linestyle=':', linewidth=1.0, color='r')
axs[0].plot(totSeq_0[0::3], zSeq_0[0::3]/zi, label='sowfa-tot', marker='', markersize=1, linestyle='-', linewidth=1.0, color='r')
axs[0].plot(rsvSeq_4, zSeq_4/zi, label='palm-rsv', marker='', markersize=1, linestyle='--', linewidth=1.0, color='b')
axs[0].plot(sgsSeq_4, zSeq_4/zi, label='palm-sgs', marker='', markersize=1, linestyle=':', linewidth=1.0, color='b')
axs[0].plot(totSeq_4, zSeq_4/zi, label='palm-tot', marker='', markersize=1, linestyle='-', linewidth=1.0, color='b')
#axs[0].set_xlim(0.0,0.5)
axs[0].set_ylim(0.0,1.0)
#axs[0].set_xticklabels([0.0,0.2,0.4],fontsize=20)
for tick in axs[0].xaxis.get_major_ticks():
                tick.label.set_fontsize(20)
axs[0].set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=20)
axs[0].set_xlabel(r'$\mathrm{e}$ $(\mathrm{m^2/s^2})$', fontsize=20)
axs[0].set_ylabel(r'$\mathrm{z_i}$', fontsize=20)
axs[0].grid()
# axs[0].legend(loc='upper right', bbox_to_anchor=(0.9,0.9), ncol=1, mode='None', borderaxespad=0, fontsize=12)

axs[1].plot(funcs.flt_seq(rsvSeq_0[0::3]/totSeq_0[0::3]*100,0), zSeq_0[0::3]/zi, label='sowfa', marker='', markersize=1, linestyle='-', linewidth=1.0, color='r')
axs[1].plot(rsvSeq_4/totSeq_4*100, zSeq_4/zi, label='palm', marker='', markersize=1, linestyle='-', linewidth=1.0, color='b')
axs[1].set_xlim(60.0,100.0)
axs[1].set_ylim(0.0,1.0); axs[1].set_yticklabels([])
axs[1].set_xticklabels([60,70,80,90,100],fontsize=20)
axs[1].set_xlabel(r'$\mathrm{e_{rsv}/e_{tot}}$ (%)', fontsize=20)
axs[1].grid()
# axs[1].legend(loc='upper left', bbox_to_anchor=(0.1,0.9), ncol=1, mode='None', borderaxespad=0, fontsize=12)

handles, labels = axs[0].get_legend_handles_labels()
lgdord = [0,3,1,4,2,5]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,0.86), ncol=1, mode='None', borderaxespad=0, fontsize=18)

saveDir = '/scratch/projects/deepwind/photo/review'
saveName = 'fig4_TKE.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()