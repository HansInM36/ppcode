import os
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/standard')
sys.path.append('/scratch/ppcode/standard/palm_std')
sys.path.append('/scratch/ppcode/standard/sowfa_std')
import imp
import palm_data_ext
from palm_data_ext import *
import sowfa_data_ext_L2
from sowfa_data_ext_L2 import *
import numpy as np
import funcs
import matplotlib.pyplot as plt


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0, coors_0 = PSD_data_sowfa(ppDir_0, 'prbg1', ((0,0,0),30.0), 'U', 0)
# averaged coherence
freq_0, coh_av_0, coh_std_0, co_coh_av_0, co_coh_std_0, phase_av_0, phase_std_0 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.1), tSeq_0, varSeq_0, 240)
freq_1, coh_av_1, coh_std_1, co_coh_av_1, co_coh_std_1, phase_av_1, phase_std_1 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.2), tSeq_0, varSeq_0, 240)
freq_2, coh_av_2, coh_std_2, co_coh_av_2, co_coh_std_2, phase_av_2, phase_std_2 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.5), tSeq_0, varSeq_0, 240)
freq_3, coh_av_3, coh_std_3, co_coh_av_3, co_coh_std_3, phase_av_3, phase_std_3 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 1.0), tSeq_0, varSeq_0, 240)


# plot
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(freq_0[0:], co_coh_av_0[0:], linestyle='-', marker='', markersize=1, color='r', label=r'$f_s$=10Hz')
ax.plot(freq_1[0:], co_coh_av_1[0:], linestyle='-', marker='', markersize=1, color='b', label=r'$f_s$=5Hz')
ax.plot(freq_2[0:], co_coh_av_2[0:], linestyle='-', marker='', markersize=1, color='g', label=r'$f_s$=2Hz')
ax.plot(freq_3[0:], co_coh_av_3[0:], linestyle='-', marker='', markersize=1, color='orange', label=r'$f_s$=1Hz')
ax.fill_between(freq_0[0:], co_coh_av_0[0:]-co_coh_std_0[0:], co_coh_av_0[0:]+co_coh_std_0[0:], color='salmon', alpha=0.8)
ax.fill_between(freq_1[0:], co_coh_av_1[0:]-co_coh_std_1[0:], co_coh_av_1[0:]+co_coh_std_1[0:], color='lightskyblue', alpha=0.6)
ax.fill_between(freq_2[0:], co_coh_av_2[0:]-co_coh_std_2[0:], co_coh_av_2[0:]+co_coh_std_2[0:], color='lightgreen', alpha=0.5)
ax.fill_between(freq_3[0:], co_coh_av_3[0:]-co_coh_std_3[0:], co_coh_av_3[0:]+co_coh_std_3[0:], color='gold', alpha=0.4)
# ax.errorbar(freq_0[0:], co_coh_av_0[0:], yerr=co_coh_std_0[0:], linestyle='-', fmt='-o', capsize=3, color='r', label='fs=10')
# ax.errorbar(freq_1[0:], co_coh_av_1[0:], yerr=co_coh_std_1[0:], linestyle='-', fmt='-o', capsize=3, color='b', label='fs=5')
# ax.errorbar(freq_2[0:], co_coh_av_2[0:], yerr=co_coh_std_2[0:], linestyle='-', fmt='-o', capsize=3, color='g', label='fs=2')
# ax.errorbar(freq_3[0:], co_coh_av_3[0:], yerr=co_coh_std_3[0:], linestyle='-', fmt='-o', capsize=3, color='y', label='fs=1')
ax.set_xlabel('f (1/s)', fontsize=18)
ax.set_ylabel('co-coh', fontsize=18)
xaxis_min = 0
xaxis_max = 0.5
xaxis_d = 0.1
yaxis_min = -0.4
yaxis_max = 1.0
yaxis_d = 0.2
ax.set_ylim(yaxis_min, yaxis_max)
ax.set_xlim(xaxis_min, xaxis_max)
ax.set_xticks(list(np.linspace(xaxis_min, xaxis_max, int(np.round((xaxis_max-xaxis_min)/xaxis_d)+1))))
ax.set_yticks(list(np.linspace(yaxis_min, yaxis_max, int(np.round((yaxis_max-yaxis_min)/yaxis_d)+1))))
for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(18)
plt.legend(bbox_to_anchor=(0.06,0.8), loc=6, borderaxespad=0, ncol=2, fontsize=16) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
ax.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveDir = '/scratch/projects/deepwind/photo/review'
saveName = 'fig6b_coh_cmp_fs.png'
plt.savefig(saveDir + '/' + saveName)
plt.show()
plt.close()