import os
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/standard')
sys.path.append('/scratch/ppcode/standard/sowfa/dataExt_L2')
import imp
import sowfa_data_ext_L2
from sowfa_data_ext_L2 import *
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'examples'
jobName = 'NBL'
jobDir = '/scratch/sowfadata/JOBS/' + prjName + '/' + jobName
tSeq, xSeq, ySeq, zSeq, uSeq, coors = PSD_data_sowfa(jobDir, 'prbg0', ((0,0,0),30.0), 'U', 0)
f_seq, PSD_u_seq = PSD_sowfa((18000.0, 20400.0, 0.5), tSeq, 4, xSeq.size, ySeq.size, uSeq, int(0.4*tSeq.size))


fig, ax = plt.subplots(figsize=(6,6))
plt.loglog(f_seq, PSD_u_seq, label='NBL', linewidth=1.0, linestyle='-', color='k')

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
#saveName = 'fig3_PSD_cmp_gs.png'
#saveDir  = '/scratch/projects/deepwind/photo/review'
#if not os.path.exists(saveDir):
#    os.makedirs(saveDir)
#plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()