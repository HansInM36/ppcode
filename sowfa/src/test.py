import sys
sys.path.append('/scratch/ppcode/sowfa/src')
import imp
import numpy as np
import pickle
import sliceDataClass as sdc
import matplotlib.pyplot as plt


# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
jobName = 'pcr_NBL_U10'
ppDir = '/scratch/sowfadata/pp/' + jobName + '/data'


readDir = ppDir + '/'
readName = 'Ny0'
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()

test = sdc.Slice(data_org, 1)

plotdata = test.meshITP_Ny((0,2000,200), (0,960,48), test.data['U'][-1][:,0])


fig, axs = plt.subplots(figsize=(8,8))
cbreso = 100 # resolution of colorbar
CS = axs.contourf(plotdata[0], plotdata[1], plotdata[2], cbreso, cmap='coolwarm')
cbar = plt.colorbar(CS, ax=axs, shrink=1.0)
cbar.ax.set_ylabel('u (m/s)')
# xaxis_min = xseq
# xaxis_max = 2
# xaxis_d = 1
# yaxis_min = 0
# yaxis_max = 1000.0
# yaxis_d = 200
# plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
# plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
# plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.ylabel('y (m)')
plt.xlabel('x (m)')
# fig.tight_layout() # adjust the layout
# plt.title(jobname)
# saveDir = '/scratch/palmdata/pp/' + jobname + '/'
# saveName = varname + '_' + str(int(tseq[tind])) +'_contour_ny_' + str(int(yseq[yind])) + '.png'
# plt.savefig(saveDir + saveName)
plt.show()
