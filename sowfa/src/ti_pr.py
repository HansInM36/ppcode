import sys
sys.path.append('/scratch/ppcode/sowfa/src')
import imp
import numpy as np
import pickle
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
jobName = 'pcr_NBL_U10'
ppDir = '/scratch/sowfadata/pp/' + jobName + '/data'

varname = r'$TI_u$'
varunit = '%'
varName_save = 'TIu'

fr = open(ppDir + '/' + 'aveData', 'rb')
aveData = pickle.load(fr)
fr.close()

zSeq = aveData['H']

tSeq = aveData['time']
tNum = tSeq.size
tDelta = tSeq[1] - tSeq[0]

uuSeq = aveData['uu_mean']
uSeq = aveData['U_mean']

varSeq = 100 * np.power(uuSeq,0.5) / uSeq

### plot
ave_itv = 3600.0 # by default, the averaging interval is 3600s

tplot_start = 3600.0*8
tplot_end = 432000.0
tplot_delta = 3600.0*8

tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

varplotList = []
for tplot in tplotList:
    varplot = np.zeros(zNum)
    for zInd in range(zNum):
        f = interp1d(tSeq, varSeq[:,zInd], kind='cubic')
        tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
        varplot[zInd] = f(tplotSeq).mean()
    varplotList.append(varplot)

fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))

for i in range(tplotNum):
    plt.plot(varplotList[i], zSeq, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
plt.axhline(y=102, ls='--', c='black')
plt.xlabel(varname + ' (' + varunit + ')')
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 10
xaxis_d = 2
yaxis_min = 0
yaxis_max = 800.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveDir = '/scratch/sowfadata/pp/' + jobName + '/'
saveName = varName_save + '_pr.png'
plt.savefig(saveDir + saveName)
plt.show()
