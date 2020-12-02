import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/sowfa/src')
import imp
import numpy as np
import pickle
from scipy.interpolate import interp1d
import funcs
import matplotlib.pyplot as plt


# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs20'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName

varname = r'$TI_u$'
varunit = '%'
varName_save = 'TI_u'

hubH = 90.0

# coordinate transmation
O = (0,0,0)
alpha = 30.0

alpha = np.pi/180 * alpha

fr = open(ppDir + '/data/' + 'aveData', 'rb')
aveData = pickle.load(fr)
fr.close()

zSeq = aveData['H']
zNum = zSeq.size

tSeq = aveData['time']
tNum = tSeq.size
tDelta = tSeq[1] - tSeq[0]

uuSeq = aveData['uu_mean']
vvSeq = aveData['vv_mean']
uvSeq = aveData['uv_mean']
wwSeq = aveData['ww_mean']
uSeq = aveData['U_mean']
vSeq = aveData['V_mean']

varianceSeq = uuSeq*np.power(np.cos(alpha),2) + 2*uvSeq*np.cos(alpha)*np.sin(alpha) + vvSeq*np.power(np.sin(alpha),2)
umeanSeq = uSeq*np.cos(alpha) + vSeq*np.sin(alpha)
varSeq = 100 * np.power(varianceSeq ,0.5) / umeanSeq

### plot
ave_itv = 3600.0 # by default, the averaging interval is 3600s

tplot_start = 3600.0*6 # must larger than ave_itv
tplot_end = 3600.0*6*20
tplot_delta = 3600.0*6

tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

varplotList = []
for tplot in tplotList:
    varplot = np.zeros(zNum)
    for zInd in range(zNum):
        f = interp1d(tSeq, varSeq[:,zInd], kind='cubic', fill_value='extrapolate')
        tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
        varplot[zInd] = f(tplotSeq).mean()
    varplotList.append(varplot)

fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))

for i in range(tplotNum):
    plt.plot(varplotList[i], zSeq, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
plt.axhline(y=hubH, ls='--', c='black')
plt.xlabel(varname + ' (' + varunit + ')')
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 12
xaxis_d = 2
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '_pr.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()


### TI in 3 dimensions at certain timestep
ave_itv = 3600.0
tplot = 432000.0

uvarianceSeq = uuSeq*np.power(np.cos(alpha),2) + 2*uvSeq*np.cos(alpha)*np.sin(alpha) + vvSeq*np.power(np.sin(alpha),2)
vvarianceSeq = uuSeq*np.power(np.sin(alpha),2) + 2*uvSeq*np.sin(alpha)*np.cos(alpha) + vvSeq*np.power(np.cos(alpha),2)
wvarianceSeq = wwSeq
umeanSeq = uSeq*np.cos(alpha) + vSeq*np.sin(alpha)
TIuSeq = 100 * np.power(uvarianceSeq ,0.5) / umeanSeq
TIvSeq = 100 * np.power(vvarianceSeq ,0.5) / umeanSeq
TIwSeq = 100 * np.power(wvarianceSeq ,0.5) / umeanSeq

TIuplot = np.zeros(zNum)
TIvplot = np.zeros(zNum)
TIwplot = np.zeros(zNum)
for zInd in range(zNum):
    f = interp1d(tSeq, TIuSeq[:,zInd], kind='cubic', fill_value='extrapolate')
    tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
    TIuplot[zInd] = f(tplotSeq).mean()

    f = interp1d(tSeq, TIvSeq[:,zInd], kind='cubic', fill_value='extrapolate')
    tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
    TIvplot[zInd] = f(tplotSeq).mean()

    f = interp1d(tSeq, TIwSeq[:,zInd], kind='cubic', fill_value='extrapolate')
    tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
    TIwplot[zInd] = f(tplotSeq).mean()


fig, ax = plt.subplots(figsize=(6,6))
plt.plot(TIuplot, zSeq, label='TIu', linewidth=1.0, color='r')
plt.plot(TIvplot, zSeq, label='TIv', linewidth=1.0, color='b')
plt.plot(TIwplot, zSeq, label='TIw', linewidth=1.0, color='g')
plt.axhline(y=hubH, ls='--', c='black')
plt.xlabel('TI' + ' (' + varunit + ')')
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 12
xaxis_d = 2
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(0.8,0.9), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'TI_uvw' + '_pr.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()
