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

# coordinate transmation
O = (0,0,0)
alpha = 30.0

varName = "u-component vertical momentum flux"
varD = 0 # 0:uw, 1:vw, 2:ww
varName_save = 'cfl_uw_flux'
varUnit = r'$m^2/s^2$'

hubH = 90.0

fr = open(ppDir + '/data/' + 'aveData', 'rb')
aveData = pickle.load(fr)
fr.close()

zSeq = aveData['H']
zNum = zSeq.size

tSeq = aveData['time']
tNum = tSeq.size
tDelta = tSeq[1] - tSeq[0]

uwSeq = aveData['uw_mean']
vwSeq = aveData['vw_mean']
wwSeq = aveData['ww_mean']
R13Seq = aveData['R13_mean']
R23Seq = aveData['R23_mean']
R33Seq = aveData['R33_mean']

rsvSeq = np.zeros((tNum,zNum))
sgsSeq = np.zeros((tNum,zNum))

for zInd in range(zNum):
    tmp = np.concatenate((uwSeq[:,zInd].reshape(tNum,1), vwSeq[:,zInd].reshape(tNum,1)), axis=1)
    tmp = np.concatenate((tmp, wwSeq[:,zInd].reshape(tNum,1)), axis=1)
    tmp_ = funcs.trs(tmp,O,alpha)
    rsvSeq[:,zInd] = tmp_[:,varD]

    tmp = np.concatenate((R13Seq[:,zInd].reshape(tNum,1), R23Seq[:,zInd].reshape(tNum,1)), axis=1)
    tmp = np.concatenate((tmp, R33Seq[:,zInd].reshape(tNum,1)), axis=1)
    tmp_ = funcs.trs(tmp,O,alpha)
    sgsSeq[:,zInd] = tmp_[:,varD]




### plot
ave_itv = 3600.0 # by default, the averaging interval is 3600s
tplot = 432000.0

rsvplot = np.zeros(zNum)
sgsplot = np.zeros(zNum)
for zInd in range(zNum):
    f = interp1d(tSeq, rsvSeq[:,zInd], kind='cubic', fill_value='extrapolate')
    tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
    rsvplot[zInd] = f(tplotSeq).mean()

    f = interp1d(tSeq, sgsSeq[:,zInd], kind='cubic', fill_value='extrapolate')
    tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
    sgsplot[zInd] = f(tplotSeq).mean()

cflInd = 9
fig, ax = plt.subplots(figsize=(6,6))
plt.plot(rsvplot[:cflInd], zSeq[:cflInd], label='resolved', linestyle='--', linewidth=1.0, color='r')
plt.plot(sgsplot[:cflInd], zSeq[:cflInd], label='SGS', linestyle=':', linewidth=1.0, color='b')
plt.plot(rsvplot[:cflInd]+sgsplot[:cflInd], zSeq[:cflInd], label='total', linestyle='-', linewidth=1.0, color='k')
plt.axhline(y=hubH, ls='--', c='black')
plt.xlabel(varName + ' (' + varUnit + ')')
plt.ylabel('z (m)')
xaxis_min = -0.1
xaxis_max = 0.02
xaxis_d = 0.02
yaxis_min = 0
yaxis_max = 180.0
yaxis_d = 20.0
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
