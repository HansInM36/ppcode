import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/sowfa/src')
import imp
import numpy as np
import pickle
from scipy.interpolate import interp1d
import funcs
import matplotlib.pyplot as plt


def velo_ave(dir, trs_para, varD):
    """ extract horizontal average of velocity at various times and heights """
    # coordinate transmation
    O = trs_para[0]
    alpha = trs_para[1]

    fr = open(dir + '/data/' + 'aveData', 'rb')
    aveData = pickle.load(fr)
    fr.close()

    zSeq = aveData['H']
    zNum = zSeq.size

    tSeq = aveData['time']
    tNum = tSeq.size
    tDelta = tSeq[1] - tSeq[0]

    uSeq = aveData['U_mean']
    vSeq = aveData['V_mean']
    wSeq = aveData['W_mean']

    varSeq = np.zeros((tNum,zNum))
    for zInd in range(zNum):
        tmp = np.concatenate((uSeq[:,zInd].reshape(tNum,1), vSeq[:,zInd].reshape(tNum,1)), axis=1)
        tmp = np.concatenate((tmp, wSeq[:,zInd].reshape(tNum,1)), axis=1)
        tmp_ = funcs.trs(tmp,O,alpha)
        varSeq[:,zInd] = tmp_[:,varD]

    return tSeq, tNum, tDelta, zSeq, zNum, varSeq

def velo_pr(tplot_para, tSeq, tDelta, zNum, varSeq):

    ave_itv = tplot_para[0]
    tplot_start = tplot_para[1]
    tplot_end = tplot_para[2]
    tplot_delta = tplot_para[3]
    tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
    tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

    # compute the averaged velocity at a certain time and height
    varplotList = []
    for tplot in tplotList:
        varplot = np.zeros(zNum)
        for zind in range(zNum):
            f = interp1d(tSeq, varSeq[:,zind], kind='linear', fill_value='extrapolate')
            tplot_tmp = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
            varplot[zind] = f(tplot_tmp).mean()
        varplotList.append(varplot)

    return tplotList, tplotNum, varplotList


prjName = 'deepwind'

### plot history of velo_pr of one single case

jobName_0 = 'gs10'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, tNum_0, tDelta_0, zSeq_0, zNum_0, varSeq_0 =  velo_ave(ppDir_0, ((0,0,0),30.0), 0)
tplotList_0, tplotNum_0, varplotList_0 = velo_pr((3600.0, 3600.0*1, 3600.0*1*12, 3600.0*1), tSeq_0, tDelta_0, zNum_0, varSeq_0)

jobName_1 = 'gs10_stdsgs'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, tNum_1, tDelta_1, zSeq_1, zNum_1, varSeq_1 =  velo_ave(ppDir_1, ((0,0,0),30.0), 0)
tplotList_1, tplotNum_1, varplotList_1 = velo_pr((3600.0, 3600.0*1, 3600.0*1*12, 3600.0*1), tSeq_1, tDelta_1, zNum_1, varSeq_1)

tplotList = tplotList_0
tplotNum = tplotNum_0
varplotList = varplotList_0
zSeq = zSeq_0
varSeq = varSeq_0
ppDir = ppDir_0

fig, ax = plt.subplots(figsize=(4,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))
# # initial profile
# plt.plot(varSeq[0,:], zSeq, label='t = 0s', linewidth=1.0, color='k')
for i in range(tplotNum):
    # # add the data of ground as 0
    # zero = np.zeros(1)
    # v_ = np.concatenate((zero, varplotList[i]))
    # z_ = np.concatenate((zero, zSeq))
    # plt.plot(v_, z_, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
    plt.plot(varplotList[i], zSeq, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
# plt.axhline(y=hubH, ls='--', c='black')
plt.xlabel(r'$\overline{\mathrm{u}}$' + ' (' + 'm/s' + ')')
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 12
xaxis_d = 2.0
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(0.05,0.7), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'u' + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()


### investigate the evolution of averaged velocities at various heights
jobName_0 = 'gs10'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, tNum_0, tDelta_0, zSeq_0, zNum_0, varSeq_0 =  velo_ave(ppDir_0, ((0,0,0),30.0), 0)
tplotList_0, tplotNum_0, varplotList_0 = velo_pr((3600.0, 3600.0*1, 3600.0*1*12, 3600.0*1), tSeq_0, tDelta_0, zNum_0, varSeq_0)
jobName_1 = 'gs10_stdsgs'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, tNum_1, tDelta_1, zSeq_1, zNum_1, varSeq_1 =  velo_ave(ppDir_1, ((0,0,0),30.0), 0)
tplotList_1, tplotNum_1, varplotList_1 = velo_pr((3600.0, 3600.0*1, 3600.0*1*12, 3600.0*1), tSeq_1, tDelta_1, zNum_1, varSeq_1)

tplotList = tplotList_1
varplotList = varplotList_1
ppDir = ppDir_1

zList = [20.0, 40.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]
zIndNum = len(zList)
uList = []
for z in zList:
    u_tmp = []
    for tInd in range(tplotNum):
        f = interp1d(zSeq, varplotList[tInd], kind='linear', fill_value='extrapolate')
        u_tmp.append(f(z))
    uList.append(u_tmp)

fig, ax = plt.subplots(figsize=(6,4))
colors = plt.cm.jet(np.linspace(0,1,zIndNum))
for i in range(zIndNum):
    ax.plot(np.array(tplotList)/3600, uList[i], linewidth=1.0, linestyle='-', marker='', color=colors[i], label='H = ' + str(zList[i]) + 'm')
ax.set_xlabel('t (h)')
ax.set_ylabel(r'$\overline{\mathrm{u}}$' + ' (m/s)')
xaxis_min = 0
xaxis_max = 12
xaxis_d = 1
yaxis_min = 5.0
yaxis_max = 10.0
yaxis_d = 0.5
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0)
plt.grid()
fig.tight_layout() # adjust the layout
saveName = 'velo_av_evo.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()

# # get variance of averaged velocity at various heights
# for i in range(zIndNum):
#     print(np.var(np.array(uList[i])))



### plot of velo_pr of multi cases

### investigate the evolution of averaged velocities at various heights
jobName_0 = 'gs10'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, tNum_0, tDelta_0, zSeq_0, zNum_0, varSeq_0 =  velo_ave(ppDir_0, ((0,0,0),30.0), 0)
tplotList_0, tplotNum_0, varplotList_0 = velo_pr((3600.0, 3600.0*1, 3600.0*1*12, 3600.0*1), tSeq_0, tDelta_0, zNum_0, varSeq_0)
jobName_1 = 'gs10_stdsgs'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, tNum_1, tDelta_1, zSeq_1, zNum_1, varSeq_1 =  velo_ave(ppDir_1, ((0,0,0),30.0), 0)
tplotList_1, tplotNum_1, varplotList_1 = velo_pr((3600.0, 3600.0*1, 3600.0*1*12, 3600.0*1), tSeq_1, tDelta_1, zNum_1, varSeq_1)

fig, ax = plt.subplots(figsize=(4,6))
# # add the data of ground as 0
# zero = np.zeros(1)
# v_ = np.concatenate((zero, varplotList[i]))
# z_ = np.concatenate((zero, zSeq))
# plt.plot(v_, z_, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
plt.plot(varplotList_0[tplotNum_0-1], zSeq_0, label='oneEquation', linewidth=1.0, color='r')
plt.plot(varplotList_1[tplotNum_1-1], zSeq_1, label='standard', linewidth=1.0, color='b')
# plt.axhline(y=hubH, ls='--', c='black')
plt.xlabel(r'$\overline{\mathrm{u}}$' + ' (' + 'm/s' + ')')
plt.ylabel('z (m)')
xaxis_min = 0
xaxis_max = 12
xaxis_d = 2.0
yaxis_min = 0
yaxis_max = 1000.0
yaxis_d = 100.0
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(0.05,0.9), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'u' + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()

















### plot non-dimensional u gradient profile
startH = 0.001
topH = 200.0
zNum_ = 21
uStar = 0.4
kappa = 0.4

fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))
for i in range(tplotNum):
    zero = np.zeros(1)
    v_ = np.concatenate((zero, varplotList[i]))
    z_ = np.concatenate((zero, zSeq))
    f = interp1d(z_, v_, kind='linear', fill_value='extrapolate')
    z_ = np.linspace(startH,topH,zNum_)
    dz = (topH - startH) / (zNum_-1)
    v_ = funcs.calc_deriv_1st_FD(dz, f(z_))
    v_ = v_ * kappa * z_ / uStar
    plt.plot(v_, z_, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
plt.xlabel(r"$\mathrm{\phi_m}$")
plt.ylabel('z (m)')
xaxis_min = -3
xaxis_max = 5
xaxis_d = 2
yaxis_min = 0
yaxis_max = 200.0
yaxis_d = 20.0
plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'phi_m' + '_pr.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()
