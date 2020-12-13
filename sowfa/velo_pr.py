import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/sowfa/src')
import imp
import numpy as np
import pickle
from scipy.interpolate import interp1d
import funcs
import matplotlib.pyplot as plt

### inputs
# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs20'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName

# coordinate transmation
O = (0,0,0)
alpha = 30.0

# names and dimensions for variable
varName = r"$\mathrm{\overline{u}}$"
varName_save = 'u'
varD = 0 # 0:u, 1:v, 2:w
varUnit = 'm/s'


# hub height
hubH = 90.0

# damp height
dampH = 700.0

# time steps for plotting
ave_itv = 3600.0 # by default, the averaging interval is 3600s
                 # ind_back_num = np.floor(ave_itv / tDelta)
tplot_start = 3600.0*6 # must larger than ave_itv
tplot_end = 3600.0*6*20
tplot_delta = 3600.0*6
tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))
###


fr = open(ppDir + '/data/' + 'aveData', 'rb')
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


varplotList = []
for tplot in tplotList:
    varplot = np.zeros(zNum)
    for zind in range(zNum):
        f = interp1d(tSeq, varSeq[:,zind], kind='linear', fill_value='extrapolate')
        tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
        varplot[zind] = f(tplotSeq).mean()
    varplotList.append(varplot)

# varplotList = []
# for tplot in tplotList:
#     tind_start =  np.where(tSeq==(tplot-ave_itv))[0][0]
#     tind_plot = np.where(tSeq==tplot)[0][0]
#     tindList = list(range(tind_start,tind_plot + 1,1))
#     tindNum = len(tindList)
#     varplot = np.mean(varSeq[tind_start:tind_plot+1], axis=0)
#     varplotList.append(varplot)

fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))

# # initial profile
# plt.plot(varSeq[0,:], zSeq, label='t = 0s', linewidth=1.0, color='k')

for i in range(tplotNum):
    # zero = np.zeros(1)
    # v_ = np.concatenate((zero, varplotList[i]))
    # z_ = np.concatenate((zero, zSeq))
    # plt.plot(v_, z_, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
    plt.plot(varplotList[i], zSeq, label='t = ' + str(int(tplotList[i])) + 's', linewidth=1.0, color=colors[i])
# plt.axhline(y=hubH, ls='--', c='black')
plt.axhline(y=dampH, ls=':', c='black')
plt.xlabel(varName + ' (' + varUnit + ')')
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
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = varName_save + '_pr.png'
plt.savefig(ppDir + '/' + saveName)
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




### investigate the evolution of averaged velocities at various heights
# time steps for plotting
ave_itv = 3580.0 # by default, the averaging interval is 3600s
                 # ind_back_num = np.floor(ave_itv / tDelta)
tplot_start = 3600.0*1 # must larger than ave_itv
tplot_end = 3600.0*1*120
tplot_delta = 3600.0*1
tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

varplotList = []
for tplot in tplotList:
    print(tplot)
    varplot = np.zeros(zNum)
    for zind in range(zNum):
        f = interp1d(tSeq, varSeq[:,zind], kind='linear', fill_value='extrapolate')
        # tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
        tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/6))
        varplot[zind] = f(tplotSeq).mean()
    varplotList.append(varplot)


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
ax.set_ylabel(varName + ' (m/s)')
xaxis_min = 0
xaxis_max = 120
xaxis_d = 20
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
# plt.savefig(ppDir + '/' + saveName)
plt.show()

# # get variance of averaged velocity at various heights
# for i in range(zIndNum):
#     print(np.var(np.array(uList[i])))
