import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
from numpy import fft
from scipy.interpolate import interp1d
import sliceDataClass as sdc
import funcs
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs20'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName

prbg = 'prbg0'

# coordinate transmation
O = (0,0,0)
alpha = 30.0

var = 'U'
varD = 0 # u:0, v:1, w:2
varName = 'coherence'
varUnit = ''
varName_save = 'uu_coh_av'


# read data
readDir = ppDir + '/data/'
readName = prbg
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()

# coordinate transformation
prbNum = data_org['coors'].shape[0]
for p in range(prbNum):
    tmp = data_org[var][p]
    data_org[var][p] = funcs.trs(tmp,O,alpha)

# choose the probegroup to be used in plotting
xSeq = np.array(data_org['info'][2])
ySeq = np.array(data_org['info'][3])
zSeq = np.array(data_org['info'][4])
xNum = xSeq.size
yNum = ySeq.size
zNum = zSeq.size

data = data_org
# del data_org

tSeq = data['time']
tNum = tSeq.size

t_start = 432000.0
t_end = 434400.0
t_delta = 0.1
fs = 1 / t_delta # sampling frequency
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)


coors = data['coors']

# choose height
zInd = 2

pInd_start = xNum*yNum*zInd
pInd_end = xNum*yNum*(zInd+1)

dInd = 1

coh = []
co_coh = []
phase = []

for p0_id in range(pInd_start, pInd_end - dInd):

    p1_id = p0_id + dInd

    p0_coor = coors[p0_id]
    p1_coor = coors[p1_id]

    dp = p1_coor - p0_coor
    dp = dp.reshape(1,3)
    dp = funcs.trs(dp,O,alpha)


    dx = dp[0,0]
    dy = dp[0,1]

    p0 = '(' + str(np.round(p0_coor[0],1)) + ', ' + str(np.round(p0_coor[1],1)) + ', ' + str(np.round(p0_coor[2],1)) + ')'
    p1 = '(' + str(np.round(p1_coor[0],1)) + ', ' + str(np.round(p1_coor[1],1)) + ', ' + str(np.round(p1_coor[2],1)) + ')'

    u0_ = data[var][p0_id][:,varD]
    u1_ = data[var][p1_id][:,varD]

    # time interpolation
    method_ = 'linear' # 'linear' or 'cubic'
    f0 = interp1d(tSeq, u0_, kind=method_, fill_value='extrapolate')
    f1 = interp1d(tSeq, u1_, kind=method_, fill_value='extrapolate')
    u0 = f0(t_seq)
    u1 = f1(t_seq)

    # calculate coherence and phase
    segNum = 1200
    freq, coh_, co_coh_, phase_ = funcs.coherence(u0, u1, fs, segNum)

    coh.append(coh_)
    co_coh.append(co_coh_)
    phase.append(phase_)

coh = np.average(np.array(coh), axis=0)
co_coh = np.average(np.array(co_coh), axis=0)
phase = np.average(np.array(phase), axis=0)


""" plot coherence and fitting curve """
def fitting_func(x, a, alpha):
    return a * np.exp(- alpha * x)

f_out = 0.4
tmp = abs(freq - f_out)
ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(freq[1:], coh[1:], linestyle=':', marker='o', markersize=3, color='k')
popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out], bounds=(0, [1, 100]))
ax.plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='k',
     label='a=%5.3f, alpha=%5.3f' % tuple(popt))

plt.xlabel('f (1/s)', fontsize=12)
plt.ylabel('coherence', fontsize=12)
xaxis_min = 0
xaxis_max = 5
xaxis_d = 0.5
yaxis_min = 0
yaxis_max = 1.0
yaxis_d = 0.1
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=10)
ax.text(0.0, 1.02, 'fs = ' + str(fs) + 'Hz' + ', ' 'nperseg = ' + str(segNum), transform=ax.transAxes, fontsize=12)
# ax.text(0.0, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm', transform=ax.transAxes, fontsize=12)
ax.text(0.56, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm' + ', ' + 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
plt.legend(bbox_to_anchor=(0.5,0.9), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = varName_save + '_' + str(int(H)) + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()
#
#
#
# """ plot co-coherence """
# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(freq[1:], co_coh[1:], linestyle='-', marker='o', markersize=3, color='r')
#
# plt.xlabel('f (1/s)', fontsize=12)
# plt.ylabel('co-coherence', fontsize=12)
# xaxis_min = 0
# xaxis_max = 0.5
# xaxis_d = 0.05
# yaxis_min = -1.0
# yaxis_max = 1.0
# yaxis_d = 0.2
# plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
# plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
# plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=10)
# ax.text(0.0, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm', transform=ax.transAxes, fontsize=12)
# ax.text(0.8, 1.02, 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
# # plt.legend(bbox_to_anchor=(0.5,0.8), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
# plt.grid()
# plt.title('')
# fig.tight_layout() # adjust the layout
# # saveName = varName_save + '_' + str(int(H)) + '_pr.png'
# # plt.savefig(ppDir + '/' + saveName)
# plt.show()
#
#
#
# """ plot phase """
# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(freq[1:], phase[1:], linestyle='-', marker='o', markersize=3, color='b')
#
# plt.xlabel('f (1/s)', fontsize=12)
# plt.ylabel('phase', fontsize=12)
# xaxis_min = 0
# xaxis_max = 0.5
# xaxis_d = 0.05
# yaxis_min = -1.0*np.pi
# yaxis_max = 1.0*np.pi
# yaxis_d = np.pi/4
# plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
# plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
# plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=10)
# ax.text(0.0, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm', transform=ax.transAxes, fontsize=12)
# ax.text(0.8, 1.02, 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
# # plt.legend(bbox_to_anchor=(0.5,0.8), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
# plt.grid()
# plt.title('')
# fig.tight_layout() # adjust the layout
# # saveName = varName_save + '_' + str(int(H)) + '_pr.png'
# # plt.savefig(ppDir + '/' + saveName)
# plt.show()



""" plot coherence, co-coherence, phase in one figure """
f_out = 0.4
tmp = abs(freq - f_out)
ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

rNum, cNum = (1,3)
fig, axs = plt.subplots(rNum,cNum, constrained_layout=False)
fig.set_figwidth(12)
fig.set_figheight(4)

# coherence
axs[0].plot(freq[1:], coh[1:], linestyle='-', marker='o', markersize=3, color='k')
# popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out], bounds=(0, [1, 100]))
# axs[0].plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='k',
#      label='a=%5.3f, alpha=%5.3f' % tuple(popt))

axs[0].tick_params(axis='both', which='major', labelsize=10)
xaxis_min = 0
xaxis_max = 0.5
xaxis_d = 0.1
yaxis_min = 0
yaxis_max = 1.0
yaxis_d = 0.1
axs[0].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[0].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
axs[0].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
axs[0].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[0].grid()

axs[0].set_xlabel('f (1/s)', fontsize=12)
axs[0].set_ylabel('coherence', fontsize=12)

# axs[0].legend(bbox_to_anchor=(0.2,0.9), loc=6, borderaxespad=0, fontsize=10)

# co-coherence
axs[1].plot(freq[1:], co_coh[1:], linestyle='-', marker='o', markersize=3, color='r')

axs[1].tick_params(axis='both', which='major', labelsize=10)
xaxis_min = 0
xaxis_max = 0.5
xaxis_d = 0.1
yaxis_min = -1.0
yaxis_max = 1.0
yaxis_d = 0.2
axs[1].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[1].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
axs[1].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
axs[1].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[1].grid()

axs[1].set_xlabel('f (1/s)', fontsize=12)
axs[1].set_ylabel('co-coherence', fontsize=12)

axs[1].set_title('dx = ' + str(np.round(dx,1)) + 'm' + ', ' + 'h = ' + str(np.round(p0_coor[2])) + 'm', fontsize=12)

# phase
axs[2].plot(freq[1:], phase[1:], linestyle='-', marker='o', markersize=3, color='b')

axs[2].tick_params(axis='both', which='major', labelsize=10)
xaxis_min = 0
xaxis_max = 0.5
xaxis_d = 0.1
yaxis_min = -1.0*np.pi
yaxis_max = 1.0*np.pi
yaxis_d = np.pi/4
axs[2].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[2].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
axs[2].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
axs[2].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
# axs[2].set_yticks(np.arange(0, 2*np.pi+0.01, np.pi/4))
labels = ['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
          r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
axs[2].set_yticklabels(labels)
axs[2].grid()

axs[2].set_xlabel('f (1/s)', fontsize=12)
axs[2].set_ylabel('phase', fontsize=12)

fig.tight_layout()
saveName = 'coh_co-coh_phase_av_f0.5' + '_dx_' + str(np.round(dx,1)) + '_h_' + str(np.round(p0_coor[2])) + '_pr.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()
