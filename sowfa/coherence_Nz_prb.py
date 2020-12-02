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

# coordinate transmation
O = (0,0,0)
alpha = 30.0

var = 'U'
varD = 0 # u:0, v:1, w:2
varName = 'coherence'
varUnit = ''
varName_save = 'uu_coh'


segNum = 512

# read data
readDir = ppDir + '/data/'
readName = 'probeData'
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()

# coordinate transformation
prbgNames = data_org['prbgNames']
prbgNum = len(prbgNames)
for i in range(prbgNum):
    data_ = data_org[prbgNames[i]]
    prbNum = data_['coors'].shape[0]
    for j in range(prbNum):
        tmp = data_org[prbgNames[i]][var][j]
        data_org[prbgNames[i]][var][j] = funcs.trs(tmp,O,alpha)


prbgName = 'probeNx6'

data = data_org[prbgName]
# del data_org

tSeq = data['time']
tNum = tSeq.size

# tSeq = 432000 + 0.1*np.arange(0,tNum)

t_start = 432100.0
t_end = 433000.0
t_delta = 0.1
fs = 1 / t_delta # sampling frequency
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)


coors = data['coors']

p0_id = 0
p1_id = 10

p0_coor = coors[p0_id]
p1_coor = coors[p1_id]

dx = (p1_coor - p0_coor)[0]
dy = (p1_coor - p0_coor)[1]

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


# # check time series
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(tSeq, u0_, 'r-')
ax.plot(t_seq, u0, 'b-')
ax.plot(tSeq, u1_, 'r:')
ax.plot(t_seq, u1, 'b:')
plt.grid()
plt.show()
plt.close()



# calculate coherence and phase
freq, coh, phase_ = funcs.coherence(u0, u1, fs, segNum)

def fitting_func(x, a, alpha):
    return a * np.exp(- alpha * x)

f_out = 5.0
tmp = abs(freq - f_out)
ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(freq[1:ind_out], coh[1:ind_out], linestyle=':', marker='o', markersize=3, color='k')
popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out], bounds=(0, [1, 999]))
ax.plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='k',
     label='a=%5.3f, alpha=%5.3f' % tuple(popt))
# plt.axvline(x=22.1/100/np.pi, ls='--', c='black')
# plt.axvline(x=2*22.1/100/np.pi, ls='--', c='black')
# plt.axvline(x=3*22.1/100/np.pi, ls='--', c='black')
plt.xlabel('f (1/s)')
plt.ylabel('Coherence')
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = 0
yaxis_max = 1.0
yaxis_d = 0.1
# plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
# ax.text(0.0, 1.02, 'dx = ' + str(dx) + 'm', transform=ax.transAxes, fontsize=12)
# ax.text(0.8, 1.02, 'h = ' + str(int(H)) + 'm', transform=ax.transAxes, fontsize=12)
plt.legend(bbox_to_anchor=(0.5,0.8), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = varName_save + '_' + str(int(H)) + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()



















































# def fitting_func(x, a, alpha):
#     return a * np.exp(- alpha * x)
#
#
# for slice in sliceList:
#
#     readDir = ppDir + '/data/'
#     readName = slice
#     fr = open(readDir + readName, 'rb')
#     data_org = pickle.load(fr)
#     fr.close()
#
#     slc = sdc.Slice(data_org, 2)
#     H = slc.N_location # height of this plane
#
#     # ptCoorList = [np.array([980, 1000, H]), np.array([1000, 1000, H]), np.array([1020, 1000, H]), np.array([1000, 980, H]), np.array([1000, 1020, H])]
#     # ptCoorList = [np.array([900, 1000, H]), np.array([1000, 1000, H]), np.array([1100, 1000, H]), np.array([1000, 900, H]), np.array([1000, 1100, H])]
#     ptCoorList = [np.array([700, 1000, H]), np.array([800, 1000, H]), np.array([900, 1000, H]), np.array([1000, 1000, H]), np.array([1100, 1000, H]), np.array([1200, 1000, H])]
#     ptNum = len(ptCoorList)
#     ptIDList = []
#     dList = []
#     for pt in range(ptNum):
#         tmp0, tmp1 = slc.p_nearest(ptCoorList[pt])
#         ptIDList.append(tmp0)
#         dList.append(tmp1)
#
#     v_seq_list = []
#
#     for pt in range(ptNum):
#         vSeq = slc.data['U'][:,ptIDList[pt],0]
#         f = interp1d(tSeq, vSeq, fill_value='extrapolate')
#         v_seq = f(t_seq)
#         v_seq = v_seq
#         v_seq_list.append(v_seq)
#
#     plotDataList = []
#     for i in range(ptNum-1):
#         plotDataList.append(funcs.coherence(v_seq_list[0], v_seq_list[1+i], fs, segNum))
#
#
#     # plot
#     fig, ax = plt.subplots(figsize=(6,6))
#     colors = plt.cm.jet(np.linspace(0,1,ptNum-1))
#     label = ['dx = 100m', 'dx = 200m', 'dx = 300m', 'dx = 400m', 'dx = 500m',]
#
#     # for i in range(ptNum-1):
#     i=0
#     freq = plotDataList[i][0]
#     coh = plotDataList[i][1]
#     ax.plot(freq, coh, label=label[i], linewidth=1.0, color=colors[i])
#
#     ind_in, ind_out = 0, 40
#     freq_ = freq[ind_in:ind_out]
#     coh_ = coh[ind_in:ind_out]
#     popt, pcov = curve_fit(fitting_func, freq_, coh_, bounds=([0, 0], [1, 1e3]))
#     plt.plot(freq_, fitting_func(freq_, *popt), 'k-',
#          label='a=%5.3f, alpha=%5.3f' % tuple(popt))
#
#     plt.xlabel('f (1/s)')
#     plt.ylabel(varName)
#     # xaxis_min = 0
#     # xaxis_max = 0.25
#     # yaxis_min = 0
#     # yaxis_max = 1
#     # plt.ylim(yaxis_min, yaxis_max)
#     # plt.xlim(xaxis_min, xaxis_max)
#     plt.legend(bbox_to_anchor=(0.5,0.9), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
#     plt.grid()
#     plt.title('z = ' + str(H) + ', dx = 400')
#     fig.tight_layout() # adjust the layout
#     saveName = varName_save + '_dx400_' + str(H) + '.png'
#     # plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
#     plt.show()
