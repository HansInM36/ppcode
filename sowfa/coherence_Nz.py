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
jobName = 'pcr_NBL_U10'
ppDir = '/scratch/sowfadata/pp/' + jobName

sliceList = ['Nz0', 'Nz1', 'Nz2', 'Nz3', 'Nz4', 'Nz5', 'Nz6', 'Nz7']
sliceNum = len(sliceList)

var = 'U'
varD = 0 # u:0, v:1, w:2
varName = 'coherence'
varUnit = ''
varName_save = 'uu_coh'


# readDir = ppDir + '/data/'
# readName = sliceList[0]
# fr = open(readDir + readName, 'rb')
# data_org = pickle.load(fr)
# fr.close()
# slc = sdc.Slice(data_org, 2)
# tSeq = slc.data['time']


readDir = ppDir + '/data/'
readName = sliceList[0] + '_ITP'
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()
# tSeq = data_org['time']
xSeq = data_org['x']
xNum = xSeq.size
dx = xSeq[1] - xSeq[0]
ySeq = data_org['y']
yNum = ySeq.size
dy = ySeq[1] - ySeq[0]


HList = []
for slice in sliceList:
    readDir = ppDir + '/data/'
    readName = slice
    fr = open(readDir + readName, 'rb')
    data_org = pickle.load(fr)
    fr.close()
    slc = sdc.Slice(data_org, 2)
    HList.append(slc.N_location)


t_start = 432000.0
t_end = 435600.0
t_delta = 2.0
fs = 1 / t_delta # sampling frequency
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)

dIndList = [5,10,15,20]
dIndNum = len(dIndList)


def fitting_func(x, a, alpha):
    return a * np.exp(- alpha * x)


### calculate horizontally averaged uu_x coherence
plotDataList = []

for slice in sliceList:
    readDir = ppDir + '/data/'
    readName = slice + '_ITP'
    fr = open(readDir + readName, 'rb')
    data_org = pickle.load(fr)
    fr.close()

    cohList = []
    for dInd in dIndList:

        xInd_start = 25
        xInd_end = xNum - 26
        yInd_start = 25
        yInd_end = yNum - 26

        coh = []
        yInd = yInd_start
        while yInd <= yInd_end:
            coh_temp = []
            xInd0 = xInd_start
            xInd1 = xInd0 + dInd
            while xInd1 < xNum:
                u0 = data_org['u'][:,yInd,xInd0]
                u1 = data_org['u'][:,yInd,xInd1]
                # time interpolation
                f0 = interp1d(tSeq, u0, kind='cubic', fill_value='extrapolate')
                f1 = interp1d(tSeq, u1, kind='cubic', fill_value='extrapolate')
                u0 = f0(t_seq)
                u1 = f1(t_seq)
                # calculate coherence and phase
                freq, coh_, phase_ = funcs.coherence(u0, u1, fs)

                coh_temp.append(coh_)

                xInd0 += 1
                xInd1 += 1

            coh_temp = sum(coh_temp)/len(coh_temp)

            coh.append(coh_temp)

            yInd += 1

        coh = sum(coh)/len(coh)
        cohList.append(coh)

    plotDataList.append(cohList)


for i in range(sliceNum):

    fig, ax = plt.subplots(figsize=(6,6))
    colors = plt.cm.jet(np.linspace(0,1,dIndNum))

    for j in range(dIndNum):
        coh = plotDataList[i][j]
        ax.plot(freq, coh, label='dx = ' + str(dx*dIndList[j]) + 'm', linestyle='', marker='o', markersize=3, color=colors[j])
        ind_in, ind_out = 0, 77
        popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out], bounds=(0, [1, 9999]))
        ax.plot(freq[ind_in:ind_out], fitting_func(freq[ind_in:ind_out], *popt), linestyle=':', color=colors[j],
             label='a=%5.3f, alpha=%5.3f' % tuple(popt))

    plt.xlabel('f (1/s)')
    plt.ylabel('Coherence')
    # xaxis_min = 5
    # xaxis_max = 10
    # xaxis_d = 0.5
    yaxis_min = 0
    yaxis_max = 1.0
    yaxis_d = 0.1
    plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
    # plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
    # plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    ax.text(0.8, 1.02, 'h = ' + str(int(HList[i])) + 'm', transform=ax.transAxes, fontsize=12)
    plt.legend(bbox_to_anchor=(0.5,0.8), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('')
    fig.tight_layout() # adjust the layout
    saveName = varName_save + '_' + str(int(HList[i])) + '_pr.png'
    plt.savefig(ppDir + '/' + saveName)
    # plt.show()





### group plot
# rNum, cNum = (4,2)
# fig, axs = plt.subplots(rNum,cNum, constrained_layout=False)
# fig.set_figwidth(8)
# fig.set_figheight(12)
#
# colors = plt.cm.jet(np.linspace(0,1,dIndNum))
#
# for i in range(sliceNum):
#     rNo = int(np.floor(i/cNum))
#     cNo = int(i - rNo*cNum)
#
#     for j in range(dIndNum):
#         coh = plotDataList[i][j]
#         axs[rNo,cNo].plot(freq, coh, label='dx = ' + str(dx*dIndList[j]) + 'm', linewidth=1.0, color=colors[j])
#         ind_in, ind_out = 0, 77
#         popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out])
#         axs[rNo,cNo].plot(freq[ind_in:ind_out], fitting_func(freq[ind_in:ind_out], *popt), linestyle=':', color=colors[j],
#              label='a=%5.3f, alpha=%5.3f' % tuple(popt))
#
#     # axs[rNo,cNo].set_xlim(binMin_, binMax_)
#     # axs[rNo,cNo].tick_params(axis='y', labelcolor=color0)
#     # if rNo != rNum - 1:
#     #     axs[rNo,cNo].set_xticks([])
#     # if cNo != 0:
#     #     axs[rNo,cNo].set_yticks([])
#
#     # if cNo != cNum - 1:
#     #     ax1.set_yticks([])
#
#     axs[rNo,cNo].text(0.72, 0.88, 'h = ' + str(int(HList[i])) + 'm', transform=axs[rNo,cNo].transAxes, fontsize=10)
#
# # fig.text(0.5, 0.06, varName + ' (' + varUnit + ')', ha='center', fontsize=12)
# # fig.text(0.04, 0.5, 'Probability Distribution', va='center', rotation='vertical', fontsize=12, color=color0)
# # fig.text(0.96, 0.5, 'Cumulative Distribution', va='center', rotation='vertical', fontsize=12, color=color1)
#
# fig.suptitle('')
# # fig.tight_layout() # adjust the layout
# # saveName = 'statistics' + '.png'
# # plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
# plt.show()
