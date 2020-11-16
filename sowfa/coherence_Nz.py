import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
from numpy import fft
from scipy.interpolate import interp1d
import sliceDataClass as sdc
from funcs import *
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

readDir = ppDir + '/data/'
readName = sliceList[0]
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()
slc = sdc.Slice(data_org, 2)
tSeq = slc.data['time']

t_start = 432000.0
t_end = 435600.0
t_delta = 2.0
t_num = int((t_end - t_start) / t_delta + 1)
t_seq = np.linspace(t_start, t_end, t_num)

def fitting_func(x, a, alpha):
    return a * np.exp(- alpha * x)

for slice in sliceList:

    readDir = ppDir + '/data/'
    readName = slice
    fr = open(readDir + readName, 'rb')
    data_org = pickle.load(fr)
    fr.close()

    slc = sdc.Slice(data_org, 2)
    H = slc.N_location # height of this plane

    # ptCoorList = [np.array([980, 1000, H]), np.array([1000, 1000, H]), np.array([1020, 1000, H]), np.array([1000, 980, H]), np.array([1000, 1020, H])]
    # ptCoorList = [np.array([900, 1000, H]), np.array([1000, 1000, H]), np.array([1100, 1000, H]), np.array([1000, 900, H]), np.array([1000, 1100, H])]
    ptCoorList = [np.array([700, 1000, H]), np.array([800, 1000, H]), np.array([900, 1000, H]), np.array([1000, 1000, H]), np.array([1100, 1000, H]), np.array([1200, 1000, H])]
    ptNum = len(ptCoorList)
    ptIDList = []
    dList = []
    for pt in range(ptNum):
        tmp0, tmp1 = slc.p_nearest(ptCoorList[pt])
        ptIDList.append(tmp0)
        dList.append(tmp1)

    v_seq_list = []

    for pt in range(ptNum):
        vSeq = slc.data['U'][:,ptIDList[pt],0]
        f = interp1d(tSeq, vSeq)
        v_seq = f(t_seq)
        v_seq = v_seq
        v_seq_list.append(v_seq)

    freq, coh, phase = coherence(v_seq_list[0], v_seq_list[0], 0.5)
    freq0, coh0, phase = coherence(v_seq_list[0], v_seq_list[1], 0.5)
    freq1, coh1, phase = coherence(v_seq_list[0], v_seq_list[2], 0.5)
    freq2, coh2, phase = coherence(v_seq_list[0], v_seq_list[3], 0.5)
    freq3, coh3, phase = coherence(v_seq_list[0], v_seq_list[4], 0.5)
    freq4, coh4, phase = coherence(v_seq_list[0], v_seq_list[5], 0.5)


    # plot
    fig, ax = plt.subplots(figsize=(6,6))
    colors = plt.cm.jet(np.linspace(0,1,5))

    # for zInd in range(zNum):
    #     f_ = plotDataList[zInd][0] / (2*np.pi) # convert from omega to frequency
    #     ESD_ = plotDataList[zInd][1] * 2*np.pi
    #     plt.loglog(f_, ESD_, label='h = ' + str(int(HList[zInd])) + 'm', linewidth=1.0, color=colors[zInd])
    # -5/3 law
    # plt.plot(freq0, coh0, label='coh01', linewidth=1.0, color=colors[0])
    # plt.plot(freq1, coh1, label='coh02', linewidth=1.0, color=colors[1])
    # plt.plot(freq2, coh2, label='coh03', linewidth=1.0, color=colors[2])
    plt.plot(freq3, coh3, label='coh04', linewidth=1.0, color=colors[3])
    # plt.plot(freq4, coh4, label='coh05', linewidth=1.0, color=colors[4])
    ind_in, ind_out = 0, 77
    popt, pcov = curve_fit(fitting_func, freq3[ind_in:ind_out], coh3[ind_in:ind_out])
    plt.plot(freq0[ind_in:ind_out], fitting_func(freq0[ind_in:ind_out], *popt), 'k-',
         label='a=%5.3f, alpha=%5.3f' % tuple(popt))

    plt.xlabel('f (1/s)')
    plt.ylabel(varName)
    xaxis_min = 0
    xaxis_max = 0.25
    yaxis_min = 0
    yaxis_max = 1
    plt.ylim(yaxis_min, yaxis_max)
    plt.xlim(xaxis_min, xaxis_max)
    plt.legend(bbox_to_anchor=(0.5,0.9), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('z = ' + str(H) + ', dx = 400')
    fig.tight_layout() # adjust the layout
    saveName = varName_save + '_dx400_' + str(H) + '.png'
    plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
    plt.show()
