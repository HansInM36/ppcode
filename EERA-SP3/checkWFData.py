import os
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/palm')
import imp
import numpy as np
import pickle
from netCDF4 import Dataset
from numpy import fft
from scipy.interpolate import interp1d
import scipy.signal
import funcs
import palm_funcs
import matplotlib.pyplot as plt


def find_nearest_smaller_value_ind(arr, v):
    """ find the indice of the nearest element in an ascending array \ 
    to a given value v and the element must be smaller or equal than v"""
    ind = (np.abs(arr - v)).argmin()
    if arr[ind] <= v:
        return ind
    else:
        return ind-1

def find_nearest_larger_value_ind(arr, v):
    """ find the indice of the nearest element in an ascending array \ 
    to a given value v and the element must be larger or equal than v"""
    ind = (np.abs(arr - v)).argmin()
    if arr[ind] > v:
        return ind
    else:
        return ind+1

t_plot = np.arange(1,12) * 3600 # times of profiles that are going to be plotted

""" EERASP3_1_yw_wf """
jobDir = '/scratch/palmdata/JOBS/EERASP3_1_yw_wf'
jobName  = 'EERASP3_1_yw_wf'

wfData_1_yw = palm_funcs.getWFData_palm(jobDir, jobName, ['.000','.001','.002','.003'], 'rotor_power', [0,86400])
P_yw_org = wfData_1_yw[2]['rotor_power']
T_yw_org = wfData_1_yw[2]['rotor_thrust']

P_yw_av, P_yw_std = np.zeros((t_plot.size, wfData_1_yw[1].size)), np.zeros((t_plot.size, wfData_1_yw[1].size))
T_yw_av, T_yw_std = np.zeros((t_plot.size, wfData_1_yw[1].size)), np.zeros((t_plot.size, wfData_1_yw[1].size))

tSeq = wfData_1_yw[0]

for i in range(t_plot.size):
    tInd_0 = find_nearest_larger_value_ind(tSeq, t_plot[i]-3600)
    tInd_1 = find_nearest_smaller_value_ind(tSeq, t_plot[i])
    
    t = tSeq[tInd_0:tInd_1+1]
    
    P = P_yw_org[tInd_0:tInd_1+1]
    T = T_yw_org[tInd_0:tInd_1+1]
    
    P_av = np.average(P, axis=0)
    T_av = np.average(T, axis=0)
    
    for wt in range(wfData_1_yw[1].size):
        P_ = P[:,wt] - P_av[wt]
        T_ = T[:,wt] - T_av[wt]
        
        P_ = funcs.detrend(t, P_)
        T_ = funcs.detrend(t, T_)
        
        P_std = np.std(P_)
        T_std = np.std(T_)
        
        P_yw_av[i,wt], P_yw_std[i,wt] = P_av[wt], P_std
        T_yw_av[i,wt], T_yw_std[i,wt] = T_av[wt], T_std
        

""" EERASP3_1_ow_wf """
jobDir = '/scratch/palmdata/JOBS/EERASP3_1_ow_wf'
jobName  = 'EERASP3_1_ow_wf'

wfData_1_ow = palm_funcs.getWFData_palm(jobDir, jobName, ['.000','.001','.002','.003'], 'rotor_power', [0,86400])
P_ow_org = wfData_1_ow[2]['rotor_power']
T_ow_org = wfData_1_ow[2]['rotor_thrust']

P_ow_av, P_ow_std = np.zeros((t_plot.size, wfData_1_ow[1].size)), np.zeros((t_plot.size, wfData_1_ow[1].size))
T_ow_av, T_ow_std = np.zeros((t_plot.size, wfData_1_ow[1].size)), np.zeros((t_plot.size, wfData_1_ow[1].size))

tSeq = wfData_1_ow[0]

for i in range(t_plot.size):
    tInd_0 = find_nearest_larger_value_ind(tSeq, t_plot[i]-3600)
    tInd_1 = find_nearest_smaller_value_ind(tSeq, t_plot[i])
    
    t = tSeq[tInd_0:tInd_1+1]
    
    P = P_ow_org[tInd_0:tInd_1+1]
    T = T_ow_org[tInd_0:tInd_1+1]
    
    P_av = np.average(P, axis=0)
    T_av = np.average(T, axis=0)
    
    for wt in range(wfData_1_ow[1].size):
        P_ = P[:,wt] - P_av[wt]
        T_ = T[:,wt] - T_av[wt]
        
        P_ = funcs.detrend(t, P_)
        T_ = funcs.detrend(t, T_)
        
        P_std = np.std(P_)
        T_std = np.std(T_)
        
        P_ow_av[i,wt], P_ow_std[i,wt] = P_av[wt], P_std
        T_ow_av[i,wt], T_ow_std[i,wt] = T_av[wt], T_std




""" cmp rotor power """
for i in range(t_plot.size):
    
    labels = ['WT' + str(wt+1) for wt in range(wfData_1_yw[1].size)]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, P_yw_av[i], width, label='yw')
    rects2 = ax.bar(x + width/2, P_ow_av[i], width, label='ow')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Rotor Power')
    ax.set_title('t =' + str(np.round(t_plot[i]/3600,1)) + 'h')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    
    fig.tight_layout()
    saveName = 'P_av_t' + str(np.round(t_plot[i]/3600,1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/P_av/1st'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()


""" cmp rotor power std """
for i in range(t_plot.size):
    
    labels = ['WT' + str(wt+1) for wt in range(wfData_1_yw[1].size)]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, P_yw_std[i], width, label='yw')
    rects2 = ax.bar(x + width/2, P_ow_std[i], width, label='ow')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Rotor Power STD')
    ax.set_title('t =' + str(np.round(t_plot[i]/3600,1)) + 'h')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    
    fig.tight_layout()
    saveName = 'P_std_t' + str(np.round(t_plot[i]/3600,1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/P_std/1st'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()


