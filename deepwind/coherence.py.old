import os
import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import pickle
from netCDF4 import Dataset
from numpy import fft
from scipy.interpolate import interp1d
import scipy.signal
import sliceDataClass as sdc
import funcs
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def coh_kaimal(freq, delta, U):
    a = 12
    b = 0.12
    L = 73.5
    coh = np.exp(-a * np.power((freq*delta/U)**2 + (b*delta/L)**2,0.5))
    return coh

def getData_sowfa(dir, prbg, trs_para, var, varD):
    """ extract velocity data of specified probe groups """
    # coordinate transmation
    O = trs_para[0]
    alpha = trs_para[1]

    # read data
    readDir = dir + '/data/'
    readName = prbg
    fr = open(readDir + readName, 'rb')
    data_org = pickle.load(fr)
    fr.close()

    coors = data_org['coors']

    # coordinate transformation
    prbNum = coors.shape[0]
    for p in range(prbNum):
        tmp = data_org[var][p]
        data_org[var][p] = funcs.trs(tmp,O,alpha)

    xSeq = np.array(data_org['info'][2])
    ySeq = np.array(data_org['info'][3])
    zSeq = np.array(data_org['info'][4])
    xNum = xSeq.size
    yNum = ySeq.size
    zNum = zSeq.size
    varSeq = data_org[var][:,:,varD]
    tSeq = data_org['time']
    tNum = tSeq.size
    return tSeq, xSeq, ySeq, zSeq, varSeq, coors

def coh_sowfa(p0_ind, p1_ind, xNum, yNum, t_para, tSeq, varSeq, tL):
    p0_id = p0_ind[2]*xNum*yNum + p0_ind[1]*xNum + p0_ind[0]
    p1_id = p1_ind[2]*xNum*yNum + p1_ind[1]*xNum + p1_ind[0]

    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    u0 = varSeq[p0_id]
    u1 = varSeq[p1_id]
    f0 = interp1d(tSeq, u0, kind='linear', fill_value='extrapolate')
    f1 = interp1d(tSeq, u1, kind='linear', fill_value='extrapolate')
    u0_ = f0(t_seq)
    u1_ = f1(t_seq)

    segNum = tL*fs
    freq, coh, co_coh, phase = funcs.coherence(u0_, u1_, fs, segNum)
    return t_seq, u0_, u1_, freq, coh, co_coh, phase

def getData_palm(dir, jobName, maskID, run_no_list, var):
    """ extract velocity data of specified probe groups """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        varSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))

    # dimensions = list(nc_file_list[0].dimensions
    # vars = list(nc_file_list[0].variables
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][:], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][:], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(run_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, xSeq, ySeq, zSeq, varSeq

def coh_palm(p0_ind, p1_ind, t_para, tSeq, varSeq, tL):
    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    u0 = varSeq[:,p0_ind[2],p0_ind[1],p0_ind[0]]
    u1 = varSeq[:,p1_ind[2],p1_ind[1],p1_ind[0]]
    f0 = interp1d(tSeq, u0, kind='linear', fill_value='extrapolate')
    f1 = interp1d(tSeq, u1, kind='linear', fill_value='extrapolate')
    u0_ = f0(t_seq)
    u1_ = f1(t_seq)

    segNum = tL*fs
    freq, coh, co_coh, phase = funcs.coherence(u0_, u1_, fs, segNum)
    return t_seq, u0_, u1_, freq, coh, co_coh, phase

def coh_av_palm_x(xInd_start, xInd_end, dxInd, yInd, zInd, t_para, tSeq, varSeq):
    N = xInd_end - xInd_start - dxInd + 1
    coh = []
    co_coh = []
    phase = []
    for i in range(N):
        results = coh_palm((xInd_start+i,yInd,zInd), (xInd_start+i+dxInd,yInd,zInd), t_para, tSeq, varSeq)
        coh.append(results[4])
        co_coh.append(results[5])
        phase.append(results[6])
        freq = results[3]
    coh = np.array(coh)
    coh_av = np.mean(coh, axis=0)
    coh_std = np.std(coh, axis=0)
    co_coh = np.array(co_coh)
    co_coh_av = np.mean(co_coh, axis=0)
    co_coh_std = np.std(co_coh, axis=0)
    phase = np.array(phase)
    phase_av = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std
def coh_av_palm_y(xInd, yInd_start, yInd_end, dyInd, zInd, t_para, tSeq, varSeq, tL):
    N = yInd_end - yInd_start - dyInd + 1
    coh = []
    co_coh = []
    phase = []
    for i in range(N):
        results = coh_palm((xInd,yInd_start+i,zInd), (xInd,yInd_start+i+dyInd,zInd), t_para, tSeq, varSeq, tL)
        coh.append(results[4])
        co_coh.append(results[5])
        phase.append(results[6])
        freq = results[3]
    coh = np.array(coh)
    coh_av = np.mean(coh, axis=0)
    coh_std = np.std(coh, axis=0)
    co_coh = np.array(co_coh)
    co_coh_av = np.mean(co_coh, axis=0)
    co_coh_std = np.std(co_coh, axis=0)
    phase = np.array(phase)
    phase_av = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std

def coh_av_sowfa_x(xInd_start, xInd_end, dxInd, yInd, zInd, xNum, yNum, t_para, tSeq, varSeq):
    N = xInd_end - xInd_start - dxInd + 1
    coh = []
    co_coh = []
    phase = []
    for i in range(N):
        results = coh_sowfa((xInd_start+i,yInd,zInd), (xInd_start+i+dxInd,yInd,zInd), xNum, yNum, t_para, tSeq, varSeq)
        coh.append(results[4])
        co_coh.append(results[5])
        phase.append(results[6])
        freq = results[3]
    coh = np.array(coh)
    coh_av = np.mean(coh, axis=0)
    coh_std = np.std(coh, axis=0)
    co_coh = np.array(co_coh)
    co_coh_av = np.mean(co_coh, axis=0)
    co_coh_std = np.std(co_coh, axis=0)
    phase = np.array(phase)
    phase_av = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std
def coh_av_sowfa_y(xInd, yInd_start, yInd_end, dyInd, zInd, xNum, yNum, t_para, tSeq, varSeq):
    N = yInd_end - yInd_start - dyInd + 1
    coh = []
    co_coh = []
    phase = []
    for i in range(N):
        results = coh_sowfa((xInd,yInd_start+i,zInd), (xInd,yInd_start+i+dyInd,zInd), xNum, yNum, t_para, tSeq, varSeq)
        coh.append(results[4])
        co_coh.append(results[5])
        phase.append(results[6])
        freq = results[3]
    coh = np.array(coh)
    coh_av = np.mean(coh, axis=0)
    coh_std = np.std(coh, axis=0)
    co_coh = np.array(co_coh)
    co_coh_av = np.mean(co_coh, axis=0)
    co_coh_std = np.std(co_coh, axis=0)
    phase = np.array(phase)
    phase_av = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std

def plot_coh_av(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std):
    f_out = 0.4
    tmp = abs(freq - f_out)
    ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

    rNum, cNum = (1,3)
    fig, axs = plt.subplots(rNum,cNum, constrained_layout=False)
    fig.set_figwidth(12)
    fig.set_figheight(4)

    #coherence
    axs[0].plot(freq[1:], coh_av[1:], linestyle='-', marker='', markersize=3, color='k')
    # popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out], bounds=(0, [1, 100]))
    # axs[0].plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='k',
    # label='a=%5.3f, alpha=%5.3f' % tuple(popt))
    axs[0].fill_between(freq[1:], coh_av[1:]-coh_std[1:], coh_av[1:]+coh_std[1:], color='gray')
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    xaxis_min = 0
    xaxis_max = 5.0
    xaxis_d = 0.5
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
    #axs[0].legend(bbox_to_anchor=(0.2,0.9), loc=6, borderaxespad=0, fontsize=10)

    #co-coherence
    axs[1].plot(freq[1:], co_coh_av[1:], linestyle='-', marker='', markersize=3, color='r')
    axs[1].fill_between(freq[1:], co_coh_av[1:]-co_coh_std[1:], co_coh_av[1:]+co_coh_std[1:], color='salmon')
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    xaxis_min = 0
    xaxis_max = 5.0
    xaxis_d = 0.5
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
    # axs[1].set_title('dx = ' + str(np.round(dx,1)) + 'm' + ', ' + 'h = ' + str(np.round(p0_coor[2])) + 'm', fontsize=12)

    #phase
    axs[2].plot(freq[1:], phase_av[1:], linestyle='-', marker='', markersize=3, color='b')
    axs[2].fill_between(freq[1:], phase_av[1:]-phase_std[1:], phase_av[1:]+phase_std[1:], color='lightskyblue')
    axs[2].tick_params(axis='both', which='major', labelsize=10)
    xaxis_min = 0
    xaxis_max = 5.0
    xaxis_d = 0.5
    yaxis_min = -1.0*np.pi
    yaxis_max = 1.0*np.pi
    yaxis_d = np.pi/4
    axs[2].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    axs[2].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    axs[2].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    axs[2].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    #axs[2].set_yticks(np.arange(0, 2*np.pi+0.01, np.pi/4))
    labels = ['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
              r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
    axs[2].set_yticklabels(labels)
    axs[2].grid()
    axs[2].set_xlabel('f (1/s)', fontsize=12)
    axs[2].set_ylabel('phase', fontsize=12)
    fig.tight_layout()
    # saveName = 'coh_co-coh_phase_f5.0' + '_dx_' + str(np.round(dx,1)) + '_h_' + str(np.round(p0_coor[2])) + '_pr.png'
    #plt.savefig(ppDir + '/' + saveName)
    plt.show()
    plt.close()

def plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std):
    f_out = 0.4
    tmp = abs(freq - f_out)
    ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

    rNum, cNum = (2,3)
    fig, axs = plt.subplots(rNum,cNum, constrained_layout=False)
    fig.set_figwidth(12)
    fig.set_figheight(4)

    #coherence
    axs[0,0].plot(freq[1:], coh_av[1:], linestyle='-', marker='', markersize=3, color='k')
    # popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out], bounds=(0, [1, 100]))
    # axs[0].plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='k',
    # label='a=%5.3f, alpha=%5.3f' % tuple(popt))
    axs[0,0].fill_between(freq[1:], coh_av[1:]-coh_std[1:], coh_av[1:]+coh_std[1:], color='gray')
    axs[0,0].tick_params(axis='both', which='major', labelsize=10)
    xaxis_min = 0
    xaxis_max = 1.0
    xaxis_d = 0.1
    yaxis_min = 0
    yaxis_max = 1.0
    yaxis_d = 0.1
    axs[0,0].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    axs[0,0].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    axs[0,0].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    axs[0,0].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    axs[0,0].grid()
    axs[0,0].set_xlabel('f (1/s)', fontsize=12)
    axs[0,0].set_ylabel('coherence', fontsize=12)
    #axs[0,0].legend(bbox_to_anchor=(0.2,0.9), loc=6, borderaxespad=0, fontsize=10)

    axs[1,0].plot(freq[1:], coh_av[1:], linestyle='-', marker='', markersize=3, color='k')
    # popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out], bounds=(0, [1, 100]))
    # axs[0].plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='k',
    # label='a=%5.3f, alpha=%5.3f' % tuple(popt))
    axs[1,0].fill_between(freq[1:], coh_av[1:]-coh_std[1:], coh_av[1:]+coh_std[1:], color='gray')
    xaxis_min = 0
    xaxis_max = 0.5
    xaxis_d = 0.1
    yaxis_min = 0
    yaxis_max = 1.0
    yaxis_d = 0.1
    axs[1,0].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    axs[1,0].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    axs[1,0].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    axs[1,0].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    axs[1,0].grid()
    axs[1,0].set_xlabel('f (1/s)', fontsize=12)
    axs[1,0].set_ylabel('coherence', fontsize=12)

    #co-coherence
    axs[0,1].plot(freq[1:], co_coh_av[1:], linestyle='-', marker='', markersize=3, color='r')
    axs[0,1].fill_between(freq[1:], co_coh_av[1:]-co_coh_std[1:], co_coh_av[1:]+co_coh_std[1:], color='salmon')
    axs[0,1].tick_params(axis='both', which='major', labelsize=10)
    xaxis_min = 0
    xaxis_max = 1.0
    xaxis_d = 0.1
    yaxis_min = -1.0
    yaxis_max = 1.0
    yaxis_d = 0.2
    axs[0,1].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    axs[0,1].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    axs[0,1].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    axs[0,1].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    axs[0,1].grid()
    axs[0,1].set_xlabel('f (1/s)', fontsize=12)
    axs[0,1].set_ylabel('co-coherence', fontsize=12)
    # axs[0,1].set_title('dx = ' + str(np.round(dx,1)) + 'm' + ', ' + 'h = ' + str(np.round(p0_coor[2])) + 'm', fontsize=12)

    axs[1,1].plot(freq[1:], co_coh_av[1:], linestyle='-', marker='', markersize=3, color='r')
    axs[1,1].fill_between(freq[1:], co_coh_av[1:]-co_coh_std[1:], co_coh_av[1:]+co_coh_std[1:], color='salmon')
    axs[1,1].tick_params(axis='both', which='major', labelsize=10)
    xaxis_min = 0
    xaxis_max = 0.5
    xaxis_d = 0.1
    yaxis_min = -1.0
    yaxis_max = 1.0
    yaxis_d = 0.2
    axs[1,1].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    axs[1,1].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    axs[1,1].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    axs[1,1].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    axs[1,1].grid()
    axs[1,1].set_xlabel('f (1/s)', fontsize=12)
    axs[1,1].set_ylabel('co-coherence', fontsize=12)

    #phase
    axs[0,2].plot(freq[1:], phase_av[1:], linestyle='-', marker='', markersize=3, color='b')
    axs[0,2].fill_between(freq[1:], phase_av[1:]-phase_std[1:], phase_av[1:]+phase_std[1:], color='lightskyblue')
    axs[0,2].tick_params(axis='both', which='major', labelsize=10)
    xaxis_min = 0
    xaxis_max = 1.0
    xaxis_d = 0.1
    yaxis_min = -1.0*np.pi
    yaxis_max = 1.0*np.pi
    yaxis_d = np.pi/4
    axs[0,2].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    axs[0,2].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    axs[0,2].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    axs[0,2].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    # axs[2].set_yticks(np.arange(0, 2*np.pi+0.01, np.pi/4))
    labels = ['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
              r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
    axs[0,2].set_yticklabels(labels)
    axs[0,2].grid()
    axs[0,2].set_xlabel('f (1/s)', fontsize=12)
    axs[0,2].set_ylabel('phase', fontsize=12)

    axs[1,2].plot(freq[1:], phase_av[1:], linestyle='-', marker='', markersize=3, color='b')
    axs[1,2].fill_between(freq[1:], phase_av[1:]-phase_std[1:], phase_av[1:]+phase_std[1:], color='lightskyblue')
    axs[1,2].tick_params(axis='both', which='major', labelsize=10)
    xaxis_min = 0
    xaxis_max = 0.5
    xaxis_d = 0.1
    yaxis_min = -1.0*np.pi
    yaxis_max = 1.0*np.pi
    yaxis_d = np.pi/4
    axs[1,2].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    axs[1,2].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    axs[1,2].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
    axs[1,2].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    #axs[2].set_yticks(np.arange(0, 2*np.pi+0.01, np.pi/4))
    labels = ['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
              r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
    axs[1,2].set_yticklabels(labels)
    axs[1,2].grid()
    axs[1,2].set_xlabel('f (1/s)', fontsize=12)
    axs[1,2].set_ylabel('phase', fontsize=12)

    fig.tight_layout()
    # saveName = 'coh_co-coh_phase_f5.0' + '_dx_' + str(np.round(dx,1)) + '_h_' + str(np.round(p0_coor[2])) + '_pr.png'
    #plt.savefig(ppDir + '/' + saveName)
    plt.show()
    plt.close()

def calc_coh0(v0,v1,segNum):
    N0 = v0.size
    N1 = v1.size
    v0List = []
    v1List = []
    i = 0
    while (i + segNum < N0):
        v0List.append(v0[i:i+segNum])
        v1List.append(v1[i:i+segNum])
        i += segNum//2
    L0 = len(v0List)
    L1 = len(v1List)
    a2 = 0
    b2 = 0
    ab = 0
    for i in range(L0):
        a2 += np.power(v0List[i].mean(),2) / L0
        b2 += np.power(v1List[i].mean(),2) / L0
        ab += v0List[i].mean() * v1List[i].mean() / L0
        print(v0List[i].mean(),v1List[i].mean())
    return np.power(ab,2)/a2/b2




""" example """
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg1', ((0,0,0),30.0), 'U', 0)
# calculate coherence
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_sowfa((0,24,0), (0,25,0), xSeq_0.size, ySeq_0.size, (144000.0, 146400, 0.1), tSeq_0, varSeq_0)
# # averaged coherence
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_sowfa_x(0, 50, 2, 0, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400, 0.1), tSeq_0, varSeq_0)
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_sowfa_y(0, 0, 50, 1, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400, 0.1), tSeq_0, varSeq_0)
plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std)

# funcs.group_plot_0(t_seq_1 - 144000, 10, u0_1-u0_1.mean(), u1_1-u1_1.mean())
# funcs.group_plot_1(t_seq_0 - 144000, 10, u0_0-u0_0.mean(), u1_0-u1_0.mean())


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs10'
dir_1 = prjDir + '/' + jobName
tSeq_1, xSeq_1, ySeq_1, zSeq_1, varSeq_1 = getData_palm(dir_1, jobName, 'M04', ['.022'], 'u')
# calculate coherence
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_palm((0,0,4), (0,2,4), (288000.0, 290400, 0.1), tSeq_1, varSeq_1)
funcs.group_plot_0(t_seq_1 - 288000, 10, u0_1-u0_1.mean(), u1_1-u1_1.mean())
# averaged coherence
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_palm_y(0, 0, 50, 1, 4, (288000.0, 290400, 0.1), tSeq_1, varSeq_1)
plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std)

# funcs.group_plot_0(t_seq_1 - 144000, 10, u0_1-u0_1.mean(), u1_1-u1_1.mean())
# funcs.group_plot_1(t_seq_1 - 144000, 10, u0_1-u0_1.mean(), u1_1-u1_1.mean())


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_NBL'
dir_1 = prjDir + '/' + jobName
tSeq_1, xSeq_1, ySeq_1, zSeq_1, varSeq_1 = getData_palm(dir_1, jobName, 'M04', ['.001','.002'], 'u')
# calculate coherence
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_palm((0,0,4), (0,2,4), (144000.0, 146400, 0.1), tSeq_1, varSeq_1, 240)
funcs.group_plot_0(t_seq_1 - 144000, 10, u0_1-u0_1.mean(), u1_1-u1_1.mean())
# averaged coherence
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_palm_y(0, 0, 50, 1, 4, (144000.0, 146400, 0.1), tSeq_1, varSeq_1)
plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std)

# funcs.group_plot_0(t_seq_1 - 144000, 10, u0_1-u0_1.mean(), u1_1-u1_1.mean())
# funcs.group_plot_1(t_seq_1 - 144000, 10, u0_1-u0_1.mean(), u1_1-u1_1.mean())


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5'
dir_1 = prjDir + '/' + jobName
tSeq_1, xSeq_1, ySeq_1, zSeq_1, varSeq_1 = getData_palm(dir_1, jobName, 'M03', ['.011','.012','.013','.014'], 'u')
# calculate coherence
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_palm((0,0,4), (1,0,4), (72000.0, 74400, 0.1), tSeq_1, varSeq_1)
funcs.group_plot_0(t_seq_1 - 72000, 10, u0_1-u0_1.mean(), u1_1-u1_1.mean())
# averaged coherence
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_palm_x(0, 50, 8, 0, 4, (72000.0, 74400, 0.1), tSeq_1, varSeq_1)
plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std)







""" multi separations at certain height """
### longitudinal
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs10'
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName

# calculate coherence (long)
tSeq, xSeq, ySeq, zSeq, varSeq, coors = getData_sowfa(ppDir, 'prbg0', ((0,0,0),30.0), 'U', 0)
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_sowfa((24,0,4), (26,0,4), xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq, 240)
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_sowfa((23,0,4), (27,0,4), xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq, 240)
# t_seq_2, u0_2, u1_2, freq, coh_2, co_coh_2, phase_2 = coh_sowfa((21,0,4), (29,0,4), xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq)

# calculate coherence (lat)
tSeq, xSeq, ySeq, zSeq, varSeq, coors = getData_sowfa(ppDir, 'prbg1', ((0,0,0),30.0), 'U', 0)
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_sowfa((0,36,4), (0,38,4), xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq, 240)
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_sowfa((0,23,4), (0,27,4), xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq, 240)
# t_seq_2, u0_2, u1_2, freq, coh_2, co_coh_2, phase_2 = coh_sowfa((0,21,4), (0,29,4), xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq, 240)

# calculate averaged coherence (lat)
tSeq, xSeq, ySeq, zSeq, varSeq, coors = getData_sowfa(ppDir, 'prbg1', ((0,0,0),30.0), 'U', 0)
freq, coh_av_0, coh_std_0, co_coh_av_0, co_coh_std_0, phase_av_0, phase_std_0 = coh_av_sowfa_y(0, 0, 50, 1, 4, xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq)
freq, coh_av_1, coh_std_1, co_coh_av_1, co_coh_std_1, phase_av_1, phase_std_1 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq)
freq, coh_av_2, coh_std_2, co_coh_av_2, co_coh_std_2, phase_av_2, phase_std_2 = coh_av_sowfa_y(0, 0, 50, 3, 4, xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq)

# calculate coherence (ver)
tSeq, xSeq, ySeq, zSeq, varSeq, coors = getData_sowfa(ppDir, 'prbg2', ((0,0,0),30.0), 'U', 0)
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_sowfa((0,0,4), (0,0,6), xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq, 240)
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_sowfa((0,0,4), (0,0,8), xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq, 240)
# t_seq_2, u0_2, u1_2, freq, coh_2, co_coh_2, phase_2 = coh_sowfa((0,0,4), (0,0,8), xSeq.size, ySeq.size, (144000.0, 146400, 0.1), tSeq, varSeq, 240)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_NBL'
dir = prjDir + '/' + jobName

# calculate coherence (long)
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M03', ['.001','.002'], 'u')
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_palm((24,0,4), (26,0,4), (144000.0, 146400, 0.1), tSeq, varSeq, 120)
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_palm((23,0,4), (27,0,4), (144000.0, 146400, 0.1), tSeq, varSeq, 120)
t_seq_2, u0_2, u1_2, freq, coh_2, co_coh_2, phase_2 = coh_palm((21,0,4), (29,0,4), (144000.0, 146400, 0.1), tSeq, varSeq, 120)

# calculate coherence (lat)
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M04', ['.001','.002'], 'u')
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_palm((0,24,4), (0,26,4), (144000.0, 146400, 0.1), tSeq, varSeq, 240)
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_palm((0,23,4), (0,27,4), (144000.0, 146400, 0.1), tSeq, varSeq, 240)
t_seq_2, u0_2, u1_2, freq, coh_2, co_coh_2, phase_2 = coh_palm((0,21,4), (0,29,4), (144000.0, 146400, 0.1), tSeq, varSeq, 240)
coh0_0 = calc_coh0(u0_0,u1_0,2400)
coh0_1 = calc_coh0(u0_1,u1_1,2400)
coh0_2 = calc_coh0(u0_2,u1_2,2400)

# # calculate averaged coherence (lat)
# freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_palm_y(0, 0, 50, 2, 4, (144000.0, 146400, 0.1), tSeq, varSeq, 240)

# calculate coherence (ver)
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M05', ['.001','.002'], 'u')
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_palm((0,0,4), (0,0,5), (144000.0, 146400, 0.1), tSeq, varSeq, 240)
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_palm((0,0,4), (0,0,6), (144000.0, 146400, 0.1), tSeq, varSeq, 240)
t_seq_2, u0_2, u1_2, freq, coh_2, co_coh_2, phase_2 = coh_palm((0,0,4), (0,0,8), (144000.0, 146400, 0.1), tSeq, varSeq, 240)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5'
dir = prjDir + '/' + jobName

# calculate coherence (lat)
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M04', ['.011','.012','.013','.014'], 'u')
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_palm((0,24,4), (0,26,4), (72000.0, 74400, 0.1), tSeq, varSeq, 240)
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_palm((0,24,4), (0,28,4), (72000.0, 74400, 0.1), tSeq, varSeq, 240)

# calculate coherence (ver)
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M05', ['.011','.012','.013','.014'], 'u')
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_palm((0,0,4), (0,0,6), (72000.0, 74400, 0.1), tSeq, varSeq, 240)
t_seq_1, u0_1, u1_1, freq, coh_1, co_coh_1, phase_1 = coh_palm((0,0,2), (0,0,6), (72000.0, 74400, 0.1), tSeq, varSeq, 240)



""" plot coherence and fitting curve """
fig, ax = plt.subplots(figsize=(5.2,3))

# # coherence at frequency 0
# coh_0 = np.concatenate((np.array([coh0_0]),coh_0[1:]))
# coh_1 = np.concatenate((np.array([coh0_1]),coh_1[1:]))
# coh_2 = np.concatenate((np.array([coh0_2]),coh_2[1:]))

# # use reduced frequency
d_0 = 40
d_1 = 80
# d_2 = 160
U = (u0_0.mean() + u1_0.mean())/2
# ax.plot(freq[0:]*d_0/U, coh_0[0:], linestyle='-', marker='', markersize=1, color='r', label='40m')
# ax.plot(freq[0:]*d_1/U, coh_1[0:], linestyle='-', marker='', markersize=1, color='b', label='80m')
# ax.plot(freq[0:]*d_2/U, coh_2[0:], linestyle='-', marker='', markersize=1, color='g', label='160m')

ax.plot(freq[0:], coh_0[0:], linestyle='-', marker='', markersize=1, color='r', label='40m')
ax.plot(freq[0:], coh_1[0:], linestyle='-', marker='', markersize=1, color='b', label='80m')
# ax.plot(freq[0:], coh_2[0:], linestyle='-', marker='', markersize=1, color='g', label='80m')

# # averaged coh
# ax.plot(freq[1:], coh_av_0[1:], linestyle='-', marker='', markersize=1, color='r', label='20m')
# ax.fill_between(freq[1:], coh_av_0[1:]-coh_std_0[1:], coh_av_0[1:]+coh_std_0[1:], color='salmon')
# ax.plot(freq[1:], coh_av_1[1:], linestyle='-', marker='', markersize=1, color='b', label='40m')
# ax.fill_between(freq[1:], coh_av_1[1:]-coh_std_1[1:], coh_av_1[1:]+coh_std_1[1:], color='lightskyblue')
# ax.plot(freq[1:], coh_av_2[1:], linestyle='-', marker='', markersize=1, color='g', label='60m')
# ax.fill_between(freq[1:], coh_av_2[1:]-coh_std_2[1:], coh_av_2[1:]+coh_std_2[1:], color='lightgreen')

# IEC standard coherence model (kaimal model)
# iec_coh_0 = funcs.IEC_coh(freq, 40, 8.11, 8.1*42)
# iec_coh_1 = funcs.IEC_coh(freq, 80, 8.11, 8.1*42)
# zoomin_ax.plot(freq[0:], iec_coh_0[0:], linestyle='-', marker='', markersize=1, color='k', label='IEC-40m')
# zoomin_ax.plot(freq[0:], iec_coh_1[0:], linestyle='--', marker='', markersize=1, color='k', label='IEC-80m')


# # fitting
# f_out = 0.02
# tmp = abs(freq - f_out)
# ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]
# def fitting_func(x, a, alpha):
#     return a * np.exp(- alpha * x)

# popt_0, pcov_0 = curve_fit(fitting_func, freq[ind_in:ind_out]*d_0/U, coh_0[ind_in:ind_out], bounds=(0, [1, 500]))
# ax.plot(freq, fitting_func(freq*d_0/U, *popt_0), linestyle='-', color='r',
#      label='a=%5.3f, alpha=%5.3f' % tuple(popt_0))
# popt_1, pcov_1 = curve_fit(fitting_func, freq[ind_in:ind_out]*d_1/U, coh_1[ind_in:ind_out], bounds=(0, [1, 500]))
# ax.plot(freq, fitting_func(freq*d_1/U, *popt_1), linestyle='-', color='b',
#      label='a=%5.3f, alpha=%5.3f' % tuple(popt_1))
# popt_2, pcov_2 = curve_fit(fitting_func, freq[ind_in:ind_out]*d_2/U, coh_2[ind_in:ind_out], bounds=(0, [1, 100]))
# ax.plot(freq, fitting_func(freq*d_2/U, *popt_2), linestyle='-', color='g',
#      label='a=%5.3f, alpha=%5.3f' % tuple(popt_2))





# this is an inset zoomin plot over the main axes
zoomin_ax = fig.add_axes([.5, .5, .36, .36])
zoomin_ax.plot(freq[0:], coh_0[0:], linestyle='-', marker='', markersize=1, color='r', label='40m')
zoomin_ax.plot(freq[0:], coh_1[0:], linestyle='-', marker='', markersize=1, color='b', label='80m')
# zoomin_ax.plot(freq[0:], coh_2[0:], linestyle='-', marker='', markersize=1, color='g', label='80m')
# zoomin_ax.plot(freq, fitting_func(freq*d_0/U, *popt_0), linestyle='--', color='k',
#      label=r'a=%5.3f, $\alpha$=%5.3f' % tuple(popt_0))
# zoomin_ax.plot(freq, fitting_func(freq*d_1/U, *popt_1), linestyle=':', color='k',
#      label=r'a=%5.3f, $\alpha$=%5.3f' % tuple(popt_1))

xaxis_min = 0
xaxis_max = 0.1
xaxis_d = 0.02
yaxis_min = 0
yaxis_max = 1.0
yaxis_d = 0.1
zoomin_ax.set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
zoomin_ax.set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)

# plt.xlabel(r'$\mathrm{fd / \overline{u}}$', fontsize=12)
ax.set_xlabel('f (1/s)', fontsize=12)
ax.set_ylabel('coherence', fontsize=12)
xaxis_min = 0
xaxis_max = 0.5
xaxis_d = 0.1
yaxis_min = 0
yaxis_max = 1.0
yaxis_d = 0.1
ax.set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
ax.set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
# ax.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
# ax.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=10)
# plt.legend(bbox_to_anchor=(0.5,0.75), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
zoomin_ax.legend(bbox_to_anchor=(0.52,0.72), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
ax.grid()
plt.title('')
# fig.tight_layout() # adjust the layout
# saveName = 'coh_ver_h100_palm.png'
# plt.savefig('/scratch/projects/deepwind/photo/coherence' + '/' + saveName)
plt.show()
plt.close()



""" plot co-coherence """
fig, ax = plt.subplots(figsize=(5.2,3))

# # use reduced frequency
# d_0 = 20
# d_1 = 40
# d_2 = 60
# U = (u0_0.mean() + u1_0.mean())/2
# ax.plot(freq[1:]*d_0/U, co_coh_0[1:], linestyle='', marker='o', markersize=1, color='r', label='20m')
# ax.plot(freq[1:]*d_1/U, co_coh_1[1:], linestyle='', marker='o', markersize=1, color='b', label='40m')
# ax.plot(freq[1:]*d_2/U, co_coh_2[1:], linestyle='', marker='o', markersize=1, color='g', label='60m')

# use frequency
ax.plot(freq[0:], co_coh_0[0:], linestyle='-', marker='', markersize=1, color='r', label='40m')
ax.plot(freq[0:], co_coh_1[0:], linestyle='-', marker='', markersize=1, color='b', label='80m')
# ax.plot(freq[0:], co_coh_2[0:], linestyle='-', marker='', markersize=1, color='g', label='80m')

plt.xlabel('f (1/s)', fontsize=12)
plt.ylabel('co-coherence', fontsize=12)
xaxis_min = 0
xaxis_max = 0.5
xaxis_d = 0.1
yaxis_min = -1.0
yaxis_max = 1.0
yaxis_d = 0.2
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=10)
# ax.text(0.0, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm', transform=ax.transAxes, fontsize=12)
# ax.text(0.8, 1.02, 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
# plt.legend(bbox_to_anchor=(0.5,0.8), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(0.7,0.85), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'cocoh_ver_h100_palm.png'
# plt.savefig('/scratch/projects/deepwind/photo/coherence' + '/' + saveName)
plt.show()
plt.close()


""" plot phase """
fig, ax = plt.subplots(figsize=(6,4))

# # use reduced frequency
# d_0 = 40
# d_1 = 80
# d_2 = 160
# U = (u0_0.mean() + u1_0.mean())/2
# ax.plot(freq[1:]*d_0/U, phase_0[1:], linestyle='', marker='o', markersize=1, color='r', label='40m')
# ax.plot(freq[1:]*d_1/U, phase_1[1:], linestyle='', marker='o', markersize=1, color='b', label='80m')
# ax.plot(freq[1:]*d_2/U, phase_2[1:], linestyle='', marker='o', markersize=1, color='g', label='160m')

# use frequency
ax.plot(freq[0:], phase_0[0:], linestyle='', marker='o', markersize=2, color='r', label='40m')
ax.plot(freq[0:], phase_1[0:], linestyle='', marker='o', markersize=2, color='b', label='80m')
# ax.plot(freq[0:], phase_2[0:], linestyle='', marker='o', markersize=2, color='g', label='80m')

plt.xlabel('f (1/s)', fontsize=12)
plt.ylabel('phase', fontsize=12)
xaxis_min = 0
xaxis_max = 0.5
xaxis_d = 0.1
yaxis_min = -1.0*np.pi
yaxis_max = 1.0*np.pi
yaxis_d = np.pi/4
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), \
['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=10)
# ax.text(0.0, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm', transform=ax.transAxes, fontsize=12)
# ax.text(0.8, 1.02, 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
plt.legend(bbox_to_anchor=(0.7,0.85), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'phase_ver_h100_palm.png'
# plt.savefig('/scratch/projects/deepwind/photo/coherence' + '/' + saveName)
plt.show()
plt.close()


































""" check Maria's coherence results """
### check time series
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(t_seq_1, u0_1)
ax.plot(t_seq_1, u1_1)
plt.grid()
plt.show()

### case 1
prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_u12_cbl'
# coh_long
dir = prjDir + '/' + jobName
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M01', ['.001'], 'u')
p0 = (0,0,0) # coor:(0,2565,115)
p1 = (4,0,0) # coor:(40,2565,115)
t_para = (86400, 86400+2400, 0.5)
# # calculate coherence
# t_seq, var0, var1, freq, coh, co_coh, phase = coh_palm(p0, p1, t_para, tSeq, varSeq)
# funcs.group_plot_0(t_seq - t_para[0], 2, var0-var0.mean(), var1-var1.mean())
# averaged coherence
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_palm_x(0, 255, 1, 0, 0, t_para, tSeq, varSeq)
plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std)

# coh_lat
dir = prjDir + '/' + jobName
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M02', ['.001'], 'u')
p0 = (0,0,0) # coor:(2560,5,115)
p1 = (0,4,0) # coor:(2560,45,115)
t_para = (86400, 86400+2400, 0.5)
# # calculate coherence
# t_seq, var0, var1, freq, coh, co_coh, phase = coh_palm(p0, p1, t_para, tSeq, varSeq)
# funcs.group_plot_0(t_seq - t_para[0], 2, var0-var0.mean(), var1-var1.mean())
# averaged coherence
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_palm_y(0, 0, 255, 1, 0, t_para, tSeq, varSeq)
plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std)

# coh_ver
dir = prjDir + '/' + jobName
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M03', ['.001'], 'u')
p0 = (0,0,10) # coor:(2560,2565,95)
p1 = (0,0,18) # coor:(2560,2565,135)
t_para = (86400, 86400+2400, 0.5)
# calculate coherence
t_seq, var0, var1, freq, coh, co_coh, phase = coh_palm(p0, p1, t_para, tSeq, varSeq)
funcs.group_plot_0(t_seq - t_para[0], 2, var0-var0.mean(), var1-var1.mean())


### case 2
prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_u12_sbl'
# coh_long
dir = prjDir + '/' + jobName
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M01', ['.001'], 'u')
p0 = (0,0,0) # coor:(0,1285,115)
p1 = (4,0,0) # coor:(40,1285,115)
t_para = (43200, 43200+2400, 0.5)
# # calculate coherence
# t_seq, var0, var1, freq, coh, co_coh, phase = coh_palm(p0, p1, t_para, tSeq, varSeq)
# funcs.group_plot_0(t_seq - t_para[0], 2, var0-var0.mean(), var1-var1.mean())
# averaged coherence
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_palm_x(0, 255, 1, 0, 0, t_para, tSeq, varSeq)
plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std)

# coh_lat
dir = prjDir + '/' + jobName
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M02', ['.001'], 'u')
p0 = (0,0,0) # coor:(1280,5,115)
p1 = (0,4,0) # coor:(1280,45,115)
t_para = (43200, 43200+2400, 0.5)
# # calculate coherence
# t_seq, var0, var1, freq, coh, co_coh, phase = coh_palm(p0, p1, t_para, tSeq, varSeq)
# funcs.group_plot_0(t_seq - t_para[0], 2, var0-var0.mean(), var1-var1.mean())
# averaged coherence
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_palm_y(0, 0, 255, 1, 0, t_para, tSeq, varSeq)
plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std)

# coh_ver
dir = prjDir + '/' + jobName
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M03', ['.001'], 'u')
p0 = (0,0,10) # coor:(1280,1285,95)
p1 = (0,0,18) # coor:(1280,1285,135)
t_para = (43200, 43200+2400, 0.5)
# calculate coherence
t_seq, var0, var1, freq, coh, co_coh, phase = coh_palm(p0, p1, t_para, tSeq, varSeq)
funcs.group_plot_0(t_seq - t_para[0], 2, var0-var0.mean(), var1-var1.mean())


### case 3
prjDir = '/scratch/palmdata/JOBS'
jobName  = 'pcr_NBL_U10'
# coh_long
dir = prjDir + '/' + jobName
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M03', ['.001','.002'], 'u')
p0 = (0,0,0) # coor:(780,1285,115)
p1 = (2,0,0) # coor:(820,1285,115)
t_para = (144000.76, 144000.65+2400, 0.1)
# # calculate coherence
# t_seq, var0, var1, freq, coh, co_coh, phase = coh_palm(p0, p1, t_para, tSeq, varSeq)
# funcs.group_plot_0(t_seq - t_para[0], 2, var0-var0.mean(), var1-var1.mean())
# averaged coherence
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_palm_x(0, 50, 1, 0, 0, t_para, tSeq, varSeq)
plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std)

# coh_lat
dir = prjDir + '/' + jobName
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M04', ['.001','.002'], 'u')
p0 = (0,0,0) # coor:(1280,780,115)
p1 = (0,2,0) # coor:(1280,820,115)
t_para = (144000, 144000+2400, 0.1)
# # calculate coherence
# t_seq, var0, var1, freq, coh, co_coh, phase = coh_palm(p0, p1, t_para, tSeq, varSeq)
# funcs.group_plot_0(t_seq - t_para[0], 2, var0-var0.mean(), var1-var1.mean())
# averaged coherence
freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std = coh_av_palm_y(0, 0, 50, 8, 0, t_para, tSeq, varSeq)
plot_coh_av_zoomin(freq, coh_av, coh_std, co_coh_av, co_coh_std, phase_av, phase_std)

# coh_ver
dir = prjDir + '/' + jobName
tSeq, xSeq, ySeq, zSeq, varSeq = getData_palm(dir, jobName, 'M04', ['.001','.002'], 'u')
p0 = (0,0,1) # coor:(1280,1285,95)
p1 = (0,0,2) # coor:(1280,1285,135)
t_para = (144000, 144000+2400, 0.1)
# calculate coherence
t_seq, var0, var1, freq, coh, co_coh, phase = coh_palm(p0, p1, t_para, tSeq, varSeq)
funcs.group_plot_0(t_seq - t_para[0], 2, var0-var0.mean(), var1-var1.mean())









""" plot psd and csd """
fs = 10
# PSD
freq, S0_0 = scipy.signal.csd(u0_0, u0_0, fs, nperseg=120*fs, noverlap=None)
freq, S1_0 = scipy.signal.csd(u1_0, u1_0, fs, nperseg=120*fs, noverlap=None)
freq, S0_1 = scipy.signal.csd(u0_1, u0_1, fs, nperseg=120*fs, noverlap=None)
freq, S1_1 = scipy.signal.csd(u1_1, u1_1, fs, nperseg=120*fs, noverlap=None)
# CSD
freq_, S01_0 = scipy.signal.csd(u0_0, u1_0, fs, nperseg=120*fs, noverlap=None)
freq_, S01_1 = scipy.signal.csd(u0_1, u1_1, fs, nperseg=120*fs, noverlap=None)


fig, ax = plt.subplots(figsize=(6,4))
plt.loglog(freq, S0_0, label='sowfa-0', linewidth=1.0, linestyle='-', color='r')
plt.loglog(freq, S1_0, label='sowfa-1', linewidth=1.0, linestyle='-', color='b')
plt.loglog(freq_, abs(S01_0), label='sowfa-01', linewidth=1.0, linestyle='-', color='g')
plt.loglog(freq, S0_1, label='palm-0', linewidth=1.0, linestyle='--', color='r')
plt.loglog(freq, S1_1, label='palm-1', linewidth=1.0, linestyle='--', color='b')
plt.loglog(freq_, abs(S01_1), label='palm-01', linewidth=1.0, linestyle='--', color='g')
# -5/3 law
f_ = np.linspace(1e-2,1e0,100)
plt.loglog(f_, 1e-1*np.power(f_, -5/3), label='-5/3 law', linewidth=2.0, color='k')
plt.xlabel('f (1/s)')
plt.ylabel('Spectra' + ' (' + r'$\mathrm{m^2/s}$' + ')')
xaxis_min = 1e-3
xaxis_max = 5 # f_seq.max()
yaxis_min = 1e-16
yaxis_max = 1e3
plt.ylim(yaxis_min, yaxis_max)
plt.xlim(xaxis_min, xaxis_max)
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'Su' + '_f_h' + str(int(zSeq_0[4])) + '.png'
# plt.savefig(ppDir + '/' + saveName, bbox_inches='tight')
plt.show()


""" plot coherence and fitting curve """
def fitting_func(x, a, alpha):
    return a * np.exp(- alpha * x)

f_out = 1.0
tmp = abs(freq - f_out)
ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(freq[1:], coh_0[1:], linestyle='', marker='o', markersize=1, color='r', label='sowfa')
popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh_0[ind_in:ind_out], bounds=(0, [1, 600]))
ax.plot(freq, fitting_func(freq, *popt), linestyle='-', color='r',
     label='a=%5.3f, alpha=%5.3f' % tuple(popt))
ax.plot(freq[1:], coh_1[1:], linestyle='', marker='o', markersize=1, color='b', label='palm')
# popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh_1[ind_in:ind_out], bounds=(0, [1, 600]))
# ax.plot(freq, fitting_func(freq, *popt), linestyle='-', color='b',
#      label='a=%5.3f, alpha=%5.3f' % tuple(popt))

plt.xlabel('f (1/s)', fontsize=12)
plt.ylabel('coherence', fontsize=12)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = 0
yaxis_max = 1.0
yaxis_d = 0.1
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=10)
# ax.text(0.0, 1.02, 'fs = ' + str(fs) + 'Hz' + ', ' 'nperseg = ' + str(segNum), transform=ax.transAxes, fontsize=12)
# ax.text(0.56, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm' + ', ' + 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
plt.legend(bbox_to_anchor=(0.5,0.75), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'uu_coh_long' + '_' + str(int(zSeq[zInd])) + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()

""" plot co-coherence """
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(freq[1:], co_coh_0[1:], linestyle='', marker='o', markersize=1, color='r', label='sowfa')
ax.plot(freq[1:], co_coh_1[1:], linestyle='', marker='o', markersize=1, color='b', label='palm')

plt.xlabel('f (1/s)', fontsize=12)
plt.ylabel('co-coherence', fontsize=12)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = -1.0
yaxis_max = 1.0
yaxis_d = 0.2
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=10)
# ax.text(0.0, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm', transform=ax.transAxes, fontsize=12)
# ax.text(0.8, 1.02, 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
# plt.legend(bbox_to_anchor=(0.5,0.8), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.legend(bbox_to_anchor=(0.7,0.85), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'uu_cocoh_long' + '_' + str(int(zSeq[zInd])) + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()

""" plot phase """
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(freq[1:], phase_0[1:], linestyle='', marker='o', markersize=1, color='r', label='sowfa')
ax.plot(freq[1:], phase_1[1:], linestyle='', marker='o', markersize=1, color='b', label='palm')

plt.xlabel('f (1/s)', fontsize=12)
plt.ylabel('phase', fontsize=12)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = -1.0*np.pi
yaxis_max = 1.0*np.pi
yaxis_d = np.pi/4
plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=10)
plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), \
['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=10)
# ax.text(0.0, 1.02, 'dx = ' + str(np.round(dx,1)) + 'm', transform=ax.transAxes, fontsize=12)
# ax.text(0.8, 1.02, 'h = ' + str(np.round(p0_coor[2])) + 'm', transform=ax.transAxes, fontsize=12)
plt.legend(bbox_to_anchor=(0.7,0.85), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
# saveName = 'uu_phase_long' + '_' + str(int(zSeq[zInd])) + '_pr.png'
# plt.savefig(ppDir + '/' + saveName)
plt.show()
