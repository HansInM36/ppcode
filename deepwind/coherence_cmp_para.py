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

def coh_av_palm_x(xInd_start, xInd_end, dxInd, yInd, zInd, t_para, tSeq, varSeq, tL):
    N = xInd_end - xInd_start - dxInd + 1
    coh = []
    co_coh = []
    phase = []
    for i in range(N):
        results = coh_palm((xInd_start+i,yInd,zInd), (xInd_start+i+dxInd,yInd,zInd), t_para, tSeq, varSeq, tL)
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

def coh_av_sowfa_x(xInd_start, xInd_end, dxInd, yInd, zInd, xNum, yNum, t_para, tSeq, varSeq, tL):
    N = xInd_end - xInd_start - dxInd + 1
    coh = []
    co_coh = []
    phase = []
    for i in range(N):
        results = coh_sowfa((xInd_start+i,yInd,zInd), (xInd_start+i+dxInd,yInd,zInd), xNum, yNum, t_para, tSeq, varSeq, tL)
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
def coh_av_sowfa_y(xInd, yInd_start, yInd_end, dyInd, zInd, xNum, yNum, t_para, tSeq, varSeq, tL):
    N = yInd_end - yInd_start - dyInd + 1
    coh = []
    co_coh = []
    phase = []
    for i in range(N):
        results = coh_sowfa((xInd,yInd_start+i,zInd), (xInd,yInd_start+i+dyInd,zInd), xNum, yNum, t_para, tSeq, varSeq, tL)
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



"""
plot averaged co_coherence to investigate the effect of sampling frequency
using SOWFA data, lateral separation 40m
"""
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg1', ((0,0,0),30.0), 'U', 0)
# calculate coherence
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_sowfa((0,20,4), (0,24,4), xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.1), tSeq_0, varSeq_0, 240)
# averaged coherence
freq_0, coh_av_0, coh_std_0, co_coh_av_0, co_coh_std_0, phase_av_0, phase_std_0 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.1), tSeq_0, varSeq_0, 240)
freq_1, coh_av_1, coh_std_1, co_coh_av_1, co_coh_std_1, phase_av_1, phase_std_1 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.2), tSeq_0, varSeq_0, 240)
freq_2, coh_av_2, coh_std_2, co_coh_av_2, co_coh_std_2, phase_av_2, phase_std_2 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.5), tSeq_0, varSeq_0, 240)
freq_3, coh_av_3, coh_std_3, co_coh_av_3, co_coh_std_3, phase_av_3, phase_std_3 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 1.0), tSeq_0, varSeq_0, 240)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(freq_0[0:], co_coh_av_0[0:], linestyle='-', marker='', markersize=1, color='r', label='fs=10Hz')
ax.plot(freq_1[0:], co_coh_av_1[0:], linestyle='-', marker='', markersize=1, color='b', label='fs=5Hz')
ax.plot(freq_2[0:], co_coh_av_2[0:], linestyle='-', marker='', markersize=1, color='g', label='fs=2Hz')
ax.plot(freq_3[0:], co_coh_av_3[0:], linestyle='-', marker='', markersize=1, color='orange', label='fs=1Hz')
ax.fill_between(freq_0[0:], co_coh_av_0[0:]-co_coh_std_0[0:], co_coh_av_0[0:]+co_coh_std_0[0:], color='salmon', alpha=0.8)
ax.fill_between(freq_1[0:], co_coh_av_1[0:]-co_coh_std_1[0:], co_coh_av_1[0:]+co_coh_std_1[0:], color='lightskyblue', alpha=0.6)
ax.fill_between(freq_2[0:], co_coh_av_2[0:]-co_coh_std_2[0:], co_coh_av_2[0:]+co_coh_std_2[0:], color='lightgreen', alpha=0.5)
ax.fill_between(freq_3[0:], co_coh_av_3[0:]-co_coh_std_3[0:], co_coh_av_3[0:]+co_coh_std_3[0:], color='gold', alpha=0.4)
# ax.errorbar(freq_0[0:], co_coh_av_0[0:], yerr=co_coh_std_0[0:], linestyle='-', fmt='-o', capsize=3, color='r', label='fs=10')
# ax.errorbar(freq_1[0:], co_coh_av_1[0:], yerr=co_coh_std_1[0:], linestyle='-', fmt='-o', capsize=3, color='b', label='fs=5')
# ax.errorbar(freq_2[0:], co_coh_av_2[0:], yerr=co_coh_std_2[0:], linestyle='-', fmt='-o', capsize=3, color='g', label='fs=2')
# ax.errorbar(freq_3[0:], co_coh_av_3[0:], yerr=co_coh_std_3[0:], linestyle='-', fmt='-o', capsize=3, color='y', label='fs=1')
ax.set_xlabel('f (1/s)', fontsize=12)
ax.set_ylabel('co-coh', fontsize=12)
xaxis_min = 0
xaxis_max = 0.5
xaxis_d = 0.1
yaxis_min = -0.4
yaxis_max = 1.0
yaxis_d = 0.2
ax.set_ylim(yaxis_min, yaxis_max)
ax.set_xlim(xaxis_min, xaxis_max)
ax.set_xticks(list(np.linspace(xaxis_min, xaxis_max, int(np.round((xaxis_max-xaxis_min)/xaxis_d)+1))))
ax.set_yticks(list(np.linspace(yaxis_min, yaxis_max, int(np.round((yaxis_max-yaxis_min)/yaxis_d)+1))))
plt.legend(bbox_to_anchor=(0.84,0.8), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
ax.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'co-coh_av_cmp_fs.png'
plt.savefig('/scratch/projects/deepwind/photo/coherence_sensitivity' + '/' + saveName)
plt.show()
plt.close()


"""
plot averaged co_coherence to investigate the effect of sampling time
using SOWFA data, lateral separation 40m
"""
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg1', ((0,0,0),30.0), 'U', 0)
# calculate coherence
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_sowfa((0,20,4), (0,24,4), xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
# averaged coherence
freq_0, coh_av_0, coh_std_0, co_coh_av_0, co_coh_std_0, phase_av_0, phase_std_0 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
freq_1, coh_av_1, coh_std_1, co_coh_av_1, co_coh_std_1, phase_av_1, phase_std_1 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 147600.0, 0.5), tSeq_0, varSeq_0, 240)
freq_2, coh_av_2, coh_std_2, co_coh_av_2, co_coh_std_2, phase_av_2, phase_std_2 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.5), tSeq_0, varSeq_0, 240)
freq_3, coh_av_3, coh_std_3, co_coh_av_3, co_coh_std_3, phase_av_3, phase_std_3 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 145200.0, 0.5), tSeq_0, varSeq_0, 240)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(freq_0[0:], co_coh_av_0[0:], linestyle='-', marker='', markersize=1, color='r', label='Ts=7200s')
ax.plot(freq_1[0:], co_coh_av_1[0:], linestyle='-', marker='', markersize=1, color='b', label='Ts=3600s')
ax.plot(freq_2[0:], co_coh_av_2[0:], linestyle='-', marker='', markersize=1, color='g', label='Ts=2400s')
ax.plot(freq_3[0:], co_coh_av_3[0:], linestyle='-', marker='', markersize=1, color='orange', label='Ts=1200s')
ax.fill_between(freq_0[0:], co_coh_av_0[0:]-co_coh_std_0[0:], co_coh_av_0[0:]+co_coh_std_0[0:], color='salmon', alpha=0.8)
ax.fill_between(freq_1[0:], co_coh_av_1[0:]-co_coh_std_1[0:], co_coh_av_1[0:]+co_coh_std_1[0:], color='lightskyblue', alpha=0.6)
ax.fill_between(freq_2[0:], co_coh_av_2[0:]-co_coh_std_2[0:], co_coh_av_2[0:]+co_coh_std_2[0:], color='lightgreen', alpha=0.5)
ax.fill_between(freq_3[0:], co_coh_av_3[0:]-co_coh_std_3[0:], co_coh_av_3[0:]+co_coh_std_3[0:], color='gold', alpha=0.4)
# ax.errorbar(freq_0[0:], co_coh_av_0[0:], yerr=co_coh_std_0[0:], linestyle='-', fmt='-o', capsize=3, color='r', label='Ts=800s')
# ax.errorbar(freq_1[0:], co_coh_av_1[0:], yerr=co_coh_std_1[0:], linestyle='-', fmt='-o', capsize=3, color='b', label='Ts=1200s')
# ax.errorbar(freq_2[0:], co_coh_av_2[0:], yerr=co_coh_std_2[0:], linestyle='-', fmt='-o', capsize=3, color='g', label='Ts=1800s')
# ax.errorbar(freq_3[0:], co_coh_av_3[0:], yerr=co_coh_std_3[0:], linestyle='-', fmt='-o', capsize=3, color='y', label='Ts=2400s')
ax.set_xlabel('f (1/s)', fontsize=12)
ax.set_ylabel('co-coh', fontsize=12)
xaxis_min = 0
xaxis_max = 0.5
xaxis_d = 0.1
yaxis_min = -0.4
yaxis_max = 1.0
yaxis_d = 0.2
ax.set_ylim(yaxis_min, yaxis_max)
ax.set_xlim(xaxis_min, xaxis_max)
ax.set_xticks(list(np.linspace(xaxis_min, xaxis_max, int(np.round((xaxis_max-xaxis_min)/xaxis_d)+1))))
ax.set_yticks(list(np.linspace(yaxis_min, yaxis_max, int(np.round((yaxis_max-yaxis_min)/yaxis_d)+1))))
plt.legend(bbox_to_anchor=(0.8,0.8), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
ax.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'co-coh_av_cmp_Ts.png'
plt.savefig('/scratch/projects/deepwind/photo/coherence_sensitivity' + '/' + saveName)
plt.show()
plt.close()


"""
plot averaged co_coherence to investigate the effect of segment length
using SOWFA data, lateral separation 40m
"""
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg1', ((0,0,0),30.0), 'U', 0)
# calculate coherence
t_seq_0, u0_0, u1_0, freq, coh_0, co_coh_0, phase_0 = coh_sowfa((0,20,4), (0,24,4), xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
# averaged coherence
freq_0, coh_av_0, coh_std_0, co_coh_av_0, co_coh_std_0, phase_av_0, phase_std_0 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.5), tSeq_0, varSeq_0, 480)
freq_1, coh_av_1, coh_std_1, co_coh_av_1, co_coh_std_1, phase_av_1, phase_std_1 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.5), tSeq_0, varSeq_0, 240)
freq_2, coh_av_2, coh_std_2, co_coh_av_2, co_coh_std_2, phase_av_2, phase_std_2 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.5), tSeq_0, varSeq_0, 120)
freq_3, coh_av_3, coh_std_3, co_coh_av_3, co_coh_std_3, phase_av_3, phase_std_3 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 146400.0, 0.5), tSeq_0, varSeq_0, 60)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(freq_0[0:], co_coh_av_0[0:], linestyle='-', marker='', markersize=1, color='r', label='segL=480s')
ax.plot(freq_1[0:], co_coh_av_1[0:], linestyle='-', marker='', markersize=1, color='b', label='segL=240s')
ax.plot(freq_2[0:], co_coh_av_2[0:], linestyle='-', marker='', markersize=1, color='g', label='segL=120s')
ax.plot(freq_3[0:], co_coh_av_3[0:], linestyle='-', marker='', markersize=1, color='orange', label='segL=60s')
ax.fill_between(freq_0[0:], co_coh_av_0[0:]-co_coh_std_0[0:], co_coh_av_0[0:]+co_coh_std_0[0:], color='salmon', alpha=0.8)
ax.fill_between(freq_1[0:], co_coh_av_1[0:]-co_coh_std_1[0:], co_coh_av_1[0:]+co_coh_std_1[0:], color='lightskyblue', alpha=0.6)
ax.fill_between(freq_2[0:], co_coh_av_2[0:]-co_coh_std_2[0:], co_coh_av_2[0:]+co_coh_std_2[0:], color='lightgreen', alpha=0.5)
ax.fill_between(freq_3[0:], co_coh_av_3[0:]-co_coh_std_3[0:], co_coh_av_3[0:]+co_coh_std_3[0:], color='gold', alpha=0.4)
# ax.errorbar(freq_0[0:], co_coh_av_0[0:], yerr=co_coh_std_0[0:], linestyle='-', fmt='-o', capsize=3, color='r', label='segL=60s')
# ax.errorbar(freq_1[0:], co_coh_av_1[0:], yerr=co_coh_std_1[0:], linestyle='-', fmt='-o', capsize=3, color='b', label='segL=120s')
# ax.errorbar(freq_2[0:], co_coh_av_2[0:], yerr=co_coh_std_2[0:], linestyle='-', fmt='-o', capsize=3, color='g', label='segL=240s')
# ax.errorbar(freq_3[0:], co_coh_av_3[0:], yerr=co_coh_std_3[0:], linestyle='-', fmt='-o', capsize=3, color='y', label='segL=480s')
ax.set_xlabel('f (1/s)', fontsize=12)
ax.set_ylabel('co-coh', fontsize=12)
xaxis_min = 0
xaxis_max = 0.5
xaxis_d = 0.1
yaxis_min = -0.4
yaxis_max = 1.0
yaxis_d = 0.2
ax.set_ylim(yaxis_min, yaxis_max)
ax.set_xlim(xaxis_min, xaxis_max)
ax.set_xticks(list(np.linspace(xaxis_min, xaxis_max, int(np.round((xaxis_max-xaxis_min)/xaxis_d)+1))))
ax.set_yticks(list(np.linspace(yaxis_min, yaxis_max, int(np.round((yaxis_max-yaxis_min)/yaxis_d)+1))))
plt.legend(bbox_to_anchor=(0.8,0.8), loc=6, borderaxespad=0, fontsize=10) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
ax.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'co-coh_av_cmp_segL.png'
plt.savefig('/scratch/projects/deepwind/photo/coherence_sensitivity' + '/' + saveName)
plt.show()
plt.close()
