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
    freq, coh, co_coh, quad_coh, phase = funcs.coherence(u0_, u1_, fs, segNum)
    return t_seq, u0_, u1_, freq, coh, co_coh, quad_coh, phase
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
    freq, coh, co_coh, quad_coh, phase = funcs.coherence(u0_, u1_, fs, segNum)
    return t_seq, u0_, u1_, freq, coh, co_coh, quad_coh, phase

def coh_av_palm_x(xInd_start, xInd_end, dxInd, yInd, zInd, t_para, tSeq, varSeq, tL):
    N = xInd_end - xInd_start - dxInd + 1
    coh = []
    co_coh = []
    quad_coh = []
    phase = []
    for i in range(N):
        results = coh_palm((xInd_start+i,yInd,zInd), (xInd_start+i+dxInd,yInd,zInd), t_para, tSeq, varSeq, tL)
        coh.append(results[4])
        co_coh.append(results[5])
        quad_coh.append(results[6])
        phase.append(results[7])
        freq = results[3]
    # coh = np.array(coh)
    # coh_av = np.mean(coh, axis=0)
    # coh_std = np.std(coh, axis=0)
    co_coh = np.array(co_coh)
    co_coh_av = np.mean(co_coh, axis=0)
    co_coh_std = np.std(co_coh, axis=0)
    quad_coh = np.array(quad_coh)
    quad_coh_av = np.mean(quad_coh, axis=0)
    quad_coh_std = np.std(quad_coh, axis=0)
    coh_av = np.power(np.power(co_coh_av,2)+np.power(quad_coh_av,2),0.5)
    coh_std = np.power(np.power(co_coh_std,2)+np.power(quad_coh_std,2),0.5)
    phase = np.array(phase)
    phase_av = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return freq, coh_av, coh_std, co_coh_av, co_coh_std, quad_coh_av, quad_coh_std, phase_av, phase_std
def coh_av_palm_y(xInd, yInd_start, yInd_end, dyInd, zInd, t_para, tSeq, varSeq, tL):
    N = yInd_end - yInd_start - dyInd + 1
    coh = []
    co_coh = []
    quad_coh = []
    phase = []
    for i in range(N):
        results = coh_palm((xInd,yInd_start+i,zInd), (xInd,yInd_start+i+dyInd,zInd), t_para, tSeq, varSeq, tL)
        coh.append(results[4])
        co_coh.append(results[5])
        quad_coh.append(results[6])
        phase.append(results[7])
        freq = results[3]
    # coh = np.array(coh)
    # coh_av = np.mean(coh, axis=0)
    # coh_std = np.std(coh, axis=0)
    co_coh = np.array(co_coh)
    co_coh_av = np.mean(co_coh, axis=0)
    co_coh_std = np.std(co_coh, axis=0)
    quad_coh = np.array(quad_coh)
    quad_coh_av = np.mean(quad_coh, axis=0)
    quad_coh_std = np.std(quad_coh, axis=0)
    coh_av = np.power(np.power(co_coh_av,2)+np.power(quad_coh_av,2),0.5)
    coh_std = np.power(np.power(co_coh_std,2)+np.power(quad_coh_std,2),0.5)
    phase = np.array(phase)
    phase_av = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return freq, coh_av, coh_std, co_coh_av, co_coh_std, quad_coh_av, quad_coh_std, phase_av, phase_std
def coh_av_palm_z(xInd, yInd_start, yInd_end, zInd_start, zInd_end, t_para, tSeq, varSeq, tL):
    N = yInd_end - yInd_start + 1
    coh = []
    co_coh = []
    quad_coh = []
    phase = []
    for i in range(N):
        results = coh_palm((xInd,i,zInd_start), (xInd,i,zInd_end), t_para, tSeq, varSeq, tL)
        coh.append(results[4])
        co_coh.append(results[5])
        quad_coh.append(results[6])
        phase.append(results[7])
        freq = results[3]
    # coh = np.array(coh)
    # coh_av = np.mean(coh, axis=0)
    # coh_std = np.std(coh, axis=0)
    co_coh = np.array(co_coh)
    co_coh_av = np.mean(co_coh, axis=0)
    co_coh_std = np.std(co_coh, axis=0)
    quad_coh = np.array(quad_coh)
    quad_coh_av = np.mean(quad_coh, axis=0)
    quad_coh_std = np.std(quad_coh, axis=0)
    coh_av = np.power(np.power(co_coh_av,2)+np.power(quad_coh_av,2),0.5)
    coh_std = np.power(np.power(co_coh_std,2)+np.power(quad_coh_std,2),0.5)
    phase = np.array(phase)
    phase_av = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return freq, coh_av, coh_std, co_coh_av, co_coh_std, quad_coh_av, quad_coh_std, phase_av, phase_std

def coh_av_sowfa_x(xInd_start, xInd_end, dxInd, yInd, zInd, xNum, yNum, t_para, tSeq, varSeq, tL):
    N = xInd_end - xInd_start - dxInd + 1
    coh = []
    co_coh = []
    quad_coh = []
    phase = []
    for i in range(N):
        results = coh_sowfa((xInd_start+i,yInd,zInd), (xInd_start+i+dxInd,yInd,zInd), xNum, yNum, t_para, tSeq, varSeq, tL)
        coh.append(results[4])
        co_coh.append(results[5])
        quad_coh.append(results[6])
        phase.append(results[7])
        freq = results[3]
    # coh = np.array(coh)
    # coh_av = np.mean(coh, axis=0)
    # coh_std = np.std(coh, axis=0)
    co_coh = np.array(co_coh)
    co_coh_av = np.mean(co_coh, axis=0)
    co_coh_std = np.std(co_coh, axis=0)
    quad_coh = np.array(quad_coh)
    quad_coh_av = np.mean(quad_coh, axis=0)
    quad_coh_std = np.std(quad_coh, axis=0)
    coh_av = np.power(np.power(co_coh_av,2)+np.power(quad_coh_av,2),0.5)
    coh_std = np.power(np.power(co_coh_std,2)+np.power(quad_coh_std,2),0.5)
    phase = np.array(phase)
    phase_av = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return freq, coh_av, coh_std, co_coh_av, co_coh_std, quad_coh_av, quad_coh_std, phase_av, phase_std
def coh_av_sowfa_y(xInd, yInd_start, yInd_end, dyInd, zInd, xNum, yNum, t_para, tSeq, varSeq, tL):
    N = yInd_end - yInd_start - dyInd + 1
    coh = []
    co_coh = []
    quad_coh = []
    phase = []
    for i in range(N):
        results = coh_sowfa((xInd,yInd_start+i,zInd), (xInd,yInd_start+i+dyInd,zInd), xNum, yNum, t_para, tSeq, varSeq, tL)
        coh.append(results[4])
        co_coh.append(results[5])
        quad_coh.append(results[6])
        phase.append(results[7])
        freq = results[3]
    # coh = np.array(coh)
    # coh_av = np.mean(coh, axis=0)
    # coh_std = np.std(coh, axis=0)
    co_coh = np.array(co_coh)
    co_coh_av = np.mean(co_coh, axis=0)
    co_coh_std = np.std(co_coh, axis=0)
    quad_coh = np.array(quad_coh)
    quad_coh_av = np.mean(quad_coh, axis=0)
    quad_coh_std = np.std(quad_coh, axis=0)
    coh_av = np.power(np.power(co_coh_av,2)+np.power(quad_coh_av,2),0.5)
    coh_std = np.power(np.power(co_coh_std,2)+np.power(quad_coh_std,2),0.5)
    phase = np.array(phase)
    phase_av = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return freq, coh_av, coh_std, co_coh_av, co_coh_std, quad_coh_av, quad_coh_std, phase_av, phase_std
def coh_av_sowfa_z(xInd, yInd_start, yInd_end, zInd_start, zInd_end, xNum, yNum, t_para, tSeq, varSeq, tL):
    N = yInd_end - yInd_start + 1
    coh = []
    co_coh = []
    quad_coh = []
    phase = []
    for i in range(N):
        results = coh_sowfa((xInd,i,zInd_start), (xInd,i,zInd_end), xNum, yNum, t_para, tSeq, varSeq, tL)
        coh.append(results[4])
        co_coh.append(results[5])
        quad_coh.append(results[6])
        phase.append(results[7])
        freq = results[3]
    # coh = np.array(coh)
    # coh_av = np.mean(coh, axis=0)
    # coh_std = np.std(coh, axis=0)
    co_coh = np.array(co_coh)
    co_coh_av = np.mean(co_coh, axis=0)
    co_coh_std = np.std(co_coh, axis=0)
    quad_coh = np.array(quad_coh)
    quad_coh_av = np.mean(quad_coh, axis=0)
    quad_coh_std = np.std(quad_coh, axis=0)
    coh_av = np.power(np.power(co_coh_av,2)+np.power(quad_coh_av,2),0.5)
    coh_std = np.power(np.power(co_coh_std,2)+np.power(quad_coh_std,2),0.5)
    phase = np.array(phase)
    phase_av = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return freq, coh_av, coh_std, co_coh_av, co_coh_std, quad_coh_av, quad_coh_std, phase_av, phase_std

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
def calc_uz_sowfa(xNum,yNum,zInd,uSeq):
    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)
    uz = np.mean(uSeq[pInd_start:pInd_end])
    return uz
def calc_uz_palm(zInd,uSeq):
    uz = np.mean(uSeq[:,zInd,:,:])
    return uz
def coh_kaimal(f, d, U, L, a, b):
    return np.exp(-a * np.power(np.power(f*d/U,2) + np.power(b*d/L,2),0.5))



""" SOWFA """
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_0 = 'gs10_0.0001_refined'
ppDir_0 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_0
tSeq_0, xSeq_0, ySeq_0, zSeq_0, varSeq_0, coors_0 = getData_sowfa(ppDir_0, 'prbg1', ((0,0,0),30.0), 'U', 0)
# # calculate coherence (lat)
t_seq_0, u0_0, u1_0, freq, coh_0_d40h20, co_coh_0_d40h20, quad_coh_0_d40h20, phase_0_d40h20 = coh_sowfa((0,24,0), (0,26,0), xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
t_seq_0, u0_0, u1_0, freq, coh_0_d80h20, co_coh_0_d80h20, quad_coh_0_d80h20, phase_0_d80h20 = coh_sowfa((0,24,0), (0,28,0), xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
t_seq_0, u0_0, u1_0, freq, coh_0_d40h100, co_coh_0_d40h100, quad_coh_0_d40h100, phase_0_d40h100 = coh_sowfa((0,24,4), (0,26,4), xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
t_seq_0, u0_0, u1_0, freq, coh_0_d80h100, co_coh_0_d80h100, quad_coh_0_d80h100, phase_0_d80h100 = coh_sowfa((0,24,4), (0,28,4), xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
t_seq_0, u0_0, u1_0, freq, coh_0_d40h180, co_coh_0_d40h180, quad_coh_0_d40h180, phase_0_d40h180 = coh_sowfa((0,24,8), (0,26,8), xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
t_seq_0, u0_0, u1_0, freq, coh_0_d80h180, co_coh_0_d80h180, quad_coh_0_d80h180, phase_0_d80h180 = coh_sowfa((0,24,8), (0,28,8), xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
# averaged coherence (lat)
freq, coh_av_0_d40h20, coh_std_0_d40h20, co_coh_av_0_d40h20, co_coh_std_0_d40h20, quad_coh_av_0_d40h20, quad_coh_std_0_d40h20, phase_av_0_d40h20, phase_std_0_d40h20 = coh_av_sowfa_y(0, 0, 50, 2, 0, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
freq, coh_av_0_d80h20, coh_std_0_d80h20, co_coh_av_0_d80h20, co_coh_std_0_d80h20, quad_coh_av_0_d80h20, quad_coh_std_0_d80h20, phase_av_0_d80h20, phase_std_0_d80h20 = coh_av_sowfa_y(0, 0, 50, 4, 0, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
freq, coh_av_0_d40h100, coh_std_0_d40h100, co_coh_av_0_d40h100, co_coh_std_0_d40h100, quad_coh_av_0_d40h100, quad_coh_std_0_d40h100, phase_av_0_d40h100, phase_std_0_d40h100 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
freq, coh_av_0_d80h100, coh_std_0_d80h100, co_coh_av_0_d80h100, co_coh_std_0_d80h100, quad_coh_av_0_d80h100, quad_coh_std_0_d80h100, phase_av_0_d80h100, phase_std_0_d80h100 = coh_av_sowfa_y(0, 0, 50, 4, 4, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
freq, coh_av_0_d40h180, coh_std_0_d40h180, co_coh_av_0_d40h180, co_coh_std_0_d40h180, quad_coh_av_0_d40h180, quad_coh_std_0_d40h180, phase_av_0_d40h180, phase_std_0_d40h180 = coh_av_sowfa_y(0, 0, 50, 2, 8, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)
freq, coh_av_0_d80h180, coh_std_0_d80h180, co_coh_av_0_d80h180, co_coh_std_0_d80h180, quad_coh_av_0_d80h180, quad_coh_std_0_d80h180, phase_av_0_d80h180, phase_std_0_d80h180 = coh_av_sowfa_y(0, 0, 50, 4, 8, xSeq_0.size, ySeq_0.size, (144000.0, 151200.0, 0.5), tSeq_0, varSeq_0, 240)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_1 = 'gs10_refined'
ppDir_1 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_1
tSeq_1, xSeq_1, ySeq_1, zSeq_1, varSeq_1, coors_1 = getData_sowfa(ppDir_1, 'prbg1', ((0,0,0),30.0), 'U', 0)
# calculate coherence (lat)
t_seq_1, u0_1, u1_1, freq, coh_1_d40h20, co_coh_1_d40h20, quad_coh_1_d40h20, phase_1_d40h20 = coh_sowfa((0,24,0), (0,26,0), xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
t_seq_1, u0_1, u1_1, freq, coh_1_d80h20, co_coh_1_d80h20, quad_coh_1_d80h20, phase_1_d80h20 = coh_sowfa((0,24,0), (0,28,0), xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
t_seq_1, u0_1, u1_1, freq, coh_1_d40h100, co_coh_1_d40h100, quad_coh_1_d40h100, phase_1_d40h100 = coh_sowfa((0,24,4), (0,26,4), xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
t_seq_1, u0_1, u1_1, freq, coh_1_d80h100, co_coh_1_d80h100, quad_coh_1_d80h100, phase_1_d80h100 = coh_sowfa((0,24,4), (0,28,4), xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
t_seq_1, u0_1, u1_1, freq, coh_1_d40h180, co_coh_1_d40h180, quad_coh_1_d40h180, phase_1_d40h180 = coh_sowfa((0,24,8), (0,26,8), xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
t_seq_1, u0_1, u1_1, freq, coh_1_d80h180, co_coh_1_d80h180, quad_coh_1_d80h180, phase_1_d80h180 = coh_sowfa((0,24,8), (0,28,8), xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
# averaged coherence (lat)
freq, coh_av_1_d20h20, coh_std_1_d20h20, co_coh_av_1_d20h20, co_coh_std_1_d20h20, quad_coh_av_1_d20h20, quad_coh_std_1_d20h20, phase_av_1_d20h20, phase_std_1_d20h20 = coh_av_sowfa_y(0, 0, 50, 1, 0, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d40h20, coh_std_1_d40h20, co_coh_av_1_d40h20, co_coh_std_1_d40h20, quad_coh_av_1_d40h20, quad_coh_std_1_d40h20, phase_av_1_d40h20, phase_std_1_d40h20 = coh_av_sowfa_y(0, 0, 50, 2, 0, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d80h20, coh_std_1_d80h20, co_coh_av_1_d80h20, co_coh_std_1_d80h20, quad_coh_av_1_d80h20, quad_coh_std_1_d80h20, phase_av_1_d80h20, phase_std_1_d80h20 = coh_av_sowfa_y(0, 0, 50, 4, 0, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d20h100, coh_std_1_d20h100, co_coh_av_1_d20h100, co_coh_std_1_d20h100, quad_coh_av_1_d20h100, quad_coh_std_1_d20h100, phase_av_1_d20h100, phase_std_1_d20h100 = coh_av_sowfa_y(0, 0, 50, 1, 4, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d40h100, coh_std_1_d40h100, co_coh_av_1_d40h100, co_coh_std_1_d40h100, quad_coh_av_1_d40h100, quad_coh_std_1_d40h100, phase_av_1_d40h100, phase_std_1_d40h100 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d80h100, coh_std_1_d80h100, co_coh_av_1_d80h100, co_coh_std_1_d80h100, quad_coh_av_1_d80h100, quad_coh_std_1_d80h100, phase_av_1_d80h100, phase_std_1_d80h100 = coh_av_sowfa_y(0, 0, 50, 4, 4, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d20h180, coh_std_1_d20h180, co_coh_av_1_d20h180, co_coh_std_1_d20h180, quad_coh_av_1_d20h180, quad_coh_std_1_d20h180, phase_av_1_d20h180, phase_std_1_d20h180 = coh_av_sowfa_y(0, 0, 50, 1, 8, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d40h180, coh_std_1_d40h180, co_coh_av_1_d40h180, co_coh_std_1_d40h180, quad_coh_av_1_d40h180, quad_coh_std_1_d40h180, phase_av_1_d40h180, phase_std_1_d40h180 = coh_av_sowfa_y(0, 0, 50, 2, 8, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)
freq, coh_av_1_d80h180, coh_std_1_d80h180, co_coh_av_1_d80h180, co_coh_std_1_d80h180, quad_coh_av_1_d80h180, quad_coh_std_1_d80h180, phase_av_1_d80h180, phase_std_1_d80h180 = coh_av_sowfa_y(0, 0, 50, 4, 8, xSeq_1.size, ySeq_1.size, (144000.0, 151200.0, 0.5), tSeq_1, varSeq_1, 240)


prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName_2 = 'gs10_0.01_refined'
ppDir_2 = '/scratch/sowfadata/pp/' + prjName + '/' + jobName_2
tSeq_2, xSeq_2, ySeq_2, zSeq_2, varSeq_2, coors_2 = getData_sowfa(ppDir_2, 'prbg1', ((0,0,0),30.0), 'U', 0)
# calculate coherence (lat)
t_seq_2, u0_2, u1_2, freq, coh_2_d40h20, co_coh_2_d40h20, quad_coh_2_d40h20, phase_2_d40h20 = coh_sowfa((0,24,0), (0,26,0), xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
t_seq_2, u0_2, u1_2, freq, coh_2_d80h20, co_coh_2_d80h20, quad_coh_2_d80h20, phase_2_d80h20 = coh_sowfa((0,24,0), (0,28,0), xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
t_seq_2, u0_2, u1_2, freq, coh_2_d40h100, co_coh_2_d40h100, quad_coh_2_d40h100, phase_2_d40h100 = coh_sowfa((0,24,4), (0,26,4), xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
t_seq_2, u0_2, u1_2, freq, coh_2_d80h100, co_coh_2_d80h100, quad_coh_2_d80h100, phase_2_d80h100 = coh_sowfa((0,24,4), (0,28,4), xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
t_seq_2, u0_2, u1_2, freq, coh_2_d40h180, co_coh_2_d40h180, quad_coh_2_d40h180, phase_2_d40h180 = coh_sowfa((0,24,8), (0,26,8), xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
t_seq_2, u0_2, u1_2, freq, coh_2_d80h180, co_coh_2_d80h180, quad_coh_2_d80h180, phase_2_d80h180 = coh_sowfa((0,24,8), (0,28,8), xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
# averaged coherence (lat)
freq, coh_av_2_d20h20, coh_std_2_d20h20, co_coh_av_2_d20h20, co_coh_std_2_d20h20, quad_coh_av_2_d20h20, quad_coh_std_2_d20h20, phase_av_2_d20h20, phase_std_2_d20h20 = coh_av_sowfa_y(0, 0, 50, 1, 0, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d40h20, coh_std_2_d40h20, co_coh_av_2_d40h20, co_coh_std_2_d40h20, quad_coh_av_2_d40h20, quad_coh_std_2_d40h20, phase_av_2_d40h20, phase_std_2_d40h20 = coh_av_sowfa_y(0, 0, 50, 2, 0, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d80h20, coh_std_2_d80h20, co_coh_av_2_d80h20, co_coh_std_2_d80h20, quad_coh_av_2_d80h20, quad_coh_std_2_d80h20, phase_av_2_d80h20, phase_std_2_d80h20 = coh_av_sowfa_y(0, 0, 50, 4, 0, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d20h100, coh_std_2_d20h100, co_coh_av_2_d20h100, co_coh_std_2_d20h100, quad_coh_av_2_d20h100, quad_coh_std_2_d20h100, phase_av_2_d20h100, phase_std_2_d20h100 = coh_av_sowfa_y(0, 0, 50, 1, 4, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d40h100, coh_std_2_d40h100, co_coh_av_2_d40h100, co_coh_std_2_d40h100, quad_coh_av_2_d40h100, quad_coh_std_2_d40h100, phase_av_2_d40h100, phase_std_2_d40h100 = coh_av_sowfa_y(0, 0, 50, 2, 4, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d80h100, coh_std_2_d80h100, co_coh_av_2_d80h100, co_coh_std_2_d80h100, quad_coh_av_2_d80h100, quad_coh_std_2_d80h100, phase_av_2_d80h100, phase_std_2_d80h100 = coh_av_sowfa_y(0, 0, 50, 4, 4, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d20h180, coh_std_2_d20h180, co_coh_av_2_d20h180, co_coh_std_2_d20h180, quad_coh_av_2_d20h180, quad_coh_std_2_d20h180, phase_av_2_d20h180, phase_std_2_d20h180 = coh_av_sowfa_y(0, 0, 50, 1, 8, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d40h180, coh_std_2_d40h180, co_coh_av_2_d40h180, co_coh_std_2_d40h180, quad_coh_av_2_d40h180, quad_coh_std_2_d40h180, phase_av_2_d40h180, phase_std_2_d40h180 = coh_av_sowfa_y(0, 0, 50, 2, 8, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)
freq, coh_av_2_d80h180, coh_std_2_d80h180, co_coh_av_2_d80h180, co_coh_std_2_d80h180, quad_coh_av_2_d80h180, quad_coh_std_2_d80h180, phase_av_2_d80h180, phase_std_2_d80h180 = coh_av_sowfa_y(0, 0, 50, 4, 8, xSeq_2.size, ySeq_2.size, (72000.0, 79200.0, 0.5), tSeq_2, varSeq_2, 240)



prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5_0.0001_main'
dir_3 = prjDir + '/' + jobName
tSeq_3, xSeq_3, ySeq_3, zSeq_3, varSeq_3 = getData_palm(dir_3, jobName, 'M04', ['.001'], 'u')
# calculate coherence
t_seq_3, u0_3, u1_3, freq, coh_3_d40h20, co_coh_3_d40h20, quad_coh_3_d40h20, phase_3_d40h20 = coh_palm((0,24,0), (0,26,0), (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
t_seq_3, u0_3, u1_3, freq, coh_3_d80h20, co_coh_3_d80h20, quad_coh_3_d80h20, phase_3_d80h20 = coh_palm((0,24,0), (0,28,0), (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
t_seq_3, u0_3, u1_3, freq, coh_3_d40h100, co_coh_3_d40h100, quad_coh_3_d40h100, phase_3_d40h100 = coh_palm((0,24,4), (0,26,4), (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
t_seq_3, u0_3, u1_3, freq, coh_3_d80h100, co_coh_3_d80h100, quad_coh_3_d80h100, phase_3_d80h100 = coh_palm((0,24,4), (0,28,4), (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
t_seq_3, u0_3, u1_3, freq, coh_3_d40h180, co_coh_3_d40h180, quad_coh_3_d40h180, phase_3_d40h180 = coh_palm((0,24,8), (0,26,8), (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
t_seq_3, u0_3, u1_3, freq, coh_3_d80h180, co_coh_3_d80h180, quad_coh_3_d80h180, phase_3_d80h180 = coh_palm((0,24,8), (0,28,8), (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
# averaged coherence
freq, coh_av_3_d20h20, coh_std_3_d20h20, co_coh_av_3_d20h20, co_coh_std_3_d20h20, quad_coh_av_3_d20h20, quad_coh_std_3_d20h20, phase_av_3_d20h20, phase_std_3_d20h20 = coh_av_palm_y(0, 0, 50, 1, 0, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d40h20, coh_std_3_d40h20, co_coh_av_3_d40h20, co_coh_std_3_d40h20, quad_coh_av_3_d40h20, quad_coh_std_3_d40h20, phase_av_3_d40h20, phase_std_3_d40h20 = coh_av_palm_y(0, 0, 50, 2, 0, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d80h20, coh_std_3_d80h20, co_coh_av_3_d80h20, co_coh_std_3_d80h20, quad_coh_av_3_d80h20, quad_coh_std_3_d80h20, phase_av_3_d80h20, phase_std_3_d80h20 = coh_av_palm_y(0, 0, 50, 4, 0, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d20h100, coh_std_3_d20h100, co_coh_av_3_d20h100, co_coh_std_3_d20h100, quad_coh_av_3_d20h100, quad_coh_std_3_d20h100, phase_av_3_d20h100, phase_std_3_d20h100 = coh_av_palm_y(0, 0, 50, 1, 4, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d40h100, coh_std_3_d40h100, co_coh_av_3_d40h100, co_coh_std_3_d40h100, quad_coh_av_3_d40h100, quad_coh_std_3_d40h100, phase_av_3_d40h100, phase_std_3_d40h100 = coh_av_palm_y(0, 0, 50, 2, 4, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d80h100, coh_std_3_d80h100, co_coh_av_3_d80h100, co_coh_std_3_d80h100, quad_coh_av_3_d80h100, quad_coh_std_3_d80h100, phase_av_3_d80h100, phase_std_3_d80h100 = coh_av_palm_y(0, 0, 50, 4, 4, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d20h180, coh_std_3_d20h180, co_coh_av_3_d20h180, co_coh_std_3_d20h180, quad_coh_av_3_d20h180, quad_coh_std_3_d20h180, phase_av_3_d20h180, phase_std_3_d20h180 = coh_av_palm_y(0, 0, 50, 1, 8, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d40h180, coh_std_3_d40h180, co_coh_av_3_d40h180, co_coh_std_3_d40h180, quad_coh_av_3_d40h180, quad_coh_std_3_d40h180, phase_av_3_d40h180, phase_std_3_d40h180 = coh_av_palm_y(0, 0, 50, 2, 8, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)
freq, coh_av_3_d80h180, coh_std_3_d80h180, co_coh_av_3_d80h180, co_coh_std_3_d80h180, quad_coh_av_3_d80h180, quad_coh_std_3_d80h180, phase_av_3_d80h180, phase_std_3_d80h180 = coh_av_palm_y(0, 0, 50, 4, 8, (3600.0, 10800.0, 0.5), tSeq_3, varSeq_3, 240)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5_main'
dir_4 = prjDir + '/' + jobName
tSeq_4, xSeq_4, ySeq_4, zSeq_4, varSeq_4 = getData_palm(dir_4, jobName, 'M04', ['.001'], 'u')
# calculate coherence
t_seq_4, u0_4, u1_4, freq, coh_4_d40h20, co_coh_4_d40h20, quad_coh_4_d40h20, phase_4_d40h20 = coh_palm((0,24,0), (0,26,0), (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
t_seq_4, u0_4, u1_4, freq, coh_4_d80h20, co_coh_4_d80h20, quad_coh_4_d80h20, phase_4_d80h20 = coh_palm((0,24,0), (0,28,0), (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
t_seq_4, u0_4, u1_4, freq, coh_4_d40h100, co_coh_4_d40h100, quad_coh_4_d40h100, phase_4_d40h100 = coh_palm((0,24,4), (0,26,4), (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
t_seq_4, u0_4, u1_4, freq, coh_4_d80h100, co_coh_4_d80h100, quad_coh_4_d80h100, phase_4_d80h100 = coh_palm((0,24,4), (0,28,4), (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
t_seq_4, u0_4, u1_4, freq, coh_4_d40h180, co_coh_4_d40h180, quad_coh_4_d40h180, phase_4_d40h180 = coh_palm((0,24,8), (0,26,8), (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
t_seq_4, u0_4, u1_4, freq, coh_4_d80h180, co_coh_4_d80h180, quad_coh_4_d80h180, phase_4_d80h180 = coh_palm((0,24,8), (0,28,8), (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
# averaged coherence
freq, coh_av_4_d20h20, coh_std_4_d20h20, co_coh_av_4_d20h20, co_coh_std_4_d20h20, quad_coh_av_4_d20h20, quad_coh_std_4_d20h20, phase_av_4_d20h20, phase_std_4_d20h20 = coh_av_palm_y(0, 0, 50, 1, 0, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d40h20, coh_std_4_d40h20, co_coh_av_4_d40h20, co_coh_std_4_d40h20, quad_coh_av_4_d40h20, quad_coh_std_4_d40h20, phase_av_4_d40h20, phase_std_4_d40h20 = coh_av_palm_y(0, 0, 50, 2, 0, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d80h20, coh_std_4_d80h20, co_coh_av_4_d80h20, co_coh_std_4_d80h20, quad_coh_av_4_d80h20, quad_coh_std_4_d80h20, phase_av_4_d80h20, phase_std_4_d80h20 = coh_av_palm_y(0, 0, 50, 4, 0, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d20h100, coh_std_4_d20h100, co_coh_av_4_d20h100, co_coh_std_4_d20h100, quad_coh_av_4_d20h100, quad_coh_std_4_d20h100, phase_av_4_d20h100, phase_std_4_d20h100 = coh_av_palm_y(0, 0, 50, 1, 4, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d40h100, coh_std_4_d40h100, co_coh_av_4_d40h100, co_coh_std_4_d40h100, quad_coh_av_4_d40h100, quad_coh_std_4_d40h100, phase_av_4_d40h100, phase_std_4_d40h100 = coh_av_palm_y(0, 0, 50, 2, 4, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d80h100, coh_std_4_d80h100, co_coh_av_4_d80h100, co_coh_std_4_d80h100, quad_coh_av_4_d80h100, quad_coh_std_4_d80h100, phase_av_4_d80h100, phase_std_4_d80h100 = coh_av_palm_y(0, 0, 50, 4, 4, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d20h180, coh_std_4_d20h180, co_coh_av_4_d20h180, co_coh_std_4_d20h180, quad_coh_av_4_d20h180, quad_coh_std_4_d20h180, phase_av_4_d20h180, phase_std_4_d20h180 = coh_av_palm_y(0, 0, 50, 1, 8, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d40h180, coh_std_4_d40h180, co_coh_av_4_d40h180, co_coh_std_4_d40h180, quad_coh_av_4_d40h180, quad_coh_std_4_d40h180, phase_av_4_d40h180, phase_std_4_d40h180 = coh_av_palm_y(0, 0, 50, 2, 8, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)
freq, coh_av_4_d80h180, coh_std_4_d80h180, co_coh_av_4_d80h180, co_coh_std_4_d80h180, quad_coh_av_4_d80h180, quad_coh_std_4_d80h180, phase_av_4_d80h180, phase_std_4_d80h180 = coh_av_palm_y(0, 0, 50, 4, 8, (3600.0, 10800.0, 0.5), tSeq_4, varSeq_4, 240)


prjDir = '/scratch/palmdata/JOBS'
jobName  = 'deepwind_gs5_0.01_main'
dir_5 = prjDir + '/' + jobName
tSeq_5, xSeq_5, ySeq_5, zSeq_5, varSeq_5 = getData_palm(dir_5, jobName, 'M04', ['.001'], 'u')
# calculate coherence
t_seq_5, u0_5, u1_5, freq, coh_5_d40h20, co_coh_5_d40h20, quad_coh_5_d40h20, phase_5_d40h20 = coh_palm((0,24,0), (0,26,0), (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
t_seq_5, u0_5, u1_5, freq, coh_5_d80h20, co_coh_5_d80h20, quad_coh_5_d80h20, phase_5_d80h20 = coh_palm((0,24,0), (0,28,0), (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
t_seq_5, u0_5, u1_5, freq, coh_5_d40h100, co_coh_5_d40h100, quad_coh_5_d40h100, phase_5_d40h100 = coh_palm((0,24,4), (0,26,4), (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
t_seq_5, u0_5, u1_5, freq, coh_5_d80h100, co_coh_5_d80h100, quad_coh_5_d80h100, phase_5_d80h100 = coh_palm((0,24,4), (0,28,4), (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
t_seq_5, u0_5, u1_5, freq, coh_5_d40h180, co_coh_5_d40h180, quad_coh_5_d40h180, phase_5_d40h180 = coh_palm((0,24,8), (0,26,8), (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
t_seq_5, u0_5, u1_5, freq, coh_5_d80h180, co_coh_5_d80h180, quad_coh_5_d80h180, phase_5_d80h180 = coh_palm((0,24,8), (0,28,8), (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
# averaged coherence
freq, coh_av_5_d20h20, coh_std_5_d20h20, co_coh_av_5_d20h20, co_coh_std_5_d20h20, quad_coh_av_5_d20h20, quad_coh_std_5_d20h20, phase_av_5_d20h20, phase_std_5_d20h20 = coh_av_palm_y(0, 0, 50, 1, 0, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d40h20, coh_std_5_d40h20, co_coh_av_5_d40h20, co_coh_std_5_d40h20, quad_coh_av_5_d40h20, quad_coh_std_5_d40h20, phase_av_5_d40h20, phase_std_5_d40h20 = coh_av_palm_y(0, 0, 50, 2, 0, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d80h20, coh_std_5_d80h20, co_coh_av_5_d80h20, co_coh_std_5_d80h20, quad_coh_av_5_d80h20, quad_coh_std_5_d80h20, phase_av_5_d80h20, phase_std_5_d80h20 = coh_av_palm_y(0, 0, 50, 4, 0, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d20h100, coh_std_5_d20h100, co_coh_av_5_d20h100, co_coh_std_5_d20h100, quad_coh_av_5_d20h100, quad_coh_std_5_d20h100, phase_av_5_d20h100, phase_std_5_d20h100 = coh_av_palm_y(0, 0, 50, 1, 4, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d40h100, coh_std_5_d40h100, co_coh_av_5_d40h100, co_coh_std_5_d40h100, quad_coh_av_5_d40h100, quad_coh_std_5_d40h100, phase_av_5_d40h100, phase_std_5_d40h100 = coh_av_palm_y(0, 0, 50, 2, 4, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d80h100, coh_std_5_d80h100, co_coh_av_5_d80h100, co_coh_std_5_d80h100, quad_coh_av_5_d80h100, quad_coh_std_5_d80h100, phase_av_5_d80h100, phase_std_5_d80h100 = coh_av_palm_y(0, 0, 50, 4, 4, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d20h180, coh_std_5_d20h180, co_coh_av_5_d20h180, co_coh_std_5_d20h180, quad_coh_av_5_d20h180, quad_coh_std_5_d20h180, phase_av_5_d20h180, phase_std_5_d20h180 = coh_av_palm_y(0, 0, 50, 1, 8, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d40h180, coh_std_5_d40h180, co_coh_av_5_d40h180, co_coh_std_5_d40h180, quad_coh_av_5_d40h180, quad_coh_std_5_d40h180, phase_av_5_d40h180, phase_std_5_d40h180 = coh_av_palm_y(0, 0, 50, 2, 8, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)
freq, coh_av_5_d80h180, coh_std_5_d80h180, co_coh_av_5_d80h180, co_coh_std_5_d80h180, quad_coh_av_5_d80h180, quad_coh_std_5_d80h180, phase_av_5_d80h180, phase_std_5_d80h180 = coh_av_palm_y(0, 0, 50, 4, 8, (3600.0, 10800.0, 0.5), tSeq_5, varSeq_5, 240)



""" 3*3 plots of co-coh_av """
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(8)
rNum, cNum = (3,3)
axs = fig.subplots(nrows=rNum, ncols=cNum)

## height = 20m
uz_0_20 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_20 = calc_uz_palm(4,varSeq_3) # palm

axs[0,0].plot(freq[1:]*40/uz_0_20, co_coh_av_0_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,0].plot(freq[1:]*80/uz_0_20, co_coh_av_0_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,0].plot(freq[1:]*40/uz_3_20, co_coh_av_3_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,0].plot(freq[1:]*80/uz_3_20, co_coh_av_3_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[0,0].set_xlim(0, 1.0); axs[0,0].set_xticklabels([])
axs[0,0].set_ylim(-0.2, 1.0)
axs[0,0].set_ylabel(r"co-coh", fontsize=12)
axs[0,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,0].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_20 = calc_uz_palm(4,varSeq_4) # palm

axs[0,1].plot(freq[1:]*40/uz_1_20, co_coh_av_1_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,1].plot(freq[1:]*80/uz_1_20, co_coh_av_1_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,1].plot(freq[1:]*40/uz_4_20, co_coh_av_4_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,1].plot(freq[1:]*80/uz_4_20, co_coh_av_4_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[0,1].set_xlim(0, 1.0); axs[0,1].set_xticklabels([])
axs[0,1].set_ylim(-0.2, 1.0); axs[0,1].set_yticklabels([])
axs[0,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,1].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_20 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_20 = calc_uz_palm(4,varSeq_5) # palm

axs[0,2].plot(freq[1:]*40/uz_2_20, co_coh_av_2_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,2].plot(freq[1:]*80/uz_2_20, co_coh_av_2_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,2].plot(freq[1:]*40/uz_5_20, co_coh_av_5_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,2].plot(freq[1:]*80/uz_5_20, co_coh_av_5_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[0,2].set_xlim(0, 1.0); axs[0,2].set_xticklabels([])
axs[0,2].set_ylim(-0.2, 1.0); axs[0,2].set_yticklabels([])
axs[0,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,2].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

## height = 100m
uz_0_100 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_100 = calc_uz_palm(4,varSeq_3) # palm

axs[1,0].plot(freq[1:]*40/uz_0_100, co_coh_av_0_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,0].plot(freq[1:]*80/uz_0_100, co_coh_av_0_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,0].plot(freq[1:]*40/uz_3_100, co_coh_av_3_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,0].plot(freq[1:]*80/uz_3_100, co_coh_av_3_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,0].set_xlim(0, 1.0); axs[1,0].set_xticklabels([])
axs[1,0].set_ylim(-0.2, 1.0)
axs[1,0].set_ylabel(r"co-coh", fontsize=12)
axs[1,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,0].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_100 = calc_uz_palm(4,varSeq_4) # palm

axs[1,1].plot(freq[1:]*40/uz_1_100, co_coh_av_1_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,1].plot(freq[1:]*80/uz_1_100, co_coh_av_1_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,1].plot(freq[1:]*40/uz_4_100, co_coh_av_4_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,1].plot(freq[1:]*80/uz_4_100, co_coh_av_4_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,1].set_xlim(0, 1.0); axs[1,1].set_xticklabels([])
axs[1,1].set_ylim(-0.2, 1.0); axs[1,1].set_yticklabels([])
axs[1,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,1].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_100 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_100 = calc_uz_palm(4,varSeq_5) # palm

axs[1,2].plot(freq[1:]*40/uz_2_100, co_coh_av_2_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,2].plot(freq[1:]*80/uz_2_100, co_coh_av_2_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,2].plot(freq[1:]*40/uz_5_100, co_coh_av_5_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,2].plot(freq[1:]*80/uz_5_100, co_coh_av_5_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,2].set_xlim(0, 1.0); axs[1,2].set_xticklabels([])
axs[1,2].set_ylim(-0.2, 1.0); axs[1,2].set_yticklabels([])
axs[1,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,2].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

## height = 180m
uz_0_180 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_180 = calc_uz_palm(4,varSeq_3) # palm

axs[2,0].plot(freq[1:]*40/uz_0_180, co_coh_av_0_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,0].plot(freq[1:]*80/uz_0_180, co_coh_av_0_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,0].plot(freq[1:]*40/uz_3_180, co_coh_av_3_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,0].plot(freq[1:]*80/uz_3_180, co_coh_av_3_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,0].set_xlim(0, 1.0)
axs[2,0].set_ylim(-0.2, 1.0)
axs[2,0].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,0].set_ylabel(r"co-coh", fontsize=12)
axs[2,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,0].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_180 = calc_uz_palm(4,varSeq_4) # palm

axs[2,1].plot(freq[1:]*40/uz_1_180, co_coh_av_1_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,1].plot(freq[1:]*80/uz_1_180, co_coh_av_1_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,1].plot(freq[1:]*40/uz_4_180, co_coh_av_4_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,1].plot(freq[1:]*80/uz_4_180, co_coh_av_4_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,1].set_xlim(0, 1.0)
axs[2,1].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,1].set_ylim(-0.2, 1.0); axs[2,1].set_yticklabels([])
axs[2,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,1].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_180 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_180 = calc_uz_palm(4,varSeq_5) # palm

axs[2,2].plot(freq[1:]*40/uz_2_180, co_coh_av_2_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,2].plot(freq[1:]*80/uz_2_180, co_coh_av_2_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,2].plot(freq[1:]*40/uz_5_180, co_coh_av_5_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,2].plot(freq[1:]*80/uz_5_180, co_coh_av_5_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,2].set_xlim(0, 1.0)
axs[2,2].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,2].set_ylim(-0.2, 1.0); axs[2,2].set_yticklabels([])
axs[2,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,2].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

for i in range(3):
    for j in range(3):
        axs[i,j].grid(True)

# plt.legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
handles, labels = axs[0,0].get_legend_handles_labels()
lgdord = [0,1,2,3]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,0.94), ncol=4, mode='None', borderaxespad=0, fontsize=12)
saveDir = '/scratch/projects/deepwind/photo/coherence'
saveName = 'co-coh_av_lat_gp.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()



""" 3*3 plots of quad-coh_av """
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(8)
rNum, cNum = (3,3)
axs = fig.subplots(nrows=rNum, ncols=cNum)

## height = 20m
uz_0_20 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_20 = calc_uz_palm(4,varSeq_3) # palm

axs[0,0].plot(freq[1:]*40/uz_0_20, quad_coh_av_0_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,0].plot(freq[1:]*80/uz_0_20, quad_coh_av_0_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,0].plot(freq[1:]*40/uz_3_20, quad_coh_av_3_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,0].plot(freq[1:]*80/uz_3_20, quad_coh_av_3_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[0,0].set_xlim(0, 1.0); axs[0,0].set_xticklabels([])
axs[0,0].set_ylim(-0.2, 1.0)
axs[0,0].set_ylabel(r"quad-coh", fontsize=12)
axs[0,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,0].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_20 = calc_uz_palm(4,varSeq_4) # palm

axs[0,1].plot(freq[1:]*40/uz_1_20, quad_coh_av_1_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,1].plot(freq[1:]*80/uz_1_20, quad_coh_av_1_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,1].plot(freq[1:]*40/uz_4_20, quad_coh_av_4_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,1].plot(freq[1:]*80/uz_4_20, quad_coh_av_4_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[0,1].set_xlim(0, 1.0); axs[0,1].set_xticklabels([])
axs[0,1].set_ylim(-0.2, 1.0); axs[0,1].set_yticklabels([])
axs[0,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,1].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_20 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_20 = calc_uz_palm(4,varSeq_5) # palm

axs[0,2].plot(freq[1:]*40/uz_2_20, quad_coh_av_2_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,2].plot(freq[1:]*80/uz_2_20, quad_coh_av_2_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,2].plot(freq[1:]*40/uz_5_20, quad_coh_av_5_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,2].plot(freq[1:]*80/uz_5_20, quad_coh_av_5_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[0,2].set_xlim(0, 1.0); axs[0,2].set_xticklabels([])
axs[0,2].set_ylim(-0.2, 1.0); axs[0,2].set_yticklabels([])
axs[0,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,2].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

## height = 100m
uz_0_100 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_100 = calc_uz_palm(4,varSeq_3) # palm

axs[1,0].plot(freq[1:]*40/uz_0_100, quad_coh_av_0_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,0].plot(freq[1:]*80/uz_0_100, quad_coh_av_0_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,0].plot(freq[1:]*40/uz_3_100, quad_coh_av_3_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,0].plot(freq[1:]*80/uz_3_100, quad_coh_av_3_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,0].set_xlim(0, 1.0); axs[1,0].set_xticklabels([])
axs[1,0].set_ylim(-0.2, 1.0)
axs[1,0].set_ylabel(r"quad-coh", fontsize=12)
axs[1,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,0].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_100 = calc_uz_palm(4,varSeq_4) # palm

axs[1,1].plot(freq[1:]*40/uz_1_100, quad_coh_av_1_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,1].plot(freq[1:]*80/uz_1_100, quad_coh_av_1_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,1].plot(freq[1:]*40/uz_4_100, quad_coh_av_4_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,1].plot(freq[1:]*80/uz_4_100, quad_coh_av_4_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,1].set_xlim(0, 1.0); axs[1,1].set_xticklabels([])
axs[1,1].set_ylim(-0.2, 1.0); axs[1,1].set_yticklabels([])
axs[1,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,1].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_100 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_100 = calc_uz_palm(4,varSeq_5) # palm

axs[1,2].plot(freq[1:]*40/uz_2_100, quad_coh_av_2_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,2].plot(freq[1:]*80/uz_2_100, quad_coh_av_2_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,2].plot(freq[1:]*40/uz_5_100, quad_coh_av_5_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,2].plot(freq[1:]*80/uz_5_100, quad_coh_av_5_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,2].set_xlim(0, 1.0); axs[1,2].set_xticklabels([])
axs[1,2].set_ylim(-0.2, 1.0); axs[1,2].set_yticklabels([])
axs[1,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,2].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

## height = 180m
uz_0_180 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_180 = calc_uz_palm(4,varSeq_3) # palm

axs[2,0].plot(freq[1:]*40/uz_0_180, quad_coh_av_0_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,0].plot(freq[1:]*80/uz_0_180, quad_coh_av_0_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,0].plot(freq[1:]*40/uz_3_180, quad_coh_av_3_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,0].plot(freq[1:]*80/uz_3_180, quad_coh_av_3_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,0].set_xlim(0, 1.0)
axs[2,0].set_ylim(-0.2, 1.0)
axs[2,0].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,0].set_ylabel(r"quad-coh", fontsize=12)
axs[2,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,0].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_180 = calc_uz_palm(4,varSeq_4) # palm

axs[2,1].plot(freq[1:]*40/uz_1_180, quad_coh_av_1_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,1].plot(freq[1:]*80/uz_1_180, quad_coh_av_1_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,1].plot(freq[1:]*40/uz_4_180, quad_coh_av_4_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,1].plot(freq[1:]*80/uz_4_180, quad_coh_av_4_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,1].set_xlim(0, 1.0)
axs[2,1].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,1].set_ylim(-0.2, 1.0); axs[2,1].set_yticklabels([])
axs[2,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,1].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_180 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_180 = calc_uz_palm(4,varSeq_5) # palm

axs[2,2].plot(freq[1:]*40/uz_2_180, quad_coh_av_2_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,2].plot(freq[1:]*80/uz_2_180, quad_coh_av_2_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,2].plot(freq[1:]*40/uz_5_180, quad_coh_av_5_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,2].plot(freq[1:]*80/uz_5_180, quad_coh_av_5_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,2].set_xlim(0, 1.0)
axs[2,2].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,2].set_ylim(-0.2, 1.0); axs[2,2].set_yticklabels([])
axs[2,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,2].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

for i in range(3):
    for j in range(3):
        axs[i,j].grid(True)

# plt.legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
handles, labels = axs[0,0].get_legend_handles_labels()
lgdord = [0,1,2,3]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,0.94), ncol=4, mode='None', borderaxespad=0, fontsize=12)
saveDir = '/scratch/projects/deepwind/photo/coherence'
saveName = 'quad-coh_av_lat_gp.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()



""" 3*3 plots of coh_av """
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(8)
rNum, cNum = (3,3)
axs = fig.subplots(nrows=rNum, ncols=cNum)

## height = 20m
uz_0_20 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_20 = calc_uz_palm(4,varSeq_3) # palm

axs[0,0].plot(freq[1:]*40/uz_0_20, coh_av_0_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,0].plot(freq[1:]*80/uz_0_20, coh_av_0_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,0].plot(freq[1:]*40/uz_3_20, coh_av_3_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,0].plot(freq[1:]*80/uz_3_20, coh_av_3_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[0,0].plot(freq[1:]*40/uz_0_20, coh_kaimal(freq[1:], 40, uz_0_20, 8.1*0.7*20, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[0,0].plot(freq[1:]*80/uz_0_20, coh_kaimal(freq[1:], 80, uz_0_20, 8.1*0.7*20, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[0,0].set_xlim(0, 1.0); axs[0,0].set_xticklabels([])
axs[0,0].set_ylim(-0.2, 1.0)
axs[0,0].set_ylabel(r"coh", fontsize=12)
axs[0,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,0].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_20 = calc_uz_palm(4,varSeq_4) # palm

axs[0,1].plot(freq[1:]*40/uz_1_20, coh_av_1_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,1].plot(freq[1:]*80/uz_1_20, coh_av_1_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,1].plot(freq[1:]*40/uz_4_20, coh_av_4_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,1].plot(freq[1:]*80/uz_4_20, coh_av_4_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[0,1].plot(freq[1:]*40/uz_1_20, coh_kaimal(freq[1:], 40, uz_1_20, 8.1*0.7*20, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[0,1].plot(freq[1:]*80/uz_1_20, coh_kaimal(freq[1:], 80, uz_1_20, 8.1*0.7*20, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[0,1].set_xlim(0, 1.0); axs[0,1].set_xticklabels([])
axs[0,1].set_ylim(-0.2, 1.0); axs[0,1].set_yticklabels([])
axs[0,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,1].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_20 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_20 = calc_uz_palm(4,varSeq_5) # palm

axs[0,2].plot(freq[1:]*40/uz_2_20, coh_av_2_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,2].plot(freq[1:]*80/uz_2_20, coh_av_2_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,2].plot(freq[1:]*40/uz_5_20, coh_av_5_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,2].plot(freq[1:]*80/uz_5_20, coh_av_5_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[0,2].plot(freq[1:]*40/uz_2_20, coh_kaimal(freq[1:], 40, uz_2_20, 8.1*0.7*20, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[0,2].plot(freq[1:]*80/uz_2_20, coh_kaimal(freq[1:], 80, uz_2_20, 8.1*0.7*20, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[0,2].set_xlim(0, 1.0); axs[0,2].set_xticklabels([])
axs[0,2].set_ylim(-0.2, 1.0); axs[0,2].set_yticklabels([])
axs[0,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,2].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

## height = 100m
uz_0_100 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_100 = calc_uz_palm(4,varSeq_3) # palm

axs[1,0].plot(freq[1:]*40/uz_0_100, coh_av_0_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,0].plot(freq[1:]*80/uz_0_100, coh_av_0_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,0].plot(freq[1:]*40/uz_3_100, coh_av_3_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,0].plot(freq[1:]*80/uz_3_100, coh_av_3_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,0].plot(freq[1:]*40/uz_0_100, coh_kaimal(freq[1:], 40, uz_0_100, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[1,0].plot(freq[1:]*80/uz_0_100, coh_kaimal(freq[1:], 80, uz_0_100, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[1,0].set_xlim(0, 1.0); axs[1,0].set_xticklabels([])
axs[1,0].set_ylim(-0.2, 1.0)
axs[1,0].set_ylabel(r"coh", fontsize=12)
axs[1,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,0].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_100 = calc_uz_palm(4,varSeq_4) # palm

axs[1,1].plot(freq[1:]*40/uz_1_100, coh_av_1_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,1].plot(freq[1:]*80/uz_1_100, coh_av_1_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,1].plot(freq[1:]*40/uz_4_100, coh_av_4_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,1].plot(freq[1:]*80/uz_4_100, coh_av_4_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,1].plot(freq[1:]*40/uz_1_100, coh_kaimal(freq[1:], 40, uz_1_100, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[1,1].plot(freq[1:]*80/uz_1_100, coh_kaimal(freq[1:], 80, uz_1_100, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[1,1].set_xlim(0, 1.0); axs[1,1].set_xticklabels([])
axs[1,1].set_ylim(-0.2, 1.0); axs[1,1].set_yticklabels([])
axs[1,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,1].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_100 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_100 = calc_uz_palm(4,varSeq_5) # palm

axs[1,2].plot(freq[1:]*40/uz_2_100, coh_av_2_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,2].plot(freq[1:]*80/uz_2_100, coh_av_2_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,2].plot(freq[1:]*40/uz_5_100, coh_av_5_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,2].plot(freq[1:]*80/uz_5_100, coh_av_5_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,2].plot(freq[1:]*40/uz_2_100, coh_kaimal(freq[1:], 40, uz_2_100, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[1,2].plot(freq[1:]*80/uz_2_100, coh_kaimal(freq[1:], 80, uz_2_100, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[1,2].set_xlim(0, 1.0); axs[1,2].set_xticklabels([])
axs[1,2].set_ylim(-0.2, 1.0); axs[1,2].set_yticklabels([])
axs[1,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,2].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

## height = 180m
uz_0_180 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_180 = calc_uz_palm(4,varSeq_3) # palm

axs[2,0].plot(freq[1:]*40/uz_0_180, coh_av_0_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,0].plot(freq[1:]*80/uz_0_180, coh_av_0_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,0].plot(freq[1:]*40/uz_3_180, coh_av_3_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,0].plot(freq[1:]*80/uz_3_180, coh_av_3_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,0].plot(freq[1:]*40/uz_0_180, coh_kaimal(freq[1:], 40, uz_0_180, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[2,0].plot(freq[1:]*80/uz_0_180, coh_kaimal(freq[1:], 80, uz_0_180, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[2,0].set_xlim(0, 1.0)
axs[2,0].set_ylim(-0.2, 1.0)
axs[2,0].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,0].set_ylabel(r"coh", fontsize=12)
axs[2,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,0].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_180 = calc_uz_palm(4,varSeq_4) # palm

axs[2,1].plot(freq[1:]*40/uz_1_180, coh_av_1_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,1].plot(freq[1:]*80/uz_1_180, coh_av_1_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,1].plot(freq[1:]*40/uz_4_180, coh_av_4_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,1].plot(freq[1:]*80/uz_4_180, coh_av_4_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,1].plot(freq[1:]*40/uz_1_180, coh_kaimal(freq[1:], 40, uz_1_180, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[2,1].plot(freq[1:]*80/uz_1_180, coh_kaimal(freq[1:], 80, uz_1_180, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[2,1].set_xlim(0, 1.0)
axs[2,1].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,1].set_ylim(-0.2, 1.0); axs[2,1].set_yticklabels([])
axs[2,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,1].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_180 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_180 = calc_uz_palm(4,varSeq_5) # palm

axs[2,2].plot(freq[1:]*40/uz_2_180, coh_av_2_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,2].plot(freq[1:]*80/uz_2_180, coh_av_2_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,2].plot(freq[1:]*40/uz_5_180, coh_av_5_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,2].plot(freq[1:]*80/uz_5_180, coh_av_5_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,2].plot(freq[1:]*40/uz_2_180, coh_kaimal(freq[1:], 40, uz_2_180, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[2,2].plot(freq[1:]*80/uz_2_180, coh_kaimal(freq[1:], 80, uz_2_180, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[2,2].set_xlim(0, 1.0)
axs[2,2].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,2].set_ylim(-0.2, 1.0); axs[2,2].set_yticklabels([])
axs[2,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,2].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

for i in range(3):
    for j in range(3):
        axs[i,j].grid(True)

# plt.legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
handles, labels = axs[0,0].get_legend_handles_labels()
lgdord = [0,1,2,3,4,5]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,0.98), ncol=3, mode='None', borderaxespad=0, fontsize=12)
saveDir = '/scratch/projects/deepwind/photo/coherence'
saveName = 'coh_av_lat_gp.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()


""" 3*3 plots of phase_av """
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(8)
rNum, cNum = (3,3)
axs = fig.subplots(nrows=rNum, ncols=cNum)

yaxis_min = -1.0*np.pi
yaxis_max = 1.0*np.pi
yaxis_d = np.pi/4
ylabels = ['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
          r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']

## height = 20m
uz_0_20 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_20 = calc_uz_palm(4,varSeq_3) # palm

axs[0,0].plot(freq[1:]*40/uz_0_20, phase_av_0_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,0].plot(freq[1:]*80/uz_0_20, phase_av_0_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,0].plot(freq[1:]*40/uz_3_20, phase_av_3_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,0].plot(freq[1:]*80/uz_3_20, phase_av_3_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[0,0].set_xlim(0, 1.0); axs[0,0].set_xticklabels([])
axs[0,0].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[0,0].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[0,0].set_yticklabels(ylabels)
axs[0,0].set_ylabel(r"phase", fontsize=12)
axs[0,0].text(0.5, 0.75*np.pi, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,0].text(0.5, 0.5*np.pi, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_20 = calc_uz_palm(4,varSeq_4) # palm

axs[0,1].plot(freq[1:]*40/uz_1_20, phase_av_1_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,1].plot(freq[1:]*80/uz_1_20, phase_av_1_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,1].plot(freq[1:]*40/uz_4_20, phase_av_4_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,1].plot(freq[1:]*80/uz_4_20, phase_av_4_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[0,1].set_xlim(0, 1.0); axs[0,1].set_xticklabels([])
axs[0,1].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[0,1].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[0,1].set_yticklabels([])
axs[0,1].text(0.5, 0.75*np.pi, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,1].text(0.5, 0.5*np.pi, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_20 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_20 = calc_uz_palm(4,varSeq_5) # palm

axs[0,2].plot(freq[1:]*40/uz_2_20, phase_av_2_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,2].plot(freq[1:]*80/uz_2_20, phase_av_2_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,2].plot(freq[1:]*40/uz_5_20, phase_av_5_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,2].plot(freq[1:]*80/uz_5_20, phase_av_5_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[0,2].set_xlim(0, 1.0); axs[0,2].set_xticklabels([])
axs[0,2].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[0,2].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[0,2].set_yticklabels([])
axs[0,2].text(0.5, 0.75*np.pi, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,2].text(0.5, 0.5*np.pi, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

## height = 100m
uz_0_100 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_100 = calc_uz_palm(4,varSeq_3) # palm

axs[1,0].plot(freq[1:]*40/uz_0_100, phase_av_0_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,0].plot(freq[1:]*80/uz_0_100, phase_av_0_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,0].plot(freq[1:]*40/uz_3_100, phase_av_3_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,0].plot(freq[1:]*80/uz_3_100, phase_av_3_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[1,0].set_xlim(0, 1.0); axs[1,0].set_xticklabels([])
axs[1,0].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[1,0].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[1,0].set_yticklabels(ylabels)
axs[1,0].set_ylabel(r"phase", fontsize=12)
axs[1,0].text(0.5, 0.75*np.pi, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,0].text(0.5, 0.5*np.pi, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_100 = calc_uz_palm(4,varSeq_4) # palm

axs[1,1].plot(freq[1:]*40/uz_1_100, phase_av_1_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,1].plot(freq[1:]*80/uz_1_100, phase_av_1_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,1].plot(freq[1:]*40/uz_4_100, phase_av_4_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,1].plot(freq[1:]*80/uz_4_100, phase_av_4_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[1,1].set_xlim(0, 1.0); axs[1,1].set_xticklabels([])
axs[1,1].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[1,1].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[1,1].set_yticklabels([])
axs[1,1].text(0.5, 0.75*np.pi, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,1].text(0.5, 0.5*np.pi, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_100 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_100 = calc_uz_palm(4,varSeq_5) # palm

axs[1,2].plot(freq[1:]*40/uz_2_100, phase_av_2_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,2].plot(freq[1:]*80/uz_2_100, phase_av_2_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,2].plot(freq[1:]*40/uz_5_100, phase_av_5_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,2].plot(freq[1:]*80/uz_5_100, phase_av_5_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[1,2].set_xlim(0, 1.0); axs[1,2].set_xticklabels([])
axs[1,2].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[1,2].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[1,2].set_yticklabels([])
axs[1,2].text(0.5, 0.75*np.pi, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,2].text(0.5, 0.5*np.pi, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

## height = 180m
uz_0_180 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_180 = calc_uz_palm(4,varSeq_3) # palm

axs[2,0].plot(freq[1:]*40/uz_0_180, phase_av_0_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,0].plot(freq[1:]*80/uz_0_180, phase_av_0_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,0].plot(freq[1:]*40/uz_3_180, phase_av_3_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,0].plot(freq[1:]*80/uz_3_180, phase_av_3_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[2,0].set_xlim(0, 1.0)
axs[2,0].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[2,0].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[2,0].set_yticklabels(ylabels)
axs[2,0].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,0].set_ylabel(r"phase", fontsize=12)
axs[2,0].text(0.5, 0.75*np.pi, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,0].text(0.5, 0.5*np.pi, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_180 = calc_uz_palm(4,varSeq_4) # palm

axs[2,1].plot(freq[1:]*40/uz_1_180, phase_av_1_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,1].plot(freq[1:]*80/uz_1_180, phase_av_1_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,1].plot(freq[1:]*40/uz_4_180, phase_av_4_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,1].plot(freq[1:]*80/uz_4_180, phase_av_4_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[2,1].set_xlim(0, 1.0)
axs[2,1].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,1].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[2,1].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[2,1].set_yticklabels([])
axs[2,1].text(0.5, 0.75*np.pi, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,1].text(0.5, 0.5*np.pi, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_180 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_180 = calc_uz_palm(4,varSeq_5) # palm

axs[2,2].plot(freq[1:]*40/uz_2_180, phase_av_2_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,2].plot(freq[1:]*80/uz_2_180, phase_av_2_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,2].plot(freq[1:]*40/uz_5_180, phase_av_5_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,2].plot(freq[1:]*80/uz_5_180, phase_av_5_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')

axs[2,2].set_xlim(0, 1.0)
axs[2,2].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,2].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[2,2].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[2,2].set_yticklabels([])
axs[2,2].text(0.5, 0.75*np.pi, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,2].text(0.5, 0.5*np.pi, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

for i in range(3):
    for j in range(3):
        axs[i,j].grid(True)

# plt.legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
handles, labels = axs[0,0].get_legend_handles_labels()
lgdord = [0,1,2,3]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,0.94), ncol=4, mode='None', borderaxespad=0, fontsize=12)
saveDir = '/scratch/projects/deepwind/photo/coherence'
saveName = 'phase_av_lat_gp.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()




""" 3*3 plots of coh """
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(8)
rNum, cNum = (3,3)
axs = fig.subplots(nrows=rNum, ncols=cNum)

## height = 20m
uz_0_20 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_20 = calc_uz_palm(4,varSeq_3) # palm

axs[0,0].plot(freq[1:]*40/uz_0_20, coh_0_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,0].plot(freq[1:]*80/uz_0_20, coh_0_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,0].plot(freq[1:]*40/uz_3_20, coh_3_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,0].plot(freq[1:]*80/uz_3_20, coh_3_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[0,0].plot(freq[1:]*40/uz_0_20, coh_kaimal(freq[1:], 40, uz_0_20, 8.1*0.7*20, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[0,0].plot(freq[1:]*80/uz_0_20, coh_kaimal(freq[1:], 80, uz_0_20, 8.1*0.7*20, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[0,0].set_xlim(0, 1.0); axs[0,0].set_xticklabels([])
axs[0,0].set_ylim(-0.2, 1.0)
axs[0,0].set_ylabel(r"coh", fontsize=12)
axs[0,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,0].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_20 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_20 = calc_uz_palm(4,varSeq_4) # palm
axs[0,1].plot(freq[1:]*40/uz_1_20, coh_1_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,1].plot(freq[1:]*80/uz_1_20, coh_1_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,1].plot(freq[1:]*40/uz_4_20, coh_4_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,1].plot(freq[1:]*80/uz_4_20, coh_4_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[0,1].plot(freq[1:]*40/uz_1_20, coh_kaimal(freq[1:], 40, uz_1_20, 8.1*0.7*20, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[0,1].plot(freq[1:]*80/uz_1_20, coh_kaimal(freq[1:], 80, uz_1_20, 8.1*0.7*20, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[0,1].set_xlim(0, 1.0); axs[0,1].set_xticklabels([])
axs[0,1].set_ylim(-0.2, 1.0); axs[0,1].set_yticklabels([])
axs[0,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,1].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_20 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_20 = calc_uz_palm(4,varSeq_5) # palm
axs[0,2].plot(freq[1:]*40/uz_2_20, coh_2_d40h20[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[0,2].plot(freq[1:]*80/uz_2_20, coh_2_d80h20[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[0,2].plot(freq[1:]*40/uz_5_20, coh_5_d40h20[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[0,2].plot(freq[1:]*80/uz_5_20, coh_5_d80h20[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[0,2].plot(freq[1:]*40/uz_2_20, coh_kaimal(freq[1:], 40, uz_2_20, 8.1*0.7*20, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[0,2].plot(freq[1:]*80/uz_2_20, coh_kaimal(freq[1:], 80, uz_2_20, 8.1*0.7*20, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[0,2].set_xlim(0, 1.0); axs[0,2].set_xticklabels([])
axs[0,2].set_ylim(-0.2, 1.0); axs[0,2].set_yticklabels([])
axs[0,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[0,2].text(0.5, 0.6, r"$\mathrm{h = 20m}$", fontsize=12) # transform=axs[0,0].transAxes

## height = 100m
uz_0_100 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_100 = calc_uz_palm(4,varSeq_3) # palm
axs[1,0].plot(freq[1:]*40/uz_0_100, coh_0_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,0].plot(freq[1:]*80/uz_0_100, coh_0_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,0].plot(freq[1:]*40/uz_3_100, coh_3_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,0].plot(freq[1:]*80/uz_3_100, coh_3_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,0].plot(freq[1:]*40/uz_0_100, coh_kaimal(freq[1:], 40, uz_0_100, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[1,0].plot(freq[1:]*80/uz_0_100, coh_kaimal(freq[1:], 80, uz_0_100, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[1,0].set_xlim(0, 1.0); axs[1,0].set_xticklabels([])
axs[1,0].set_ylim(-0.2, 1.0)
axs[1,0].set_ylabel(r"coh", fontsize=12)
axs[1,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,0].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_100 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_100 = calc_uz_palm(4,varSeq_4) # palm
axs[1,1].plot(freq[1:]*40/uz_1_100, coh_1_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,1].plot(freq[1:]*80/uz_1_100, coh_1_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,1].plot(freq[1:]*40/uz_4_100, coh_4_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,1].plot(freq[1:]*80/uz_4_100, coh_4_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,1].plot(freq[1:]*40/uz_1_100, coh_kaimal(freq[1:], 40, uz_1_100, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[1,1].plot(freq[1:]*80/uz_1_100, coh_kaimal(freq[1:], 80, uz_1_100, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[1,1].set_xlim(0, 1.0); axs[1,1].set_xticklabels([])
axs[1,1].set_ylim(-0.2, 1.0); axs[1,1].set_yticklabels([])
axs[1,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,1].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_100 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_100 = calc_uz_palm(4,varSeq_5) # palm
axs[1,2].plot(freq[1:]*40/uz_2_100, coh_2_d40h100[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[1,2].plot(freq[1:]*80/uz_2_100, coh_2_d80h100[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[1,2].plot(freq[1:]*40/uz_5_100, coh_5_d40h100[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[1,2].plot(freq[1:]*80/uz_5_100, coh_5_d80h100[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[1,2].plot(freq[1:]*40/uz_2_100, coh_kaimal(freq[1:], 40, uz_2_100, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[1,2].plot(freq[1:]*80/uz_2_100, coh_kaimal(freq[1:], 80, uz_2_100, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[1,2].set_xlim(0, 1.0); axs[1,2].set_xticklabels([])
axs[1,2].set_ylim(-0.2, 1.0); axs[1,2].set_yticklabels([])
axs[1,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[1,2].text(0.5, 0.6, r"$\mathrm{h = 100m}$", fontsize=12) # transform=axs[0,0].transAxes

## height = 180m
uz_0_180 = calc_uz_sowfa(xSeq_0.size,ySeq_0.size,4,varSeq_0) # sowfa
uz_3_180 = calc_uz_palm(4,varSeq_3) # palm
axs[2,0].plot(freq[1:]*40/uz_0_180, coh_0_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,0].plot(freq[1:]*80/uz_0_180, coh_0_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,0].plot(freq[1:]*40/uz_3_180, coh_3_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,0].plot(freq[1:]*80/uz_3_180, coh_3_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,0].plot(freq[1:]*40/uz_0_180, coh_kaimal(freq[1:], 40, uz_0_180, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[2,0].plot(freq[1:]*80/uz_0_180, coh_kaimal(freq[1:], 80, uz_0_180, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[2,0].set_xlim(0, 1.0)
axs[2,0].set_ylim(-0.2, 1.0)
axs[2,0].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,0].set_ylabel(r"coh", fontsize=12)
axs[2,0].text(0.5, 0.8, r"$\mathrm{z_0 = 0.0001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,0].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_1_180 = calc_uz_sowfa(xSeq_1.size,ySeq_1.size,4,varSeq_1) # sowfa
uz_4_180 = calc_uz_palm(4,varSeq_4) # palm
axs[2,1].plot(freq[1:]*40/uz_1_180, coh_1_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,1].plot(freq[1:]*80/uz_1_180, coh_1_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,1].plot(freq[1:]*40/uz_4_180, coh_4_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,1].plot(freq[1:]*80/uz_4_180, coh_4_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,1].plot(freq[1:]*40/uz_1_180, coh_kaimal(freq[1:], 40, uz_1_180, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[2,1].plot(freq[1:]*80/uz_1_180, coh_kaimal(freq[1:], 80, uz_1_180, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[2,1].set_xlim(0, 1.0)
axs[2,1].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,1].set_ylim(-0.2, 1.0); axs[2,1].set_yticklabels([])
axs[2,1].text(0.5, 0.8, r"$\mathrm{z_0 = 0.001m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,1].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

uz_2_180 = calc_uz_sowfa(xSeq_2.size,ySeq_2.size,4,varSeq_2) # sowfa
uz_5_180 = calc_uz_palm(4,varSeq_5) # palm
axs[2,2].plot(freq[1:]*40/uz_2_180, coh_2_d40h180[1:], linestyle='-', marker='', markersize=1, color='r', label='sowfa-40m')
axs[2,2].plot(freq[1:]*80/uz_2_180, coh_2_d80h180[1:], linestyle='--', marker='', markersize=1, color='r', label='sowfa-80m')
axs[2,2].plot(freq[1:]*40/uz_5_180, coh_5_d40h180[1:], linestyle='-', marker='', markersize=1, color='b', label='palm-40m')
axs[2,2].plot(freq[1:]*80/uz_5_180, coh_5_d80h180[1:], linestyle='--', marker='', markersize=1, color='b', label='palm-80m')
axs[2,2].plot(freq[1:]*40/uz_2_180, coh_kaimal(freq[1:], 40, uz_2_180, 8.1*42, 12, 0.12), linestyle='-', marker='', markersize=1, color='k', label='Kaimal-40m')
axs[2,2].plot(freq[1:]*80/uz_2_180, coh_kaimal(freq[1:], 80, uz_2_180, 8.1*42, 12, 0.12), linestyle='--', marker='', markersize=1, color='k', label='Kaimal-80m')

axs[2,2].set_xlim(0, 1.0)
axs[2,2].set_xlabel(r'$\mathrm{f\delta/\overline{u}}$', fontsize=12)
axs[2,2].set_ylim(-0.2, 1.0); axs[2,2].set_yticklabels([])
axs[2,2].text(0.5, 0.8, r"$\mathrm{z_0 = 0.01m}$", fontsize=12) # transform=axs[0,0].transAxes
axs[2,2].text(0.5, 0.6, r"$\mathrm{h = 180m}$", fontsize=12) # transform=axs[0,0].transAxes

for i in range(3):
    for j in range(3):
        axs[i,j].grid(True)

# plt.legend(bbox_to_anchor=(0.0,1.06,1.,.12), loc=9, ncol=2, mode='expand', borderaxespad=0, fontsize=12)
handles, labels = axs[0,0].get_legend_handles_labels()
lgdord = [0,1,2,3,4,5]
fig.legend([handles[i] for i in lgdord], [labels[i] for i in lgdord], loc='upper center', bbox_to_anchor=(0.5,0.98), ncol=3, mode='None', borderaxespad=0, fontsize=12)
saveDir = '/scratch/projects/deepwind/photo/coherence'
saveName = 'coh_lat_gp.png'
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
