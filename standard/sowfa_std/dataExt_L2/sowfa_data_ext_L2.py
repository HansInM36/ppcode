#!/usr/bin/python3.6
import sys
sys.path.append('/scratch/ppcode')
sys.path.append('/scratch/ppcode/sowfa/src')
import sliceDataClass as sdc
import pickle
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
import funcs


### ----------------------------------------------------------- velocity profile ---------------------------------------------------- ###

def velo_pr_sowfa(dir, trs_para, varD):
    """ 
    Extract horizontal average of velocity at various times and heights based on aveData from SOWFA
    INPUT
        dir: job directory, e.g. '/scratch/sowfadata/pp/tutorials/example_nbl'
        trs_para: coordinate transform parameters, e.g. ((0,0,0),30.0), the origin and the conterclockwise rotation degree
        varD: the dimension of which the target vector variable will be outputted
    OUTPUT:
        tSeq: 1D array of times
        zSeq: 1D array of heights
        varSeq: 2D array of time and horizontal average variables, 1st dim time, 2nd dim height
    """
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

    return tSeq, zSeq, varSeq




### ------------------------------------------------------- turbulence intensity ---------------------------------------------------- ###

def TI_pr_sowfa(dir, trs_para):
    """
    Derive turbulence intensity in 3 directions based on aveData from SOWFA
    INPUT:
        dir: job directory, e.g. '/scratch/sowfadata/pp/tutorials/example_nbl'
        trs_para: coordinate transform parameters, e.g. ((0,0,0),30.0), the origin and the conterclockwise rotation degree
    OUTPUT
        tSeq: 1D array of times
        zSeq: 1D array of heights, zero height is excluded
        TIuSeq: TI 2d array in x direction, 1st dim time, 2nd dim height
        TIvSeq: TI 2d array in y direction, 1st dim time, 2nd dim height
        TIwSeq: TI 2d array in z direction, 1st dim time, 2nd dim height    
    """
    # coordinate transmation
    O = trs_para[0]
    alpha = trs_para[1]*np.pi/180

    fr = open(dir + '/data/' + 'aveData', 'rb')
    aveData = pickle.load(fr)
    fr.close()

    zSeq = aveData['H']
    zNum = zSeq.size

    tSeq = aveData['time']
    tNum = tSeq.size
    tDelta = tSeq[1] - tSeq[0]

    uuSeq = aveData['uu_mean']
    vvSeq = aveData['vv_mean']
    uvSeq = aveData['uv_mean']
    wwSeq = aveData['ww_mean']
    uSeq = aveData['U_mean']
    vSeq = aveData['V_mean']

    uvarianceSeq = uuSeq*np.power(np.cos(alpha),2) + 2*uvSeq*np.cos(alpha)*np.sin(alpha) + vvSeq*np.power(np.sin(alpha),2)
    vvarianceSeq = uuSeq*np.power(np.sin(alpha),2) + 2*uvSeq*np.sin(alpha)*np.cos(alpha) + vvSeq*np.power(np.cos(alpha),2)
    wvarianceSeq = wwSeq
    umeanSeq = uSeq*np.cos(alpha) + vSeq*np.sin(alpha)
    TIuSeq = 100 * np.power(uvarianceSeq,0.5) / umeanSeq
    TIvSeq = 100 * np.power(vvarianceSeq,0.5) / umeanSeq
    TIwSeq = 100 * np.power(wvarianceSeq,0.5) / umeanSeq
    return tSeq, zSeq, TIuSeq, TIvSeq, TIwSeq


### ------------------------------------------------------- contours ---------------------------------------------------- ###

def getSliceData_Nz_sowfa(dir, slice, var, varD, trs_para, tInd, xcoor, ycoor):
    """
    Extract data from specified slice_Nz
    INPUT
        dir: job directory, e.g. '/scratch/sowfadata/pp/tutorials/example_nbl'
        slice: the name of the slice, e.g. 'Nz2'
        var: name of the target variable, e.g. 'U'
        varD: the dimension of which the target vector variable will be outputted
        trs_para: coordinate transform parameters, e.g. ((0,0,0),30.0), the origin and the conterclockwise rotation degree
        tInd: a tuple containing the start time index and end time index of the output data (not including the end tInd)
        xcoor: a tuple containing the start x, end x and delta x of the output data
        ycoor: a tuple containing the start x, end x and delta x of the output data
    OUTPUT
        tSeq: 1D array of times
        ySeq: 1D array of y positions
        xSeq: 1D array of x positions
        H: the height of the slice
        varSeq: 3D array of the target variable, 1st dim time, 2nd dim y, 3rd x
    """
    readDir = dir + '/data/'
    readName = slice
    fr = open(readDir + readName, 'rb')
    data = pickle.load(fr)
    fr.close()
    slc = sdc.Slice(data, 2); del data # 2 means z-axis
    tSeq = slc.data['time']
    tSeq = tSeq[tInd[0]:tInd[1]]
    H = slc.N_location # height of this plane

    O, alpha = trs_para[0], trs_para[1]
    varSeq = []
    for t_ind in range(tInd[0],tInd[1]):
        print('processing: ' + str(t_ind) + ' ...')
        tmp = slc.data[var][t_ind]
        tmp = funcs.trs(tmp,O,alpha)
        tmp1 = slc.meshITP_Nz(xcoor, ycoor, tmp[:,varD], method_='linear')
        xSeq, ySeq, varSeq_ = tmp1[0], tmp1[1], tmp1[2]
        varSeq.append(varSeq_)
    varSeq = np.array(varSeq)
    return tSeq, xSeq, ySeq, H, varSeq


### ------------------------------------------------------- PSD ---------------------------------------------------- ###

def PSD_data_sowfa(dir, prbg, trs_para, var, varD):
    """
    Extract velocity data of specified probe groups
    INPUT
        dir: job directory, e.g. '/scratch/sowfadata/pp/tutorials/example_nbl'
        prbg: name of the probe group, e.g. 'prbg0'
        trs_para: coordinate transform parameters, e.g. ((0,0,0),30.0), the origin and the conterclockwise rotation degree
        var: name of the target variable, e.g. 'U'
        varD: the dimension of which the target vector variable will be outputted
    OUTPUT
        tSeq: 1D array of times
        xSeq: 1D array of x positions
        ySeq: 1D array of y positions
        zSeq: 1D array of heights
        varSeq: 2D array of the target variable, 1st dim point ID, 2nd dim time
        coors: 2D array of coordinates of the probes, 1st dim point ID, 2nd dim x, y, z
    """
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


def PSD_sowfa(t_para, tSeq, zInd, xNum, yNum, varSeq, segNum):
    """
    Calculate power spectral density
    INPUT
        t_para: time parameters, tuple, e.g. (144000.0, 146400, 0.1)
        tSeq: 1D array of times
        zInd: height
        xNum: length of xSeq
        yNum: length of ySeq
        varSeq: 2D array of the target variable, 1st dim point ID, 2nd dim time
        segNum: length of the segment
    OUTPUT
        f_seq: frequency sequence
        PSD_seq: PSD sequence
    """
    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    data = varSeq
    tNum = tSeq.size
    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)
    # coors = data_org['coors'][xNum*yNum*zInd:xNum*yNum*(zInd+1)]
    # pNum = coors.shape[0]
    PSD_list = []
    for p in range(pInd_start, pInd_end):
        vSeq = varSeq[p]
        # interpolate
        f = interp1d(tSeq, vSeq, kind='linear', fill_value='extrapolate')
        v_seq = f(t_seq)
        # detrend
        deg_ = 1
        polyFunc = np.poly1d(np.polyfit(t_seq, v_seq, deg=deg_))
        tmp = v_seq - polyFunc(t_seq)
        tmp = tmp - tmp.mean()
        # bell tapering
        tmp = funcs.window_weight(tmp)
        # FFT
        # omega_seq, tmp = PSD_omega(t_seq,tmp)
        f_seq, tmp = scipy.signal.csd(tmp, tmp, fs, nperseg=segNum, noverlap=None)
        PSD_list.append(tmp)
    PSD_seq = np.average(np.array(PSD_list), axis=0)
    return f_seq, PSD_seq



### ------------------------------------------------------- TKE ---------------------------------------------------- ###

def TKE_sowfa(dir, trs_para, varD):
    """
    Extract TKE related data from aveData
    INPUT
        dir: job directory, e.g. '/scratch/sowfadata/pp/tutorials/example_nbl'
        trs_para: coordinate transform parameters, e.g. ((0,0,0),30.0), the origin and the conterclockwise rotation degree
        varD: the dimension of which the target vector variable will be outputted
    OUTPUT
        tSeq: time array
        zSeq: height array
        rsvSeq: resolved TKE array
        sgsSeq: SGS TKE array
        totSeq: total TKE array
    """
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

    uuSeq = aveData['uu_mean']
    vvSeq = aveData['vv_mean']
    wwSeq = aveData['ww_mean']
    R11Seq = aveData['R11_mean']
    R22Seq = aveData['R22_mean']
    R33Seq = aveData['R33_mean']

    rsvSeq = 0.5 * (uuSeq + vvSeq + wwSeq)
    sgsSeq = 0.5 * (R11Seq + R22Seq + R33Seq)
    totSeq = rsvSeq + sgsSeq
    return tSeq, zSeq, rsvSeq, sgsSeq, totSeq


def TKE_av_sowfa(TKESeq, tSeq, zNum, t_para):
    """
    Use aveData to calculate temporally average TKE at a certain time
    INPUT
        TKEseq: TKE array
        tSeq: time array
        zNum: number of height levels
        t_para: time parameters, tuple, 1st element average interval, 2nd element time
    OUTPUT
        TKESeq_av: the averaged TKE array
    """
    tDelta = tSeq[1] - tSeq[0]
    ave_itv = t_para[0]
    tplot = t_para[1]
    TKESeq_av = np.zeros(zNum)
    for zInd in range(zNum):
        f = interp1d(tSeq, TKESeq[:,zInd], kind='linear', fill_value='extrapolate')
        tplotSeq = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
        TKESeq_av[zInd] = f(tplotSeq).mean()
    return TKESeq_av



### ------------------------------------------------------- MOST ---------------------------------------------------- ###

def calc_uStar_sowfa(xNum,yNum,zSeq,uSeq,zMO_ind=0,kappa=0.4,z0=0.001):
    zMO = zSeq[zMO_ind]
    pInd_start = xNum*yNum*zMO_ind
    pInd_end = xNum*yNum*(zMO_ind+1)
    uMO = np.mean(uSeq[pInd_start:pInd_end])
    uStar = kappa * uMO / np.log(zMO/z0)
    return uStar

def calc_uz_sowfa(xNum,yNum,zInd,uSeq):
    pInd_start = xNum*yNum*zInd
    pInd_end = xNum*yNum*(zInd+1)
    uz = np.mean(uSeq[pInd_start:pInd_end])
    return uz



### ------------------------------------------------------- coherence ---------------------------------------------------- ###

def coh_sowfa(p0_ind, p1_ind, xNum, yNum, t_para, tSeq, varSeq, tL):
    """
    Calculate coherence, co-coherence, phase based on interpolated data
    INPUT
        p0_ind: indice of the probe0
        p1_ind: indice of the probe1
        xNum: length of xSeq
        yNum: length of ySeq
        t_para: parameters of the interpolated time array, tuple, (start time, end time, time step)
        tSeq: original time array
        varSeq: variable array
        tL: length of subsegment (s)
    OUTPUT
        t_seq: interpolated time array
        u0_: 1st velocity array
        u1_: 2nd velocity array
        freq: frequency array
        coh: coherence array
        co_coh: co-coh array
        quad_coh: quad-coh array
        phase: phase array
    """
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

def coh_av_sowfa_x(xInd_start, xInd_end, dxInd, yInd, zInd, xNum, yNum, t_para, tSeq, varSeq, tL):
    """
    Calculate the probe-group-averaged coh, co-coh, phase and their standard deviation along x axis
    INPUT
        xInd_start: start of the x index of the target probe group
        xInd_end: end of the x index of the target probe group
        dxInd: step size of the x index
        yInd: y index of the target probe group
        zInd: height index of the target probe group
        xNum: number of total x indice
        yNum: number of total y indice
        t_para: parameters of the interpolated time array, tuple, (start time, end time, time step)
        tSeq: the original time array
        varSeq: the original variable array
        tL: length of subsegment (s)
    OUPUT
        freq: frequency array
        coh_av: averaged coh
        coh_std: standard deviation of coh
        co-coh_av: averaged co-coh
        co-coh_std: standard deviation of co-coh
        quad_coh_av: averaged quad_coh
        quad_coh_std: standard deviation of quad-coh
        phase_av: averaged phase
        phase_std: standard deviation of phase
    """
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
#    coh = np.array(coh)
#    coh_av = np.mean(coh, axis=0)
#    coh_std = np.std(coh, axis=0)
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
    """
    Calculate the probe-group-averaged coh, co-coh, phase and their standard deviation along y axis
    """
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
#    coh = np.array(coh)
#    coh_av = np.mean(coh, axis=0)
#    coh_std = np.std(coh, axis=0)
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
