#!/usr/bin/python3.6
import sys
sys.path.append('/scratch/ppcode')
from netCDF4 import Dataset
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
import funcs

### ----------------------------------------------------------- velocity profile ---------------------------------------------------- ###

def velo_pr_palm(jobDir, jobName, run_no_list, var, dimprint=False, varprint=False, vardimprint=False):
    """
    Extract time and horizontal average of velocity at various times and heights based on profile data from PALM
    INPUT
        jobDir: job directory, e.g. '/scratch/palmdata/JOBS/example_nbl'
        jobName: jobName, e.g. 'example_nbl'
        run_no_list: list of suffixes of the INPUT data files, e.g. ['.000', '.001']
        var: name of the target variable, e.g. 'u'
        dimprint: all the dimension names will be printed out if True
        varprint: all the variable names will be printed out if True
        vardimprint: all the dimension names related to the target variable will be printed out if True
    OUTPUT
        tSeq: 1D array of times
        zSeq: 1D array of heights
        varSeq: 2D array of time and horizontal average variables, 1st dim time, 2nd dim height
    """
    run_num = len(run_no_list)
    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []
    for i in range(run_num):
        input_file = jobDir + "/OUTPUT/" + jobName + "_pr" + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        varSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(run_num)], axis=0)
    varSeq = varSeq.astype(float)

    if dimprint: # list all dimensions
        dimensions = list(nc_file_list[0].dimensions)
        print(list(nc_file_list[0].dimensions))
    if varprint: # list all variables
        vars = list(nc_file_list[0].variables)
        print(list(nc_file_list[0].variables))
    if vardimprint: # list dimensions of a specified variable
        print(list(nc_file_list[0].variables[var].dimensions))
    return tSeq, zSeq, varSeq


### ------------------------------------------------------- turbulence intensity ---------------------------------------------------- ###

def TI_pr_palm(jobDir, jobName, run_no_list):
    """
    Derive turbulence intensity in 3 directions based on profile data from PALM
    INPUT
        jobDir: job directory, e.g. '/scratch/palmdata/JOBS/example_nbl'
        jobName: jobName, e.g. 'example_nbl'
        run_no_list: list of suffixes of the INPUT data files, e.g. ['.000', '.001']
    OUTPUT
        tSeq: 1D array of times
        zSeq: 1D array of heights, zero height is excluded
        TIuSeq: TI 2d array in x direction, 1st dim time, 2nd dim height
        TIvSeq: TI 2d array in y direction, 1st dim time, 2nd dim height
        TIwSeq: TI 2d array in z direction, 1st dim time, 2nd dim height
    """
    run_num = len(run_no_list)
    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    uSeq_list = []
    uuSeq_list = []
    vSeq_list = []
    vvSeq_list = []
    wSeq_list = []
    wwSeq_list = []
    for i in range(run_num):
        input_file = jobDir + "/OUTPUT/" + jobName + "_pr" + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        uSeq_list.append(np.array(nc_file_list[i].variables['u'][:], dtype=type(nc_file_list[i].variables['u'])))
        uuSeq_list.append(np.array(nc_file_list[i].variables['u*2'][:], dtype=type(nc_file_list[i].variables['u*2'])))
        vSeq_list.append(np.array(nc_file_list[i].variables['v'][:], dtype=type(nc_file_list[i].variables['v'])))
        vvSeq_list.append(np.array(nc_file_list[i].variables['v*2'][:], dtype=type(nc_file_list[i].variables['v*2'])))
        wSeq_list.append(np.array(nc_file_list[i].variables['w'][:], dtype=type(nc_file_list[i].variables['w'])))
        wwSeq_list.append(np.array(nc_file_list[i].variables['w*2'][:], dtype=type(nc_file_list[i].variables['w*2'])))

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables['u'].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)

    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    uSeq = np.concatenate([uSeq_list[i] for i in range(run_num)], axis=0)
    uSeq = uSeq.astype(float)
    uuSeq = np.concatenate([uuSeq_list[i] for i in range(run_num)], axis=0)
    uuSeq = uuSeq.astype(float)
    vSeq = np.concatenate([vSeq_list[i] for i in range(run_num)], axis=0)
    vSeq = vSeq.astype(float)
    vvSeq = np.concatenate([vvSeq_list[i] for i in range(run_num)], axis=0)
    vvSeq = vvSeq.astype(float)
    wSeq = np.concatenate([wSeq_list[i] for i in range(run_num)], axis=0)
    wSeq = wSeq.astype(float)
    wwSeq = np.concatenate([wwSeq_list[i] for i in range(run_num)], axis=0)
    wwSeq = wwSeq.astype(float)

    TIuSeq = 100 * np.power(uuSeq[:,1:], 0.5) / uSeq[:,1:] # somehow negative var2 values appear in the first two time steps; TI(z=0) should be specified to 0
    TIvSeq = 100 * np.power(vvSeq[:,1:], 0.5) / uSeq[:,1:]
    TIwSeq = 100 * np.power(wwSeq[:,1:], 0.5) / uSeq[:,1:]

    return tSeq, zSeq[1:], TIuSeq, TIvSeq, TIwSeq


### ------------------------------------------------------- contours ---------------------------------------------------- ###

def getDataInfo_palm(file_in, var):
    """
    Get information of x,y,z,t to decide how much data we should extract
    INPUT
        file_in: directory of the file
        var: name of the target variable, e.g. 'u'
    OUTPUT
        tSeq: 1D array of times
        xSeq: 1D array of x positions
        ySeq: 1D array of y positions
        zSeq: 1D array of heights
    """
    input_file = Dataset(file_in, "r", format="NETCDF4")
    zName = list(input_file.variables[var].dimensions)[1]
    zSeq = np.array(input_file.variables[zName][:], dtype=type(input_file.variables[zName]))
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    yName = list(input_file.variables[var].dimensions)[2] # the height name string
    ySeq = np.array(input_file.variables[yName][:], dtype=type(input_file.variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(input_file.variables[var].dimensions)[3] # the height name string
    xSeq = np.array(input_file.variables[xName][:], dtype=type(input_file.variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size
    tSeq = np.array(input_file.variables['time'][:], dtype=type(input_file.variables['time']))
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    print('xMin = ', xSeq[0], ', ', 'xMax = ', xSeq[-1], ', ', 'xNum = ', xNum)
    print('yMin = ', ySeq[0], ', ', 'yMax = ', ySeq[-1], ', ', 'yNum = ', yNum)
    print('zMin = ', zSeq[0], ', ', 'zMax = ', zSeq[-1], ', ', 'zNum = ', zNum)
    print('tMin = ', tSeq[0], ', ', 'tMax = ', tSeq[-1], ', ', 'tNum = ', tNum)
    return tSeq, xSeq, ySeq, zSeq


def mask_data_palm(jobDir, jobName, maskID, run_no_list, var, tInd, xInd, yInd, zInd):
    """
    Extract masked data files
    INPUT
        jobDir: job directory, e.g. '/scratch/palmdata/JOBS/example_nbl'
        jobName: jobName, e.g. 'example_nbl'
        maskID: mask ID, e.g. 'M01'
        run_no_list: list of suffixes of the INPUT data files, e.g. ['.000', '.001']
        var: name of the target variable, e.g. 'u'
        tInd: a tuple containing the start time index and end time index of the output data (not including the end tInd)
        xInd: a tuple containing the start x index and end x index of the output data (not including the end xInd)
        yInd: a tuple containing the start y index and end y index of the output data (not including the end yInd)
        zInd: a tuple containing the start z index and end z index of the output data (not including the end zInd)
    OUTPUT
        tSeq: 1D array of times
        xSeq: 1D array of x positions
        ySeq: 1D array of y positions
        zSeq: 1D array of heights
        varSeq: 4D array of the target variable, 1st dim time, 2nd dim height, 3rd dim y, 4th dim x
    """
    """ to opt """
    tInd = list(tInd)
    run_num = len(run_no_list)
    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []

    tInd_start = 0
    list_num = 0
    for i in range(run_num):
        input_file = jobDir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))

        tSeq_tmp = np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time']))
        tInd_end = tInd_start + tSeq_tmp.size -1

        if tInd[0] >= tInd_start + tSeq_tmp.size:
            tInd_start += tSeq_tmp.size
            continue
        else:
            if tInd[1] < tInd_start + tSeq_tmp.size:
                tSeq_list.append(tSeq_tmp[tInd[0]-tInd_start:tInd[1]-tInd_start])
                varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:tInd[1]-tInd_start, zInd[0]:zInd[1], yInd[0]:yInd[1], xInd[0]:xInd[1]],
                                                                           dtype=type(nc_file_list[i].variables[var])))
                list_num += 1
                break
            else:
                tSeq_list.append(tSeq_tmp[tInd[0]-tInd_start:])
                varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:, zInd[0]:zInd[1], yInd[0]:yInd[1], xInd[0]:xInd[1]],
                                                                           dtype=type(nc_file_list[i].variables[var])))
                tInd[0] = tInd_start + tSeq_tmp.size
                tInd_start += tSeq_tmp.size
                list_num += 1
    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][zInd[0]:zInd[1]], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][yInd[0]:yInd[1]], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][xInd[0]:xInd[1]], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size
    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(list_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(list_num)], axis=0)
    varSeq = varSeq.astype(float)
    return tSeq, xSeq, ySeq, zSeq, varSeq


def sec_data_palm(jobDir, jobName, suffix, run_no_list, var, tInd, xInd, yInd, zInd):
    """
    similar to mask_data_palm but based on cross section data in PALM
    """
    tInd = list(tInd)
    run_num = len(run_no_list)
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []
    
    tInd_start = 0
    list_num = 0
    for i in range(run_num):
        input_file = jobDir + "/OUTPUT/" + jobName + suffix + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_tmp = np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time']))
        tInd_end = tInd_start + tSeq_tmp.size -1
        
        if tInd[0] >= tInd_start + tSeq_tmp.size:
            tInd_start += tSeq_tmp.size
            continue
        else:
            if tInd[1] < tInd_start + tSeq_tmp.size:
                tSeq_list.append(tSeq_tmp[tInd[0]-tInd_start:tInd[1]-tInd_start])
                varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:tInd[1]-tInd_start, zInd[0]:zInd[1], yInd[0]:yInd[1], xInd[0]:xInd[1]],
                                                                           dtype=type(nc_file_list[i].variables[var])))
                list_num += 1
                break
            else:
                tSeq_list.append(tSeq_tmp[tInd[0]-tInd_start:])
                varSeq_list.append(np.array(nc_file_list[i].variables[var][tInd[0]-tInd_start:, zInd[0]:zInd[1], yInd[0]:yInd[1], xInd[0]:xInd[1]],
                                                                           dtype=type(nc_file_list[i].variables[var])))
                tInd[0] = tInd_start + tSeq_tmp.size
                tInd_start += tSeq_tmp.size
                list_num += 1
    
    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][zInd[0]:zInd[1]], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zSeq = zSeq.astype(float)
    zNum = zSeq.size
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][:], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    yNum = ySeq.size
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][xInd[0]:xInd[1]], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    xNum = xSeq.size
    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(list_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(list_num)], axis=0)
    varSeq = varSeq.astype(float)
    return tSeq, xSeq, ySeq, zSeq, varSeq


### ------------------------------------------------------- PSD ---------------------------------------------------- ###

def PSD_data_palm(jobDir, jobName, maskID, run_no_list, var):
    """
    Extract masked data files for PSD plot
    INPUT
        jobDir: job directory, e.g. '/scratch/palmdata/JOBS/example_nbl'
        jobName: jobName, e.g. 'example_nbl'
        maskID: mask ID, e.g. 'M01'
        run_no_list: list of suffixes of the INPUT data files, e.g. ['.000', '.001']
        var: name of the target variable, e.g. 'u'
    """
    run_num = len(run_no_list)
    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []
    for i in range(run_num):
        input_file = jobDir + "/OUTPUT/" + jobName + "_masked_" + maskID + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        varSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))

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

def PSD_palm(t_para, tSeq, zInd, xNum, yNum, varSeq, segNum):
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
    """
    t_start = t_para[0]
    t_end = t_para[1]
    t_delta = t_para[2]
    fs = 1 / t_delta # sampling frequency
    t_num = int((t_end - t_start) / t_delta + 1)
    t_seq = np.linspace(t_start, t_end, t_num)

    PSD_list = []
    for yInd in range(yNum):
        for xInd in range(xNum):
            vSeq = varSeq[:,zInd,yInd,xInd]
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

def TKE_palm(jobDir, jobName, run_no_list):
    """
    Extract TKE data from profile data
    INPUT
        jobDir: job directory, e.g. '/scratch/palmdata/JOBS/example_nbl'
        jobName: jobName, e.g. 'example_nbl'
        run_no_list: list of suffixes of the INPUT data files, e.g. ['.000', '.001']
    OUTPUT
        tSeq: time array
        zSeq: height array
        rsvSeq: resolved TKE array
        sgsSeq: SGS TKE array
        totSeq: total TKE array        
    """
    run_num = len(run_no_list)
    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    rsvSeq_list = []
    sgsSeq_list = []
    for i in range(run_num):
        input_file = jobDir + "/OUTPUT/" + jobName + "_pr" + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        rsvSeq_list.append(np.array(nc_file_list[i].variables['e*'][:], dtype=type(nc_file_list[i].variables['e*'])))
        sgsSeq_list.append(np.array(nc_file_list[i].variables['e'][:], dtype=type(nc_file_list[i].variables['e'])))

    height = list(nc_file_list[0].variables['e*'].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[height][:], dtype=type(nc_file_list[0].variables[height])) # array of height levels
    zSeq = zSeq.astype(float)
    zNum = zSeq.size

    # concatenate arraies of all run_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size

    rsvSeq = np.concatenate([rsvSeq_list[i] for i in range(run_num)], axis=0)
    rsvSeq = rsvSeq.astype(float)
    sgsSeq = np.concatenate([sgsSeq_list[i] for i in range(run_num)], axis=0)
    sgsSeq = sgsSeq.astype(float)
    totSeq = rsvSeq + sgsSeq
    return tSeq, zSeq, rsvSeq, sgsSeq, totSeq

def TKE_plot_palm(TKESeq, tSeq, zNum, tplot):
    """
    calculate TKE at a certain time by extrapolation or interpolation based on profile data
    INPUT
        TKEseq: TKE array
        tSeq: time array
        zNum: number of height levels
        tplot: the target time
    OUTPUT
        TKESeq_: the averaged TKE array
    """
    TKESeq_ = np.zeros(zNum)
    for zInd in range(zNum):
        f = interp1d(tSeq[1:], TKESeq[1:,zInd], kind='linear', fill_value='extrapolate')
        TKESeq_[zInd] = f(tplot)
    return TKESeq_



### ------------------------------------------------------- MOST ---------------------------------------------------- ###

def calc_uStar_palm(zSeq,uSeq,zMO_ind=0,kappa=0.4,z0=0.001):
    zMO = zSeq[zMO_ind]
    uMO = np.mean(uSeq[:,zMO_ind,:,:])
    uStar = kappa * uMO / np.log(zMO/z0)
    return uStar

def calc_uz_palm(zInd,uSeq):
    uz = np.mean(uSeq[:,zInd,:,:])
    return uz


### ------------------------------------------------------- coherence ---------------------------------------------------- ###

def coh_palm(p0_ind, p1_ind, t_para, tSeq, varSeq, tL):
    """
    Calculate coherence, co-coherence, phase based on interpolated data
    INPUT
        p0_ind: indice of the probe0
        p1_ind: indice of the probe1
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
        quad_coh: quad_coh array
        phase: phase array
    """
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
    """
    Calculate the probe-group-averaged coh, co-coh, phase and their standard deviation along x axis
    INPUT
        xInd_start: start of the x index of the target probe group
        xInd_end: end of the x index of the target probe group
        dxInd: step size of the x index
        yInd: y index of the target probe group
        zInd: height index of the target probe group
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
        results = coh_palm((xInd_start+i,yInd,zInd), (xInd_start+i+dxInd,yInd,zInd), t_para, tSeq, varSeq, tL)
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

def coh_av_palm_y(xInd, yInd_start, yInd_end, dyInd, zInd, t_para, tSeq, varSeq, tL):
    """
    Calculate the probe-group-averaged coh, co-coh, phase and their standard deviation along y axis
    """
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

def coh_av_palm_z(xInd, yInd_start, yInd_end, zInd_start, zInd_end, t_para, tSeq, varSeq, tL):
    """
    Calculate the probe-group-averaged coh, co-coh, phase and their standard deviation along z axis
    """
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