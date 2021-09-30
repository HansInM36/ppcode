#!/usr/bin/python3.6
import numpy as np
from scipy.interpolate import interp1d

def palm_pr_evo(tplot_para, tSeq, zSeq, varSeq):
    """
    calculate time-average profile at various times based on the PALM pr data
    INPUT:
        tplot_para: start time, end time and time step of the averaged variable array (0.0,12000.0,1200.0)
        tSeq: time series of the PALM pr data
        zSeq: height array of the PALM pr data
        varSeq: the target variable array
    """
    tplot_start = tplot_para[0]
    tplot_end = tplot_para[1]
    tplot_delta = tplot_para[2]
    
    tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
    tplotSeq = np.linspace(tplot_start, tplot_end, tplotNum)
    
    plotArr = np.zeros((tplotNum,zSeq.size))
    for zInd in range(zSeq.size):
        f = interp1d(tSeq, varSeq[:,zInd], fill_value='extrapolate')
        for tplotInd in range(tplotSeq.size):
            plotArr[tplotInd,zInd] = f(tplotSeq[tplotInd])
    return tplotSeq, plotArr
        

def velo_pr_ave(tplot_para, tSeq, tDelta, zNum, varSeq):
    """
    Calculate temporally averaged horizontal average variable at various times and heights by extrapolation and interpolation
    INPUT:
        tplot_para: the averge time interval, start time, end time and time step of the new averaged variable array (3600,151200,151200,1e6)
        tSeq: time array of the input data
        tDelta: time step of the input data
        zNum: number of height levels of the input data
        varSeq: 2d array input data, 1st dim time, 2nd dim height
    OUTPUT:
        t_seq: 1d time array
        varSeq: 2d array of temporally averaged variable array, 1st dim time, 2st dim height
    """
    ave_itv = tplot_para[0]
    tplot_start = tplot_para[1]
    tplot_end = tplot_para[2]
    tplot_delta = tplot_para[3]
    tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
    tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

    # compute the averaged velocity at a certain time and height
    varplotList = []
    for tplot in tplotList:
        varplot = np.zeros(zNum)
        for zInd in range(zNum):
            f = interp1d(tSeq, varSeq[:,zInd], kind='linear', fill_value='extrapolate')
            tplot_tmp = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
            varplot[zInd] = f(tplot_tmp).mean()
        varplotList.append(varplot)
    t_seq = np.array(tplotList)
    varSeq = np.array(varplotList)
    return t_seq, varSeq


def TI_pr_ave(tplot_para, tSeq, tDelta, zNum, TIuSeq, TIvSeq, TIwSeq):
    """
    Calculate temporally averaged turbulence intensity at various times and heights by extrapolation and interpolation
    INPUT:
        tplot_para: the averge time interval, start time, end time and time step of the new averaged variable array (3600,151200,151200,1e6)
        tSeq: time array of the input data
        tDelta: time step of the input data
        zNum: number of height levels of the input data
        TIuSeq: 2d array TIu data, 1st dim time, 2nd dim height
        TIvSeq: 2d array TIv data, 1st dim time, 2nd dim height
        TIwSeq: 2d array TIw data, 1st dim time, 2nd dim height
    OUTPUT:
        t_seq: 1d time array
        TIuSeq: 2d array of temporally averaged TIu array, 1st dim time, 2st dim height   
        TIvSeq: 2d array of temporally averaged TIv array, 1st dim time, 2st dim height
        TIwSeq: 2d array of temporally averaged TIw array, 1st dim time, 2st dim height
    """
    ave_itv = tplot_para[0]
    tplot_start = tplot_para[1]
    tplot_end = tplot_para[2]
    tplot_delta = tplot_para[3]
    tplotNum = int((tplot_end - tplot_start)/tplot_delta+1)
    tplotList = list(np.linspace(tplot_start, tplot_end, tplotNum))

    # compute the averaged velocity at a certain time and height
    TIuplotList = []
    TIvplotList = []
    TIwplotList = []
    for tplot in tplotList:
        TIuplot = np.zeros(zNum)
        TIvplot = np.zeros(zNum)
        TIwplot = np.zeros(zNum)
        for zInd in range(zNum):
            f1 = interp1d(tSeq, TIuSeq[:,zInd], kind='linear', fill_value='extrapolate')
            f2 = interp1d(tSeq, TIvSeq[:,zInd], kind='linear', fill_value='extrapolate')
            f3 = interp1d(tSeq, TIwSeq[:,zInd], kind='linear', fill_value='extrapolate')
            tplot_tmp = np.linspace(tplot - ave_itv, tplot, int(ave_itv/tDelta))
            TIuplot[zInd] = f1(tplot_tmp).mean()
            TIvplot[zInd] = f2(tplot_tmp).mean()
            TIwplot[zInd] = f3(tplot_tmp).mean()
        TIuplotList.append(TIuplot)
        TIvplotList.append(TIvplot)
        TIwplotList.append(TIwplot)
    t_seq = np.array(tplotList)
    TIuSeq = np.array(TIuplotList)
    TIvSeq = np.array(TIvplotList)
    TIwSeq = np.array(TIwplotList)
    return t_seq, TIuSeq, TIvSeq, TIwSeq