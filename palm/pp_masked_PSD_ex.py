#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

# wrapper

import os
import sys
import math
import numpy as np
from numpy import fft
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import pickle

jobname  = 'pcr_NBL_U10_gs20'
rnList = ['.005'] # "" for initial run, ".001" for first cycle, etc.
run_num = len(rnList)

maskid = 'M01'

varname = 'w'

zlist = [0,1,2,3,4,5,6,7] # specify the list of height level index where you want to extract the PSD

# read the dimension 'time' of all rnList and check what is the timeslot
file_rn = []
tseq_rn = []
for i in range(run_num):
    input_file = "/scratch/palmdata/JOBS/" + jobname + "/OUTPUT/" + jobname + "_masked_" + maskid + rnList[i] + ".nc"
    file_rn.append(Dataset(input_file, "r", format="NETCDF4"))
    tseq_rn.append(np.array(file_rn[i].variables['time'][:], dtype=type(file_rn[i].variables['time'])))
# concatenate arraies of all rnList along the first dimension (axis=0), i.e. time
tseq = np.concatenate([tseq_rn[i] for i in range(run_num)], axis=0)
tlen = tseq.size

# set the interval, start and end of sampling
t_delta = 2.0
t_start = 432000.0
t_end = 435600.0

# print(list(file_rn[0].dimensions)) #list all dimensions
# print(list(file_rn[0].variables)) #list all the variables
# print(list(file_rn[0].variables['u'].dimensions)) #list dimensions of a specified variable


print("+++ Preparing plots for run " + jobname + "...")

# get the frequency series (only keep the half)
tlen_ip = t_end - t_start
tnum = int(tlen_ip / t_delta) + 1
tseq_ip = np.linspace(t_start, t_end, tnum)
fseq = np.linspace(0, 1/t_delta, tnum)
fseq = fseq[:math.ceil(tnum/2)]

# extract the values of all dimensions of the var1
zname = list(file_rn[0].variables[varname].dimensions)[1] # the height name string
zseq = np.array(file_rn[0].variables[zname][:], dtype=type(file_rn[0].variables[zname])) # array of height levels
yname = list(file_rn[0].variables[varname].dimensions)[2] # the height name string
yseq = np.array(file_rn[0].variables[yname][:], dtype=type(file_rn[0].variables[yname])) # array of height levels
xname = list(file_rn[0].variables[varname].dimensions)[3] # the height name string
xseq = np.array(file_rn[0].variables[xname][:], dtype=type(file_rn[0].variables[xname])) # array of height levels
xlen = xseq.size
ylen = yseq.size
seqnum = int(xlen*ylen)

# PSD Dict restoring PSD of var at all height levels
PSDD = {}

for iz in zlist:
    # extract the part of the var and concatenate arraies of all rnList
    PSD = np.zeros(fseq.size)
    for ix in range(0,xlen):
        for iy in range(0,ylen):
            print( 'Height: ', zseq[iz], '', '%' + str(round(((ix+1)*ylen + (iy+1))/seqnum*100, 2)) + ' finished ...' )
            varseq_list = []
            for i in range(run_num):
                varseq_list.append(np.array(file_rn[i].variables[varname][:,iz,iy,ix], dtype=type(file_rn[i].variables[varname])))
            varseq = np.concatenate([varseq_list[i] for i in range(run_num)], axis=0)
            f = interp1d(tseq.astype('float'), varseq.astype('float'), kind='cubic', fill_value="extrapolate")
            varseq_ip = f(tseq_ip)
            varseq_ip = varseq_ip - np.mean(varseq_ip)
            var_PSD = np.power(abs(fft.fft(varseq_ip)),2) / tlen # the square of FFT series over time interval is power spectrum density
            var_PSD = var_PSD[:math.ceil(tnum/2)] # keep the half
            PSD = PSD + var_PSD/seqnum

    PSDD[str(zseq[iz])] = PSD

PSDD['fseq'] = fseq


saveDir = '/scratch/palmdata/pp/' + jobname + '/data/'
saveName = varname +'_PSDD'
fw = open(saveDir + saveName, "wb")
pickle.dump(PSDD, fw, 2)
fw.close()
