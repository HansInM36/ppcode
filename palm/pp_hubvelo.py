#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

# wrapper

import os
import sys
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm

jobname  = 'pcr_NBL_U10_gs20'
cycle_no_list = ['.004'] # "" for initial run, ".001" for first cycle, etc.
cycle_num = len(cycle_no_list)

#time     = "7200"           # in s
#height   = "90"             # z-position of plot (nacelle at 90 m)
#xpos     = "1030"           # x-position of plot (turbine at 2000 m)
#ypos     = "2400"           # y-position of plot (turbine at 2000 m)

var1 = 'u'
var2 = 'v'
zhub = 102

print("+++ Preparing plots for run " + jobname + "...")

# read the output data of all cycle_no_list
nc_file_list = []
tseq_list = []
var1_list = []
var2_list = []
for i in range(cycle_num):
    input_file = "/scratch/palmdata/JOBS/" + jobname + "/OUTPUT/" + jobname + "_pr" + cycle_no_list[i] + ".nc"
    nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
    tseq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
    var1_list.append(np.array(nc_file_list[i].variables[var1][:], dtype=type(nc_file_list[i].variables[var1])))
    var2_list.append(np.array(nc_file_list[i].variables[var2][:], dtype=type(nc_file_list[i].variables[var2])))

height = list(nc_file_list[0].variables[var1].dimensions)[1] # the height name string
zseq = np.array(nc_file_list[0].variables[height][:], dtype=type(nc_file_list[0].variables[height])) # array of height levels

# concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
tseq = np.concatenate([tseq_list[i] for i in range(cycle_num)], axis=0)
tlen = tseq.size

var1seq = np.concatenate([var1_list[i] for i in range(cycle_num)], axis=0)
var2seq = np.concatenate([var2_list[i] for i in range(cycle_num)], axis=0)

### compute the horizontal velocity and direction at given height
for tind in range(tlen):
    f1 = interp1d(zseq.astype(float), var1seq[tind].astype('float'), kind='cubic', fill_value="extrapolate")
    f2 = interp1d(zseq.astype(float), var2seq[tind].astype('float'), kind='cubic', fill_value="extrapolate")
    print('Time: ', round(tseq[tind],2), 's ', 'u_hub = ', round(float(f1(zhub)),3), 'v_hub = ', round(float(f2(zhub)),3))
