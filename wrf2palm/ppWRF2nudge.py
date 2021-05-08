import os
import sys
import datetime
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt

def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.datetime.fromordinal(int(datenum)) \
           + datetime.timedelta(days=int(days)) \
           + datetime.timedelta(hours=int(hours)) \
           + datetime.timedelta(minutes=int(minutes)) \
           + datetime.timedelta(seconds=round(seconds)) \
           - datetime.timedelta(days=366)


""" ppWRF data """
readDir = '/scratch/projects/EERA-SP3/data/WRFpp'
readName = "WRFOUT_NODA_20150701.nc"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimList = list(data.dimensions)
varList = list(data.variables)

time_org = data.variables['time'][:]
dateTime = []
time = []
for tInd in range(time_org.size):
    dateTime.append(datenum_to_datetime(time_org[tInd]))
    time.append(dateTime[tInd].timestamp() - dateTime[0].timestamp())

Z = data.variables['Z'][:]
U = data.variables['U'][:]
V = data.variables['V'][:]
THETA = data.variables['THETA'][:] # potential temperature?
Q = data.variables['Q'][:] 


### write nudging data (constant)

writeDir = '/scratch/palmdata/JOBS/EERASP3_1_ndg/INPUT'
writeName = 'EERASP3_1_ndg_nudge'

tInd = 6 # tInd of the target state
tau = 3600.0

W = 0.0

itemList = ['zu (m)', 'tau (s)', 'u (m/s)', 'v (m/s)', 'w (m/s)', 'lpt (K)', 'q (kg/kg)']

out = open(writeDir + '/' + writeName,"w")
out.write("# dara obtained from " + readDir + '/' + readName + "\n")
out.write("#" + ('{:>18}'*len(itemList)).format(*itemList) + "\n")          
for i in [0,1]:
    if i == 0:
        time = 0
    else:
        time = 999999999
    out.write("# " + str(np.round(time,2)) + "\n")            
    for zInd in range(Z.shape[1]):
        tmp = []
        tmp += [str(np.round(Z[tInd, zInd],4))]
        tmp += [str(np.round(tau,2))]
        tmp += [str(np.round(U[tInd, zInd],6))]
        tmp += [str(np.round(V[tInd, zInd],6))]
        tmp += [str(np.round(W,6))]
        tmp += [str(np.round(THETA[tInd, zInd],6))]
        tmp += [str(np.round(Q[tInd, zInd],9))]
        out.write(" " + ('{:>18}'*len(tmp)).format(*tmp) + "\n")
    out.write("\n")
out.close()


### write nudging data (time-dependent)

writeDir = '/scratch/palmdata/JOBS/EERASP3_ndg/INPUT'
writeName = 'EERASP3_ndg_nudge'

tIndList = [6]
tau = 21600.0

timeSkip = 21600.0

W = 0.0

itemList = ['zu (m)', 'tau (s)', 'u (m/s)', 'v (m/s)', 'w (m/s)', 'lpt (K)', 'q (kg/kg)']

out = open(writeDir + '/' + writeName,"w")
out.write("# dara obtained from " + readDir + '/' + readName + "\n")
out.write("#" + ('{:>18}'*len(itemList)).format(*itemList) + "\n")          
for tInd in tIndList:
    out.write("# " + str(np.round(time[tInd]-timeSkip,2)) + "\n")            
    for zInd in range(Z.shape[1]):
        tmp = []
        tmp += [str(np.round(Z[tInd, zInd],4))]
        tmp += [str(np.round(tau,2))]
        tmp += [str(np.round(U[tInd, zInd],6))]
        tmp += [str(np.round(V[tInd, zInd],6))]
        tmp += [str(np.round(W,6))]
        tmp += [str(np.round(THETA[tInd, zInd],6))]
        tmp += [str(np.round(Q[tInd, zInd],9))]
        out.write(" " + ('{:>18}'*len(tmp)).format(*tmp) + "\n")
    out.write("\n")
out.close()