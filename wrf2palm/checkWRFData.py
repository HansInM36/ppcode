import os
import sys
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt

### test file
readDir = '/scratch/palmdata/JOBS/WRFnesting_new/WRF/WRFoutput/org'
readName = "wrfout_d01_2015-07-01_00:00:00"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

attr = lambda a: getattr(data, a)
attr('TRUELAT1')

dimlist = list(data.dimensions)
varlist = list(data.variables)

time = data.variables['Times'][:]
time.tobytes().decode("utf-8")

U = data.variables['U'][:]
V = data.variables['V'][:]
W = data.variables['W'][:]

PH = data.variables['PH'][:]

P_TOP = data.variables['P_TOP'][:]


### example file
readDir = '/scratch/palmdata/JOBS/WRFnesting_new/WRF/WRFoutput'
readName = "wrfout_d01_2015-07-01_00:00:00"

data_ = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

time = data_.variables['Times'][:]

P_TOP = data_.variables['P_TOP'][:]

data.close()
