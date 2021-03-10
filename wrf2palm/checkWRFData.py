import os
import sys
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
from pyproj import Proj, transform
import matplotlib.pyplot as plt


# reference coordinate system of PALM and lon-lat coordinate system
proj_palm = "EPSG:32633"
proj_wgs84 = 'EPSG:4326'
inproj = Proj('+init='+proj_palm)
lonlatproj = Proj('+init='+proj_wgs84)



### test file
readDir = '/scratch/palmdata/JOBS/WRFPALM_20150701/WRF/WRFoutput'
readName = "wrfout_d01_2015-07-01_00:00:00"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

attr = lambda a: getattr(data, a)
#attr('TRUELAT1')

dimlist = list(data.dimensions)
varlist = list(data.variables)


time = data.variables['Times'][:]
time.tobytes().decode("utf-8")

XLONG = data.variables['XLONG'][:]
XLAT = data.variables['XLAT'][:]


# lon and lat of center point
CENT_LONG = XLONG[:,XLONG.shape[1]//2,XLONG.shape[2]//2].mean()
CENT_LAT = XLAT[:,XLAT.shape[1]//2,XLAT.shape[2]//2].mean()
# palm coordinates of center point
cent_x, cent_y = transform(lonlatproj, inproj, CENT_LONG, CENT_LAT)


HGT = data.variables['HGT'][:]

U = data.variables['U'][:]
V = data.variables['V'][:]
T = data.variables['T'][:]


PH = data.variables['PH'][:]

P_TOP = data.variables['P_TOP'][:]


### example file
readDir = '/scratch/palmdata/JOBS/WRFnesting_new/WRF/WRFoutput'
readName = "wrfout_d01_2015-07-01_00:00:00"

data_ = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

time = data_.variables['Times'][:]

P_TOP = data_.variables['P_TOP'][:]

data.close()
