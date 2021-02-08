import os
import sys
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt

readDir = '/home/xni001/palm/current_version/trunk/UTIL/Bergen'
readName = "Bergen_dynamic_driver_d10_test.nc"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")
time = data.variables['time'][:]

init_atmosphere_pt = np.array(data.variables['init_atmosphere_pt'])
init_atmosphere_qv = np.array(data.variables['init_atmosphere_qv'])

init_atmosphere_u = np.array(data.variables['init_atmosphere_u'])
init_atmosphere_v = np.array(data.variables['init_atmosphere_v'])
init_atmosphere_w = np.array(data.variables['init_atmosphere_w'])

ls_forcing_ug = np.array(data.variables['ls_forcing_ug'])
ls_forcing_vg = np.array(data.variables['ls_forcing_vg'])

ls_forcing_left_pt = np.array(data.variables['ls_forcing_left_pt'])
ls_forcing_right_pt = np.array(data.variables['ls_forcing_right_pt'])
ls_forcing_south_pt = np.array(data.variables['ls_forcing_south_pt'])
ls_forcing_north_pt = np.array(data.variables['ls_forcing_north_pt'])
ls_forcing_top_pt = np.array(data.variables['ls_forcing_top_pt'])

ls_forcing_left_qv = np.array(data.variables['ls_forcing_left_qv'])
ls_forcing_right_qv = np.array(data.variables['ls_forcing_right_qv'])
ls_forcing_south_qv = np.array(data.variables['ls_forcing_south_qv'])
ls_forcing_north_qv = np.array(data.variables['ls_forcing_north_qv'])
ls_forcing_top_qv = np.array(data.variables['ls_forcing_top_qv'])

ls_forcing_left_u = np.array(data.variables['ls_forcing_left_u'])
ls_forcing_right_u = np.array(data.variables['ls_forcing_right_u'])
ls_forcing_south_u = np.array(data.variables['ls_forcing_south_u'])
ls_forcing_north_u = np.array(data.variables['ls_forcing_north_u'])
ls_forcing_top_u = np.array(data.variables['ls_forcing_top_u'])

ls_forcing_left_v = np.array(data.variables['ls_forcing_left_v'])
ls_forcing_right_v = np.array(data.variables['ls_forcing_right_v'])
ls_forcing_south_v = np.array(data.variables['ls_forcing_south_v'])
ls_forcing_north_v = np.array(data.variables['ls_forcing_north_v'])
ls_forcing_top_v = np.array(data.variables['ls_forcing_top_v'])

ls_forcing_left_w = np.array(data.variables['ls_forcing_left_w'])
ls_forcing_right_w = np.array(data.variables['ls_forcing_right_w'])
ls_forcing_south_w = np.array(data.variables['ls_forcing_south_w'])
ls_forcing_north_w = np.array(data.variables['ls_forcing_north_w'])
ls_forcing_top_w = np.array(data.variables['ls_forcing_top_w'])

data.close()
