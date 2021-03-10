import os
import sys
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt

readDir = '/scratch/palmdata/JOBS/WRFPALM_20150701/INPUT'
readName = "WRFPALM_20150701_dynamic"

nx, ny, nz = 256, 256, 96
dx, dy, dz = 10, 10, 10

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimlist = list(data.dimensions)
varlist = list(data.variables)


tSeq = data.variables['time'][:]

zSeq = data.variables['z'][:]
zwSeq = data.variables['zw'][:]


pSeq = data.variables['surface_forcing_surface_pressure'][:]

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

### check mass conservation
zstag_all = np.r_[0., zwSeq, (zwSeq[-1]-zwSeq[-2]) + zwSeq[-1]] # ztop = (zwSeq[-1]-zwSeq[-2])*1.08 + zwSeq[-1]
zwidths = zstag_all[1:] - zstag_all[:-1]
areas_xb = np.zeros((zSeq.size, 1))
areas_xb[:,0] = zwidths * dy
areas_yb = np.zeros((zSeq.size, 1))
areas_yb[:,0] = zwidths * dx
areas_zb = dx*dy

for tInd in range(tSeq.size):  
    flux_left = np.sum(ls_forcing_left_u[tInd,:,:] * areas_xb)
    flux_right = - np.sum(ls_forcing_right_u[tInd,:,:] * areas_xb)
    flux_south = np.sum(ls_forcing_south_v[tInd,:,:] * areas_yb)
    flux_north = - np.sum(ls_forcing_north_v[tInd,:,:] * areas_yb)
    flux_top = - np.sum(ls_forcing_top_w[tInd,:,:] * areas_zb)
    mass_err = flux_left + flux_right + flux_south + flux_north + flux_top
    print(mass_err)

