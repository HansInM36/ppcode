import os
import sys
sys.path.append('/scratch/ppcode')
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
from pyproj import Proj, transform
import funcs
import matplotlib.pyplot as plt

P0 = 1e5
THETA0 = 300
rd = 287
rd_cp = 2/7
_ax = np.newaxis

# reference coordinate system of PALM and lon-lat coordinate system
proj_palm = "EPSG:32633"
proj_wgs84 = 'EPSG:4326'
inproj = Proj('+init='+proj_palm)
lonlatproj = Proj('+init='+proj_wgs84)

""" read WRF original data """
readDir = '/scratch/projects/EERA-SP3/data/WRF'
readName = "wrfout_d03_2017-10-14_06:00:00"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

attr = lambda a: getattr(data, a)
#attr('TRUELAT1')

dimlist = list(data.dimensions)
varlist = list(data.variables)


time = data.variables['Times'][:]
time.tobytes().decode("utf-8")

XLONG = data.variables['XLONG'][:] 
XLAT = data.variables['XLAT'][:]
x, y = funcs.lonlat2cts(XLONG, XLAT)

XLONG_U = data.variables['XLONG_U'][:]
XLAT_U = data.variables['XLAT_U'][:]
x_U, y_U = funcs.lonlat2cts(XLONG_U, XLAT_U)

XLONG_V = data.variables['XLONG_V'][:]
XLAT_V = data.variables['XLAT_V'][:]
x_V, y_V = funcs.lonlat2cts(XLONG_V, XLAT_V)

ETAW = data.variables['ZNW'][:] 
ETA = data.variables['ZNU'][:]

# surface dry air colomn
MU = data.variables['MU'][:] + data.variables['MUB'][:]

# geopotential
PH = data.variables['PH'][:]
PHB = data.variables['PHB'][:]
GPW = PH + PHB

# dry air mass in colomn at various levels
DAMCW = MU[:,_ax,:,:] * ETAW[:,:,_ax,_ax]
DAMC = MU[:,_ax,:,:] * ETA[:,:,_ax,_ax]

# compute the absolute temperature
P = data.variables['P'][:] + data.variables['PB'][:] # pressure
THETA = data.variables['T'][:]
T = (THETA + THETA0) * np.power(P0/P, rd_cp)

GP = GPW[:,:-1,:,:] - np.log(DAMC[:,:,:,:]/DAMCW[:,:-1,:,:]) * rd * T

HGT = GP/9.81

# lon and lat of center point
CENT_LONG = XLONG[:,XLONG.shape[1]//2,XLONG.shape[2]//2].mean()
CENT_LAT = XLAT[:,XLAT.shape[1]//2,XLAT.shape[2]//2].mean()
# palm coordinates of center point
cent_x, cent_y = transform(lonlatproj, inproj, CENT_LONG, CENT_LAT)


HGT = data.variables['HGT'][:]

U = data.variables['U'][:]
V = data.variables['V'][:]


U10 = data.variables['U10'][:]

PH = data.variables['PH'][:]
PHB = data.variables['PHB'][:]


P_TOP = data.variables['P_TOP'][:]



""" read dynamic data """
readDir = '/scratch/palmdata/JOBS/EERASP3_1/INPUT'
readName = "EERASP3_1_dynamic.org"

nx, ny, nz = 384, 384, 56
dx, dy, dz = 40, 40, 5

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimlist = list(data.dimensions)
varlist = list(data.variables)

tSeq = data.variables['time'][:]

xuSeq = data.variables['xu'][:]

ySeq = data.variables['y'][:]

zSeq = data.variables['z'][:]
zwSeq = data.variables['zw'][:]


pSeq = data.variables['surface_forcing_surface_pressure'][:]

init_pt = np.array(data.variables['init_atmosphere_pt'])
init_qv = np.array(data.variables['init_atmosphere_qv'])

init_u = np.array(data.variables['init_atmosphere_u'])
init_v = np.array(data.variables['init_atmosphere_v'])
init_w = np.array(data.variables['init_atmosphere_w'])


""" compare WRF org data and dynamic data """
fig, axs = plt.subplots(1,1, constrained_layout=False)
fig.set_figwidth(8)
fig.set_figheight(6)

### contour
vMin, vMax, vDelta = (8.5, 9.0, 0.1)
cbreso = 100 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

CS = axs.contourf(xuSeq, ySeq, init_u[0], cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs.set_xlim([0,15360])
axs.set_ylim([0,15360])
axs.set_ylabel('y (m)', fontsize=12)
axs.set_xlabel('x (m)', fontsize=12)
axs.text(1.02, 0.5, 'PALM', transform=axs.transAxes, rotation=90, fontsize=12)

cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.ax.set_ylabel(r"$\mathrm{u}$" + ' (m/s)', fontsize=12)

#    saveName = 't' + str(np.round(time[tInd],1)) + '.png'
#    saveDir = '/scratch/projects/EERA-SP3/photo/cmp_bc'
#    if not os.path.exists(saveDir):
#        os.makedirs(saveDir)
#    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()