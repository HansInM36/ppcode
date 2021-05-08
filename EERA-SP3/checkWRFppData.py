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


""" WRF data """
readDir = '/scratch/projects/EERA-SP3/data/WRF'
readName = "WRFOUT_NODA_20150701.nc"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimlist = list(data.dimensions)
varlist = list(data.variables)

time_org = data.variables['time'][:]
dateTime = []
time = []
for tInd in range(time_org.size):
    dateTime.append(datenum_to_datetime(time_org[tInd]))
    time.append(dateTime[tInd].timestamp() - dateTime[0].timestamp())
tplot = np.arange(len(time))*3600

Z = np.array(data.variables['Z'][:])
zplot = Z[0,:31]

U = np.array(data.variables['U'][:])
V = np.array(data.variables['V'][:])
uvplot = np.zeros((len(time),zplot.size))
for tInd in range(len(time)):
    Fu = interp1d(Z[tInd], U[tInd], kind='linear', fill_value='extrapolate')
    Fv = interp1d(Z[tInd], V[tInd], kind='linear', fill_value='extrapolate')
    u = Fu(zplot)
    v = Fv(zplot)
    uvplot[tInd,:] = np.sqrt(np.power(u,2)+np.power(v,2))


THETA = data.variables['THETA'][:]

U10 = data.variables['U10'][:]
V10 = data.variables['V10'][:]




""" evolution of the horizontal wind speed profile """
fig, axs = plt.subplots(1,1, constrained_layout=False)
fig.set_figwidth(8)
fig.set_figheight(6)

### contour
vMin, vMax, vDelta = (0.0, 24.0, 4.0)
cbreso = 50 # resolution of colorbar
levels = np.linspace(vMin, vMax, cbreso + 1)

CS = axs.contourf(tplot/3600, zplot, np.transpose(uvplot), cbreso, levels=levels, cmap='jet', vmin=vMin, vmax=vMax)
axs.set_xlim([0,312])
axs.set_ylim([0,1600])
axs.set_ylabel('z (m)', fontsize=12)
axs.set_xlabel('t (h)', fontsize=12)

cbartickList = np.linspace(vMin, vMax, int((vMax-vMin)/vDelta)+1)
cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
cbar.ax.set_ylabel(r"$\mathrm{uv}$" + ' (m/s)', fontsize=12)

plt.title('evolution of the horizontal wind speed profile')
saveName = 'velo_pr_evo.png'
saveDir = '/scratch/projects/EERA-SP3/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()


""" evolution of the horizontal wind speed at 98.6m height """
fig, axs = plt.subplots(1,1, constrained_layout=False)
fig.set_figwidth(8)
fig.set_figheight(6)
plt.plot(tplot/3600, uvplot[:,10], 'k')
plt.xlim([0,312])
plt.ylim([0.0,20.0])
plt.xlabel('t (h)')
plt.ylabel('uv (m/s)')
plt.grid()
plt.title('evolution of the horizontal wind speed at 98.6m height')
saveName = 'velo_h98.6_evo.png'
saveDir = '/scratch/projects/EERA-SP3/photo'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()
plt.close()