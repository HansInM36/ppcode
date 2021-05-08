import os
import sys
sys.path.append('/scratch/ppcode')
import datetime
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import scipy.signal
import funcs
import matplotlib.pyplot as plt


""" WRF data """
readDir = '/scratch/projects/EERA-SP3/data/WRF'
readName = "wrfout_d01_2015-07-01_00:00:00"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimlist = list(data.dimensions)
varlist = list(data.variables)

time = data.variables['Times'][:]
time.tobytes().decode("utf-8")

XLONG = data.variables['XLONG'][0]
XLAT = data.variables['XLAT'][0]
x, y = funcs.lonlat2cts(XLONG, XLAT) # this is for scalar
# interpolate to u position for only interval points (without the leftmost and rightmost points)
xu = (x[:,:-1] + x[:,1:]) / 2
yu = (y[:,:-1] + y[:,1:]) / 2
# interpolate to v position for only interval points (without the southmost and northmost points)
xv = (x[:-1,:] + x[1:,:]) / 2
yv = (y[:-1,:] + y[1:,:]) / 2

z = list(np.arange(0,25,10)) + [29,33] + list(np.arange(35,55,8)) + list(np.arange(60,250,15)) + \
    list(np.arange(300,400,50)) + list(np.arange(450,1000,100)) + list(np.arange(1300,3500,500)) + \
    list(np.arange(4000,18000,1500))
z = np.array(z)

U = data.variables['U'][:,:,:,1:-1]
V = data.variables['V'][:,:,1:-1,:]
T = data.variables['T'][:,:,:,:] + 300


""" FINO1 data """
### FINO1 site coordinates
lon = funcs.hms2std(6,35,15.5)
lat = funcs.hms2std(54,0,53.5)
x0, y0 = funcs.lonlat2cts(lon, lat)

### find the closest point to FINO1
Ju,Iu = 0, 0
Jv,Iv = 0, 0
J,I = 0, 0

# u position
Min = 1e10
for j in range(U.shape[2]):
    for i in range(U.shape[3]):
        d2 = (xu[j,i] - x0)**2 + (yu[j,i] - y0)**2
        if d2 < Min:
            Min = d2
            Ju, Iu = j, i # J, I is the index of the point closest to the FINO1

# v position
Min = 1e10
for j in range(V.shape[2]):
    for i in range(V.shape[3]):
        d2 = (xv[j,i] - x0)**2 + (yv[j,i] - y0)**2
        if d2 < Min:
            Min = d2
            Jv, Iv = j, i # J, I is the index of the point closest to the FINO1

# scalar position
Min = 1e10
for j in range(T.shape[2]):
    for i in range(T.shape[3]):
        d2 = (x[j,i] - x0)**2 + (y[j,i] - y0)**2
        if d2 < Min:
            Min = d2
            J, I = j, i # J, I is the index of the point closest to the FINO1


U_wrf = U[:,:,Ju,Iu]
V_wrf = V[:,:,Jv,Iv]
T_wrf = T[:,:,J,I]
UV_wrf = np.sqrt(np.power(U_wrf,2) + np.power(V_wrf,2))
WD_wrf = funcs.wd(U_wrf,V_wrf)


""" profile of horizontal velocity """
for tInd in range(0,time.shape[0],3):
    fig, ax = plt.subplots(figsize=(4.5,6))
    plt.plot(UV_wrf[tInd], z, label='WRF', marker='', linestyle='-', linewidth=1.0, color='k')
    plt.xlabel(r"$\mathrm{\overline{u}_h}$ (m/s)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    xaxis_min = 0.0
    xaxis_max = 20.0
    xaxis_d = 2.0
    yaxis_min = 0.0
    yaxis_max = 1200.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.36,0.88), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(tInd) + 'h')
    fig.tight_layout() # adjust the layout
    saveName = 'velo_pr_' + str(tInd) + 'hr' + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/wrf_velo_pr'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()


""" profile of horizontal wind direction """
for tInd in range(0,time.shape[0],3):
    fig, ax = plt.subplots(figsize=(4.5,6))
    plt.plot(WD_wrf[tInd][1:], z[1:], label='WRF', marker='', linestyle='-', linewidth=1.0, color='k')
    plt.xlabel(r"wind direction ($\degree$)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    xaxis_min = 0.0
    xaxis_max = 360.0
    xaxis_d = 60.0
    yaxis_min = 0
    yaxis_max = 1200.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.36,0.88), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(tInd) + 'h')
    fig.tight_layout() # adjust the layout
    saveName = 'wd_pr_' + str(tInd) + 'hr' + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/wrf_wd_pr'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()


""" profile of theta """
for tInd in range(0,time.shape[0],3):
    fig, ax = plt.subplots(figsize=(3.0,4.5))
    plt.plot(T_wrf[tInd], z, label='WRF', linewidth=1.0, linestyle='-', color='k')
    plt.xlabel(r"$\mathrm{\overline{\theta}}$ (K)", fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    xaxis_min = 270.0
    xaxis_max = 320.0
    xaxis_d = 10.0
    yaxis_min = 0.0
    yaxis_max = 1200.0
    yaxis_d = 100.0
    plt.ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
    plt.xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
    plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)), fontsize=12)
    plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)), fontsize=12)
    plt.legend(bbox_to_anchor=(0.36,0.88), loc=6, borderaxespad=0, fontsize=12) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
    plt.grid()
    plt.title('t = ' + str(tInd) + 'h')
    fig.tight_layout() # adjust the layout
    saveName = 'theta_pr_' + str(tInd) + 'hr' + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/wrf_theta_pr'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close()