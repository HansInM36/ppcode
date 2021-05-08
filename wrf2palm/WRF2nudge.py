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

# reference coordinate system of PALM and lon-lat coordinate system
proj_palm = "EPSG:32633"
proj_wgs84 = 'EPSG:4326'
inproj = Proj('+init='+proj_palm)
lonlatproj = Proj('+init='+proj_wgs84)


""" WRF data """
readDir = '/scratch/projects/EERA-SP3/data/WRF'
readName = "wrfout_d01_2015-07-01_00:00:00"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimlist = list(data.dimensions)
varlist = list(data.variables)

time = data.variables['Times'][:6]
time.tobytes().decode("utf-8")

XLONG = data.variables['XLONG'][0]
XLAT = data.variables['XLAT'][0]
X, Y = funcs.lonlat2cts(XLONG, XLAT) # this is for scalar
# interpolate to u position for only interval points (without the leftmost and rightmost points)
XU = (x[:,:-1] + x[:,1:]) / 2
YU = (y[:,:-1] + y[:,1:]) / 2
# interpolate to v position for only interval points (without the southmost and northmost points)
XV = (x[:-1,:] + x[1:,:]) / 2
YV = (y[:-1,:] + y[1:,:]) / 2

Z = list(np.arange(0,25,10)) + [29,33] + list(np.arange(35,55,8)) + list(np.arange(60,250,15)) + \
    list(np.arange(300,400,50)) + list(np.arange(450,1000,100)) + list(np.arange(1300,3500,500)) + \
    list(np.arange(4000,18000,1500))
Z = np.array(Z)

#U = data.variables['U'][0:13,:,:,1:-1]
#V = data.variables['V'][0:13,:,1:-1,:]
#TH2 = data.variables['TH2'][0:13,:,:]
#THETA = data.variables['T'][0:13,:,:,:] + 300 # potential temperature
#Q = data.variables['QVAPOR'][0:13,:,:,:] + data.variables['QCLOUD'][0:13,:,:,:] + \
#    data.variables['QRAIN'][0:13,:,:,:] + data.variables['QICE'][0:13,:,:,:] + \
#    data.variables['QSNOW'][0:13,:,:,:] + data.variables['QGRAUP'][0:13,:,:,:]

U = data.variables['U'][126:139,:,:,1:-1]
V = data.variables['V'][126:139,:,1:-1,:]
THETA = data.variables['T'][126:139,:,:,:] + 300 # potential temperature
Q = data.variables['QVAPOR'][126:139,:,:,:] + data.variables['QCLOUD'][126:139,:,:,:] + \
    data.variables['QRAIN'][126:139,:,:,:] + data.variables['QICE'][126:139,:,:,:] + \
    data.variables['QSNOW'][126:139,:,:,:] + data.variables['QGRAUP'][126:139,:,:,:]

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
        d2 = (XV[j,i] - x0)**2 + (YV[j,i] - y0)**2
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


U = U[:,:,Ju,Iu]
V = V[:,:,Jv,Iv]
THETA = THETA[:,:,J,I]
Q = Q[:,:,J,I]



### write nudging data (constant)
writeDir = '/scratch/palmdata/JOBS/EERASP3_2_ndg/INPUT'
writeName = 'EERASP3_2_ndg_nudge'

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
    for zInd in range(Z.size):
        tmp = []
        tmp += [str(np.round(Z[zInd],4))]
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