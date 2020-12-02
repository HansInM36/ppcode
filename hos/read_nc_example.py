from netCDF4 import Dataset
import numpy as np

prjDir = "/scratch/HOSdata/JOBS/"
prjName = "regular"
jobName = "ka01"
suffix = ""


readDir = '/scratch/HOSdata/pp/' + prjName + '/' + jobName + '/data/'
readName = "2Ddata_Dict" + suffix + '.nc'
data = Dataset(readDir + readName, "r", format="NETCDF4")

tSeq = np.array(data.variables['time'][:]).astype(float)
ySeq = np.array(data.variables['y'][:]).astype(float)
xSeq = np.array(data.variables['x'][:]).astype(float)

etaSeq = np.array(data.variables['eta'][:,:,:]).astype(float)
phiSeq = np.array(data.variables['phi'][:,:,:]).astype(float)
