import os
import sys
sys.path.append('/scratch/ppcode')
import numpy as np
from netCDF4 import Dataset
import datetime
from scipy.interpolate import interp1d
from scipy.stats import kurtosis
from scipy.stats import skew
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

def pr_palm(dir, jobName, run_no_list, var):
    """ extract horizontal average of velocity at various times and heights """
    run_num = len(run_no_list)

    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    varSeq_list = []
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_pr" + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        varSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))

    # dimensions = list(nc_file_list[0].dimensions)
    # vars = list(nc_file_list[0].variables)
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable

    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zNum = zSeq.size
    zSeq = zSeq.astype(float)
    # concatenate arraies of all cycle_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0)
    tSeq = tSeq.astype(float)
    tNum = tSeq.size
    varSeq = np.concatenate([varSeq_list[i] for i in range(run_num)], axis=0)
    varSeq = varSeq.astype(float)

    return tSeq, zSeq, varSeq

""" WRF data """
readDir = '/scratch/projects/EERA-SP3'
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

Z = data.variables['Z'][:]
U = data.variables['U'][:]
V = data.variables['V'][:]
THETA = data.variables['THETA'][:] # it seems that THETA is not potential temperature

UV = np.sqrt(np.power(U,2) + np.power(V,2))
WD = 270 - np.arctan(V / (U + 1e-6)) * 180/np.pi

""" PALM data """
jobName = 'WRFPALM_20150701'
dir = '/scratch/palmdata/JOBS/' + jobName
tSeq, zSeq, uSeq = pr_palm(dir, jobName, ['.000','.001','.002','.003','.004','.005','.006','.007'], 'u')
tSeq, zSeq, vSeq = pr_palm(dir, jobName, ['.000','.001','.002','.003','.004','.005','.006','.007'], 'v')
tSeq, zSeq, thetaSeq = pr_palm(dir, jobName, ['.000','.001','.002','.003','.004','.005','.006','.007'], 'theta')


""" histogram """
tMax = 13
zMax = 30
z_seq = Z[0][:zMax]

for zInd in range(zMax):
    u_wrf = U[:tMax,zInd]
    f = interp1d(zSeq, uSeq, axis=1, kind='linear', fill_value='extrapolate')
    u_palm = f(z_seq[zInd])

    varMin = np.min((u_wrf.min(), u_palm.min()))
    varMax = np.max((u_wrf.max(), u_palm.max()))
    
    binMin = np.floor(varMin)
    binMax = np.ceil(varMax)
    binNum = 7
    binWidth = (binMax - binMin) / binNum
    bins = list(np.linspace(binMin, binMax, binNum))
    
    #counts_wrf, bins_wrf = np.histogram(u_wrf, bins=binNum, range=(binMin,binMax), density=False)
    #counts_palm, bins_palm = np.histogram(u_palm, bins=binNum, range=(binMin,binMax), density=False)

    
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.hist(u_wrf, bins=bins, weights=np.ones(len(u_wrf))/len(u_wrf)*100, color='b', alpha=0.6, label='WRF')
    ax.hist(u_palm, bins=bins, weights=np.ones(len(u_palm))/len(u_palm)*100, color='r', alpha=0.6, label='PALM')
    plt.xlabel("u (m/s)", fontsize=12)
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0, fontsize=12)
    saveName = 'hist_H' + str(np.round(z_seq[zInd],1)) + '.png'
    saveDir = '/scratch/projects/EERA-SP3/photo/hist'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.close()


