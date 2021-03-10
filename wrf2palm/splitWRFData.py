""" This script read the original WRFoutput file and split it into individual one-hour data files  """

import os
import sys
from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np

""" input the directory and name of the original WRF output """
readDir = '/scratch/wrfdata/20150701'
readName = "wrfout_d01_2015-07-01_00:00:00"

""" input the directory for splited WRF data """
writeDir = '/scratch/palmdata/JOBS/WRFPALM_20150701/WRF/WRFoutput'


""" input the start and end time (in hour) """
t_start = 0
t_end = 12

org = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimlist = list(org.dimensions)
varlist = list(org.variables)

times = org['Times']

for tInd in range(t_start,t_end+1): # the number of time steps that will be copied, in total tN
    t_wrf = org['Times'][tInd]
    t_str = t_wrf.tobytes().decode("utf-8")
    writeName = 'wrfout_d01_'+t_str    
    
    out = Dataset(writeDir + '/' + writeName, "w", format="NETCDF4" )
    
    for ncattr in org.ncattrs():
        out.setncattr(ncattr, org.getncattr(ncattr))
    
    # copy dimensions
    for dim in dimlist:
        if dim == 'Time':
            out.createDimension(dim, 1)
        else:
            out.createDimension(dim, len(org.dimensions[dim]))
    
    # copy variables of certain time step
    for var in varlist:
        vardims = org[var].dimensions
        if var == 'Times':
            var_ = out.createVariable(var, '|S1', vardims)
            for ncattr in org.variables[var].ncattrs():
                var_.setncattr(ncattr, org.variables[var].getncattr(ncattr))
            tmp = org[var][tInd]
            tmp = tmp[np.newaxis,:]
            var_[:] = np.copy(tmp)
        else:
            var_ = out.createVariable(var, 'f4', vardims)
            for ncattr in org.variables[var].ncattrs():
                var_.setncattr(ncattr, org.variables[var].getncattr(ncattr))
            if vardims[0] == 'Time':
                tmp = org[var][tInd]
                try:
                    tmp = tmp[np.newaxis,:]
                except:
                    pass
                var_[:] = np.copy(tmp)
            else:
                var_[:] = org[var][:]
                
    out.close()
org.close()