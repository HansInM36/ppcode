import os
import sys
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt

### test file
readDir = '/scratch/palmdata/JOBS/example_cbl/OUTPUT'
readName = "example_cbl_masked_M02.000.nc"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

attr = lambda a: getattr(data, a)
#attr('TRUELAT1')

dimlist = list(data.dimensions)
varlist = list(data.variables)


u2 = data.variables['u2'][:]

data.close()

