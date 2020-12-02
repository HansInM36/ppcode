from netCDF4 import Dataset
import numpy as np



ncfile = Dataset('/scratch/ppcode/test/new.nc',mode='w',format='NETCDF4_CLASSIC')
print(ncfile)

lat_dim = ncfile.createDimension('lat', 73)     # latitude axis
lon_dim = ncfile.createDimension('lon', 144)    # longitude axis
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
for dim in ncfile.dimensions.items():
    print(dim)

ncfile.title='My model data'
print(ncfile.title)

ncfile.subtitle="My model data subtitle"
print(ncfile.subtitle)
print(ncfile)

# Define variables with the same names as dimensions,
# a conventional way to define "coordinate variables".
lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'hours since 1800-01-01'
time.long_name = 'time'

# Define a 3D variable to hold the data
temp = ncfile.createVariable('temp',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
temp.units = 'K' # degrees Kelvin
temp.standard_name = 'air_temperature' # this is a CF standard name
print(temp)

nlats = len(lat_dim); nlons = len(lon_dim); ntimes = 3
# Write latitudes, longitudes.
# Note: the ":" is necessary in these "write" statements
lat[:] = -90. + (180./nlats)*np.arange(nlats) # south pole to north pole
lon[:] = (180./nlats)*np.arange(nlons) # Greenwich meridian eastward
# create a 3D array of random numbers
data_arr = np.random.uniform(low=280,high=330,size=(ntimes,nlats,nlons))
# Write the data.  This writes the whole 3D netCDF variable all at once.
temp[:,:,:] = data_arr  # Appends data along unlimited dimension
print("-- Wrote data, temp.shape is now ", temp.shape)
# read data back from variable (by slicing it), print min and max
print("-- Min/Max values:", temp[:,:,:].min(), temp[:,:,:].max())

# first print the Dataset object to see what we've got
print(ncfile)
# close the Dataset.
ncfile.close(); print('Dataset is closed!')



data = Dataset('/scratch/ppcode/test/new.nc', "r", format="NETCDF4")
tmp = np.array(data.variables['temp'][:])
