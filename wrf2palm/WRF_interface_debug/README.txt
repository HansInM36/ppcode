Scripts for processing of WRF and CAMx files to PALM dynamic driver.
Version: v.3.1, 20201102

Usage: palm_dynamic -c <config_name> [-w]

The script requires name of the case configuration on the command line.
The corresponding configuration files are placed in subdirectory
"configuration" and they are named <config_name>.conf. Examples of
the config files are supplied with the scripts, all configuration
options with are in the file palm_dynamic_defaults.py as well as
at the end of this readme file. The values which agree with defaults
need not be present in the user config. The file palm_dynamic_init.py
contains setting and calculation of standard initialization values
for particular system and can be adjusted. The optional parameter -w
allows to skip horizontal and vertical interpolation in case it is
already done.

The scripts were implemented with regards to good portability and they
only depend on standard and well-known Python libraries.
Needed modules are:

- numpy   (https://pypi.org/project/numpy)
- scipy   (https://pypi.org/project/scipy)
- pyproj  (https://pypi.org/project/pyproj)
- netCDF4 (https://pypi.org/project/netCDF4)
- metpy   (https://unidata.github.io/MetPy)

First four modules are included in all major linux distributions. In other cases,
they can be installed via pip (e.g. "pip3 install netCDF4").

The MetPy module is a well-known and widely used project for meteorological
calculations in Python maintained by Unidata. It can also be installed using
pip ("pip3 install MetPy").

In the current version, the only supported projection in WRF is Lambert
conformal conic, which is WRF default and recommended projection for
mid-latitudes. The CAMx output files, which are used optionally for the
chemical initial and boundary conditions, are expected to be the result of
a coupled CAMx simulation using the same horizontal grid as WRF, although this
is not strictly required.

The scripts support both variants of WRF vertical levels - the sigma levels
(default until WRF version 3.*) and the hybrid levels (default since WRF 4.*).
However, it is necessary to correctly configure this option via the setting
"wrf_hybrid_levs = True/False".

CONFIGURATION

Configuration files have syntax:
<parameter1> = <value1>
<parameter2> = <value2>
...
Parameters can be string, integer, float, logical or list of these values.
The <value> terms have Python syntax, the config file needs to contain only
parameters which differ from default.

Description of the particular configuration options are (defaults are in parenthesis):

# 1. Domain and case related config
domain              name of the simulation case ("")
resolution          name of the particular domain resolution scenario ("")
scenario            name of the individual scenario in the case ("")
nested_domain       False indicates parent and True nested domain. (False)
dynamic_driver_file file name of output dynamic driver (""). The default value
                    "" means the standard name assigned in the palm_dynamic_init.
grid_from_static    value True - the grid parameters are imported from the static
                    driver, False - they are prescribed in the config (True)
static_driver_file  file name of the static driver in case of grid_from_static ("").
                    The default value "" means the standard name assigned in init.
proj_palm           reference coordinate system of PALM simulation ("EPSG:32633")
proj_wgs84          reference coordinate system of lon-lat projection ("EPSG:4326")
dz                  height of the PALM vertical grid layer (0.0). The default
                    value dz = 0.0 means dz is assigned from dx.
nz                  number of vertical layers of PALM domain (200)
dz_stretch_level    height in meters from which stretching of vertical levels
                    starts in PALM (5000.0)
dz_stretch_factor   coefficient of the stretching of the vertical layers in PALM (1.0)
dz_max              max height of the stretched vertical layers (100.0)
origin_time         origin time of the PALM simulation in the format
                    YYYY-MM-DD hh:mm:ss (""). The default value "" means that
                    the value is read from the global attribute of the static driver.
simulation_hours    extent of the simulation in hours (24)

# 2. WRF and CAMx related configurations
wrf_hybrid_levs     True means hybrid levels in WRF files, False means sigma levels (True).
vinterp_terrain_smoothing
                    the standard deviation for Gaussian kernel of smoothing method
                    of the PALM terrain for WRF vertical interpolation to avoid sharp
                    horizontal gradients. Value None disables the smoothing. (None)
wrf_dir_name        files path of the wrf and camx input files (""). The default value ""
                    means that the standard path will be calculated in the init.
wrf_file_mask       file mask of the wrf input files  ("wrfout_*.e000")
radiation_from_wrf  enable or disable processing of radiation from WRF files (True).
wrf_rad_file_mask   file mask of the wrf radiation input files ("auxhist6_*").
                    The default setting reads radiation from WRF auxiliary
                    history files. This setting allows to use finer time step for WRF
                    radiation outputs than for other values.
radiation_smoothing_distance
                    smoothing distance for radiation values in m (10000.0).
camx_file_mask      file mask of the CAMx input files ("CAMx.*.nc").
species_names       PALM names of the chemical species (['NO','NO2','O3','PM10','PM25']).
                    The default value can be used for phstatp and phstatp2 chemical
                    mechanisms in PALM. The mapping and recalculation from the CAMx
                    species to PALM species is defined in the variable camx_conversions
                    (see palm_dynamic_init.py). The current mapping is adjusted for
                    CAMx v6.50 with CB05 + PM (CF,SOAP2.1,ISORROPIA) chemistry mechanisms.
                    In the case other combinations, it may be necessary to adjust
                    this mapping.

# 3. horizontal parameters of the PALM domain which have to be set in case
#    of grid_from_static = False
nx, ny              number of horizontal grids of the domain in x and y directions
dx, dy              grid cell size of the domain in x and y directions
origin_x, origin_y  origin x and y of the domain
origin_z            origin of the domain in the vertical direction

