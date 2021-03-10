#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------#
#
# Scripts for processing of WRF and CAMx files to PALM dynamic driver
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2018-2021 Institute of Computer Science
#                     of the Czech Academy of Sciences, Prague
# Authors: Krystof Eben, Jaroslav Resler, Pavel Krc
#
#------------------------------------------------------------------------------#

'''WRF (+CAMx) utility module for PALM dynamic driver generator'''

import os
import math
import numpy as np
import pyproj
import scipy.ndimage as ndimage
import netCDF4

import metpy.calc as mpcalc
from metpy.interpolate import interpolate_1d, log_interpolate_1d
from metpy.units import units

from palm_dynamic_config import *

# Constants directly equivalent to WRF code
radius = 6370000.0
g = 9.81 #m/s2
rd = 287. #dry air gas constant (J/kg/K)
rd_cp = 2./7. #from WRF v4 technote (R_d / c_p)
wrf_base_temp = 300. #NOT wrfout T00

_ax = np.newaxis

class WRFCoordTransform(object):
    'Coordinate transformer for WRFOUT files'

    def __init__(self, ncf):
        attr = lambda a: getattr(ncf, a)

        # Define grids

        # see http://www.pkrc.net/wrf-lambert.html
        #latlon_wgs84 = pyproj.Proj(proj='latlong',
        #    ellps='WGS84', datum='WGS84',
        #    no_defs=True) #don't use - WRF datum misuse

        latlon_sphere = pyproj.Proj(proj='latlong',
            a=radius, b=radius,
            towgs84='0,0,0', no_defs=True)

        lambert_grid = pyproj.Proj(proj='lcc',
            lat_1=attr('TRUELAT1'),
            lat_2=attr('TRUELAT2'),
            lat_0=attr('MOAD_CEN_LAT'),
            lon_0=attr('STAND_LON'),
            a=radius, b=radius,
            towgs84='0,0,0', no_defs=True)

        # resoltion in m
        self.dx = dx = attr('DX')
        self.dy = dy = attr('DY')

        # number of mass grid points
        self.nx = nx = attr('WEST-EAST_GRID_DIMENSION') - 1
        self.ny = ny = attr('SOUTH-NORTH_GRID_DIMENSION') - 1

        # distance between centers of mass grid points at edges
        extent_x = (nx - 1) * dx
        extent_y = (ny - 1) * dy

        # grid center in lambert
        center_x, center_y = pyproj.transform(latlon_sphere, lambert_grid,
            attr('CEN_LON'), attr('CEN_LAT'))

        # grid origin coordinates in lambert
        i0_x = center_x - extent_x*.5
        j0_y = center_y - extent_y*.5

        # Define fast transformation methods

        def latlon_to_ji(lat, lon):
            x, y = pyproj.transform(latlon_sphere, lambert_grid,
                    lon, lat)
            return (y-j0_y)/dy, (x-i0_x)/dx
        self.latlon_to_ji = latlon_to_ji

        def ji_to_latlon(j, i):
            lon, lat = pyproj.transform(lambert_grid, latlon_sphere,
                i*dx+i0_x, j*dy+j0_y)
            return lat, lon
        self.ji_to_latlon = ji_to_latlon

    def verify(self, ncf):
        lat = ncf.variables['XLAT'][0]
        lon = ncf.variables['XLONG'][0]
        j, i = np.mgrid[0:self.ny, 0:self.nx]

        jj, ii = self.latlon_to_ji(lat, lon)
        d = np.hypot(jj-j, ii-i)
        print('error for ll->ji: max {0} m, avg {1} m.'.format(d.max(), d.mean()))

        llat, llon = self.ji_to_latlon(j, i)
        d = np.hypot(llat - lat, llon - lon)
        print('error for ji->ll: max {0} deg, avg {1} deg.'.format(d.max(), d.mean()))

        lat = ncf.variables['XLAT_U'][0]
        lon = ncf.variables['XLONG_U'][0]
        j, i = np.mgrid[0:self.ny, 0:self.nx+1]
        jj, ii = self.latlon_to_ji(lat, lon)
        ii = ii + .5
        d = np.hypot(jj-j, ii-i)
        print('error for U-staggered ll->ji: max {0} m, avg {1} m.'.format(d.max(), d.mean()))

class CAMxCoordTransform(object):
    'Coordinate transformer for CAMx files running from WRF'

    def __init__(self, ncf):
        attr = lambda a: getattr(ncf, a)

        # Define grids

        latlon_sphere = pyproj.Proj(proj='latlong',
            a=radius, b=radius,
            towgs84='0,0,0', no_defs=True)

        lambert_grid = pyproj.Proj(proj='lcc',
            lat_1=attr('P_ALP'),
            lat_2=attr('P_BET'),
            lat_0=attr('YCENT'),
            lon_0=attr('P_GAM'),
            a=radius, b=radius,
            towgs84='0,0,0', no_defs=True)

        # resoltion in m
        self.dx = dx = attr('XCELL')
        self.dy = dy = attr('YCELL')

        # number of mass grid points
        self.nx = nx = attr('NCOLS')
        self.ny = ny = attr('NROWS')

        # grid origin coordinates in lambert
        i0_x = attr('XORIG')
        j0_y = attr('YORIG')

        # Define fast transformation methods

        def latlon_to_ji(lat, lon):
            x, y = pyproj.transform(latlon_sphere, lambert_grid,
                    lon, lat)
            return (y-j0_y)/dy, (x-i0_x)/dx
        self.latlon_to_ji = latlon_to_ji

        def ji_to_latlon(j, i):
            lon, lat = pyproj.transform(lambert_grid, latlon_sphere,
                i*dx+i0_x, j*dy+j0_y)
            return lat, lon
        self.ji_to_latlon = ji_to_latlon

    def verify(self, ncf):
        lat = ncf.variables['latitude'][:]
        lon = ncf.variables['longitude'][:]
        j, i = np.mgrid[0:self.ny, 0:self.nx]

        jj, ii = self.latlon_to_ji(lat, lon)
        d = np.hypot(jj-j, ii-i)
        print('error for ll->ji: max {0} m, avg {1} m.'.format(d.max(), d.mean()))

        llat, llon = self.ji_to_latlon(j, i)
        d = np.hypot(llat - lat, llon - lon)
        print('error for ji->ll: max {0} deg, avg {1} deg.'.format(d.max(), d.mean()))

class BilinearRegridder(object):
    '''Bilinear regridder for multidimensional data.

    By standard, the last two dimensions are always Y,X in that order.
    '''
    def __init__(self, projected_x, projected_y, preloaded=False):
        projected_x = np.asanyarray(projected_x)
        projected_y = np.asanyarray(projected_y)
        self.shape = projected_x.shape
        self.rank = len(self.shape)
        assert self.shape == projected_y.shape

        y0 = np.floor(projected_y)
        yd = projected_y - y0
        ydd = 1. - yd
        self.y0 = y0.astype('i8')

        x0 = np.floor(projected_x)
        xd = projected_x - x0
        xdd = 1. - xd
        self.x0 = x0.astype('i8')

        if preloaded:
            # Prepare slices for preloading from NetCDF files (for cases where
            # the range of loaded Y, X coordinates is much less than total
            # size. The regrid method then expects preloaded data.

            ybase = self.y0.min()
            self.ys = slice(ybase, self.y0.max()+2)
            self.y0 -= ybase

            xbase = self.x0.min()
            self.xs = slice(xbase, self.x0.max()+2)
            self.x0 -= xbase

        self.y1 = self.y0 + 1
        self.x1 = self.x0 + 1

        self.weights = np.array([
            ydd * xdd, #wy0x0
            ydd * xd , #wy0x1
            yd  * xdd, #wy1x0
            yd  * xd , #wy1x1
            ])

    def regrid(self, data):
        # data may contain additional dimensions (before Y,X)
        dshape = data.shape[:-2]
        drank = len(dshape)

        # Prepare array for selected data
        sel_shape = (4,) + dshape + self.shape
        selection = np.empty(sel_shape, dtype=data.dtype)

        selection[0, ...] = data[..., self.y0, self.x0]
        selection[1, ...] = data[..., self.y0, self.x1]
        selection[2, ...] = data[..., self.y1, self.x0]
        selection[3, ...] = data[..., self.y1, self.x1]

        # Slice weights to match the extra dimensions
        wslice = ((slice(None),) +      #weights
            (np.newaxis,) * drank +     #data minus Y,X
            (slice(None),) * self.rank) #regridded shape

        w = selection * self.weights[wslice]
        return w.sum(axis=0)

def print_dstat(desc, delta):
    print('Delta stats for {0} ({1:8g} ~ {2:8g}): bias = {3:8g}, MAE = {4:8g}, RMSE = {5:8g}'.format(
        desc, delta.min(), delta.max(), delta.mean(), np.abs(delta).mean(), np.sqrt((delta**2).mean())))

def barom_pres(p0, gp, gp0, t0):
    barom = 1. / (rd * t0)
    return p0 * np.exp((gp0-gp)*barom)

def barom_gp(gp0, p, p0, t0):
    baromi = rd * t0
    return gp0 - np.log(p/p0) * baromi

def calc_ph_hybrid(f, mu):
    pht = f.variables['P_TOP'][0]
    c3f = f.variables['C3F'][0]
    c4f = f.variables['C4F'][0]
    c3h = f.variables['C3H'][0]
    c4h = f.variables['C4H'][0]
    return (c3f[:,_ax,_ax]*mu[_ax,:,:] + (c4f[:,_ax,_ax] + pht),
            c3h[:,_ax,_ax]*mu[_ax,:,:] + (c4h[:,_ax,_ax] + pht))

def calc_ph_sigma(f, mu):
    pht = f.variables['P_TOP'][0]
    pht += 1e-6 # debug2 !!!!
    eta_f = f.variables['ZNW'][0] # eta values on fall(w) levels
    eta_h = f.variables['ZNU'][0] # eta values on half(mass) levels
    return (eta_f[:,_ax,_ax]*mu[_ax,:,:] + pht,
            eta_h[:,_ax,_ax]*mu[_ax,:,:] + pht)

def wrf_t(f):
    p = f.variables['P'][0,:,:,:] + f.variables['PB'][0,:,:,:] # original # pressure (Pa)
    p[p < 0] = 0 # debug1 !!!!!
    return (f.variables['T'][0,:,:,:] + wrf_base_temp) * np.power(0.00001*p, rd_cp)

def calc_gp(f, ph):
    terr = f.variables['HGT'][0,:,:]
    gp0 = terr * g
    gp = [gp0]
    t = wrf_t(f)
    for lev in range(1, ph.shape[0]):
        gp.append(barom_gp(gp[-1], ph[lev,:,:], ph[lev-1,:,:], t[lev-1,:,:]))
    return np.array(gp)

def palm_wrf_vertical_interp(infile, outfile, wrffile, z_levels, z_levels_stag,
        z_soil_levels, origin_z, terrain, wrf_hybrid_levs, vinterp_terrain_smoothing):

    # !!!!
    # infile, outfile, wrffile, z_levels, z_levels_stag, z_soil_levels, origin_z, terrain, wrf_hybrid_levs, vinterp_terrain_smoothing = hinterp_file, vinterp_file, wrf_file, z_levels, z_levels_stag, z_soil_levels, origin_z, terrain, wrf_hybrid_levs, vinterp_terrain_smoothing

    zdim = len(z_levels)
    zwdim = len(z_levels_stag)
    zsoildim = len(z_soil_levels)
    dimnames = ['z', 'zw', 'zsoil']   # dimnames of palm vertical dims
    dimensions = [zdim , zwdim, zsoildim]

    print("infile: " , infile)
    print("outfile: ", outfile)

    try:
        os.remove(outfile)
        os.remove(infile+'_vinterp.log')
    except:
        pass

    nc_infile = netCDF4.Dataset(infile, 'r')
    nc_wrf = netCDF4.Dataset(wrffile, 'r')
    nc_outfile = netCDF4.Dataset(outfile, "w", format="NETCDF4")
    nc_outfile.createDimension('Time', None)
    for dimname in ['west_east', 'south_north', 'soil_layers_stag']:
        nc_outfile.createDimension(dimname, len(nc_infile.dimensions[dimname]))
    for i in range (len(dimnames)):
        nc_outfile.createDimension(dimnames[i], dimensions[i])

    # Use hybrid ETA levels in WRF and stretch them so that the WRF terrain
    # matches either PALM terrain or flat terrain at requested height
    gpf = nc_infile.variables['PH'][0,:,:,:] + nc_infile.variables['PHB'][0,:,:,:] # geopotential (m^2*s^-2)
    wrfterr = gpf[0]*(1./g)

    if vinterp_terrain_smoothing is None:
        target_terrain = terrain
    else:
        print('Smoothing PALM terrain for the purpose of dynamic driver with sigma={0} grid points.'.format(
            vinterp_terrain_smoothing))
        target_terrain = ndimage.gaussian_filter(terrain, sigma=vinterp_terrain_smoothing, order=0)
    print('Morphing WRF terrain ({0} ~ {1}) to PALM terrain ({2} ~ {3})'.format(
        wrfterr.min(), wrfterr.max(), target_terrain.min(), target_terrain.max()))
    print_dstat('terrain shift', wrfterr - target_terrain[:,:])

    # Load original dry air column pressure
    mu = nc_infile.variables['MUB'][0,:,:] + nc_infile.variables['MU'][0,:,:] # dry air mass in column (Pa)
    pht = nc_wrf.variables['P_TOP'][0] # pressure top (Pa)

    # Shift column pressure so that it matches PALM terrain
    t = wrf_t(nc_infile) # bug1: 'nan' appears!!!!!! (because negative pressures)
    mu2 = barom_pres(mu+pht, target_terrain*g, gpf[0,:,:], t[0,:,:])-pht

    # Calculate original and shifted 3D dry air pressure
    if wrf_hybrid_levs:
        phf, phh = calc_ph_hybrid(nc_wrf, mu)
        phf2, phh2 = calc_ph_hybrid(nc_wrf, mu2)
    else:
        phf, phh = calc_ph_sigma(nc_wrf, mu)
        phf2, phh2 = calc_ph_sigma(nc_wrf, mu2)

    # Shift 3D geopotential according to delta dry air pressure
    tf = np.concatenate((t, t[-1:,:,:]), axis=0) # repeat highest layer
    gpf2 = barom_gp(gpf, phf2, phf, tf) # bug2: divided by zero appears !!!! (there are zeros in phf)
    # For half-levs, originate from gp full levs rather than less accurate gp halving
    gph2 = barom_gp(gpf[:-1,:,:], phh2, phf[:-1,:,:], t)
    zf = gpf2 * (1./g) - origin_z
    zh = gph2 * (1./g) - origin_z

    # Report
    gpdelta = gpf2 - gpf
    print('GP deltas by level:')
    for k in range(gpf.shape[0]):
        print_dstat(k, gpdelta[k])

    # Because we require levels below the lowest level from WRF, we will always
    # add one layer at zero level with repeated values from the lowest level.
    # WRF-python had some special treatment for theta in this case.
    height = np.zeros((zh.shape[0]+1,) + zh.shape[1:], dtype=zh.dtype)
    height[0,:,:] = -999. #always below terrain
    height[1:,:,:] = zh
    heightw = np.zeros((zf.shape[0]+1,) + zf.shape[1:], dtype=zf.dtype)
    heightw[0,:,:] = -999. #always below terrain
    heightw[1:,:,:] = zf

    # ======================== SPECIFIC HUMIDITY ==============================
    qv_raw = nc_infile.variables['SPECHUM'][0]
    qv_raw = np.r_[qv_raw[0:1], qv_raw]

    # Vertical interpolation to grid height levels (specified in km!)
    # Levels start at 50m (below that the interpolation looks very sketchy)
    init_atmosphere_qv = interpolate_1d(z_levels, height, qv_raw)
    vdata = nc_outfile.createVariable('init_atmosphere_qv', "f4", ("Time", "z","south_north","west_east"))
    vdata[0,:,:,:] = init_atmosphere_qv

    # ======================= POTENTIAL TEMPERATURE ==========================
    pt_raw = nc_infile.variables['T'][0] + 300.   # from perturbation pt to standard
    pt_raw = np.r_[pt_raw[0:1], pt_raw]
    init_atmosphere_pt = interpolate_1d(z_levels, height, pt_raw)

    #plt.figure(); plt.contourf(pt[0]) ; plt.colorbar() ; plt.show()
    vdata = nc_outfile.createVariable('init_atmosphere_pt', "f4", ("Time", "z","south_north","west_east"))
    vdata[0,:,:,:] = init_atmosphere_pt

    # ======================= Wind ==========================================
    u_raw = nc_infile.variables['U'][0]
    u_raw = np.r_[u_raw[0:1], u_raw]
    init_atmosphere_u = interpolate_1d(z_levels, height, u_raw)

    vdata = nc_outfile.createVariable('init_atmosphere_u', "f4", ("Time", "z","south_north","west_east"))
    vdata[0,:,:,:] = init_atmosphere_u

    v_raw = nc_infile.variables['V'][0]
    v_raw = np.r_[v_raw[0:1], v_raw]
    init_atmosphere_v = interpolate_1d(z_levels, height, v_raw)

    vdata = nc_outfile.createVariable('init_atmosphere_v', "f4", ("Time", "z","south_north","west_east"))
    #vdata.coordinates = "XLONG_V XLAT_V XTIME"
    vdata[0,:,:,:] = init_atmosphere_v

    w_raw = nc_infile.variables['W'][0]
    w_raw = np.r_[w_raw[0:1], w_raw]
    init_atmosphere_w = interpolate_1d(z_levels_stag, heightw, w_raw)

    vdata = nc_outfile.createVariable('init_atmosphere_w', "f4", ("Time", "zw","south_north","west_east"))
    #vdata.coordinates = "XLONG XLAT XTIME"
    vdata[0,:,:,:] = init_atmosphere_w

    # ===================== SURFACE PRESSURE ==================================
    surface_forcing_surface_pressure = nc_infile.variables['PSFC']
    vdata = nc_outfile.createVariable('surface_forcing_surface_pressure', "f4", ("Time", "south_north","west_east"))
    vdata[0,:,:] = surface_forcing_surface_pressure[0,:,:]

    # ======================== SOIL VARIABLES (without vertical interpolation) =============
    # soil temperature
    init_soil_t = nc_infile.variables['TSLB']
    vdata = nc_outfile.createVariable('init_soil_t', "f4", ("Time", "zsoil","south_north","west_east"))
    vdata[0,:,:,:] = init_soil_t[0,:,:,:]

    # soil moisture
    init_soil_m = nc_infile.variables['SMOIS']
    vdata = nc_outfile.createVariable('init_soil_m', "f4", ("Time","zsoil","south_north","west_east"))
    vdata[0,:,:,:] = init_soil_m[0,:,:,:]

    # zsoil
    zsoil = nc_wrf.variables['ZS']    #ZS:description = "DEPTHS OF CENTERS OF SOIL LAYERS" ;
    vdata = nc_outfile.createVariable('zsoil', "f4", ("zsoil"))
    vdata[:] = zsoil[0,:]

    # coordinates z, zw
    vdata = nc_outfile.createVariable('z', "f4", ("z"))
    vdata[:] = list(z_levels)

    vdata = nc_outfile.createVariable('zw', "f4", ("zw"))
    vdata[:] = list (z_levels_stag)

    # zsoil is taken from wrf - not need to define it

    nc_infile.close()
    nc_wrf.close()
    nc_outfile.close()

def palm_wrf_gw(f, lon, lat, levels, tidx=0):
    '''Calculate geostrophic wind from WRF using metpy'''

    # f, lon, lat, levels, tidx = nc_wrf, lon_center, lat_center, z_levels, 0

    hgts, ug, vg = calcgw_wrf(f, lat, lon, levels, tidx) # bug3, hgts (heights from WRF) cannot cover heights in PALM

    # extrapolate at the bottom
    hgts = np.r_[np.array([0.]), hgts]
    ug = np.r_[ug[0], ug]
    vg = np.r_[vg[0], vg]

    return minterp(levels, hgts, ug, vg) # !!!! hgts cannot cover levels


def minterp(interp_heights, data_heights, u, v):
    '''Interpolate wind using power law for agl levels'''

    # interp_heights, data_heights, u, v = levels, hgts, ug, vg

    pdata = data_heights ** gw_alpha
    pinterp = interp_heights ** gw_alpha
    hindex = np.searchsorted(data_heights, interp_heights, side='right')
    lindex = hindex - 1
    assert lindex[0] >= 0
    assert hindex[-1] < len(data_heights) # means the largest value in interp_heights should be less than that in data_heights
    lbound = pdata[lindex]
    hcoef = (pinterp - lbound) / (pdata[hindex] - lbound)
    #print(data_heights)
    #print(lindex)
    #print(hcoef)
    lcoef = 1. - hcoef
    iu = u[lindex] * lcoef + u[hindex] * hcoef
    iv = v[lindex] * lcoef + v[hindex] * hcoef
    return iu, iv

def get_wrf_dims(f, lat, lon, xlat, xlong):
    '''A crude method, yet satisfactory for approximate WRF surroundings'''

    sqdist = (xlat - lat)**2 + (xlong - lon)**2
    coords = np.unravel_index(sqdist.argmin(), sqdist.shape)

    xmargin = int(math.ceil(gw_wrf_margin_km * 1000 / f.DX)) #py2 ceil produces float
    ymargin = int(math.ceil(gw_wrf_margin_km * 1000 / f.DY))
    y0, y1 = coords[0] - ymargin, coords[0] + ymargin
    x0, x1 = coords[1] - xmargin, coords[1] + xmargin
    assert 0 <= y0 < y1 < sqdist.shape[0], "Point {0} + surroundings not inside domain".format(coords[0])
    assert 0 <= x0 < x1 < sqdist.shape[1], "Point {0} + surroundings not inside domain".format(coords[1])

    return coords, (slice(y0, y1+1), slice(x0, x1+1)), (ymargin, xmargin)

def calcgw_wrf(f, lat, lon, levels, tidx=0):

    # f, lat, lon, levels, tidx = nc_wrf, lat_center, lon_center, z_levels, 0

    # MFDataset removes the time dimension from XLAT, XLONG
    xlat = f.variables['XLAT']
    xlslice = (0,) * (len(xlat.shape)-2) + (slice(None), slice(None))
    xlat = xlat[xlslice]
    xlong = f.variables['XLONG'][xlslice]

    (iy, ix), area, (iby, ibx) = get_wrf_dims(f, lat, lon, xlat, xlong)
    areat = (tidx,) + area
    areatz = (tidx, slice(None)) + area
    #print('wrf coords', lat, lon, xlat[iy,ix], xlong[iy,ix])
    #print(xlat[area][iby,ibx], xlong[area][iby,ibx], areat)

    # load area
    hgt = (f.variables['PH'][areatz] + f.variables['PHB'][areatz]) / 9.81
    hgtu = (hgt[:-1] + hgt[1:]) * .5
    pres = f.variables['P'][areatz] + f.variables['PB'][areatz] # bug3 pres has steep change at the 5th level (corresponding height is hgt[5])
    terrain = f.variables['HGT'][areat]

    # find suitable pressure levels
    yminpres, xminpres = np.unravel_index(pres[0].argmin(), pres[0].shape)
    pres1 = pres[0, yminpres, xminpres] - 1.

    aglpt = hgtu[:,iby,ibx] - terrain[iby,ibx]
    pres0 = pres[np.searchsorted(aglpt, levels[-1]), iby, ibx]
    plevels = np.arange(pres1, min(pres0, pres1)-1, -1000.)

    # interpolate wrf into pressure levels
    pres[pres<=0] = 1e-6 # debug3
    phgt = log_interpolate_1d(plevels, pres, hgtu, axis=0) # bug3 (negative values in pres results in 'nan')

    # Set up some constants based on our projection, including the Coriolis parameter and
    # grid spacing, converting lon/lat spacing to Cartesian
    coriol = mpcalc.coriolis_parameter(np.deg2rad(xlat[area])).to('1/s')

    # lat_lon_grid_deltas doesn't work under py2, but for WRF grid it is still
    # not very accurate, better use direct values.
    #dx, dy = mpcalc.lat_lon_grid_deltas(xlong[area], xlat[area])
    dx = f.DX * units.m
    dy = f.DY * units.m

    # Smooth height data. Sigma=1.5 for gfs 0.5deg
    res_km = f.DX / 1000.

    ug = np.zeros(plevels.shape, 'f8')
    vg = np.zeros(plevels.shape, 'f8')
    for i in range(len(plevels)):
        sh = ndimage.gaussian_filter(phgt[i,:,:], sigma=1.5*50/res_km, order=0)
        # ugl, vgl = mpcalc.geostrophic_wind(sh * units.m, coriol, dx, dy) # bug4 (`dx` requires "[length]" but given "1 / second", `latitude` requires "[dimensionless]" but given "meter")
        ugl, vgl = mpcalc.geostrophic_wind(sh * units.m, dx, dy, xlat[area]*np.pi/180) # debug4
        ug[i] = ugl[iby, ibx].magnitude # ; print(ug[i])
        vg[i] = vgl[iby, ibx].magnitude # ; print(vg[i])

    return phgt[:,iby,ibx], ug, vg

# The following two functions calculate GW from GFS files, although this
# function is currently not implemented in PALM dynamic driver generation
# script

def calcgw_gfs(v, lat, lon):
    height, lats, lons = v.data(lat1=lat-gw_gfs_margin_deg ,lat2=lat+gw_gfs_margin_deg,
            lon1=lon-gw_gfs_margin_deg, lon2=lon+gw_gfs_margin_deg)
    i = np.searchsorted(lats[:,0], lat)
    if abs(lats[i+1,0] - lat) < abs(lats[i,0] - lat):
        i = i+1
    j = np.searchsorted(lons[0,:], lon)
    if abs(lons[0,i+1] - lon) < abs(lons[0,i] - lon):
        j = j+1
    #print('level', v.level, 'height', height[i,j], lats[i,j], lons[i,j])

    # Set up some constants based on our projection, including the Coriolis parameter and
    # grid spacing, converting lon/lat spacing to Cartesian
    f = mpcalc.coriolis_parameter(np.deg2rad(lats)).to('1/s')
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
    res_km = (dx[i,j]+dy[i,j]).magnitude / 2000.

    # Smooth height data. Sigma=1.5 for gfs 0.5deg
    height = ndimage.gaussian_filter(height, sigma=1.5*50/res_km, order=0)

    # In MetPy 0.5, geostrophic_wind() assumes the order of the dimensions is (X, Y),
    # so we need to transpose from the input data, which are ordered lat (y), lon (x).
    # Once we get the components,transpose again so they match our original data.
    geo_wind_u, geo_wind_v = mpcalc.geostrophic_wind(height * units.m, f, dx, dy)

    return height[i,j], geo_wind_u[i,j], geo_wind_v[i,j]

def combinegw_gfs(grbs, levels, lat, lon):
    heights = []
    us = []
    vs = []
    for grb in grbs:
        h, u, v = calcgw_gfs(grb, lat, lon)
        heights.append(h)
        us.append(u.magnitude)
        vs.append(v.magnitude)
    heights = np.array(heights)
    us = np.array(us)
    vs = np.array(vs)

    ug, vg = minterp(np.asanyarray(levels), heights[::-1], us[::-1], vs[::-1])
    return ug, vg

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-w', '--wrfout', help='verify wrfout file')
    parser.add_argument('-c', '--camx', help='verify camx file')
    args = parser.parse_args()

    if args.wrfout:
        f = netCDF4.Dataset(args.wrfout)

        print('Verifying coord transform:')
        t = WRFCoordTransform(f)
        t.verify(f)

        print('\nVerifying vertical levels:')
        mu = f.variables['MUB'][0,:,:] + f.variables['MU'][0,:,:]
        gp = f.variables['PH'][0,:,:,:] + f.variables['PHB'][0,:,:,:]

        print('\nUsing sigma:')
        phf, phh = calc_ph_sigma(f, mu)
        gp_calc = calc_gp(f, phf)
        delta = gp_calc - gp
        for lev in range(delta.shape[0]):
            print_dstat(lev, delta[lev])

        print('\nUsing hybrid:')
        phf, phh = calc_ph_hybrid(f, mu)
        gp_calc = calc_gp(f, phf)
        delta = gp_calc - gp
        for lev in range(delta.shape[0]):
            print_dstat(lev, delta[lev])

        f.close()

    if args.camx:
        f = netCDF4.Dataset(args.camx)
        t = CAMxCoordTransform(f)
        t.verify(f)
        f.close()
