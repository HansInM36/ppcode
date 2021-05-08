#matplotlib inline   

#config InlineBackend.figure_format='retina'

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure, show

from matplotlib.collections import LineCollection

import cartopy

import cartopy.feature as cpf

import cartopy.crs as ccrs

 

cpf_ocean = cpf.NaturalEarthFeature(

        category='physical',

        name='ocean',

        scale='50m',

        facecolor='#aaaaff',edgecolor="g",linewidth=0.5)

 

dx = 9000; dy = 9000

nx = 301; ny = 201   # just for the fist domain

map_proj = 'LambertConformal'

ref_lat   =  56.0

#ref_lon   =  6.6

ref_lon   =  3

truelat1  =  54.0

truelat2  =  59.0

stand_lon =  6.6

 

prj=ccrs.LambertConformal(central_longitude=ref_lon,

                          central_latitude=ref_lat,

                         standard_parallels=(truelat1,truelat2))

xmin = -(nx/2)*dx

xmax = (nx/2)*dx

ymin = -(ny/2)*dy

ymax = (ny/2)*dy

 

ax = plt.figure(figsize=(10,10)).gca(projection=prj)

 

ax.set_xlim([xmin,xmax])

ax.set_ylim([ymin,ymax])

 

ax.add_feature(cpf_ocean)

 

xlocs = np.arange(0,361,1)

ylocs = np.arange(-90,90,1)

gl = ax.gridlines(draw_labels=False, xlocs=xlocs, ylocs=ylocs,linewidth=0.5, color='gray')

gl.n_steps = 90

 

ax.plot([xmin,xmax,xmax,xmin,xmin],[xmin,xmin,xmax,xmax,xmin], '-', marker='+',c="b",alpha=1)

 

   

plt.show()
