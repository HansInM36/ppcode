#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

# wrapper

import os
import sys
import math
import numpy as np
from numpy import fft
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import pickle

jobname  = 'pcr_NBL_U10_gs20'

varlist = ['u', 'v', 'w']

PSDD = {}

for var in varlist:
    readDir = '/scratch/palmdata/pp/' + jobname + '/data/'
    readName = var +'_PSDD'
    fr = open(readDir + readName, 'rb')
    PSDD[var] = pickle.load(fr)
    fr.close()

law53x = np.linspace(1e-4,1,20)
law53seq = np.power(law53x, -5/3)

### plot PSD of same var at different height levels
var = 'u'
hlist = list(PSDD[var].keys())
fig, ax = plt.subplots(figsize=(8,8))
colors = plt.cm.jet(np.linspace(0,1,len(hlist)))
for ih in range(len(hlist)-1): # sometimes the data of the highest level is unphysical, in that case: -1
    plt.loglog(PSDD[var]['fseq'], PSDD[var][hlist[ih]], label='h = ' + hlist[ih] + 'm', linewidth=1.0, color=colors[ih])
# plt.loglog(law53x, law53seq*1e-2, label='-5/3', linewidth=1.0, color='k', linestyle='--')
plt.ylabel('S' + var + ' (m^2/t^3)')
plt.xlabel('f (Hz)')
xaxis_min = 1e-4
xaxis_max = 1e-1
xaxis_d = 1
yaxis_min = 1e-7
yaxis_max = 1e2
yaxis_d = 200
plt.xlim(xaxis_min, xaxis_max)
plt.ylim(yaxis_min, yaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
# plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title(jobname)
fig.tight_layout() # adjust the layout
saveDir = '/scratch/palmdata/pp/' + jobname + '/'
saveName = var + '_PSD_hs.png'
plt.savefig(saveDir + saveName)
plt.show()



### plot PSD of different vars at same height level
hlist = list(PSDD[varlist[0]].keys())
hwlist = list(PSDD['w'].keys())
hind = 7
varplot = ['u', 'v', 'w']
fig, axs = plt.subplots(figsize=(8,8))
colors = plt.cm.jet(np.linspace(0,1,len(varplot)))
for ivar in range(len(varplot)):
    if varplot[ivar] == 'w':
        plt.loglog(PSDD['w']['fseq'], PSDD['w'][hwlist[hind]], label='w', linewidth=1.0, color=colors[ivar])
    else:
        plt.loglog(PSDD[varplot[ivar]]['fseq'], PSDD[varplot[ivar]][hlist[hind]], label=varplot[ivar], linewidth=1.0, color=colors[ivar])
plt.ylabel('S (m^2/t^3)')
plt.xlabel('f (Hz)')
xaxis_min = 1e-4
xaxis_max = 1e-1
xaxis_d = 1
yaxis_min = 1e-7
yaxis_max = 1e1
yaxis_d = 200
plt.xlim(xaxis_min, xaxis_max)
plt.ylim(yaxis_min, yaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
# plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.text(0.8, 1.02, 'h = ' + hlist[hind] + 'm', transform=axs.transAxes, fontdict={'size':12})
plt.title(jobname, x=0.5, y=1.02, fontdict={'size':12})
fig.tight_layout() # adjust the layout
saveDir = '/scratch/palmdata/pp/' + jobname + '/'
saveName = 'PSD_vars_h' + hlist[hind] + '.png'
plt.savefig(saveDir + saveName)
plt.show()
