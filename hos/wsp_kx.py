import importlib as imp
import numpy as np
import math
from numpy import fft
import pickle
import matplotlib.pyplot as plt

import statistics as stt

jobname = "test"

readDir = '/scratch/HOSdata/pp/' + jobname + '/data/'
readName = "2Ddata_Dict"
fr = open(readDir + readName, 'rb')
data = pickle.load(fr)
fr.close()

tNum = data['time'].size
I = data['x'].size

kseq = stt.calcu_wsp_kx(data,0)[0]

ws = np.zeros(I)
for t in range(tNum):
    ws = ws + stt.calcu_wsp_kx(data,t)[1] / tNum


fig, ax = plt.subplots(figsize=(8,8))
plt.loglog(kseq, ws, linewidth=1.0, color='k', linestyle='-')
plt.xlim(1e-3,1e0)
plt.ylim(1e0,1e5)
plt.ylabel(r'$S_x$ $(m^3)$')
plt.xlabel(r'$k_x$ $(m^{-1})$')
plt.grid()
saveDir = '/scratch/HOSdata/pp/' + jobname + '/'
saveName = 'wave_spectrum_kx.png'
plt.savefig(saveDir + saveName)
plt.show()
