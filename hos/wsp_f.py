import sys
sys.path.append("/scratch/ppcode/HOS")
import importlib as imp
import numpy as np
import math
from numpy import fft
import pickle
import matplotlib.pyplot as plt
import statis as stt

jobname = "test"

readDir = '/scratch/HOSdata/pp/' + jobname + '/data/'
readName = "2Ddata_Dict"
fr = open(readDir + readName, 'rb')
data = pickle.load(fr)
fr.close()

ws = stt.calcu_wsp_f(data)

Tp = 10.0
Hs = 4.5
gama = 3.3
alpha = 0.3
jonswap = stt.JONSWAP(ws[0][1:], Tp, Hs, gama, alpha)
# jswp = stt.jonswap(ws[0][1:], Tp, Hs, gamma=gama,normalize=0)

fig, ax = plt.subplots(figsize=(8,8))
plt.loglog(ws[0], ws[1], linewidth=1.0, color='r', linestyle='-')
# plt.loglog(ws[0][1:], jonswap, linewidth=1.0, color='k', linestyle='-', label='JONSWAP')
plt.xlim(1e-3,1e0)
plt.ylim(1e0,1e5)
plt.ylabel(r'S $(m^2/Hz)$')
plt.xlabel('f (Hz)')
plt.grid()

saveDir = '/scratch/HOSdata/pp/' + jobname + '/'
saveName = 'wave_spectrum_f.png'
plt.savefig(saveDir + saveName)
plt.show()

valid = np.sum(ws[1])/np.power(tNum,2)*2/np.var(data['se'])
