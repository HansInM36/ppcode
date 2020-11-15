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

reso_level = 100 # resolution level of PDF: the range is divided into 100 interval

seq = data['se'].flatten()

PDF = stt.calcu_PDF(seq,reso_level)

var_ = np.var(seq)
mean_ = np.mean(seq)
gs = stt.gauss(PDF[0],mean_,var_)

fig, ax = plt.subplots(figsize=(8,8))
plt.plot(PDF[0], PDF[1], color='r', marker='o', linestyle='None', label='Simulation')
plt.plot(PDF[0], gs, linewidth=1.0, color='k', linestyle='-', label='Gaussian Distribution')
# plt.xlim(1e-3,1e0)
# plt.ylim(0,0.06)
plt.ylabel(r'$f(\eta)$')
plt.xlabel(r'$\eta$')
plt.grid()
plt.legend()
saveDir = '/scratch/HOSdata/pp/' + jobname + '/'
saveName = 'PDF.png'
plt.savefig(saveDir + saveName)
plt.show()
