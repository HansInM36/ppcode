import importlib as imp
import numpy as np
import math
from numpy import fft
import pickle
import matplotlib.pyplot as plt

import statistics as stt

prjname = "regular"
jobname = "test"

readDir = '/scratch/HOSdata/pp/' + prjname + '/' + jobname + '/data/'
readName = "2Ddata_Dict"
fr = open(readDir + readName, 'rb')
data = pickle.load(fr)
fr.close()

varnameList = ['variance', 'skewness', 'kurtosis']
varunitList = [r'$m^2$']
varnum = len(varnameList)

varseqList = [[] for var in range(varnum)]

tseq = data['time']
tNum = tseq.size

varseqList[0] = np.zeros(tNum)
for t in range(tNum):
    varseqList[0][t] = np.var(data['se'][t])

varseqList[1] = np.zeros(tNum)
for t in range(tNum):
    varseqList[1][t] = np.mean(np.power(data['se'][t],3)) / np.power(np.var(data['se'][t]),3/2)

varseqList[2] = np.zeros(tNum)
for t in range(tNum):
    varseqList[2][t] = np.mean(np.power(data['se'][t],4)) / np.power(np.var(data['se'][t]),4/2)


fig, axs = plt.subplots(varnum,1)
fig.set_figwidth(8)
fig.set_figheight(6)

# colors = plt.cm.jet(np.linspace(0,1,varnum-1))
colors = ['k','b','g']
linestyles = ['-','-','-']
labels = ['variance', 'skewness', 'kurtosis']
ylims = [(1.26,1.27),(-0.2,0.6),(2,4)]

for i in range(0,varnum):
    axs[i].plot(tseq, varseqList[i], linewidth=1.0, color=colors[i], label=labels[i])
    axs[i].set_ylim(ylims[i])
    axs[i].set_ylabel(varnameList[i], fontsize=12)
    axs[i].grid()
    axs[varnum-1].set_xlabel('time (s)', fontsize=12)
fig.tight_layout() # adjust the layout
saveDir = '/scratch/HOSdata/pp/' + jobname + '/'
saveName = 'ts.png'
plt.savefig(saveDir + saveName)
plt.show()
