import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

dataDir = '/scratch/prjdata/wind_wave_data_Mostafa/2nd'
dataName = 'WindDataDCF15_July2015.mat'

data = scipy.io.loadmat(dataDir + '/' + dataName)

keyList = list(data.keys())

tSeq = data['tii'][0]

tempSeq = data['Temp'][0]

uSeq = data['uvel'][0]
vSeq = data['vvel'][0]
wSeq = data['wvel'][0]