import numpy as np
from numpy import fft
import math
import scipy.stats as stats

def calcu_Hs(x):
    # this function evaluate the significant wave height by Hs = 4*sqrt(var(eta(t)))
    return 4*np.power(np.var(x),0.5)

def calcu_wsp_f(data):
    # this function compute the wave spectrum averaged over all grid point
    I = data['x'].size
    J = data['y'].size
    tNum = data['time'].size

    # frequency sequence
    t_delta = data['time'][1] - data['time'][0]
    fseq = np.linspace(0, 1/t_delta, tNum)
    fseq = fseq[:math.ceil(tNum/2)]

    # wave spectrum sequence
    wsp = np.zeros(tNum)
    for i in range(I):
        for j in range(J):
            tmp = fft.fft(data['se'][:,j,i])
            tmp = np.power(abs(tmp),2)
            wsp = wsp + tmp / (I*J)
    wsp = wsp[:math.ceil(tNum/2)]

    return (fseq,wsp)

def JONSWAP(f,Tp,Hs,gama,alpha):
    omg_p = 2*np.pi / Tp
    omg = 2*np.pi * f
    A = alpha * np.power(Hs,2) * np.power(omg_p,4) * np.power(omg,-5)
    B = np.exp(-5/4 * np.power(omg/omg_p,-4))
    len = f.size
    C = np.zeros(len)
    for i in range(len):
        if omg[i] < omg_p:
            sgm = 0.07
        else:
            sgm = 0.09
        C[i] = np.power(gama, np.exp(-1 * np.power(omg[i]-omg_p,2) / (2*np.power(sgm,2)*np.power(omg_p,2))))
    return A*B*C

def calcu_wsp_kx(data,tind):
    # this function compute the horizontally averaged wave spectrum in x direction
    I = data['x'].size
    J = data['y'].size
    tNum = data['time'].size

    deltax = data['x'][-1] / I
    kmin = 0
    kmax = np.pi / deltax
    kseq = np.linspace(kmin,kmax,I)

    # wave spectrum sequence
    wsp = np.zeros(I)
    for j in range(J):
        # tmp = np.power(data['se'][tind][j,:],2)
        # tmp = abs(fft.fft(tmp))
        tmp = fft.fft(data['se'][tind][j,:])
        tmp = np.power(abs(tmp),2)
        wsp = wsp + tmp / J
    wsp = wsp[:math.ceil(tNum/2)]

    return (kseq,wsp)


def calcu_PDF(x, reso):
    # this function calculate PDF with a 1D array and resolution as input
    # e.g. x = [-1, 2, 7, 8], reso = 10
    len = x.size
    x_mean = np.mean(x)
    x = x - x_mean
    x_limit = math.ceil(abs(x).max())
    xseq = np.linspace(-x_limit, x_limit, reso+1)
    pdf = np.zeros(reso+1)
    for i in range(len):
        tmp = abs(xseq - x[i])
        min_ind = np.argmin(tmp)
        pdf[min_ind] = pdf[min_ind] + 1
    delta = x_limit*2/reso
    S = (pdf[0]+pdf[-1])*0.5*delta + np.sum(pdf[1:-2] * delta)
    pdf = pdf / S
    return (xseq+x_mean,pdf)

def gauss(x,mu,var):
    # this function calculate the gaussian distribution
    A = 1 / np.power(2*np.pi*var,0.5)
    B = - np.power(x-mu,2) / 2 / var
    return A * np.exp(B)
