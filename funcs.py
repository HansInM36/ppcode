import numpy as np
from numpy import fft
import scipy.signal


def window_weight(seq, method_='bell'):
    seqNum = seq.size
    ww = np.ones(seqNum)
    if method_ == 'bell':
        for i in range(seqNum):
            if i <= 0.1*seqNum:
                ww[i] = np.power(np.sin(5*np.pi*i / seqNum),2)
            elif i >= 0.9*seqNum:
                ww[i] = np.power(np.sin(5*np.pi*i / seqNum),2)
            else:
                ww[i] = 1.0
        return seq * ww
    else:
        print('error: wrong method')

def corr(x, y0, y1, norm_=True):
    seqNum = x.size
    delta_x = (x[-1] - x[0]) / (seqNum - 1)
    vNum = int(seqNum/2)
    xSeq = np.array([i*delta_x for i in range(vNum)]).astype(float)
    y0 = y0 - y0.mean()
    y1 = y1 - y1.mean()
    vSeq = np.correlate(y0,y1,mode='same')[-vNum:]
    vSeq = np.array([vSeq[i]/(seqNum - i) for i in range(vNum)]).astype(float)
    if norm_ == True:
        vSeq = vSeq / np.power(np.var(y0) * np.var(y1),0.5)
    return (xSeq, vSeq)

# def autocorr(x, y, norm_=True): version wrote by myself
#     seqNum = x.size
#     delta_x = (x[-1] - x[0]) / (seqNum - 1)
#     if seqNum % 2 == 0:
#         vNum = int(seqNum/2) - 1
#     else:
#         vNum = int(seqNum/2)
#     xSeq = np.array([i*delta_x for i in range(vNum)]).astype(float)
#     y = y - y.mean()
#     vSeq = []
#     for i in range(vNum):
#         tmp = [y[j]*y[j+i] for j in range(seqNum - i)]
#         tmp = np.array(tmp).astype(float)
#         vSeq.append(tmp.sum() / (seqNum - i))
#     vSeq = np.array(vSeq).astype(float)
#     if norm_ == True:
#         vSeq = vSeq / np.var(y)
#     return (xSeq, vSeq)

def PSD_k(x, ks):
    L = x.size
    X = np.fft.fft(x) / L # divide by y.size to calculate power spectral density
    S = abs(np.conjugate(X) * X)
    if L%2 == 0:
        S = 2*S[0:int(L/2)]
    else:
        S = 2*S[0:int(seqNum/2)+1]
        S[-1] = S[-1]/2
    k = np.linspace(0, ks, L)
    k = k[0:S.size]
    delta_k = k[-1] / (k.size-1)
    S = S / delta_k
    return (k, S)

def ESD_k(x, ks):
    L = x.size
    X = np.fft.fft(x)
    S = abs(np.conjugate(X) * X) / L
    if L%2 == 0:
        S = 2*S[0:int(L/2)]
    else:
        S = 2*S[0:int(seqNum/2)+1]
        S[-1] = S[-1]/2
    k = np.linspace(0, ks, L)
    k = k[0:S.size]
    delta_k = k[-1] / (k.size-1)
    S = S / delta_k
    return (k, S)


def PSD_f(x, fs):
    L = x.size
    X = np.fft.fft(x) / x.size # divide by y.size to calculate power spectral density
    S = abs(np.conjugate(X) * X)
    if L%2 == 0:
        S = 2*S[0:int(L/2)]
    else:
        S = 2*S[0:int(seqNum/2)+1]
        S[-1] = S[-1]/2
    f = np.linspace(0, fs, L)
    f = f[0:S.size]
    delta_f = f[-1] / (f.size-1)
    S = S / delta_f
    return (f, S)

def ESD_f(x, fs): # self-defined energy spectral density function
    L = x.size
    X = np.fft.fft(x)
    S = abs(np.conjugate(X) * X) / L
    if L%2 == 0:
        S = 2*S[0:int(L/2)]
    else:
        S = 2*S[0:int(seqNum/2)+1]
        S[-1] = S[-1]/2
    f = np.linspace(0, fs, L)
    f = f[0:S.size]
    delta_f = f[-1] / (f.size-1)
    S = S / delta_f
    return (f, S)

# def ESD_f(x, fs): # energy spectral density function using scipy.signal.csd()
#     f, S = scipy.signal.csd(x, x, fs, nperseg=None, noverlap=None)
#     return (f, S)



def coherence(data1, data2, fs, max_freq=None):

    freq, Pxx = scipy.signal.csd(data1, data1, fs, nperseg=None, noverlap=None)
    freq, Pxy = scipy.signal.csd(data1, data2, fs, nperseg=None, noverlap=None)
    freq, Pyy = scipy.signal.csd(data2, data2, fs, nperseg=None, noverlap=None)
    freq = np.squeeze(freq).ravel()

    if max_freq is not None:
        index = np.where(freq<=max_freq)
        freq  = freq[index]
        Pxx   = Pxx[index]
        Pyy   = Pyy[index]
        Pxy   = Pxy[index]

    Rxy    = Pxy.real
    Qxy    = Pxy.imag
    coh    = abs(np.array(Pxy) * np.array(np.conj(Pxy))) / (np.array(Pxx) * np.array(Pyy))
    co_coh = np.real(Pxy / np.sqrt(np.array(Pxx) * np.array(Pyy)))
    phase  = np.arctan2(Qxy,Rxy)

    return (freq, coh, phase)
