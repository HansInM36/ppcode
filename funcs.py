import numpy as np
from numpy import fft

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

def ESD_k(x, y):
    seqNum = x.size
    L = x[-1] - x[0]
    DFT2 = np.power(abs(np.fft.fft(y)),2)
    if seqNum%2 == 0:
        ESD = 2*DFT2[0:int(seqNum/2)]
    else:
        ESD = 2*DFT2[0:int(seqNum/2)+1]
        ESD[-1] = ESD[-1]/2
    ESD_num = ESD.size
    lambda_min = L / (ESD_num-1)
    k_max = 2*np.pi / lambda_min
    kSeq = np.linspace(0, k_max, ESD_num)
    delta_k = k_max / ESD_num
    ESD = ESD / delta_k
    return (kSeq, ESD)


def ESD_omega(x, y):
    seqNum = x.size
    tL = x[-1] - x[0]
    DFT2 = np.power(abs(np.fft.fft(y)),2)
    if seqNum%2 == 0:
        ESD = 2*DFT2[0:int(seqNum/2)]
    else:
        ESD = 2*DFT2[0:int(seqNum/2)+1]
        ESD[-1] = ESD[-1]/2
    ESD_num = ESD.size
    T_min = tL / (ESD_num-1)
    omega_max = 2*np.pi / T_min
    omegaSeq = np.linspace(0, omega_max, ESD_num)
    delta_omega = omega_max / ESD_num
    ESD = ESD / delta_omega
    return (omegaSeq, ESD)
