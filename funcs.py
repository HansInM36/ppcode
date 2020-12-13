import numpy as np
from numpy import fft
import scipy.signal

def trs(M,O,alpha):
    """ horizontal coordinate transformation """
    '''
    M: vector matrix with n rows, 3 columns (x,y,z)
    O: new origin
    alpha: rotation degree (counter clockwise)
    '''
    """ notice: for vectors like velocity, etc., the change of origin shouldn't make sense, so O should be set to (0,0,0) """
    # O = (431.6, 343.9, 0)
    # alpha = 15.52
    alpha_ = 2*np.pi/360 * alpha
    trsM = np.array([[np.cos(alpha_), -np.sin(alpha_), 0], [np.sin(alpha_), np.cos(alpha_), 0], [0, 0, 1]])

    M_ = np.copy(M)

    rNum = M_.shape[0]
    for r in range(rNum):
        M_[r][0] = M_[r][0] - O[0]
        M_[r][1] = M_[r][1] - O[1]
        M_[r][2] = M_[r][2] - O[2]

    M_ = np.dot(M_, trsM)
    return M_

def detrend(x, y, method_='linear'):
    deg_ = 1
    polyFunc = np.poly1d(np.polyfit(x, y, deg=deg_))
    tmp = y - polyFunc(x)
    return tmp

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

def corr(x, y0 ,y1, norm_=True): # version wrote by myself
    """ R_{01}(k) = \frac{1}{N-k} \sum_{n=0}^[N-k-1] y0[n]y[n+k], 0 <= k <= N/2 """
    seqNum = x.size
    delta_x = (x[-1] - x[0]) / (seqNum - 1)
    vNum = int(seqNum/2)
    xSeq = np.array([i*delta_x for i in range(vNum)]).astype(float)
    y0 = y0 - y0.mean()
    y1 = y1 - y1.mean()
    vSeq = []
    for i in range(vNum):
        tmp = [y0[j]*y1[j+i] for j in range(seqNum - i)]
        tmp = np.array(tmp).astype(float)
        vSeq.append(tmp.sum() / (seqNum - i))
    vSeq = np.array(vSeq).astype(float)
    if norm_ == True:
        vSeq = vSeq / np.power(np.var(y0) * np.var(y1),0.5)
    return (xSeq, vSeq)

def autocorr_FFT(x, fs, norm_=True):
    x -= x.mean()
    N = x.size
    tau = np.arange(0, N) * 1/fs
    X = np.fft.fft(x)
    X2 = abs(X*np.conj(X))
    v = np.real(np.fft.ifft(X2)) / N
    # tau = tau[:int(N/2)]
    # v = v[:int(N/2)]
    if norm_ == True:
        v = v / np.var(x)
    return (tau, v)

def crosscorr_FFT(x0, x1, fs, norm_=True):
    x0 -= x0.mean()
    x1 -= x1.mean()
    N = x0.size
    tau = np.arange(0, N) * 1/fs
    X0 = np.fft.fft(x0)
    X1 = np.fft.fft(x1)
    X01 = X0*np.conj(X1)
    phase = np.angle(X01)
    X01_ = abs(X01)
    v = np.real(np.fft.ifft(X01)) / N
    # tau = tau[:int(N/2)]
    # v = v[:int(N/2)]
    if norm_ == True:
        v = v / np.sqrt((np.var(x0) * np.var(x1)))
    return (tau, v, phase)

# def corr(x, y0, y1, norm_=True):
#     seqNum = x.size
#     delta_x = (x[-1] - x[0]) / (seqNum - 1)
#     vNum = int(seqNum/2)
#     xSeq = np.array([i*delta_x for i in range(vNum)]).astype(float)
#     y0 = y0 - y0.mean()
#     y1 = y1 - y1.mean()
#     vSeq = np.correlate(y0,y1,mode='same')[-vNum:]
#     vSeq = np.array([vSeq[i]/(seqNum - i) for i in range(vNum)]).astype(float)
#     if norm_ == True:
#         vSeq = vSeq / np.power(np.var(y0) * np.var(y1),0.5)
#     return (xSeq, vSeq)

def PSD_k(x, ks):
    L = x.size
    X = np.fft.fft(x) / L # divide by y.size to calculate power spectral density
    S = abs(np.conjugate(X) * X)
    if L%2 == 0:
        S = 2*S[0:int(L/2)]
    else:
        S = 2*S[0:int(L/2)+1]
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



def coherence(data1, data2, fs, segNum, max_freq=None):
    """
    nperseg:
        In personal opinion, this function will divided each of the two input segments into certain sub-segments with a length of nperseg,
        and use these sub-segments to mimic two random segments and estimate cross spectrum density. if nperseg is set to tha same as the length of input segments,
        then the mimicked random segments become determined, the cross spectral density will be 1 at all frequency. the recommended value of nperseg is
        less than tenth of the input data length while hold a appropriate value that provides enough frequency resolution.
    """

    freq, Pxx = scipy.signal.csd(data1, data1, fs, nperseg=segNum, noverlap=None)
    freq, Pxy = scipy.signal.csd(data1, data2, fs, nperseg=segNum, noverlap=None)
    freq, Pyy = scipy.signal.csd(data2, data2, fs, nperseg=segNum, noverlap=None)
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

    return (freq, coh, co_coh, phase)

def calc_deriv_1st_FFT(dx,y):
    N = y.size
    L = dx * (N-1)
    Y = np.fft.fft(y)
    omg = 2*np.pi/L * np.arange(-N/2, N/2)
    omg = fft.fftshift(omg)
    dyFFT = (1j) * omg * Y
    dy = fft.ifft(dyFFT)
    return dy

def calc_deriv_1st_FD(dx,y):
    N = y.size
    dy = np.zeros(N)
    for i in range(N):
        if i == 0:
            dy[i] = (y[i+1] - y[i]) / dx
        elif i > 0 and i < N-1:
            dy[i] = (y[i+1] - y[i-1]) / (2*dx)
        else:
            dy[i] = (y[i] - y[i-1]) / dx
    return dy


""" wind spectrum """
def kaimal_u(f, uz, z, uStar):
    f_ = f*z / uz
    up = 105 * f_ * np.power(uStar,2)
    down = f * np.power(1 + 33*f_, 5/3)
    return up/down

def kaimal_v(f, uz, z, uStar):
    f_ = f*z / uz
    up = 17 * f_ * np.power(uStar,2)
    down = f * np.power(1 + 9.5*f_, 5/3)
    return up/down

def kaimal_w(f, uz, z, uStar):
    f_ = f*z / uz
    up = 2.1 * f_ * np.power(uStar,2)
    down = f * np.power(1 + 5.3*f_, 5/3)
    return up/down

def fourier_series(t_, A_, omg_, k_, x_, phi_):
    seq = np.zeros(t_.size)
    for i in range(t_.size):
        sum = 0
        for k in range(A_.size):
            sum += A_[k] * np.cos(omg_[k]*t_[i] - k_[k]*x_ + phi_[k])
        seq[i] = sum
    return seq
