import numpy as np
from numpy import fft
import scipy.fft
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

def flt_seq(x,tao):
    """ 以tao来过滤一个序列 """
    '''
    filter x with tao
    x must be a 1D array
    '''
    '''
    x = np.array([1,2,3,4,5,6,7])
    flt_seq(x,3)
    '''
    tao = int(tao)
    l = np.shape(x)[0]
    y = np.zeros(np.shape(x))

    for i in range(l):
        if i-tao < 0:
            a = 0
        else:
            a = i-tao
        if i+tao+1 > l:
            b = l
        else:
            b = i+tao+1
        a, b = int(a), int(b)
        y[i] = sum(x[a:b]) / np.shape(x[a:b])[0]
    return y


def FFT2D(v, dx, dy):
    """ 2d Fourier Transform """
    """
    v: 2d array
    dx, dy: spacing in x (the second dimension) and y (the first dimension) direction
    """
    # # this part of code is equivalent to fft2d = scipy.fft.fft2(v)
    # fft1d = []
    # for i in range(v.shape[1]):
    #     tmp = scipy.fft.fft(v[:,i])
    #     fft1d.append(tmp)
    # fft1d = np.array(fft1d)
    #
    # fft2d = []
    # for j in range(v.shape[0]):
    #     tmp = scipy.fft.fft(fft1d[:,j])
    #     fft2d.append(tmp)
    # fft2d = np.array(fft2d)
    Nx = v.shape[1]
    Ny = v.shape[0]
    fft2d = scipy.fft.fft2(v)

    kx = scipy.fft.fftfreq(Nx, dx)[:Nx//2]
    ky = scipy.fft.fftfreq(Ny, dy)[:Ny//2]
    fft2d = fft2d[:Ny//2, :Nx//2]
    # kx = np.linspace(0, 1/dx/2, Nx//2)
    # ky = np.linspace(0, 1/dy/2, Ny//2)
    return fft2d, kx, ky

def PSD2D(v, dx, dy):
    """ compute 2d power spectral density function by 2d Fourier Transform """
    """
    v: 2d array
    dx, dy: spacing in x (the second dimension) and y (the first dimension) direction
    note: not sure if the psd2d /= (dkx * dky) is perfect, may need improvement of scaling
    """
    # # this part of code is equivalent to fft2d = scipy.fft.fft2(v)
    # fft1d = []
    # for i in range(v.shape[1]):
    #     tmp = scipy.fft.fft(v[:,i])
    #     fft1d.append(tmp)
    # fft1d = np.array(fft1d)
    #
    # fft2d = []
    # for j in range(v.shape[0]):
    #     tmp = scipy.fft.fft(fft1d[:,j])
    #     fft2d.append(tmp)
    # fft2d = np.array(fft2d)
    Nx = v.shape[1]
    Ny = v.shape[0]
    fft2d = scipy.fft.fft2(v)

    psd2d = np.abs(fft2d * np.conj(fft2d)) / np.power(Nx,2) / np.power(Ny,2)

    # fft2d = fft2d[:Ny//2, :Nx//2]
    kx = scipy.fft.fftfreq(Nx, dx)[:Nx//2]*2*np.pi
    ky = scipy.fft.fftfreq(Ny, dy)[:Ny//2]*2*np.pi
    dkx = (kx[-1] - kx[0]) / (Nx//2)
    dky = (ky[-1] - ky[0]) / (Ny//2)

    # scaling
    psd2d /= (dkx * dky)

    # fold
    psd2d = psd2d[:,:Nx//2]
    psd2d[:,1:] *= 2
    psd2d = psd2d[:Ny//2,:]
    psd2d[1:,:] *= 2
    return psd2d, kx, ky

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

def autocorr_FFT(x, delta, norm_=True):
    x -= x.mean()
    N = x.size
    tau = np.arange(0, N) * delta
    X = np.fft.fft(x)
    X2 = abs(X*np.conj(X))
    v = np.real(np.fft.ifft(X2)) / N
    # tau = tau[:int(N/2)]
    # v = v[:int(N/2)]
    if norm_ == True:
        v = v / np.var(x)
    return (tau, v)

def crosscorr_FFT(x0, x1, delta, norm_=True):
    x0 -= x0.mean()
    x1 -= x1.mean()
    N = x0.size
    tau = np.arange(0, N) * delta
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
        S = 2*S[0:int(L/2)+1]
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
    coh    = np.sqrt(abs(np.array(Pxy) * np.array(np.conj(Pxy))) / (np.array(Pxx) * np.array(Pyy)))
    co_coh = np.real(Pxy / np.sqrt(np.array(Pxx) * np.array(Pyy)))
    quad_coh = np.imag(Pxy / np.sqrt(np.array(Pxx) * np.array(Pyy)))
    phase  = np.arctan2(Qxy,Rxy)

    return (freq, coh, co_coh, quad_coh, phase)

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


""" wind spectrum (ref. Velocity Spectra and Coherence Estimates in the Marine Atmospheric Boundary Layer) """
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

""" wind spectrum (ref. Recent spectra of atmospheric turbulence (Busch, Panofsky, 1968)) """
def Busch_Panofsky_w(f, uz, z, uStar, phi=1.0):
    f_ = f*z / uz
    up = 3.36 * f_ / phi * np.power(uStar,2)
    down = f * (1 + 10 * np.power(f_/phi, 5/3))
    return up/down


def IEC_coh(f, d, uz, L):
    a, b = 12, 0.12
    A = np.power(f*d/uz,2)
    B = np.power(b*d/L,2)
    coh = np.exp(-a * np.power(A+B,0.5))
    return coh

def fourier_series(t_, A_, omg_, k_, x_, phi_):
    seq = np.zeros(t_.size)
    for i in range(t_.size):
        sum = 0
        for k in range(A_.size):
            sum += A_[k] * np.cos(omg_[k]*t_[i] - k_[k]*x_ + phi_[k])
        seq[i] = sum
    return seq


def plot_ts(tSeq,u0,u1):
    import matplotlib.pyplot as plt
    ### time series
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(tSeq, u0, 'r-', label='p0')
    ax.plot(tSeq, u1, 'b-', label='p1')
    # ax.set_ylim(-2.4, 2.4)
    # ax.set_xlim(0, 1200)
    ax.set_xlabel('t (s)', fontsize=12)
    ax.set_ylabel('u (m/s)', fontsize=12)
    ax.text(0.56, 1.02, '', transform=ax.transAxes, fontsize=12)
    ax.grid()
    plt.show()



""" group plot 0 """
""" All plots in one figure """
def group_plot_0(tSeq, fs, u0, u1):

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # auto-correlation coefficient
    tau0, R0 = autocorr_FFT(u0, 1/fs)
    tau1, R1 = autocorr_FFT(u1, 1/fs)

    # auto-correlation coefficient
    tau, R, phase__ = crosscorr_FFT(u0, u1, 1/fs)

    # PSD
    segNum = int(120*fs)
    freq0, S0 = scipy.signal.csd(u0, u0, fs, nperseg=segNum, noverlap=None)
    freq1, S1 = scipy.signal.csd(u1, u1, fs, nperseg=segNum, noverlap=None)

    # CSD
    segNum = int(120*fs)
    freq, S01 = scipy.signal.csd(u0, u1, fs, nperseg=segNum, noverlap=None)
    S01_ = abs(S01)
    phase_ = np.angle(S01)

    # coherence, co-coherence, phase
    segNum = int(120*fs)
    freq, coh, co_coh, phase = coherence(u0, u1, fs, segNum)

    fig = plt.figure(figsize=(12,18),tight_layout=True)
    gs = gridspec.GridSpec(6, 5)




    ### time series
    ax = fig.add_subplot(gs[0, :])
    ax.plot(tSeq, u0, 'r-', label='p0')
    ax.plot(tSeq, u1, 'b-', label='p1')
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlim(0, 1200)
    ax.set_xlabel('t (s)', fontsize=12)
    ax.set_ylabel('u (m/s)', fontsize=12)
    ax.text(0.56, 1.02, '', transform=ax.transAxes, fontsize=12)
    ax.grid()

    ### auto-correlation coefficient
    ax = fig.add_subplot(gs[1, :3])
    ax.plot(tau0, R0, 'r-', label='p0')
    ax.plot(tau1, R1, 'b-', label='p1')
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 1200)
    ax.set_xlabel(r'$\mathrm{\tau}$ (s)', fontsize=12)
    ax.set_ylabel(r'$\mathrm{\rho_{auto}}$', fontsize=12)
    ax.grid()

    ### PSD
    # PSD = kaimal_u(freq[1:], uz, z, uStar)
    ax = fig.add_subplot(gs[1, 3:])
    ax.loglog(freq0, S0, 'r-', label='p0')
    ax.loglog(freq1, S1, 'b-', label='p1')
    # ax.loglog(freq[1:], PSD, 'k-', label='Kaimal')
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel(r'$\mathrm{S_{auto}}$ ($\mathrm{m^2/s}$)', fontsize=12)
    ax.grid()

    ### cross-correlation coefficient
    ax = fig.add_subplot(gs[2, :3])
    ax.plot(tau, R, 'g-')
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 300)
    ax.set_xlabel(r'$\mathrm{\tau}$ (s)', fontsize=12)
    ax.set_ylabel(r'$\mathrm{\rho_{cross}}$', fontsize=12)
    ax.grid()

    ### CSD
    ax = fig.add_subplot(gs[2, 3:])
    ax.loglog(freq, S01_, 'g-')
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel(r'$\mathrm{S_{cross}}$ ($\mathrm{m^2/s}$)', fontsize=12)
    ax.grid()

    ### coherence
    f_out = 0.5
    tmp = abs(freq - f_out)
    ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

    ax = fig.add_subplot(gs[3, :3])
    ax.plot(freq[1:], coh[1:], linestyle='', marker='x', markersize=3, color='k')
    ax.set_ylim(0, 1)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('coherence', fontsize=12)
    ax.grid()
    # zoom in
    ax = fig.add_subplot(gs[3, 3:])
    ax.plot(freq[ind_in:ind_out], coh[ind_in:ind_out], linestyle='-', marker='o', markersize=3, color='k')
    ax.set_ylim(0, 1)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('coherence', fontsize=12)
    ax.grid()

    ### co-coherence
    ax = fig.add_subplot(gs[4, :3])
    ax.plot(freq[1:], co_coh[1:], linestyle='', marker='x', markersize=3, color='r')
    ax.set_ylim(-1, 1)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('co-coherence', fontsize=12)
    ax.grid()
    # zoom in
    ax = fig.add_subplot(gs[4, 3:])
    ax.plot(freq[ind_in:ind_out], co_coh[ind_in:ind_out], linestyle='-', marker='o', markersize=3, color='r')
    # ax.plot(freq[ind_in:ind_out], np.sqrt(coh[ind_in:ind_out])*np.cos(phase[ind_in:ind_out]), linestyle=':', marker='', markersize=1, color='g')
    ax.set_ylim(-1, 1)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('co-coherence', fontsize=12)
    ax.grid()

    ### phase
    xaxis_min = 0
    xaxis_max = 5.0
    xaxis_d = 0.5
    yaxis_min = -1.0*np.pi
    yaxis_max = 1.0*np.pi
    yaxis_d = np.pi/4
    labels = ['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
              r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
    ax = fig.add_subplot(gs[5, :3])
    ax.plot(freq[1:], phase[1:], linestyle='', marker='x', markersize=3, color='b')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('phase', fontsize=12)
    ax.grid()
    # zoom in
    ax = fig.add_subplot(gs[5, 3:])
    ax.plot(freq[ind_in:ind_out], phase[ind_in:ind_out], linestyle='-', marker='o', markersize=3, color='b')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('phase', fontsize=12)
    ax.grid()

    plt.show()

""" group plot 1 """
""" coh, co-coh and phase in one figure """
def group_plot_1(tSeq, fs, u0, u1):

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # auto-correlation coefficient
    tau0, R0 = autocorr_FFT(u0, fs)
    tau1, R1 = autocorr_FFT(u1, fs)

    # auto-correlation coefficient
    tau, R, phase__ = crosscorr_FFT(u0, u1, fs)

    # PSD
    segNum = int(120*fs)
    freq0, S0 = scipy.signal.csd(u0, u0, fs, nperseg=segNum, noverlap=None)
    freq1, S1 = scipy.signal.csd(u1, u1, fs, nperseg=segNum, noverlap=None)

    # CSD
    segNum = int(120*fs)
    freq, S01 = scipy.signal.csd(u0, u1, fs, nperseg=segNum, noverlap=None)
    S01_ = abs(S01)
    phase_ = np.angle(S01)

    # coherence, co-coherence, phase
    segNum = int(120*fs)
    freq, coh, co_coh, phase = coherence(u0, u1, fs, segNum)

    fig = plt.figure(figsize=(10,8),tight_layout=True)
    gs = gridspec.GridSpec(3, 5)

    ### coherence
    f_out = 0.5
    tmp = abs(freq - f_out)
    ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

    ax = fig.add_subplot(gs[0, :3])
    ax.plot(freq[1:], coh[1:], linestyle='', marker='x', markersize=3, color='k')
    ax.set_ylim(0, 1)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('coherence', fontsize=12)
    ax.grid()
    # zoom in
    ax = fig.add_subplot(gs[0, 3:])
    ax.plot(freq[ind_in:ind_out], coh[ind_in:ind_out], linestyle='-', marker='o', markersize=3, color='k')
    ax.set_ylim(0, 1)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('coherence', fontsize=12)
    ax.grid()

    ### co-coherence
    ax = fig.add_subplot(gs[1, :3])
    ax.plot(freq[1:], co_coh[1:], linestyle='', marker='x', markersize=3, color='r')
    ax.set_ylim(-1, 1)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('co-coherence', fontsize=12)
    ax.grid()
    # zoom in
    ax = fig.add_subplot(gs[1, 3:])
    ax.plot(freq[ind_in:ind_out], co_coh[ind_in:ind_out], linestyle='-', marker='o', markersize=3, color='r')
    # ax.plot(freq[ind_in:ind_out], np.sqrt(coh[ind_in:ind_out])*np.cos(phase[ind_in:ind_out]), linestyle=':', marker='', markersize=1, color='g')
    ax.set_ylim(-1, 1)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('co-coherence', fontsize=12)
    ax.grid()

    ### phase
    xaxis_min = 0
    xaxis_max = 5.0
    xaxis_d = 0.5
    yaxis_min = -1.0*np.pi
    yaxis_max = 1.0*np.pi
    yaxis_d = np.pi/4
    labels = ['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
              r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
    ax = fig.add_subplot(gs[2, :3])
    ax.plot(freq[1:], phase[1:], linestyle='', marker='x', markersize=3, color='b')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('phase', fontsize=12)
    ax.grid()
    # zoom in
    ax = fig.add_subplot(gs[2, 3:])
    ax.plot(freq[ind_in:ind_out], phase[ind_in:ind_out], linestyle='-', marker='o', markersize=3, color='b')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('f (Hz)', fontsize=12)
    ax.set_ylabel('phase', fontsize=12)
    ax.grid()

    plt.show()
