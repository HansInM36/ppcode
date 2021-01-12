import sys
sys.path.append('/scratch/ppcode/sowfa/src')
sys.path.append('/scratch/ppcode')
import imp
import numpy as np
import scipy.signal
import funcs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

prjDir = '/scratch/ppcode/test/play_coherence'

""" time series and positions of the two points p0 and p1 """
dt = 0.1
fs = 1/dt

t_start = 0
t_end = 2400

x0 = 100.0
x1 = 120.0

t_N = int((t_end - t_start) / dt + 1)
tSeq = np.linspace(t_start, t_end, t_N)


""" regular waves """
uz = 8.0

A0, A1 = 0.8, 0.8
f0, f1 = 0.1, 0.1
x0, x1 = 40.0, 80.0
phi0, phi1 = 0.0, 1.0

omg0, omg1 = 2*np.pi*f0, 2*np.pi*f1
k0, k1 = omg0/uz, omg1/uz

phase0 = omg0*tSeq - k0*x0 + phi0
phase1 = omg1*tSeq - k1*x1 + phi1

# power of the signal
P0 = np.power(A0,2) / 2
P1 = np.power(A1,2) / 2

# white noise
n0 = np.random.rand(t_N)
n1 = np.random.rand(t_N)


u0 = A0*np.cos(phase0) + n0
u1 = A1*np.cos(phase1) + n1

SNR0 = np.var(u0) / np.var(n0)
SNR1 = np.var(u1) / np.var(n1)

dphase = (phase1 - phase0) / np.pi


""" fourier series """

uz = 8
z = 100
uStar = 0.4

x0, x1 = 40.0, 80.0

fNum = 100

LMin, LMax = 1, 1000
kMin, kMax = 2*np.pi/LMax, 2*np.pi/LMin
fMin, fMax = uz*kMin/2/np.pi, uz*kMax/2/np.pi


fSeq = np.geomspace(fMin, fMax, fNum)

S = np.zeros(fNum)
for i in range(fNum):
    if i == 0:
        S[i] = 0.5 * (fSeq[i+1] - fSeq[i]) * funcs.kaimal_u(fSeq[i], uz, z, uStar)
    elif i == fNum - 1:
        S[i] = 0.5 * (fSeq[i] - fSeq[i-1]) * funcs.kaimal_u(fSeq[i], uz, z, uStar)
    else:
        S[i] = 0.5 * (fSeq[i+1] - fSeq[i-1]) * funcs.kaimal_u(fSeq[i], uz, z, uStar)

ASeq = np.power(2*S, 0.5)
omgSeq = 2*np.pi*fSeq
kSeq = omgSeq/uz
phiSeq = (np.random.rand(fNum) - 0.5) * 2*np.pi


u0 = funcs.fourier_series(tSeq, ASeq, omgSeq, kSeq, x0, phiSeq)
u1 = funcs.fourier_series(tSeq, ASeq, omgSeq, kSeq, x1, phiSeq)

# white noise
n0 = np.random.rand(t_N) * 5
n1 = np.random.rand(t_N) * 5

u0 += n0
u1 += n1

SNR0 = np.var(u0) / np.var(n0)
SNR1 = np.var(u1) / np.var(n1)



""" auto-correlation coefficient """
tau0, R0 = funcs.autocorr_FFT(u0, fs)
tau1, R1 = funcs.autocorr_FFT(u1, fs)

""" auto-correlation coefficient """
tau, R, phase__ = funcs.crosscorr_FFT(u0, u1, fs)

""" PSD """
segNum = int(120*fs)
freq0, S0 = scipy.signal.csd(u0, u0, fs, nperseg=segNum, noverlap=None)
freq1, S1 = scipy.signal.csd(u1, u1, fs, nperseg=segNum, noverlap=None)

""" CSD """
segNum = int(120*fs)
freq, S01 = scipy.signal.csd(u0, u1, fs, nperseg=segNum, noverlap=None)
S01_ = abs(S01)
phase_ = np.angle(S01)

""" coherence, co-coherence, phase """
segNum = int(120*fs)
freq, coh, co_coh, phase = funcs.coherence(u0, u1, fs, segNum)




""" plot all figures """
allplot(tSeq, u0, u1, tau, tau0, tau1, R, R0, R1, freq, freq0, freq1, S0, S1, S01_, coh, co_coh, phase)





""" plot time series """
fig, ax = plt.subplots(figsize=(8,4))
# ax.plot(tau0, R0, 'r-', label='p0')
ax.plot(tau1, R1, 'b-', label='p1')
# plt.ylim(6, 10)
# plt.xlim(0, 120)
ax.set_xlabel('tau (s)', fontsize=12)
ax.set_ylabel('autocorr (m2/s2)', fontsize=12)
ax.text(0.56, 1.02, '', transform=ax.transAxes, fontsize=12)
plt.grid()
plt.legend()
# saveName = ''
# plt.savefig(ppDir + '/' + saveName)
plt.show()
plt.close()

""" plot coherence, co-coherence, phase in one figure """
f_out = 0.4
tmp = abs(freq - f_out)
ind_in, ind_out = 1, np.where(tmp == tmp.min())[0][0]

rNum, cNum = (1,3)
fig, axs = plt.subplots(rNum,cNum, constrained_layout=False)
fig.set_figwidth(12)
fig.set_figheight(4)

# coherence
axs[0].plot(freq[1:], coh[1:], linestyle='', marker='o', markersize=3, color='k')
# popt, pcov = curve_fit(fitting_func, freq[ind_in:ind_out], coh[ind_in:ind_out], bounds=(0, [1, 100]))
# axs[0].plot(freq[0:ind_out], fitting_func(freq[0:ind_out], *popt), linestyle='-', color='k',
#      label='a=%5.3f, alpha=%5.3f' % tuple(popt))

axs[0].tick_params(axis='both', which='major', labelsize=10)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = 0
yaxis_max = 1.0
yaxis_d = 0.1
axs[0].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[0].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
axs[0].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
axs[0].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[0].grid()

axs[0].set_xlabel('f (1/s)', fontsize=12)
axs[0].set_ylabel('coherence', fontsize=12)

# axs[0].legend(bbox_to_anchor=(0.2,0.9), loc=6, borderaxespad=0, fontsize=10)

# co-coherence
axs[1].plot(freq[1:], co_coh[1:], linestyle='', marker='o', markersize=3, color='r')

axs[1].tick_params(axis='both', which='major', labelsize=10)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = -1.0
yaxis_max = 1.0
yaxis_d = 0.2
axs[1].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[1].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
axs[1].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
axs[1].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
axs[1].grid()
axs[1].set_xlabel('f (1/s)', fontsize=12)
axs[1].set_ylabel('co-coherence', fontsize=12)
# axs[1].set_title('', fontsize=12)

# phase
axs[2].plot(freq[1:], phase[1:], linestyle='', marker='o', markersize=3, color='b')
axs[2].tick_params(axis='both', which='major', labelsize=10)
xaxis_min = 0
xaxis_max = 5.0
xaxis_d = 0.5
yaxis_min = -1.0*np.pi
yaxis_max = 1.0*np.pi
yaxis_d = np.pi/4
axs[2].set_ylim(yaxis_min - 0.0*yaxis_d,yaxis_max)
axs[2].set_xlim(xaxis_min - 0.0*xaxis_d,xaxis_max)
axs[2].set_xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
axs[2].set_yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
# axs[2].set_yticks(np.arange(0, 2*np.pi+0.01, np.pi/4))
labels = ['$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
          r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
axs[2].set_yticklabels(labels)
axs[2].grid()
axs[2].set_xlabel('f (1/s)', fontsize=12)
axs[2].set_ylabel('phase', fontsize=12)

fig.tight_layout()
# saveName = ''
# plt.savefig(ppDir + '/' + saveName)
plt.show()




""" All plots in one figure """
def allplot(tSeq, u0, u1, tau, tau0, tau1, R, R0, R1, freq, freq0, freq1, S0, S1, S01_, coh, co_coh, phase):
    fig = plt.figure(figsize=(12,18),tight_layout=True)
    gs = gridspec.GridSpec(6, 5)

    ### time series
    ax = fig.add_subplot(gs[0, :])
    ax.plot(tSeq, u0, 'r-', label='p0')
    ax.plot(tSeq, u1, 'b-', label='p1')
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlim(0, 120)
    ax.set_xlabel('t (s)', fontsize=12)
    ax.set_ylabel('u (m/s)', fontsize=12)
    ax.text(0.56, 1.02, '', transform=ax.transAxes, fontsize=12)
    ax.grid()

    ### auto-correlation coefficient
    ax = fig.add_subplot(gs[1, :3])
    ax.plot(tau0, R0, 'r-', label='p0')
    ax.plot(tau1, R1, 'b-', label='p1')
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 120)
    ax.set_xlabel(r'$\mathrm{\tau}$ (s)', fontsize=12)
    ax.set_ylabel(r'$\mathrm{\rho_{auto}}$', fontsize=12)
    ax.grid()

    ### PSD
    # PSD = funcs.kaimal_u(freq[1:], uz, z, uStar)
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
    ax.set_xlim(0, 120)
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
