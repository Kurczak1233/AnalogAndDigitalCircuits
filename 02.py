import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft , ifft , fftshift
from scipy.signal.windows import hann , flattop

f1 = 400 # Hz
f2 = 400.25 # Hz
f3 = 399.75 # Hz
fs = 600 # Hz
N = 3000
k = np.arange(N)
x1 = np.sin(2*np.pi*f1/fs*k)
x2 = np.sin(2*np.pi*f2/fs*k)
x3 = np.sin(2*np.pi*f3/fs*k)

wrect = np.ones(N)
whann = hann(N, sym=False)
wflattop = flattop(N, sym=False)
plt.plot(wrect, 'C0o-', ms=3, label='rect')
plt.plot(whann, 'C1o-', ms=3, label='hann')
plt.plot(wflattop, 'C2o-', ms=3, label='flattop')
plt.xlabel(r'$k$')
plt.ylabel(r'window$w[k]$')
plt.xlim(0, N)
plt.legend()
plt.grid(True)

X1wrect = fft(x1)
X2wrect = fft(x2)
X3wrect = fft(x3)
X1whann = fft(x1 * whann)
X2whann = fft(x2 * whann)
X3whann = fft(x3 * whann)
X1wflattop = fft(x1 * wflattop)
X2wflattop = fft(x2 * wflattop)
X3wflattop = fft(x3 * wflattop)

# this handling is working for N even and odd:
def fft2db(X):
    N = X.size
    Xtmp = 2/N * X  # independent of N, norm for sine amplitudes
    Xtmp[0] *= 1/2  # bin for f=0 Hz is existing only once,
    # so can cancel *2 from above
    if N % 2 == 0:  # fs/2 is included as a bin
        # fs/2 bin is existing only once, so can cancel *2 from above
        Xtmp[N//2] = Xtmp[N//2] / 2
    return 20 * np.log10(np.abs(Xtmp))  # in dB

# set up of frequency vector this way is independent of N even/odd:
df = fs / N
f = np.arange(N) * df

plt.figure(figsize=(16/1.5, 10/1.5))
plt.subplot(3, 1, 1)
plt.plot(f, fft2db(X1wrect), 'C0o-', ms=3, label='best case rect')
plt.plot(f, fft2db(X2wrect), 'C2o-', ms=3, label='central case rect')
plt.plot(f, fft2db(X3wrect), 'C3o-', ms=3, label='worst case rect')
plt.xlim(175, 225)
plt.ylim(-60, 0)
plt.xticks(np.arange(175, 230, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(f, fft2db(X1whann), 'C0o-', ms=3, label='best case hann')
plt.plot(f, fft2db(X2whann), 'C2o-', ms=3, label='central case hann')
plt.plot(f, fft2db(X3whann), 'C3o-', ms=3, label='worst case hann')
plt.xlim(175, 225)
plt.ylim(-60, 0)
plt.xticks(np.arange(175, 230, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(f, fft2db(X1wflattop), 'C0o-', ms=3, label='best case flattop')
plt.plot(f, fft2db(X2wflattop), 'C2o-', ms=3, label='central case flattop')
plt.plot(f, fft2db(X3wflattop), 'C3o-', ms=3, label='worst case flattop')
plt.xlim(175, 225)
plt.ylim(-60, 0)
plt.xticks(np.arange(175, 230, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.xlabel('f / Hz')
plt.ylabel('A / dB')
plt.grid(True)
plt.show()


# def winDTFTdB(w):
#     N = w.size  # get window length
#     Nz = 100 * N  # zero-padding length
#     W = np.zeros(Nz)  # allocate RAM
#     W[0:N] = w  # insert window
#     W = np.abs(fftshift(fft(W)))  # fft, fftshift and magnitude
#     W /= np.max(W)  # normalize to maximum, i.e., the mainlobe maximum here
#     W = 20 * np.log10(W)  # get level in dB
#     # get appropriate digital frequencies
#     Omega = 2 * np.pi / Nz * np.arange(Nz) - np.pi  # also shifted
#     return Omega, W

# plt.plot([-np.pi, +np.pi], [-3.01, -3.01], 'gray')  # mainlobe bandwidth
# plt.plot([-np.pi, +np.pi], [-13.3, -13.3], 'gray')  # rect max sidelobe
# plt.plot([-np.pi, +np.pi], [-31.5, -31.5], 'gray')  # hann max sidelobe
# plt.plot([-np.pi, +np.pi], [-93.6, -93.6], 'gray')  # flattop max sidelobe

# Omega, W = winDTFTdB(wrect)
# plt.plot(Omega, W, label='rect')
# Omega, W = winDTFTdB(whann)
# plt.plot(Omega, W, label='hann')
# Omega, W = winDTFTdB(wflattop)
# plt.plot(Omega, W, label='flattop')
# plt.xlim(-np.pi, np.pi)
# plt.ylim(-120, 10)
# plt.xlim(-np.pi/100, np.pi/100)  # zoom into mainlobe
# plt.xlabel(r'$\Omega$')
# plt.ylabel(r'$|W(\Omega)|$ / dB')
# plt.legend()
# plt.grid(True)


# import numpy as np
# import matplotlib.pyplot as plt
# from numpy.fft import fft
# from scipy.signal.windows import hann, flattop

# # Define signal parameters
# f1 = 400  # Hz
# f2 = 400.25  # Hz
# f3 = 399.75  # Hz
# fs = 600  # Hz
# N = 3000
# k = np.arange(N)
# x1 = np.sin(2 * np.pi * f1 / fs * k)
# x2 = np.sin(2 * np.pi * f2 / fs * k)
# x3 = np.sin(2 * np.pi * f3 / fs * k)

# # Define window functions
# wrect = np.ones(N)
# whann = hann(N, sym=False)
# wflattop = flattop(N, sym=False)

# # Define function to compute FFT and normalize to dB
# def fft2db(X):
#     N = X.size
#     Xtmp = 2 / N * X
#     Xtmp[0] *= 1/2
#     if N % 2 == 0:
#         Xtmp[N // 2] = Xtmp[N // 2] / 2
#     return 20 * np.log10(np.abs(Xtmp))

# # Calculate FFT for different signals and windows
# X1wrect = fft(x1 * wrect)
# X2wrect = fft(x2 * wrect)
# X3wrect = fft(x3 * wrect)
# X1whann = fft(x1 * whann)
# X2whann = fft(x2 * whann)
# X3whann = fft(x3 * whann)
# X1wflattop = fft(x1 * wflattop)
# X2wflattop = fft(x2 * wflattop)
# X3wflattop = fft(x3 * wflattop)

# # Define frequency vector
# df = fs / N
# f = np.arange(N) * df

# # Create subplots for results
# plt.figure(figsize=(16/1.5, 10/1.5))
# plt.subplot(3, 1, 1)
# plt.plot(f, fft2db(X1wrect), 'C0o-', ms=3, label='best case rect')
# plt.plot(f, fft2db(X2wrect), 'C3o-', ms=3, label='worst case rect')
# plt.plot(f, fft2db(X3wrect), 'C3o-', ms=3, label='worst case rect')
# plt.xlim(175, 225)
# plt.ylim(-60, 0)
# plt.xticks(np.arange(175, 230, 5))
# plt.yticks(np.arange(-60, 10, 10))
# plt.legend()
# plt.ylabel('A / dB')
# plt.grid(True)
# plt.subplot(3, 1, 2)
# plt.plot(f, fft2db(X1whann), 'C0o-', ms=3, label='best case hann')
# plt.plot(f, fft2db(X2whann), 'C3o-', ms=3, label='worst case hann')
# plt.plot(f, fft2db(X3whann), 'C3o-', ms=3, label='worst case hann')
# plt.xlim(175, 225)
# plt.ylim(-60, 0)
# plt.xticks(np.arange(175, 230, 5))
# plt.yticks(np.arange(-60, 10, 10))
# plt.legend()
# plt.ylabel('A / dB')
# plt.grid(True)
# plt.subplot(3, 1, 3)
# plt.plot(f, fft2db(X1wflattop), 'C0o-', ms=3, label='best case flattop')
# plt.plot(f, fft2db(X2wflattop), 'C3o-', ms=3, label='worst case flattop')
# plt.plot(f, fft2db(X3wflattop), 'C3o-', ms=3, label='worst case flattop')
# plt.xlim(175, 225)
# plt.ylim(-60, 0)
# plt.xticks(np.arange(175, 230, 5))
# plt.yticks(np.arange(-60, 10, 10))
# plt.legend()
# plt.xlabel('f / Hz')
# plt.ylabel('A / dB')
# plt.grid(True)

# plt.show()
