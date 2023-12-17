import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64
from scipy import signal, stats

# Function for cross-correlation
def my_xcorr(x, y):
    N, M = len(x), len(y)
    kappa = np.arange(N + M - 1) - (M - 1)
    ccf = signal.correlate(x, y, mode='full', method='auto')
    return kappa, ccf

# Set seed for reproducible results
seed = 1234
np.random.seed(seed)

# Create random process based on normal distribution
Ns = 2**10  # Number of sample functions for e.g. time instance k=0
loc, scale = 5, 3  # mu, sigma

theta = np.arange(-15, 25, 0.01)  # Amplitudes for plotting PDF
# Random process object with normal PDF
rv = stats.norm(loc=loc, scale=scale)
# Get random data from sample functions
x = np.random.normal(loc=loc, scale=scale, size=Ns)

# Plot
fig, ax = plt.subplots(1, 1)
hist_estimate = ax.hist(x, bins='auto', density=True, histtype='bar',
                        color='C0', alpha=0.5, label='histogram')
ax.plot(theta, rv.pdf(theta), 'C0-', lw=2, label='pdf')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\hat{p}_x(\theta,k=0)$')
ax.set_title('Normalized histogram = PDF estimate')
ax.set_xlim(-15, 25)
ax.legend()
ax.grid(True)

# Get histogram data from ax.hist()
edges = hist_estimate[1]
freq = hist_estimate[0]

# Simple ensemble averages by numeric integration
# Over histogram data as a simple estimate of the pdf
theta_num = edges[:-1]
dtheta = np.diff(edges)

# Linear mean
linear_mean = np.sum(theta_num * freq * dtheta)

# Quadratic mean and Variance estimates
quadratic_mean = np.sum((theta_num - linear_mean)**2 * freq * dtheta)  # Quadratic mean estimate
variance = quadratic_mean  # Since the linear mean is subtracted, quadratic mean is equal to variance

print('Linear mean estimate: %5.2f' % linear_mean)
print('Quadratic mean estimate: %5.2f' % quadratic_mean)
print('Variance estimate: %5.2f' % variance)

# Estimate and plot the auto-correlation function (ACF)
kappa, acf = my_xcorr(x - linear_mean, x - linear_mean)
plt.figure()
plt.stem(kappa, acf, linefmt='C0-', markerfmt='C0o', basefmt='C0:')
plt.xlabel(r'$\kappa$')
plt.ylabel(r'ACF$(\kappa)$')
plt.title('Auto-Correlation Function (ACF) Estimate')
plt.grid(True)
plt.show()
