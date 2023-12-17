import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64
from scipy import stats

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

edges = hist_estimate[1]
freq = hist_estimate[0]

theta_num = edges[:-1]
dtheta = np.diff(edges)

linear_mean = np.sum(theta_num * freq * dtheta)

quadratic_mean = np.sum((theta_num - linear_mean)**2 * freq * dtheta)
variance = quadratic_mean  

print('Quadratic mean estimate: %5.2f' % quadratic_mean)
print('Variance estimate: %5.2f' % variance)

plt.show()
