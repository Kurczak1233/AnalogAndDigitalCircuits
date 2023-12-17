import numpy as np
import matplotlib.pyplot as plt

f = 400
A = 400.25
B = 399.75
N = 3000

t = np.arange(N)

signal = (A - B) * np.sin(2 * np.pi * f * t / N) + B

plt.plot(t, signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title(f'Signal with Frequency {f} Hz and Linearly Varying Amplitude')
plt.grid(True)
plt.show()

linear_mean = np.mean(signal)
print(f'Linear Mean (Ensemble Average): {linear_mean:.4f}')