import numpy as np
import matplotlib.pyplot as plt

# xµ = np.array([6, 2, 4, 3, 4, 5, 0, 0, 0, 0], dtype=complex)

# Uncomment array you would like to extract the signal from.
# xµ = np.array([10, 5, 6, 6, 2, 4, 3, 4, 5, 0, 0, 0, 0], dtype=complex)
# xµ = np.array([6, 2, 4, 3, 4, 5, 0, 0, 0, 0], dtype=complex)
# xµ = np.array([6, 2, 4, 3, 4, 5, 0, 0, 0], dtype=complex)
# xµ = np.array([6, 4, 4, 5, 3, 4, 5, 0, 0, 0, 0], dtype=complex)
# xµ = np.array([7, 2, 4, 3, 4, 5, 0, 0, 0, 0], dtype=complex)
# xµ = np.array([6, 8, 2, 4, 3, 4, 5, 0, 0, 0], dtype=complex)
# xµ = np.array([6, 2, 4, 4, 4, 5, 0, 0, 0, 0], dtype=complex)
# xµ = np.array([6, 5, 4, 3, 4, 5, 0, 0, 0, 0], dtype=complex)
# xµ = np.array([6, 2, 4, 3, 4, 4, 0, 0, 0, 0], dtype=complex)
# xµ = np.array([6, 2, 4, 3, 4, 5, 0, 0, 0, 0], dtype=complex)
# xµ = np.array([10, 5, 6, 6, 2, 4, 3, 4, 5, 0, 0, 0, 0], dtype=complex)
# xµ = np.array([6, 2, 4, 3, 4, 5, 0, 0, 0, 0], dtype=complex)
# xµ = np.array([6, 2, 4, 3, 4, 5, 0, 0, 0], dtype=complex)
xµ = np.array([6, 4, 4, 5, 3, 4, 5, 0, 0, 0, 0], dtype=complex)

N = len(xµ)

W = np.zeros((N, N), dtype=complex)
for i in range(N):
    for j in range(N):
        W[i, j] = np.exp(-1j * 2 * np.pi * i * j / N)

K = np.conjugate(np.transpose(W))

x = (1 / N) * np.dot(K, xµ)

print("Matrix W:")
print(W)
print("\nMatrix K:")
print(K)

plt.stem(np.real(x))
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Plotted Signal')
plt.grid(True)
plt.show()

