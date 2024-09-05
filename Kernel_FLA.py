#Ejemplo de Kernel PCA
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, random_state=123)
XKPCA = KernelPCA(n_components=2, kernel='rbf', gamma=15).fit_transform(X)

plt.figure(figsize=(16, 6))

# figure 1
plt.subplot(1, 2, 1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='brown', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='pink', alpha=0.5)

plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$x_2$", fontsize=16)

plt.show()

#Ejemplo de Kernel PCA
plt.figure(figsize=(16, 6))

# figure 2
plt.subplot(1, 2, 2)
plt.scatter(XKPCA[y == 0, 0], XKPCA[y == 0, 1], color='red', alpha=0.5)
plt.scatter(XKPCA[y == 1, 0], XKPCA[y == 1, 1], color='blue', alpha=0.5)

plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$x_2$", fontsize=16)

plt.show()

X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
XKPCA = KernelPCA(n_components=2, kernel='rbf', gamma=15).fit_transform(X)

#Ejemplo de Kernel PCA
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', alpha=0.5)

plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$x_2$", fontsize=16)

plt.show()

#Ejemplo de Kernel PCA
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 2)
plt.scatter(XKPCA[y == 0, 0], XKPCA[y == 0, 1], color='red', alpha=0.5)
plt.scatter(XKPCA[y == 1, 0], XKPCA[y == 1, 1], color='blue', alpha=0.5)

plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$x_2$", fontsize=16)

plt.show()
