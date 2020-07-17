import numpy as np
import matplotlib.pyplot as plt

p = 28
# Read data and comppute the cov matrix
x = np.genfromtxt('MNIST3.csv', delimiter=',')
mu = x.mean(axis=0)
r = x.shape[0]
c = x.shape[1]
cov = np.cov(x.T) * (1 - 1 / r) # Because in the given formula in the spec, devided by N

# Find eigenvalues/vectors
w, v = np.linalg.eig(cov)
w = w.real
v = v.real
idx = np.argsort(w)[::-1]

# Pick first 4 eigenvectors
"""
v4 = np.zeros((4, p, p))
for i in range(4):
    temp = v[:, idx[i]]
    temp = np.interp(temp, (temp.min(), temp.max()), (0, 255))
    temp = np.reshape(temp,newshape=(p,p)).T
"""

# Compress the top left image with m eigenvectors
x_c = mu
m = 250
for i in range(m):
    x_c += (np.dot(x[0], v[:, idx[i]]) - np.dot(mu, v[:, idx[i]])) * v[:, idx[i]]

x_c = np.interp(x_c, (x_c.min(), x_c.max()), (0, 255))
x_c = np.reshape(x_c,newshape=(p,p)).T

# Plot a vector
plt.imshow(x_c,cmap='gray')
plt.title('M = 250')
#plt.show()
plt.savefig('x_c1')
exit()
