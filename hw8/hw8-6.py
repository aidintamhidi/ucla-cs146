import numpy as np
import matplotlib.pyplot as plt

# Read the data
data = np.genfromtxt('AdaBoost_data.csv', delimiter=',')
x = data[:, :2]
y = data[:, 2]

# Ada algorithm
K = 3
N = x.shape[0]

s = np.array([-1, -1, 1])
i = np.array([0, 0, 1])

t = np.zeros(K)
al = np.zeros(K)
d = np.zeros((K + 1, N))
d[0] = np.full((N), 1 / N)

for k in range(K):

    # Train the classifier
    start = int(np.ceil(np.min(x[:, i[k]])))
    end = int(np.floor(np.max(x[:, i[k]])))
    err = N
    for p in range(start, end+1):
        # Compute the error
        pred = np.zeros(N)
        err_temp = 0
        for n in range(N):
            pred[n] = np.sign(s[k] * (x[n, i[k]] - p))
            err_temp += d[k, i[k]] if pred[n] * y[n] == -1 else 0

        # If the new err is smaller than the previous one, keep new t
        if err_temp < err:
            err = err_temp
            t[k] = p
            y_pred = pred

    # Compute alpha
    al[k] = 0.5 * np.log(1 / err - 1)

    # Update d
    d[k + 1] = d[k] * np.exp(-al[k] * y_pred * y)
    d[k + 1] /= np.sum(d[k + 1])

    print('t = ', t[k])
    print('alpha =', al[k])
    print('d =', d[k + 1])

# Compute err of combined classifier
acc = 0
for n in range(N):
    temp = 0
    for k in range(K):
        temp += al[k] * np.sign(s[k] * (x[n, i[k]] - t[k]))
    y_pred = np.sign(temp)
    acc += 1 if y_pred * y[n] == 1 else 0

# Plot the data
for n in range(N):
    c = 'blue' if y[n] < 0 else 'red'
    plt.scatter(x[n, 0], x[n, 1],color=c,alpha=0.5)

#plt.axhline(t[2])
plt.title('k = 3')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()
#plt.savefig('plotb')

exit()
