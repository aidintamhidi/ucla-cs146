import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# Read data
data = np.genfromtxt('Data.csv', delimiter=',')
y_positive = data[data[:, 2] > 0]
y_negative = data[data[:, 2] < 0]

y = data[:, 2]
x = data[:, :2]


# Prime problem
w = cp.Variable(data.shape[1] - 1)
b = cp.Variable()
objective = cp.Minimize(0.5 * cp.norm(w, 2))
constraints = [y[i] * (w.T * x[i] + b) >= 1 for i in range(y.shape[0])]
prob = cp.Problem(objective, constraints)
prob.solve()
w_p = w.value
b_p = b.value


# Dual problem
temp = []
for i in range (y.shape[0]):
    temp.append(y[i] * x[i])
P = np.dot(np.array(temp),np.array(temp).T) + 1e-13 * np.eye(y.shape[0])

a = cp.Variable(y.shape[0])
objective = cp.Maximize(cp.sum(a) - 0.5 * cp.quad_form(a, P))
constraints = [i >= 0 for i in a] + [a.T * y == 0]
prob = cp.Problem(objective, constraints)
prob.solve()
a_d = a.value
idx = np.argwhere(a_d >= 1e-9)

# Plot
plt.scatter(y_negative[:, 0], y_negative[:, 1], label='y = -1')
plt.scatter(y_positive[:, 0], y_positive[:, 1], label='y = +1')

for i in range(y.shape[0]):
    if [i] in idx:
        plt.scatter(x[i][0], x[i][1], color='red')
n = np.linspace(0, 10)
plt.plot(n,(-w_p[0] * n - b_p) / w_p[1], c='firebrick')

plt.legend()
#plt.show()
#plt.savefig('6_c')
exit()

"""

# For Q5
n = np.array([[-1,1],[0,0],[1,1]])
p = np.array([[-3,9],[-2,4],[3,9]])
n0= np.array([[-1,0],[0,0],[1,0]])
p0 = np.array([[-3,0],[-2,0],[3,0]])
plt.scatter(n0[:,0],n0[:,1],label='negative')
plt.scatter(p0[:,0],p0[:,1],label='positive')
x=np.linspace(-3,3)
plt.plot(x,(x+9)/3)

plt.xlabel('x')
#plt.ylabel('x^2')
plt.legend()
#plt.show()
plt.savefig('5_e')
"""

"""
x = np.arange(-25, 25, 0.01)

plt.plot(x, np.exp(x*x / (-1 * 4)) * (1 / np.sqrt(4 * 3.14)), '-', label='t = 1')
plt.plot(x, np.exp(x*x / (-1 * 8)) * (1 / np.sqrt(8 * 3.14)), '--',label='t = 2')
plt.plot(x, np.exp(x*x / (-1 * 40)) * (1 / np.sqrt(40 * 3.14)),':', label='t = 10')
plt.plot(x, np.exp(x*x / (-1 * 120)) * (1 / np.sqrt(120 * 3.14)),'-.', label='t = 30')

x = np.arange(0, 1, 0.001)
plt.plot(x, 100 * (np.exp(2) - np.exp(2*x)))


#plt.legend()
#plt.show()
plt.savefig('plot1')

"""