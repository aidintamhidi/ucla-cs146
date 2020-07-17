import numpy as np
import matplotlib.pyplot as plt


def readfile(filepath):
    data = np.genfromtxt(filepath, delimiter=',')
    x = np.vstack(([data[:, 0]], [data[:, 1]])).T
    y = data[:, 2]
    return x, y


def perceptron_algo(D, MaxIter, x_train, y_train):
    # Initialize weights and bias to zero
    w = np.zeros(D)
    b = 0
    u = 0

    for i in range(MaxIter):
        for j in range(len(y_train)):
            y_pred = np.dot(w.T, x_train[j]) + b
            if y_train[j] * y_pred <= 0:
                w += y_train[j] * x_train[j]
                b += y_train[j]
                u += 1
    return w, b, u


def margine(x,w,b):
    gamma = 0
    idx = 0
    for i in range(x.shape[0]):
        d = abs(np.dot(w.T,x[i])/np.linalg.norm(w))
        if i == 0:
            gamma = d
        elif d < gamma:
            gamma = d
            idx = i

    return gamma, idx


def main():
    data = 'data3.csv'

    x_train, y_train = readfile(data)
    D = x_train.shape[1]
    w, b, u = perceptron_algo(D, 1000, x_train, y_train)
    g, closest_idx = margine(x_train, w, b)
    print('w = ', w, '\nb = ', b, '\nu = ', u, '\ngamma = ', g, '\n1/gamma^2 = ', 1/(g*g))

    # Plot the hyperplane and points
    plt.title('data3')
    for i in range(len(y_train)):
        m = 'o' if y_train[i] == 1 else 'x'
        c = 'firebrick' if i == closest_idx else 'dimgray'
        plt.scatter(x_train[i, 0], x_train[i, 1], color=c, marker=m)
    x = np.linspace(-0.6, 0.75)
    plt.plot(x, -(w[0] / w[1]) * x, c='firebrick')
    plt.savefig('data2_g')
    plt.show()

if __name__ == "__main__":
    main()
