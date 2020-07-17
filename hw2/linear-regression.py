import numpy as np
import matplotlib.pyplot as plt


def mse(x, y, w):
    return np.dot((np.dot(x, w) - y).T, np.dot(x, w) - y)


def closed_form(x_train, y_train):
    w = np.dot(np.dot(np.linalg.inv(np.dot(x_train.T, x_train)), x_train.T), y_train)
    j = mse(x_train, y_train, w)
    return w, j


def gradient_descent(x_train, y_train, max_iter, eta):
    w = w_temp = np.zeros(x_train.shape[1])
    num_iter = 0

    for i in range(max_iter):
        if max_iter == 0:
            break
        # Terminate if there is no change in J(w)
        elif abs(mse(x_train, y_train, w_temp) - mse(x_train, y_train, w)) < 0.0001 and i != 0:
            break
        w = w_temp
        w_temp = w - eta * np.matmul((np.dot(x_train,w) - y_train),x_train)
        num_iter += 1

    return w_temp, mse(x_train, y_train, w_temp),num_iter

def polynomial_regression(x_train,y_train,m):
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train.T, x_train)), x_train.T), y_train)
    j = mse(x_train, y_train, w)
    e = (mse(x_train,y_train,w)/x_train.shape[0])**(.5)
    return w,j,e

if __name__ == "__main__":
    data = np.genfromtxt('regression_train.csv', delimiter=',')
    test = np.genfromtxt('regression_test.csv', delimiter=',')

    x_train = np.array([np.ones(data.shape[0]), data[:, 0]]).T
    y_train = data[:, 1]

    x_test = np.array([np.ones(test.shape[0]), test[:, 0]]).T
    y_test = test[:, 1]

    # Closed-form solution
    # w, j = closed_form()

    # Gradient descent
    # w, j, num_iter = gradient_descent(x_train, y_train, max_iter=10, eta=0.0407)


    t = np.linspace(0.05, 0.95)
    """
    for i in range(5):
        w, j, num_iter = gradient_descent(x_train, y_train, max_iter=i*10, eta=0.0407)
        print('num of iterations = ', num_iter)
        print('w = ', w)
        print('J(w) = ', j)
        plt.plot(x, w[0] + w[1]* x, label=str(num_iter))
    """

    # Plynomial Regression
    ts = []
    tr = []
    for m in range(11):
        x = np.ones((x_train.shape[0], m + 1))
        for i in range(m):
            x[:, i + 1] = [t ** (i + 1) for t in x_train[:, 1]]
        w, j, e = polynomial_regression(x, y_train, m=m)
        for i in range(m):
            x[:, i + 1] = [t ** (i + 1) for t in x_test[:, 1]]
        e_test = (mse(x, y_test, w) / x.shape[0]) ** (.5)
        ts.append(e_test)
        tr.append(e)

    #plt.plot(t,np.poly1d(w[::-1])(t),c='firebrick',label='m = 10')
    plt.plot(tr,label='train')
    plt.plot(ts, label='test')
    plt.legend()
    #plt.scatter(data[:, 0], data[:, 1])
    plt.title('RMSE vs. m')
    plt.xlabel('m')
    plt.ylabel('RMSE')
    #plt.show()
    plt.savefig('RMSE_m')