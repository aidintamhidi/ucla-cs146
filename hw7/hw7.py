
import numpy as np
import matplotlib.pyplot as plt


def read_png(filename):
    return plt.imread(filename)[:, :, :3] * 255


def min_dist(node, nodes):
    dist = np.sum((nodes - node) ** 2, axis=1) ** (0.5)
    return [np.argmin(dist), np.min(dist)]


def max_dist(node, nodes):
    dist = np.sum((nodes - node) ** 2, axis=1) ** (0.5)
    return [np.argmax(dist), np.max(dist)]


def remove(r, x):
    temp = []
    for i in x:
        if i[0] == r[0] and i[1] == r[1] and i[2] == r[2]:
            continue
        temp.append(i)
    return np.array(temp)


def initial_mu(x, m, k):
    mu = np.zeros((k, 3))
    mu[0] = np.array(m)
    x = remove(mu[0], x)
    mu[1] = x[max_dist(mu[0], x)[0]]
    x = remove(mu[1], x)

    for i in range(2, k):
        max_ = 0
        for p in x:
            if max_ == 0:
                c = mu[:i]
                max_ = min_dist(p, mu[:i])[1]
                val = p
                continue
            if min_dist(p, mu[:i])[1] > max_:
                max_ = min_dist(p, mu[:i])[1]
                val = p
        mu[i] = val
        remove(val, x)
    return mu


def k_means(k, iter, mu, x_val):
    j = np.zeros(iter)
    r = np.zeros((x_val.shape[0], k))

    for it in range(iter):

        # Assign cluster
        for i in range(x_val.shape[0]):
            c = min_dist(x_val[i], mu)[0]
            r[i][c] = 1

        # Update mean
        n = np.zeros(3)
        d = 0
        for i in range(k):
            for m in range(x_val.shape[0]):
                if r[m][i] == 1:
                    n += x_val[m]
                    d += 1
            mu[i] = n/d

        # Compute J
        for m in range(r.shape[0]):
            j[it] += np.sum((x_val[m] - mu[np.argmax(r[m])]) ** 2, axis=0)

    return j, r

def compress(x, r,mu,row,col):
    n = 0
    for i in range(row):
        for j in range(col):
            x[i,j] = mu[np.argmax(r[n])]
            n += 1

    return x


if __name__ == '__main__':
    m = [147, 200, 250]
    k = 16

    x = read_png('UCLA_Bruin.png')
    row = x.shape[0]
    col = x.shape[1]
    x_val = np.concatenate(x, axis=0)

    mu = initial_mu(x_val,m,k)

    j, r = k_means(k=k, iter=10, mu=mu, x_val=x_val)
    print (j)
    x_ = compress(x, r,mu,row,col)


    plt.imshow(x/255)
    plt.title('k = 4')
    plt.show()
    #plt.savefig('3_2_k16')

    plt.plot(j)
    # plt.show()
    plt.ylabel('J')
    plt.xlabel('# of iterations')
    plt.savefig('3_2')

    exit()
