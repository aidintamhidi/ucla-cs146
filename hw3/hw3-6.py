import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter
import
as plt


def read_file(path):
    x = np.genfromtxt(path + '_X.csv', delimiter=',')
    y = np.genfromtxt(path + '_Y.csv', delimiter=',')

    return x, y


def majority(y_test, y_train):
    c_train = Counter(y_train)
    c_test = Counter(y_test)

    maj = c_train.most_common(1)[0]

    b_acc_train = maj[1] / y_train.shape[0]
    b_acc_test = c_test[maj[0]] / y_test.shape[0]

    return b_acc_train, b_acc_test


def decisionTree(x_train, y_train, x_test, y_test):
    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(x_train, y_train)

    acc_train = accuracy_score(y_train, dt.predict(x_train))
    acc_test = accuracy_score(y_test, dt.predict(x_test))
    cross_val = cross_val_score(dt, x_train, y_train, cv=10).mean()

    return acc_train, acc_test, cross_val


def knn(x_train, y_train, x_test, y_test,i):
    kn = KNeighborsClassifier(n_neighbors=i)
    kn.fit(x_train, y_train)

    sum = 0
    c = 0
    for i in kn.predict(x_train):
        sum += int(i == y_train[c])
        c += 1
    sum /= y_train.shape[0]

    acc_train = sum
    # acc_train = accuracy_score(y_train, kn.predict(x_train))
    acc_test = accuracy_score(y_test, kn.predict(x_test))
    cross_val = cross_val_score(kn, x_train, y_train, cv=10).mean()

    return acc_train, acc_test, cross_val


if __name__ == "__main__":
    x_train, y_train = read_file('dataTraining')
    x_test, y_test = read_file('dataTesting')

    # Majority
    # acc_train, acc_test = majority(y_test,y_train)

    # Decision Tree
    #acc_train, acc_test, acc_cross = decisionTree(x_train, y_train, x_test, y_test)

    # KNN
    """
    v_acc = []
    t_acc = []
    for i in range(1,51):
        acc_train, acc_test, acc_cross = knn(x_train, y_train, x_test, y_test,i)
        v_acc.append(1-acc_cross)
        t_acc.append(1-acc_test)
    """
    acc_train, acc_test, acc_cross = knn(x_train, y_train, x_test, y_test, 1)


    """
    k=np.linspace(1,50,50)
    plt.plot(k,v_acc,label='validation error')
    plt.plot(k,t_acc,label='test error')
    plt.legend()

    plt.xlabel('k')
    plt.savefig('50')
    """