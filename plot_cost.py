import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

path = os.getcwd() + '/ex1'
df = pd.read_csv(path + '/ex1data1.txt', names=['X', 'Y'])
X = np.array(df.X).reshape(97, 1)
Y = np.array(df.Y).reshape(97, 1)
m = len(Y)

plt.scatter(X, Y, c='g', alpha=0.5)

# Insert column of ones into X
theta = np.zeros(2)
X = np.c_[np.ones(m), X]


def computeCost(x, y, t):
    hyp = np.dot(x, t)
    l = (hyp - y)
    l = l**2
    j = np.sum(l) / (2 * m)
    return j


# Gradient descent
aplpha = 0.01
for iter in range(1500):
    hypothesis = np.dot(X, theta)
    loss = (hypothesis - df.Y)
    J = np.sum(loss**2) / (2 * m)

    gradient0 = np.dot(X[:, 0], loss) / m
    theta[0] = theta[0] - aplpha * gradient0

    gradient1 = np.dot(X[:, 1], loss) / m
    theta[1] = theta[1] - aplpha * gradient1

plt.plot(X[:, 1], np.dot(X, theta), '-')
print(theta)

# plotting cost function
fig = plt.figure(figsize=(9, 9))
ax = fig.gca(projection='3d')
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = computeCost(X, df['Y'], t)
        #print(computeCost(X, df['Y'], t))

Xs, Ys = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.reshape(J_vals, Xs.shape)

ax.set_xlabel(r't0')
ax.set_ylabel(r't1')
ax.set_zlabel(r'cost')
ax.view_init(elev=25, azim=40)
surf = ax.plot_surface(Xs, Ys, J_vals, cmap=cm.rainbow)

ax = plt.figure().gca()
theta = np.zeros(2)
ax.plot(theta[0], theta[1], 'r*')
plt.contour(Xs, Ys, J_vals, np.logspace(-3, 3, 15))

plt.show()
