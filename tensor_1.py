import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
import sklearn
import matplotlib.pyplot as plt

housing = fetch_california_housing()
m, n = housing.data.shape
print(housing.data.shape)
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
print(housing_data_plus_bias.shape)

scaler = sklearn.preprocessing.StandardScaler()
normalized_data = scaler.fit_transform(housing_data_plus_bias)

n_epochs = 2000
learning_rate = 0.1

X = tf.constant(normalized_data, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2 / m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

MSE_VALUES = []
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        mse_ev = mse.eval()
        MSE_VALUES.append(mse_ev)
        if epoch % 100 == 0:
            print("Epoch {} MSE {}".format(epoch, mse_ev))
        sess.run(training_op)

    best_theta = theta.eval()
    print(best_theta)

plt.plot(MSE_VALUES)
plt.show()
