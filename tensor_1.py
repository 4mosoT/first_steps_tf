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

housing_data_plus_bias = sklearn.preprocessing.StandardScaler().fit_transform(housing_data_plus_bias)

n_epochs = 1000
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size, data, data_y):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(data.shape[0], size=batch_size)
    X_batch = data[indices]
    y_batch = data_y.reshape(-1, 1)[indices]
    return X_batch, y_batch


# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# Manual gradients
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)

# Autodiff gradients
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients)

# Using optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

MSE_VALUES = []
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch, batch_size, housing_data_plus_bias, housing.target)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        mse_ev = mse.eval(feed_dict={X: X_batch, y: y_batch})
        MSE_VALUES.append(mse_ev)
        if epoch % 100 == 0:
            print("Epoch {} MSE {}".format(epoch, mse_ev))


    best_theta = theta.eval()
    print(best_theta)

plt.plot(MSE_VALUES)
plt.show()
