# XOR problem - manual vs TensorFlow Keras NN implementation

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import SGD

##### manual #####

# 2 input nodes (input1, input2)
# 1 hidden layer with 2 nodes (hidden1, hidden2)
# 1 output node (output)

# w1 -> input1 to hidden1
# w2 -> input1 to hidden2
# w3 -> input2 to hidden1
# w4 -> input2 to hidden2
# b1 -> hidden1 bias
# b2 -> hidden2 bias
# w5 -> hidden1 to output
# w6 -> hidden2 to output
# b3 -> output bias

# samples
X = [[0,0],
     [0,1],
     [1,0],
     [1,1]]

y = [[0],
     [1],
     [1],
     [0]]

def tanh(x):
    return np.tanh(x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# initialising weights and biases
w1, w2, w3, w4 = [-0.02998602, -0.02283859, 1.1968023, -0.7734521]
w5, w6 = [-0.65623754, 0.98991454]
b1, b2 = [0., 0.]
b3 = 0

# feedforward
def feedforward(x):
    # i1, i2 -> input1 and input2
    # z1, z2 -> hidden1 nand hidden2 net (before activation)
    # a1, a2 -> hidden1 and hidden2 (after activation)
    # zo -> output net (before activation)
    # o -> output (after activation)

    # activation functions
    def tanh(x): return np.tanh(x)
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    
    i1, i2 = x
    z1 = i1 * w1 + i2 * w3 + b1
    a1 = tanh(z1)
    z2 = i1 * w2 + i2 * w4 + b2
    a2 = tanh(z2)
    zo = a1 * w5 + a2 * w6 + b3
    o = sigmoid(zo)

    return i1, i2, z1, a1, z2, a2, zo, o

# backpropogate
def backpropagate(i1, i2, z1, a1, z2, a2, zo, o, y):
    # binary cross-entropy loss function
    # calcuate rate of change of loss function with respect to each weight and bias
    # d*_d^ -> derivative of * with respect to ^ 

    dL_do = (o - y)/(o*(1 - o)) # derivative of binary cross-entropy loss function (dL) with respect to output
    do_zo = o*(1 - o) # derivative of sigmoid
    dzo_dw5, dzo_dw6, dzo_db3 = a1, a2, 1
    dL_dw5 = dL_do * do_zo * dzo_dw5 # dL with respect to w5
    dL_dw6 = dL_do * do_zo * dzo_dw6 # dL with respect to w6
    dL_db3 = dL_do * do_zo * dzo_db3 # dL with respect to b3

    dzo_da2 = w6
    da2_dz2 = 1 - a2**2 # derivative of tanh
    dz2_dw2, dz2_dw4, dz2_db2 = i1, i2, 1
    dL_dw2 = dL_do * do_zo * dzo_da2 * da2_dz2 * dz2_dw2 # dL with respect to w2
    dL_dw4 = dL_do * do_zo * dzo_da2 * da2_dz2 * dz2_dw4 # dL with respect to w4
    dL_db2 = dL_do * do_zo * dzo_da2 * da2_dz2 * dz2_db2 # dL with respect to b2

    dzo_da1 = w5
    da1_dz1 = 1 - a1**2 # derivative of tanh
    dz1_dw1, dz1_dw3, dz1_db1 = i1, i2, 1
    dL_dw1 = dL_do * do_zo * dzo_da1 * da1_dz1 * dz1_dw1 # dL with respect to w1
    dL_dw3 = dL_do * do_zo * dzo_da1 * da1_dz1 * dz1_dw3 # dL with respect to w3
    dL_db1 = dL_do * do_zo * dzo_da1 * da1_dz1 * dz1_db1 # dL with respect to b1

    return dL_dw1, dL_dw2, dL_dw3, dL_dw4, dL_dw5, dL_dw6, dL_db1, dL_db2, dL_db3

# fit single sample and update weights and biases
def fit_sample(sample_x, sample_y, learning_rate):
    # feedforward
    results = feedforward(sample_x)
    # backpropagate
    dL_dw1, dL_dw2, dL_dw3, dL_dw4, dL_dw5, dL_dw6, dL_db1, dL_db2, dL_db3 = backpropagate(*results, sample_y)
    # gradient descent (update weights and biases)
    global w1, w2, w3, w4, w5, w6, b1, b2, b3
    w1 = w1 - learning_rate*dL_dw1
    w2 = w2 - learning_rate*dL_dw2
    w3 = w3 - learning_rate*dL_dw3
    w4 = w4 - learning_rate*dL_dw4
    w5 = w5 - learning_rate*dL_dw5
    w6 = w6 - learning_rate*dL_dw6
    b1 = b1 - learning_rate*dL_db1
    b2 = b2 - learning_rate*dL_db2
    b3 = b3 - learning_rate*dL_db3

# fit all samples with batch size of 1 (i.e. update weights and biases every sample)
def fit(X, y, epochs, learning_rate):
    for _ in range(epochs):
        for i, x_sample in enumerate(X):
            y_sample = y[i][0]
            fit_sample(x_sample, y_sample, learning_rate)

##### TensorFlow Keras #####

keras.utils.set_random_seed(1337) # set same initial weights and biases as manual implementation

model = Sequential()
model.add(InputLayer((2,))) # 2 input nodes
model.add(Dense(2, activation='tanh')) # 1 hidden layer with 2 nodes
model.add(Dense(1, activation='sigmoid')) # 1 output node
model.compile(
    loss = 'binary_crossentropy',
    optimizer = SGD(learning_rate=0.1)
)

##### comparison #####
# comparing feedforward results of manual and TensorFlow Keras model after fitting with 1000 epochs

fit(X, y, epochs=1000, learning_rate=0.1) # manual model
model.fit(X, y, batch_size=1, epochs=1000) # TensorFlow Keras model

results = [[
    feedforward(sample_x)[-1], # predicted y (manual)
    model.predict([sample_x])[0, 0], # predicted y (TensorFlow Keras)
    y[i][0] # actual y
] for i, sample_x in enumerate(X)]

print(pd.DataFrame(results, columns=['Manual', 'Keras', 'Actual'], index=['0, 0', '0, 1', '1, 0', '1, 1']))
