import network
import numpy as np
import pandas as pd

#train_data = np.genfromtxt('train.csv', delimiter=',', skiprows=0)

def vectorized_result(num_class, y):
    yv = np.zeros((num_class,1))
    yv[y] = 1.0
    return yv


data = pd.read_csv('train.csv')

train_inputs = [np.reshape(x, (784, 1)) for x in data.iloc[:32000,1:].values.tolist()]
train_results = [vectorized_result(10, y) for y in data.iloc[:32000,0].values.tolist()]
train_data = list(zip(train_inputs, train_results))

test_inputs = [np.reshape(x, (784, 1)) for x in data.iloc[32000:,1:].values.tolist()]
test_data = list(zip(test_inputs, data.iloc[32000:,0].values.tolist()))

NN = network.Network([784, 30, 10])

NN.SGD(train_data, 20, 10, 0.1, 0.003, test_data=test_data)
