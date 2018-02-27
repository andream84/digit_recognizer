import random
import numpy as np

class Network(object):
    """docstring for Network"""
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a 

    def cost(self, train_data):
        return sum(np.sum(np.power(self.feedforward(x) - y, 2)) for (x, y) in train_data)

    def cost_derivative(self, output_act, y):
        return (output_act - y) 

    def backprop(self, x, y, lambd):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #feedforward propagation
        activation = x
        activations = [x] #list of all activations
        zs = [] #list of all z's
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)
        #backward propagation
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) + lambd * self.weights[-1]
        for j in range(2, self.num_layers):
            delta = np.dot(self.weights[-j+1].transpose(), delta) * sigmoid_prime(zs[-j])
            nabla_b[-j] = delta
            nabla_w[-j] = np.dot(delta, activations[-j-1].transpose()) + lambd * self.weights[-j]
        return  (nabla_b, nabla_w)

    def SGD(self, train_data, epochs, mini_batch_size, eta, lambd, test_data = None):
        n = len(train_data)
        if test_data: n_test = len(test_data)
        for j in range(epochs):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] 
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, eta, lambd)
            eta = eta * 0.8
            if test_data:
                print("Epoch {0} completed: {1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} completed".format(j))

    def update_parameters(self, mini_batch, eta, lambd):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            del_n_b, del_n_w = self.backprop(x, y, lambd)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, del_n_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, del_n_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, data):
        ypredy = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(yp == y) for (yp, y) in ypredy)


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


