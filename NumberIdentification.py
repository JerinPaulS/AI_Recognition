import urllib.request
import gzip
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T
from PIL import Image
import dill as pickle

SOURCE = "http://yann.lecun.com/exdb/mnist/"

def load_dataset():
    def download(filename):
        print("Downloading the file: ", filename)
        urllib.request.urlretrieve(SOURCE + filename, filename)

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset = 16)
            data = data.reshape(-1, 1, 28, 28)
            return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset = 8)
        return data

    X_train = load_mnist_images("train-images-idx3-ubyte.gz")
    Y_train = load_mnist_labels("train-labels-idx1-ubyte.gz")
    X_test = load_mnist_images("t10k-images-idx3-ubyte.gz")
    Y_test = load_mnist_labels("t10k-labels-idx1-ubyte.gz")

    return X_train, Y_train, X_test, Y_test

def biuld_NN(input_var = None):

    layer_inp = lasagne.layers.InputLayer(shape = (None, 1, 28, 28), input_var = input_var)
    #layer_inp_drop = lasagne.layers.DropoutLayer(layer_inp, p = 0.2)
    layer_hidden1 = lasagne.layers.DenseLayer(layer_inp, num_units = 800, nonlinearity = lasagne.nonlinearities.rectify, W = lasagne.init.GlorotUniform())
    #layer_hidden1_drop = lasagne.layers.DropoutLayer(layer_hidden1, p = 0.5)
    layer_hidden2 = lasagne.layers.DenseLayer(layer_hidden1, num_units = 800, nonlinearity = lasagne.nonlinearities.rectify, W = lasagne.init.GlorotUniform())
    #layer_hidden2_drop = lasagne.layers.DropoutLayer(layer_hidden2, p = 0.5)
    layer_out = lasagne.layers.DenseLayer(layer_hidden2, num_units = 10, nonlinearity = lasagne.nonlinearities.softmax)

    return layer_out


X_train, Y_train, X_test, Y_test = load_dataset()
'''
plotData = X_test[1]
plotData = plotData.reshape(28, 28)
plt.gray()
plt.imshow(plotData)
plt.show()
'''
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

network = biuld_NN(input_var)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable = True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate = 0.01, momentum = 0.9)

train_fn = theano.function([input_var, target_var], loss, updates = updates)

num_training_steps = 10

for step in range(num_training_steps):
    train_err = train_fn(X_train, Y_train)
    print("Training the model! Epoch: " + str(step + 1))

test_prediction = lasagne.layers.get_output(network, deterministic = True)
val_fn = theano.function([input_var], test_prediction)
print(np.argmax(val_fn([X_test[0]])))

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis = 1), target_var), dtype = theano.config.floatX)
acc_fn = theano.function([input_var, target_var], test_acc)
print(acc_fn(X_test, Y_test))

np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
with open('model.dpkl','wb') as p_output:
   pickle.dump(network, p_output)
