import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

#Importing data
data = pd.read_csv('digit-recognizer/train.csv')

#pandas dataframe --> numpy array
data = np.array(data) 
m, n = data.shape #dimensions of the data
np.random.shuffle(data)

#Split data for validation (10%) & training (90%)
val_data = data[0:1000].T #First 1000 inputs & transpose
Y_val = val_data[0]
X_val = val_data[1:n]
X_val = X_val / 255.

#Taking the rest of the split for training
train_data = data[1000:m].T
Y_train = train_data[0]
X_train = train_data[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

#Two test to make sure the arrays are correct
#Y_train
#X_train[:, 0].shape

#define activation functions
def ReLU(z):
    return np.maximum(z, 0) 

def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a

#function to hard code the values of Y
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

#define the derivative of ReLU
def derive_ReLU(z):
    return z > 0

#Initialize parameters
def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

#after initilizing parameters, start with forward prop
def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def backwards_prop(z1, a1, z2, a2, w1, w2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = a2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(a1.T)
    dB2 = 1 / m * np.sum(dZ2)
    dZ1 = w2.T.dot(dZ2) * derive_ReLU(z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ1)
    return dW1, dB1, dW2, dB2

#third part of cycle, update the parameters
def update_params(w1, b1, w2, b2, dW1, dB1, dW2, dB2, alpha):
    w1 = w1 - alpha * dW1
    b1 = b1 - alpha * dB1
    w2 = w2 - alpha * dW2
    b2 = b2 - alpha * dB2
    return w1, b1, w2, b2

#make functions for predictions and accuracy
def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

#defining gradiant descent
def grad_desc(X, Y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dW1, dB1, dW2, dB2 = backwards_prop(z1, a1, z2, a2, w1, w2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dW1, dB1, dW2, dB2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(a2)
            print(get_accuracy(predictions, Y))
    return w1, b1, w2, b2
    
#code to run it
w1, b1, w2, b2 = grad_desc(X_train, Y_train, 500, 0.15)

#code for testing
def make_predictions(X, w1, b1, w2, b2):
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, X)
    predictions = get_predictions(a2)
    return predictions

def test_predictions(index, w1, b1, w2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], w1, b1, w2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation = 'nearest')
    plt.show()

test_predictions(0, w1, b1, w2, b2)
test_predictions(1, w1, b1, w2, b2)
test_predictions(2, w1, b1, w2, b2)
test_predictions(3, w1, b1, w2, b2)

#code for checking validation data
val_predictions = make_predictions(X_val, w1, b1, w2, b2)
get_accuracy(val_predictions, Y_val)