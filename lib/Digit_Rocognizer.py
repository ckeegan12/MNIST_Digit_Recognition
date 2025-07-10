import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Import Test/Training data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalizee Data to be between 1 and 0
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0 # Flatten and normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0 # Flatten and normalize

# Matrix of Training Data
m,n = X_train.shape

def _init_():
  hidden_units = 128  
  W1 = np.random.rand(784, hidden_units) - 0.5  
  b1 = np.random.rand(hidden_units, 1)
  W2 = np.random.rand(10, hidden_units) - 0.5
  b2 = np.random.rand(10, 1)
  lr = 0.1  # Learning Rate
  return (W1, W2, b1, b2, lr)

# Functions 
def ReLu(Z):
  return(np.maximum(Z,0))

def SoftMax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def derv_ReLu(Z):
  return(Z > 0)

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # For batch input
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T  # Transpose to shape (10, n) 
  return one_hot_Y

# Function derivatives
def derv_ReLu(X):
    return X >= 0

def derv_loss(observed, predicted):
    # SoftMax and Cross-entropy loss
    return predicted - observed

# Propagation
def fowardprop(W1, W2, b1, b2, X):
  Z1 = np.matmul(X, W1) + b1.T  # Shape (1, hidden_units)
  A1 = ReLu(Z1)
  Z2 = np.matmul(A1, W2.T) + b2.T  # Shape (1, 10)
  A2 = SoftMax(Z2)
  return (A2, A1, Z1)

def backprop(W2, A1, A2, Z1, Y, input):
  # X: Predicted value
  # Y: Actual value
  n = int(A1.shape[0])
  Y_one_hot = one_hot(Y)
  
  # Second Layer
  dZ2 = derv_loss(Y_one_hot, A2.T)  # Shape (samples, 10)
  dw2 = dZ2.dot(A1) / n # Shape (10, hidden_units)
  db2 = np.sum(dZ2.T, axis=0, keepdims=True) / n # Shape (1, 10)

  # First Layer
  dZ1 = dZ2.T.dot(W2) * derv_ReLu(Z1)  # Shape (1, hidden_units)
  dw1 = input.T.dot(dZ1) / n  # Shape (784, hidden_units)
  db1 = np.sum(dZ1, axis=0, keepdims=True) / n  # Shape (1, hidden_units)

  return(dw1, dw2, db1, db2)

# Update Parameters
def Update(W1, W2, b1, b2, dw1, dw2, db1, db2, lr):
  new_W1 = W1 - lr * dw1
  new_W2 = W2 - lr * dw2
  new_b1 = b1 - lr * db1.T
  new_b2 = b2 - lr * db2.T
  return(new_W1, new_W2, new_b1, new_b2)

# Gradient Decent
def Get_pred(A2):
  return(np.argmax(A2, axis=1))

def Pred_accuracy(predictions, test_data):
  print(predictions, test_data)
  return((np.sum(predictions == test_data) / test_data.size) * 100)

def Gradient_decent(train_data, train_labels, iterations):
  W1, W2, b1, b2, lr= _init_()
  for i in range(iterations):
    prediction = []
    A2, A1, Z1 = fowardprop(W1, W2, b1, b2, train_data)
    dw1, dw2, db1, db2 = backprop(W2, A1, A2, Z1, train_labels, train_data)
    W1, W2, b1, b2 = Update(W1, W2, b1, b2, dw1, dw2, db1, db2, lr)
    if (i % 10 == 0):
      print(f"Iteration: {i}")
      prediction = np.append(prediction, Get_pred(A2))
      print(f"Accuracy: {Pred_accuracy(prediction, train_labels):.2f}%")
  return(W1, W2, b1, b2)

# Running Model
W1, W2, b1, b2 = Gradient_decent(X_train, Y_train, 500)
 
# Testing the model
test_predictions = Get_pred(fowardprop(W1, W2, b1, b2, X_test)[0])
test_accuracy = Pred_accuracy(test_predictions, Y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")




