import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Import Test/Training data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train.head()

# Preview of Data
X_train.info()
X_train.describe()
Y_train.info()
Y_train.describe()
sample = 5 # Interchangeable 
image = X_train[sample]
plt.imshow(image, cmap='gray')
plt.show()

# Normalizee Data to be between 1 and 0
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0 # Flatten and normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0 # Flatten and normalize

# Matrix of Training Data
m,n = X_train.shape

def _init_():
  W1 = np.random.randn(10,700) * 0.01
  b1 = 0
  W2 = np.random.randn(10,700) * 0.01
  b2 = 0
  lr = 0.1 # Learning Rate
  return(W1, W2, b1, b2, lr)

# Functions 
def ReLu(Z):
  return(np.maximum(0,Z))

def SoftMax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / exp_x.sum(axis=0)

def derv_ReLu(Z):
  return(Z > 0).astype(float)

# Propagation
def fowardprop(W1, W2, b1, b2, X):
   Z1 = np.dot(X, W1.T) + b1.T
   A1 = ReLu(Z1)
   Z2 = np.dot(A1, W2.T) + b2.T
   A2 = SoftMax(Z2)
  return(A2, A1, Z2, Z1)

def backprop(W1, W2, A1, A2, Z1, Y, X):
  m = Y.size
  # Second Layer
  Y_one_hot = np.eye(10)[Y]  # One-hot encoding of labels
  dz2 = A2 - Y_one_hot  # Gradient for output layer
  dw2 = np.dot(dz2.T, A1) / m
  db2 = np.sum(dz2, axis=0, keepdims=True) / m
  # First Layer
  dz1 = np.dot(dz2, W2) * derv_ReLu(A1)
  dw1 = np.dot(dz1.T, X) / m
  db1 = np.sum(dz1, axis=0, keepdims=True) / m
  return(dw1, dw2, db1, db2)

# Update Parameters
def Update(W1, W2, b1, b2, dw1, dw2, db1, db2, lr):
  new_W1 = W1 - lr * dw1
  new_W2 = W2 - lr * dw2
  new_b1 = b1 - lr * db1
  new_b2 = b2 - lr * db2
  return(new_W1, new_W2, new_b1, new_b2)

# Gradient Decent
def Get_pred(A2):
  return(np.argmax(A2,0))

def Pred_accuracy(predictions, test_data):
  return(np.sum(predictions == test_data) / test_data.size)

def Gradient_decent(train_data, test_data, train_labels, test_labels, iterations):
  W1, W2, b1, b2, lr = _init_()
  for i in range(iterations):
    A2, A1, Z2, Z1 = fowardprop(W1, W2, b1, b2, train_data)
    dw1, dw2, db1, db2 = backprop(W1, W2, A1, A2, train_labels, train_data)
    W1, W2, b1, b2 = Update(W1, W2, b1, b2, dw1, dw2, db1, db2, lr)
    if (i % 10 == 0):
      print("Iteration: {i}")
      print("Accuracy: ", Pred_accuracy(Get_pred(A2), train_labels))
  return(W1, W2, b1, b2)

# Running Model
W1, W2, b1, b2 = Gradient_decent(X_train, X_test, Y_train, Y_test, 500)

# Testing the model
test_predictions = Get_pred(fowardprop(W1, W2, b1, b2, X_test)[0])
test_accuracy = Pred_accuracy(test_predictions, Y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
