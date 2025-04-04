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

# Matrix of Training Data
X_train = np.array(X_train)
m,n = X_train.shape

def _init_():
  W1 = np.random.randn(10,700)
  b1 = 0
  W2 = np.random.randn(10,700)
  b2 = 0
  lr = 0.1 # Learning Rate
  return(W1, W2, b1, b2, lr)

# Functions 
def ReLu(Z):
  return(np.maximum(0,Z))

def SoftMax(Z):
  return(np.exp(Z) / np.sum.exp(Z))

def derv_ReLu(Z):
  return(Z > 0)

# Propagation
def fowardprop(W1, W2, b1, b2, X):
  Z1 = W1.dot(X) + b1
  A1 = ReLu(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = SoftMax(Z2)
  return(A2, A1, Z2, Z1)

def backprop(W1, W2, A1, A2, Z1, X_act, X):
  m = X_act.size
  # Second Layer
  dz2 = A2 - X_act
  dw2 = dz2.dot(A1.T) / m
  db2 = np.sum(dz2) / m
  # First Layer
  dz1 = W2.T.dot(dz2) * derv_ReLu(Z1)
  dw1 = dz1.dot(X.T) / m
  db1 = np.sum(dz1) / m
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
  print(predictions, test_data)
  return(np.sum(predictions == test_data) / test_data.size)

def Gradient_decent(train_data, test_data, iterations):
  W1, W2, b1, b2, lr = _init_()
  for i in range(iterations):
    A2, A1, Z2, Z1 = fowardprop(W1, W2, b1, b2, train_data)
    dw1, dw2, db1, db2 = backprop(W1, W2, A1, A2, test_data, train_data)
    W1, W2, b1, b2 = Update(W1, W2, b1, b2, dw1, dw2, db1, db2, lr)
    if (i % 10 == 0):
      print("Iteration: ", i)
      print("Accuracy: ", Pred_accuracy(Get_pred(A2)))
  return(W1, W2, b1, b2)

# Running Model
W1, W2, b1, b2 = Gradient_decent(X_train, Y_train, 500)