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

# Propagation
def fowardprop():
  
  return()

def backprop():
  return()

def softmax():
  return()

def ReLu():

  return()


