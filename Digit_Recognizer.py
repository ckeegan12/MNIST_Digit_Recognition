import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import Test/Training data
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_train.head()

# Preview of Data
df_train.info()
df_train.describe()

# Matrix of Training Data
df_train = np.array(df_train)
m,n = df_train.shape

# Image Preview
