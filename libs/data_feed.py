from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pathlib
import tensorflow as tf
import os

class MnistDataSet:
  
  @staticmethod
  def get_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    ### Normalization

    x_train = ((x_train - 0) / (255-0))
    x_test = ((x_test - 0) / (255-0))

    ### One Hot Encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    y_test = enc.fit_transform(y_test.reshape((-1,1))).toarray()
    enc = OneHotEncoder(handle_unknown='ignore')
    y_train = enc.fit_transform(y_train.reshape((-1,1))).toarray()
    
    return x_train, y_train, x_test, y_test
	
  
  
  
