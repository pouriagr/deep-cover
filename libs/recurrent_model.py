import tensorflow.compat.v2 as tf

import collections
import functools
import warnings

import numpy as np
from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.saving.saved_model import layer_serialization
from keras.utils import control_flow_util
from keras.utils import generic_utils
from keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from libs.better_lstm import BetterLSTMCell
from tensorflow.keras.layers import GRUCell, SimpleRNNCell

class RecurrentModel:
  def __init__(
      self, 
      input_shape, 
      hidden_layer_size, 
      output_size, 
      weights_folder_address,
      rrn_module_name
    ):
    self.input_shape = input_shape
    self.hidden_layer_size = hidden_layer_size
    self.output_size = output_size
    self.rrn_module_name = rrn_module_name
    
    self.keras_model = self.__generate_model(return_states=False)
    self.keras_model.compile(
        optimizer='rmsprop', loss=tf.keras.losses.categorical_crossentropy, 
        metrics=['accuracy'], loss_weights=None,
        weighted_metrics=None, run_eagerly=None, steps_per_execution=None
    )
    self.keras_model.load_weights(weights_folder_address)
    
    self.keras_model_with_states = self.__generate_model(return_states=True)
    self.keras_model_with_states.compile(
        optimizer='rmsprop', loss=tf.keras.losses.categorical_crossentropy, 
        metrics=['accuracy'], loss_weights=None,
        weighted_metrics=None, run_eagerly=None, steps_per_execution=None
    )
    self.keras_model_with_states.load_weights(weights_folder_address)
  
  def __generate_model(self, return_states):
    model_input = tf.keras.Input(
        shape=self.input_shape, name='model_input', dtype=tf.float32
    )
    
    rnn_cell = (BetterLSTMCell(self.hidden_layer_size) if self.rrn_module_name=='lstm' 
       else GRUCell(self.hidden_layer_size) if self.rrn_module_name=='gru'
       else SimpleRNNCell(self.hidden_layer_size))

    if not return_states:
      rnn_layer = tf.keras.layers.RNN(
        rnn_cell,
        return_sequences=False,
        return_state=False
      )(model_input) 
    else:
      rnn_layer = tf.keras.layers.RNN(
        rnn_cell,
        return_sequences=True,
        return_state=False
      )(model_input) 

    dense = tf.keras.layers.Dense(self.output_size,activation='relu')(rnn_layer)
    model_output = tf.keras.layers.Softmax()(dense)

    if not return_states:
      return tf.keras.models.Model(
        inputs=model_input, outputs=model_output
      )
    else:
      return tf.keras.models.Model(
        inputs=model_input, outputs=[rnn_layer, model_output]
      )