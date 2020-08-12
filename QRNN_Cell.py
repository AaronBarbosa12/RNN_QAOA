import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

class QRNN_Cell(tf.keras.layers.Layer):
    def __init__(self, symbol_names, num_neurons):
        super(QRNN_Cell, self).__init__()
        # inputs -> shared layer -> produce hidden state, param prediction, and expectation value
        self.shared = tf.keras.layers.Dense(num_neurons, name="shared", activation = tf.keras.activations.elu)
        self.dropout= tf.keras.layers.Dropout(0.2)
        self.state = tf.keras.layers.Dense(num_neurons, name="state", activation = tf.keras.activations.elu)
        self.params = tf.keras.layers.Dense(2, name="params")
        self.expectation_layer = tfq.layers.Expectation()
        self.symbol_names = symbol_names 

    def call(self,inputs,training=False):
        qaoa_circuits = inputs[0]
        cost_hams = inputs[1]
        hidden_state = inputs[2]
        qaoa_parameters = inputs[3]
        prev_exp = inputs[4]

        #Create data pipeline
        input_layer = tf.keras.layers.concatenate([hidden_state, qaoa_parameters, prev_exp])        
        shared_layer_out = self.shared(input_layer)
        dropout_out = self.dropout(shared_layer_out)
        hidden_layer_out = self.state(dropout_out)
        param_layer_out = self.params(dropout_out)

        exp_out = self.expectation_layer(qaoa_circuits, symbol_names=self.symbol_names, symbol_values=param_layer_out, operators = cost_hams)
       
        return [qaoa_circuits, cost_hams, hidden_layer_out, param_layer_out, exp_out]