import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from QRNN_Cell import QRNN_Cell

class QRNN_Model(tf.keras.Model):
    def __init__(self, symbol_names, num_neurons):
        super(QRNN_Model, self).__init__()
        self.symbol_names = symbol_names
        self.rnn_0 = QRNN_Cell(symbol_names, num_neurons = num_neurons)
        self.rnn_1 = QRNN_Cell(symbol_names, num_neurons = num_neurons)
        self.rnn_2 = QRNN_Cell(symbol_names, num_neurons = num_neurons)
        self.rnn_3 = QRNN_Cell(symbol_names, num_neurons = num_neurons)
        self.rnn_4 = QRNN_Cell(symbol_names, num_neurons = num_neurons)
        self.num_layer_neurons = num_neurons
        self.average = tf.keras.layers.Average()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'symbol_names': self.symbol_names,
            'num_layer_neurons':self.num_layer_neurons,
            'rnn_0': self.rnn_0,
            'rnn_1': self.rnn_1,
            'rnn_2': self.rnn_2,
            'rnn_3': self.rnn_3,
            'rnn_4': self.rnn_4,
            'averager':self.average})
        return config

    def call(self,inputs):
        #input_layer = tf.keras.layers.concatenate([qaoa_circuits, cost_hams, hidden_state, qaoa_parameters, prev_exp])
        output_0 = self.rnn_0(inputs)
        output_1 = self.rnn_1(output_0)
        output_2 = self.rnn_2(output_1)
        output_3 = self.rnn_3(output_2)
        output_4 = self.rnn_4(output_3)

        # We want the most improvement over time
        loss_timestep = self.average([0.2 * output_0[4], 0.4 * output_1[4], 0.8 * output_2[4],1.6 * output_3[4], 3.2 * output_4[4]]) 
        loss_timestep = tf.keras.backend.flatten(0.1*loss_timestep)

        # Notice that the penalty goes up when we take longer to make a good guess...

        # We also want to penalize jumping around the landscape randomly (QNN barrier plateus)
        jumping_penalty = tf.norm(output_0[3]-output_1[3], axis=1) + tf.norm(output_1[3]-output_2[3], axis=1) + tf.norm(output_2[3]-output_3[3], axis=1) + tf.norm(output_3[3]-output_4[3], axis=1) 
        jumping_penalty = tf.keras.backend.flatten(jumping_penalty)

        #print(jumping_penalty)
        #print(loss_timestep)

        # Notice that the penalty goes up when the guesses for each time step are too different

        full_loss = loss_timestep + jumping_penalty

        #print(full_loss)

        return [output_0[3], output_1[3], output_2[3], output_3[3], output_4[3], full_loss]
