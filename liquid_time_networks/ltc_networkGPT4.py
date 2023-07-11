import tensorflow as tf
from tensorflow.keras.layers import Layer

class LTCCell(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        super(LTCCell, self).__init__(**kwargs)
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = tf.matmul(inputs, self.kernel)
        output = h + tf.matmul(prev_output, self.recurrent_kernel)

        # Nonlinear gate modulation
        gate = tf.sigmoid(output)
        output *= gate

        # Numerical differential equation solver (Backward Euler)
        output = prev_output + output

        return output, [output]

# Usage:
# lstm_cell = LTCCell(10)
# x = tf.zeros((3, 10))
# output, new_states = lstm_cell(x, [x])
