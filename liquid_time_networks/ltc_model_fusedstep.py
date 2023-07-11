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
        self.leakage = self.add_weight(shape=(self.units,), 
                                       initializer='uniform',
                                       name='leakage')
        self.time_constant = self.add_weight(shape=(self.units,),
                                             initializer='uniform',
                                             name='time_constant')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]

        # Normal RNN computation
        h = tf.matmul(inputs, self.kernel) + tf.matmul(prev_output, self.recurrent_kernel)

        # Leakage term
        leak = self.leakage * prev_output

        # Sigmoidal nonlinearity modulation
        modulated_output = tf.sigmoid(leak + h)

        # Fused step calculation
        delta_t = 1  # Assuming a normalized timestep
        denominator = 1 + delta_t / self.time_constant + modulated_output
        output = prev_output + delta_t * modulated_output / denominator

        return output, [output]
