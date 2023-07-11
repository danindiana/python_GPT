class LTCModel(Model):
    def __init__(self, units, output_dim, **kwargs):
        super(LTCModel, self).__init__(**kwargs)
        self.ltc_layer = tf.keras.layers.RNN(LTCCell(units), return_sequences=True)
        self.output_layer = Dense(output_dim)

    def call(self, inputs):
        x = self.ltc_layer(inputs)
        return self.output_layer(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)  # Compute gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))  # Apply gradients
        self.compiled_metrics.update_state(y, y_pred)  # Update metrics
        return {m.name: m.result() for m in self.metrics}
