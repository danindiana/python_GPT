from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    # Example input shape
    example_input_shape = (None, 2)  # Adjust this to match your actual data
    
    print("Building model...")
    model = build_model(example_input_shape)
    print("Model built.")
