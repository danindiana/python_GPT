import os
import logging
import asyncio
from streamz import Stream
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import plotly.graph_objects as go
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesForecaster:
    def __init__(self, seq_length: int = 20):
        self.seq_length = seq_length
        self.predictions = []
        self.actuals = []
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        self.metrics = {'loss': []}

    def _build_model(self) -> Sequential:
        model = Sequential([
            LSTM(64, input_shape=(self.seq_length, 5), return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def preprocess_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            # Add technical indicators
            data['MA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            features = ['Open', 'High', 'Low', 'Close', 'MA_50']
            
            # Normalize data
            data_normalized = self.scaler.fit_transform(data[features])
            return pd.DataFrame(data_normalized, columns=features)
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None

    async def create_windowed_input(self, source_X: Stream, source_y: Stream) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        try:
            X, y = [], []
            while len(X) < self.seq_length:
                x = await source_X.get()
                if x is None:
                    break
                X.append(x)
                y.append(await source_y.get())

            if len(X) == self.seq_length:
                X_df = self.preprocess_data(pd.DataFrame(X))
                if X_df is not None:
                    return X_df, pd.Series(y)
            
            return None, None
        except Exception as e:
            logger.error(f"Error creating windowed input: {str(e)}")
            return None, None

    async def train_and_update(self, X: pd.DataFrame, y: pd.Series) -> Optional[tf.keras.Model]:
        try:
            if X is None or y is None:
                return None

            # Reshape input data
            X_reshaped = X.values.reshape((1, self.seq_length, 5))
            y_reshaped = y.values.reshape(-1, 1)

            # Train model asynchronously
            history = await tf.keras.utils.model_to_json(self.model.fit(
                X_reshaped, y_reshaped,
                epochs=1,
                batch_size=1,
                verbose=0
            ))
            
            self.metrics['loss'].append(history.history['loss'][0])
            return self.model
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None

    def predict(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            if X is None:
                return None

            X_reshaped = X.values.reshape((1, self.seq_length, 5))
            pred = self.model.predict(X_reshaped, verbose=0)
            
            # Inverse transform prediction
            pred_original = self.scaler.inverse_transform(
                np.concatenate([X_reshaped[0, -1, :-1], pred], axis=1)
            )[:, -1]
            
            self.predictions.append(pred_original[0])
            return pred_original[0]
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None

    def save_model(self, filename: str = 'lstm_model.h5'):
        os.makedirs('models', exist_ok=True)
        self.model.save(os.path.join('models', filename))

async def stream_data(file_path: str, source_X: Stream, source_y: Stream):
    try:
        data = pd.read_csv(file_path)
        for i in range(len(data)):
            await asyncio.sleep(0.1)  # Simulate real-time data
            await source_X.emit(data.iloc[i][['Open', 'High', 'Low', 'Close']].values)
            await source_y.emit(data.iloc[i]['Close'])
    except Exception as e:
        logger.error(f"Error streaming data: {str(e)}")

async def main():
    forecaster = TimeSeriesForecaster()
    source_X = Stream()
    source_y = Stream()

    # Set up stream processing pipeline
    windowed = source_X.rate_limit(0.5).map(
        lambda x: forecaster.create_windowed_input(source_X, source_y)
    )
    trained = windowed.map(
        lambda x: forecaster.train_and_update(x[0], x[1])
    )
    predictions = trained.map(lambda x: forecaster.predict(x[0]))

    # Start data streaming
    await stream_data('data/2017-01-01.csv', source_X, source_y)

    # Plot results
    df = pd.DataFrame({
        'Actual': forecaster.actuals,
        'Predicted': forecaster.predictions
    })
    
    fig = go.Figure([
        go.Scatter(y=df['Actual'], name='Actual'),
        go.Scatter(y=df['Predicted'], name='Predicted')
    ])
    fig.update_layout(
        title='Real-time Time Series Forecasting',
        xaxis_title='Time',
        yaxis_title='Value'
    )
    fig.show()

if __name__ == '__main__':
    asyncio.run(main())
