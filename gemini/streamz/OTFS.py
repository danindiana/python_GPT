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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesForecaster:
    def __init__(self, seq_length: int = 20, feature_size: int = 5):
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.predictions = []
        self.actuals = []
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        self.metrics = {'loss': [], 'mae': []}

    def _build_model(self) -> Sequential:
        model = Sequential([
            LSTM(64, input_shape=(self.seq_length, self.feature_size), 
                 return_sequences=True),
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
            data['MA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            features = ['Open', 'High', 'Low', 'Close', 'MA_50']
            data_normalized = self.scaler.fit_transform(data[features])
            return pd.DataFrame(data_normalized, columns=features)
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return None

    async def create_window(self, source_X: Stream, source_y: Stream) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        try:
            X, y = [], []
            async for _ in range(self.seq_length):
                x = await source_X.get()
                if x is None:
                    return None, None
                X.append(x)
                y.append(await source_y.get())

            X_df = self.preprocess_data(pd.DataFrame(X))
            return (X_df, pd.Series(y)) if X_df is not None else (None, None)
        except Exception as e:
            logger.error(f"Window creation error: {str(e)}")
            return None, None

    async def train(self, X: pd.DataFrame, y: pd.Series) -> Optional[float]:
        try:
            if X is None or y is None:
                return None

            X_reshaped = X.values.reshape((1, self.seq_length, self.feature_size))
            y_reshaped = y.values.reshape(-1, 1)

            history = await self.model.fit(
                X_reshaped, y_reshaped,
                epochs=1,
                batch_size=1,
                verbose=0
            )
            
            loss = history.history['loss'][0]
            self.metrics['loss'].append(loss)
            return loss
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return None

    def predict(self, X: pd.DataFrame) -> Optional[float]:
        try:
            if X is None:
                return None

            X_reshaped = X.values.reshape((1, self.seq_length, self.feature_size))
            pred = self.model.predict(X_reshaped, verbose=0)[0][0]
            
            # Inverse transform
            pred_original = self.scaler.inverse_transform(
                np.concatenate([X_reshaped[0, -1, :-1], [[pred]]], axis=1)
            )[0, -1]
            
            self.predictions.append(pred_original)
            return pred_original
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

    def save_model(self, filename: str = 'lstm_model.h5'):
        os.makedirs('models', exist_ok=True)
        self.model.save(os.path.join('models', filename))

async def stream_data(file_path: str, source_X: Stream, source_y: Stream):
    try:
        data = pd.read_csv(file_path)
        for i in range(len(data)):
            await asyncio.sleep(0.1)
            await source_X.emit(data.iloc[i][['Open', 'High', 'Low', 'Close']].values)
            await source_y.emit(data.iloc[i]['Close'])
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")

async def main():
    forecaster = TimeSeriesForecaster()
    source_X = Stream()
    source_y = Stream()

    windowed = source_X.rate_limit(0.5).map(
        lambda x: forecaster.create_window(source_X, source_y)
    )
    trained = windowed.map(lambda x: forecaster.train(x[0], x[1]))
    predictions = trained.map(lambda x: forecaster.predict(x[0]))

    await stream_data('data/2017-01-01.csv', source_X, source_y)

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
