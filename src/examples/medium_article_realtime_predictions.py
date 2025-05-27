#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A detailed worked example of a real-time stock price prediction system using LSTM.

This is almost entirely based upon the original article written by `Abhishek Shaw and can be found on Medium`_.
It was this article that originally inspired me to begin the development of this entire project. The article offered a
tangible series of steps that I could follow to, in time, develop my own real-time stock market price prediction system.

Examples:
    Select a single company and use the preset pipeline in `PredictStockPrice.run()` to generate the output figure::

        >>> ps = PredictStockPrice("TSLA", "2020-01-01", "2025-01-01")
        >>> results = ps.run(seq_length=50)

Notes:
    Project
        Learning_Realtime_PricePrediction_toolkit
    Path
        src/examples/medium_article_realtime_predictions.py
    IDE
        PyCharm
    Version
        0.1.0

References:
    Author
        Abhishek Shaw
    Created
        10 Nov 2024

    Edited by
        Cameron Aidan McEleney <c.mceleney.1@research.gla.ac.uk>


.. _Abhishek Shaw and can be found on Medium:
    https://medium.com/@abhishekshaw020/python-project-building-a-real-time-stock-market-price-prediction-system-6ce626907342

"""

# Whole library imports
from collections import namedtuple
from functools import cached_property
import numpy as np
from pandas import DataFrame
import plotly.graph_objs as go
import yfinance as yf

# ML package imports
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Local application imports

__all__ = ["PredictStockPrice"]


class PredictStockPrice:
    """Predict real-time price of stocks using Long Short-Term Memory (LSTM) neural networks.

    This class provides methods to fetch historical stock data, preprocess it, build an LSTM model. This class was
    formed by translating the original article's code into a class structure for better organisation and reusability.

    Attributes:
        ticker_symbol:  Four-letter stock ticker symbol of the company for which we are retrieving data.
        start_date: Start date for fetching historical stock data.
        end_date: End date for fetching historical stock data.
    """
    _Preprocessed = namedtuple("Preprocessed", ["scaler", "training_data", "testing_data"])
    _PreparedData = namedtuple("PreparedData", ["x_train", "y_train", "x_test", "y_test"])

    def __init__(self, ticker_symbol='TSLA', start_date='2016-10-01', end_date='2024-10-01'):
        """Docstring explainer.

        Args:
            ticker_symbol(str): Defines the company for this instance.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        """
        # Core configuration
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date

        # Pipeline state attributes
        self._prepared_data = None
        self._model = None

    @cached_property
    def fetch_stock_data(self) -> DataFrame | None:
        """Fetch company's historical stock data.

        Return:
            DataFrame | None: A pandas DataFrame containing the historical stock data for the specified
            ticker symbol.
        """
        return yf.download(self.ticker_symbol, start=self.start_date, end=self.end_date)

    @property
    def _preprocessing(self) -> _Preprocessed:
        """Normalise and split the stock data into training and testing sets for LSTM model training.

        Returns:
            tuple: A customised namedtuple containing the following fields.

            - ``scaler`` (MinMaxScaler): The scaler used to normalise the stock data.
            - ``training_data`` (np.ndarray): Normalised training data.
            - ``testing_data`` (np.ndarray): Normalised testing data.
        """
        df = self.fetch_stock_data
        if df.empty:
            raise ValueError("No data fetched. Please check the ticker symbol and date range.")

        close_prices = df['Close'].values  # Use only the 'Close' column for price prediction

        # Normalise the dataset using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

        # Split the data into training (80%) and testing (20%) sets
        train_size = int(len(scaled_data) * 0.8)

        return self._Preprocessed(
            scaler=scaler,
            training_data=scaled_data[:train_size],
            testing_data=scaled_data[train_size:]
        )

    @cached_property
    def _scaler(self):
        """The scaler used for normalising the stock data."""
        return self._preprocessing.scaler

    def data_preparation(self, seq_length: int = 60) -> _PreparedData:
        """Create sequences from the training and test data.

        Args:
            seq_length (int): Number of previous days to consider for prediction. Defaults to 60.
        Returns:
            _PreparedData: namedtuple containing the training and testing sequences.
        """
        def create_sequences(data, seq_len) -> tuple[np.ndarray, np.ndarray]:
            """Helper function to create sequences from the data.

            Args:
                data (np.ndarray): The data to create sequences from.
                seq_len (int): The number of previous days to consider for prediction.

            Returns:
                tuple[np.ndarray, np.ndarray]: Tuple containing the input sequences and ... .
            """
            x, y = [], []
            for i in range(seq_len, len(data)):
                x.append(data[i - seq_len:i, 0])
                y.append(data[i, 0])
            return np.array(x), np.array(y)

        # Typically, cached_properties are determined here for the first time.
        x_train, y_train = create_sequences(self._preprocessing.training_data, seq_length)
        x_test, y_test = create_sequences(self._preprocessing.testing_data, seq_length)

        # Reshape the input data to be compatible with LSTM
        return self._PreparedData(
            x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)),
            y_train=y_train,
            x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)),
            y_test=y_test,
        )

    def build_model(self, epochs=20, batch_size=32) -> Sequential:
        """Initialise the model.

        Can be expensive to run, but don't cache to avoid clogging memory; we're only interested in the predictions.

        Args:
            epochs (int): Number of epochs to train the model. Defaults to 20.
            batch_size (int): Size of the batches used in training. Defaults to 32.

        Return:
            Sequential: A compiled, trained LSTM model ready to make predictions.
        """
        model = Sequential()

        # Add LSTM layers
        model.add(LSTM(units=100, return_sequences=True, input_shape=(self._prepared_data.x_train.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(0.2))

        # Add output layer
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(self._prepared_data.x_train, self._prepared_data.y_train, epochs=epochs, batch_size=batch_size)

        self._model = model

        return model

    def make_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Predict stock prices on the test data.

        Returns are cached as separate properties as we assume the user will want to access the predictions
        multiple times without retraining the model, such as for data analysis and data visualisation.

        Returns:
            tuple[np.ndarray, np.ndarray]: Predicted prices and the actual test prices.
        """
        predictions = self._model.predict(self._prepared_data.x_test)

        # Inverse transform the predictions back to original price scale
        predictions = self._scaler.inverse_transform(predictions)

        # Inverse transform the actual test data
        y_test_scaled = self._scaler.inverse_transform(self._prepared_data.y_test.reshape(-1, 1))

        return predictions, y_test_scaled

    def retrain(self, *args, **kwargs):
        self.build_model(*args, **kwargs)
        for prop in ('predictions', 'actual_prices'):
            self.__dict__.pop(prop, None)

    @cached_property
    def predictions(self):
        """Predictions made by the model.

        Cache to enable inexpensive repeated access during data visualisation.

        Return:
            np.ndarray: The predicted stock prices.
        """
        return self.make_predictions()[0]

    @cached_property
    def actual_prices(self):
        """The actual test prices; scaled back to the original price scale.

        Cache to enable inexpensive repeated access during data visualisation.

        Return:
            np.ndarray: The actual stock prices from the test dataset.
        """
        return self.make_predictions()[1]

    def visualise_results(self):
        """Create a figure.

        Visualises the actual stock prices and the predicted stock prices using Plotly.
        """

        fig = go.Figure()

        # Add trace for actual prices
        fig.add_trace(
            go.Scatter(x=self.fetch_stock_data.index[-len(self._prepared_data.y_test):],
                       y=self.actual_prices.flatten(),
                       mode='lines', name='Actual Price',)
        )

        # Add trace for predicted prices
        fig.add_trace(
            go.Scatter(x=self.fetch_stock_data.index[-len(self._prepared_data.y_test):],
                       y=self.predictions.flatten(),
                       mode='lines', name='Predicted Price',)
        )

        # Add titles and labels
        fig.update_layout(title='Tesla Stock Price Prediction',
                          xaxis_title='Date', yaxis_title='Stock Price (USD)',)

        # Show the figure
        fig.show()

    def model_evaluation(self):
        """Calculate MSE and RMSE."""

        mse = mean_squared_error(self.actual_prices, self.predictions)
        rmse = np.sqrt(mse)

        print(f'Mean Squared Error: {mse}')
        print(f'Root Mean Squared Error: {rmse}')

        return {'mse': mse, 'rmse': rmse}

    # --- Pipeline methods ---
    def run(self, seq_length=60, epochs=20, batch_size=32):
        """Full pipeline in one call."""
        self._prepared_data = self.data_preparation(seq_length)
        self.build_model(epochs, batch_size)
        self.make_predictions()
        self.visualise_results()
        return self.model_evaluation()


if __name__ == "__main__":
    ps = PredictStockPrice("TSLA", "2020-01-01", "2025-01-01")
    results = ps.run(seq_length=50)
