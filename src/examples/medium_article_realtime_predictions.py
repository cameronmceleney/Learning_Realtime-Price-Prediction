#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A detailed worked example of a real-time stock price prediction system using LSTM.

This file is based upon the original article written by `Abhishek Shaw and can be found on Medium`_.
It was this article that originally inspired me to begin the development of this entire project. The article offered a
tangible series of steps that I could follow to, in time, develop my own real-time stock market price prediction system.

Examples:
    Select a single company and use the preset pipeline in `PredictStockPrice.run()` to generate the output figure::

        >>> from src.examples.medium_article_realtime_predictions import PredictStockPrice
        >>> example_ps = PredictStockPrice("TSLA", "2020-01-01", "2025-01-01")
        >>> example_ps.run(seq_length=50)

Constants:
    PREPROCESSED_DATA (namedtuple): Holds PREPROCESSED_DATA data for LSTM model training under human-readable field
    names.

    PREPARED_DATA (namedtuple): Groups prepared training and testing data for LSTM model training.

References:
    Article author
        Abhishek Shaw

    Article created on
        10 Nov 2024

Notes:
    Version
        0.2.0
    Project
        Learning_Realtime_PricePrediction_toolkit
    Path
        src/examples/medium_article_realtime_predictions.py
    File created by
        Cameron Aidan McEleney <c.mceleney.1@research.gla.ac.uk>
    File created on
        25 May 2025
    IDE
        PyCharm

.. _Abhishek Shaw and can be found on Medium:
    https://medium.com/@abhishekshaw020/python-project-building-a-real-time-stock-market-price-prediction-system-6ce626907342
"""

__all__ = ["PredictStockPrice"]

# Standard library imports
from array import array
from collections import namedtuple
from functools import cached_property

# Third-party imports
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential

# Local application imports

# Module-level constants
PREPROCESSED_DATA = namedtuple(
    "PREPROCESSED_DATA",
    ["scaler", "training_data", "testing_data"],
)
"""(namedtuple): Groups preprocessed data for LSTM model training under human-readable field names."""

PREPARED_DATA = namedtuple(
    "PREPARED_DATA",
    ["x_train", "y_train", "x_test", "y_test"],
)
"""(namedtuple): Groups prepared training and testing data for LSTM model training."""


class PredictStockPrice:
    """Predict real-time price of stocks using Long Short-Term Memory (LSTM) neural networks.

    This class provides methods to fetch historical stock data, preprocess it, build an LSTM model. This class was
    formed by translating the original article's code into a class structure for better organisation and reusability.

    Attributes:
        ticker_symbol:  Four-letter stock ticker symbol of the company for which we are retrieving data.
        start_date: Start date for fetching historical stock data in 'YYYY-MM-DD' format.
        end_date: End date for fetching historical stock data in 'YYYY-MM-DD' format.
    """

    def __init__(self, ticker_symbol: str = 'TSLA', start_date: str = '2016-10-01', end_date: str = '2024-10-01'):
        """Initialise the PredictStockPrice instance.

        Args:
            ticker_symbol: Defines the company for this instance. Defaults to 'TSLA' (Tesla Inc.).
            start_date: Initial (oldest) date in recording period. Defaults to '2016-10-01'.
            end_date: Final (most recent) date in recording period. Defaults to '2024-10-01'.
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

        This method uses the yfinance library to download historical stock data for the ticker symbol and date-range
        specified in the class instance.

        Return:
            If successful, return a pandas DataFrame of the historical stock data for the specified company
            ticker symbol, otherwise return None.
        """
        return yf.download(self.ticker_symbol, start=self.start_date, end=self.end_date)

    @property
    def _preprocessing(self) -> PREPROCESSED_DATA:
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

        return PREPROCESSED_DATA(
            scaler=scaler,
            training_data=scaled_data[:train_size],
            testing_data=scaled_data[train_size:])

    @cached_property
    def _scaler(self) -> MinMaxScaler:
        """The scaler used for normalising the stock data."""
        return self._preprocessing.scaler

    def data_preparation(self, seq_length: int = 60) -> PREPARED_DATA:
        """Create sequences from the training and test data.

        Args:
            seq_length: Number of previous days to consider for prediction. Defaults to 60.

        Returns:
            PREPARED_DATA: namedtuple containing the training and testing sequences.
        """

        def create_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
            """Helper function to create sequences from the data.

            Args:
                data: Data to create sequences from.
                seq_len: Number of previous days to consider for prediction.

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
        return PREPARED_DATA(
            x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)),
            y_train=y_train,
            x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)),
            y_test=y_test)

    def build_model(self, epochs: int = 20, batch_size: int = 32) -> Sequential:
        """Initialise the model.

        Can be expensive to run, but don't cache to avoid clogging memory; we're only interested in the predictions.

        Args:
            epochs: Number of epochs to train the model. Defaults to 20.
            batch_size: Size of the batches used in training. Defaults to 32.

        Return:
            Sequential: A compiled, trained LSTM model ready to make predictions.
        """

        timesteps = self._prepared_data.x_train.shape[1]

        model = Sequential([
            # Declare the input shape here, not inside LSTM, for newer versions of Keras
            Input(shape=(timesteps, 1)),
            # Add LSTM layers
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            # Add output layer
            Dense(1),
        ])

        # Block comment - original code from article.
        """Original code from article. Left to help readers match article code to this code.
        
        model = Sequential()
        
        # Add LSTM layers
        model.add(LSTM(units=100, return_sequences=True, input_shape=(self._prepared_data.x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Add output layer
        model.add(Dense(units=1))
        """

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

    def retrain(self, *args, **kwargs) -> None:
        """Helper method that clears cached properties; allowing the model to be retrained with new parameters.

        Args:
            *args: Variable length argument list for the model training parameters.
            **kwargs: Arbitrary keyword arguments for the model training parameters.
        """
        self.build_model(*args, **kwargs)
        for prop in ('predictions', 'actual_prices'):
            self.__dict__.pop(prop, None)

    @cached_property
    def predictions(self) -> np.ndarray:
        """Predictions made by the model.

        Cached to enable inexpensive repeated access during data visualisation.

        Return:
            The predicted stock prices.
        """
        return self.make_predictions()[0]

    @cached_property
    def actual_prices(self) -> np.ndarray:
        """The actual test prices; scaled back to the original price scale.

        Cached to enable inexpensive repeated access during data visualisation.

        Return:
            The actual stock prices from the test dataset.
        """
        return self.make_predictions()[1]

    def visualise_results(self) -> None:
        """Generate a figure of the predicted stock price versus the actual stock price.

        This method uses Plotly to create an interactive line chart that displays the actual stock prices.

        Returns:
            Automatically displays the figure in a web browser or inline in a Jupyter notebook.
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

    def model_evaluation(self) -> dict[str, float | array | np.ndarray]:
        """Evaluate model's performance by calculating the Mean Square Error (MSE) and Root Mean Square Error (RMSE).

        Calculations of the MSE and RMSE are based on comparing the predictions made by the model to the actual prices.
        These datasets are stored in the class instance's attributes as `self.predictions` and `self.actual_prices`,
        respectively.

        Returns:
            Dictionary containing the MSE and RMSE of the predictions compared to the actual prices.
        """

        mse = mean_squared_error(self.actual_prices, self.predictions)
        rmse = np.sqrt(mse)

        print(f'Mean Squared Error: {mse}')
        print(f'Root Mean Squared Error: {rmse}')

        return {'mse': mse, 'rmse': rmse}

    # --- Pipeline methods ---
    def run(self, seq_length: int = 60, epochs: int = 20, batch_size: int = 32) -> None:
        """Automatically run the entire pipeline.

        Stages
            1. Fetch company's stock data.
            2. Prepare the data.
            3. Build and train the ML model.
            4. Make predictions using the trained model.
            5. Visualise results.
            6. Evaluate the model's performance.

        Args:
            seq_length: Number of previous days to consider for prediction. Defaults to 60.
            epochs: Number of epochs to train the model. Defaults to 20.
            batch_size: Size of the batches used in training. Defaults to 32.

        """

        self._prepared_data = self.data_preparation(seq_length)
        self.build_model(epochs, batch_size)
        self.make_predictions()
        self.visualise_results()
        self.model_evaluation()


if __name__ == "__main__":
    ps = PredictStockPrice("TSLA", "2020-01-01", "2025-01-01")
    ps.run(seq_length=50)
