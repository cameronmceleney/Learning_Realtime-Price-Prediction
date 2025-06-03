#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Detailed worked example of making stock market predictions using Long Short-Term Memory (LSTM) networks.

This file is based upon an excellent tutorial by `Thushan Ganegedara which can be found on Datacamp`_. This article
provided a great introduction to the topic, and was more complex than the LSTM example given in
`medium_article_realtime_predictions.py`. In contrast, this file is not a faithful reproduction of the tutorial, but
rather a more complex example that builds upon the concepts introduced in the tutorial.

This file is intended to be used as a reference for me in the future, and to provide a more comprehensive understanding
of LSTM networks in the context of stock market predictions.

Goals of the tutorial, and thus this file, include
    1. Downloading stock market data from Yahoo Finance.
    2. Split train-test data and also perform some data normalisation.
    3. Go over and apply a few averaging techniques that can be used for one-step-ahead predictions.

Constants:
    MODULE_LEVEL_CONSTANT1 (int): A module-level constant.

Examples:
    (Any example implementations for this file)::
        
        $ bar = 1
        $ foo = bar + 1

Todo:
    * Work through referenced (url) tutorial and implement the code in this file.

References:
    Article author: Thushan Ganegedara

    Article created on: 10 Dec 2024

    Style guide: `Google Python Style Guide`_

Notes:
    File version
        0.1.0
    Project
        Learning_Realtime-Price-Prediction_toolkit
    Path
        src/examples/datacamp_stockmarket_predictions.py
    Author
        Cameron Aidan McEleney < c.mceleney.1@research.gla.ac.uk >
    Created
        29 May 2025
    IDE
        PyCharm
        
.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html

.. _Thushan Ganegedara which can be found on Datacamp:
    https://www.datacamp.com/tutorial/lstm-python-stock-market
"""

__all__ = ["PredictStockMarket"]

# Standard library imports
import datetime as dt
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any, Optional, Union
import urllib.request

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import yfinance as yf

# Local application imports

# Module-level constants
MODULE_LEVEL_CONSTANT1: int = 1
"""A module-level constant with in-line docstring."""


class PredictStockMarket:
    """Class to predict the prices of a listed company's stock price using Long Short-Term Memory (LSTM) networks.

    Description for class.

    Attributes:
        ticker_symbol (str): Unique identifier for a listed company, which we use to download their data.

    Todo:
        *
    """

    def __init__(self, ticker_symbol: str = "AAL"):
        """Class to predict the prices of a listed company's stock price using Long Short-Term Memory (LSTM) networks.

        Args:
            ticker_symbol: Unique identifier for a listed company, which we use to download their data.
        """
        self.ticker_symbol = ticker_symbol

        self._dataframe: Union[pd.DataFrame, None] = None
        self._data_train: Union[NDArray, None] = None
        self._data_test: Union[NDArray, None] = None
        self._data_all = None

        self._split_point: int | None = None
        self._window_size: int | None = None
        self._mechanism: str | None = None

        # Containers
        self._plot_info: dict = {}

    def download_data(
            self,
            data_source: str = 'yfinance',
            start_date: str = '1970-01-02',
            end_date: str = '2025-01-01'
    ) -> pd.DataFrame | None:
        """Download historical stock data from a specified data source.

        Args:
            data_source: Repository from which to download the data. Currently only 'yfinance' is supported.
            start_date: Start date for the data download in 'YYYY-MM-DD' format.
            end_date: End date for the data download in 'YYYY-MM-DD' format.

        Todo:
            * Generate a generic API key for Alpha Vantage and use it to download data.

        Returns:
            DataFrame containing the stock data if successful, None otherwise.
        """

        filename_saved_data = f'stock_market_data-{self.ticker_symbol}.csv'

        # Columns we want to keep in the time series data as per the tutorial.
        data_cols_time_series = ['Date', 'Open', 'High', 'Low', 'Close']

        # Early check of file existence also serves as a guard against unnecessary downloads.
        if os.path.exists(filename_saved_data):
            print("File already exists. Loading data from CSV.")
            df = pd.read_csv(
                filename_saved_data,
                parse_dates=['Date'],
                usecols=data_cols_time_series
            )
            # print(df)
            self._dataframe = df
            return df

        # Data file was not found; proceed to download from the specified source.
        df = None
        if data_source == 'yfinance':
            raw = yf.download(
                tickers=self.ticker_symbol,
                start=start_date,
                end=end_date,
            )

            df = (
                raw
                .droplevel('Ticker', axis=1)
                .filter(items=data_cols_time_series, axis=1)
                .reset_index()
                .rename(columns={'index': 'Date'})
            )

        elif data_source == 'kaggle':
            df = pd.read_csv(os.path.join('Stocks', 'hpq.us.txt'), delimiter=',',
                             usecols=data_cols_time_series)

        elif data_source == 'alphavantage':
            # This is the approach favoured by the author of the tutorial, but requires an API key.
            api_key = None

            # JSON file with all the stock market data for AAL from the last 20 years
            url_string = (("https://www.alphavantage.co/query?"
                          "function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s")
                          % (self.ticker_symbol, api_key))

            # Grab the data from the url and store date, low, high, volume, close, open values to a Pandas DataFrame.
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                # extract stock market data
                data = data['Time Series (Daily)']
                df = pd.DataFrame(columns=data_cols_time_series)
                for k, v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                                float(v['4. close']), float(v['1. open'])]
                    df.loc[-1, :] = data_row
                    df.index = df.index + 1

        else:
            logging.warning(f"Data source '{data_source}' is not supported.")

        if df is not None:
            print(f"Data downloaded and saved to: {filename_saved_data}")
            # Save to CSV (no index so Date is a column)
            df.to_csv(filename_saved_data)
            self._dataframe = df

        return df
    
    def prepare_data(self) -> Any:
        """Prepare the stock data for training and testing.

        Returns:
            Description of returned objects.
        """
        # First calculate the mid prices from the highest and lowest
        high_prices = self._dataframe['High'].to_numpy()
        low_prices = self._dataframe['Low'].to_numpy()
        mid_prices = (high_prices + low_prices) / 2.0

        # Split the data; demanding up to 11000 data points for training, and the rest for testing.
        # Note: The tutorial uses 11000 data points for training, but this is not a hard limit.
        self._split_point = min(int(high_prices.shape[0] / 2), 11000)

        # print(self._split_point)

        data_train = mid_prices[:self._split_point]
        data_test = mid_prices[self._split_point:]

        self._data_train = data_train
        self._data_test = data_test

        return data_train, data_test

    def normalise_data(self) -> Any:
        """Normalise the data by splitting the full dataset into windowed segments, and then applying a ``scaler``."""

        # Scale the data to be between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))

        self._data_train = self._data_train.reshape(-1, 1)
        self._data_test = self._data_test.reshape(-1, 1)

        # Train the Scaler with training data and smooth data
        # Note: The tutorial uses a smoothing window size of 2500 for 4 iterations, but this is not a hard limit.
        di, smoothing_window_size = 0, 1000
        for di in range(0, self._data_test.shape[0] % smoothing_window_size, smoothing_window_size):
            # Added 'window' to guard against empty slices
            window = self._data_train[di:di + smoothing_window_size, :]
            if window.size == 0:
                break
            scaler.fit(window)
            self._data_train[di:di + smoothing_window_size, :] = scaler.transform(window)

        # You normalize the last bit of remaining data
        scaler.fit(self._data_train[di + smoothing_window_size:, :])
        self._data_train[di + smoothing_window_size:, :] = (
            scaler.transform(self._data_train[di + smoothing_window_size:, :])
        )

        # Reshape both train and test data
        self._data_train = self._data_train.reshape(-1)

        # Normalise test data
        self._data_test = scaler.transform(self._data_test).reshape(-1)

        # Now perform exponential moving average smoothing
        # So the data will have a smoother curve than the original ragged data
        EMA = 0.0
        gamma = 0.1
        for ti in range(self._split_point):
            EMA = gamma * self._data_train[ti] + (1 - gamma) * EMA
            self._data_train[ti] = EMA

        # Used for visualisation and test purposes
        self._data_all = np.concatenate([self._data_train, self._data_test], axis=0)

        return self._data_all

    def predict_by_averaging(
            self,
            mechanism='standard',
            window_size: int = 100
    ) -> tuple[NDArray, ...] | None:
        """Generate predictions for future stock prices.

        Averaging mechanisms allow you to predict (often a one-time step ahead) by representing the future stock price
        as an average of the previously observed stock prices. Mechanisms:

        Standard average:
            Average the stock prices in the range :math:`x_{t+1} = [x_{t-N},x_{t}]` where:
            :math:`x_{t}` was the stock price; :math:`t` is the indexed date in the time series; and :math:`N` is the
            length of the window.
            :math:`N`.

        Exponential moving average (EMA):
            Calculate
            :math:`x_{t+1} = EMA_{t} = \\gamma \\cdot EMA_{t-1} + x_{t} \\cdot (1 - \\gamma)`
            for a time :math:`t`; starting with :math:`EMA_{t=0} = 0`. The contributions of previously calculations
            to the current EMA is controlled by a unit weighting factor :math:`\\gamma = [0, 1]`.

        Args:
            mechanism: Averaging mechanism to use for predictions. Options include ['normal', 'exponential'].
            window_size: Size of the window to use for averaging. Default is 100 (days).

        Returns:
            The following data about the predictions are returned in the tuple.

            * ``dates`` - Dates
            * ``prices`` - Prices
            * ``errors`` - Mean square error (MSE) of the predicted prices.
        """

        mechanism = mechanism.lower()

        valid_mechanisms = {'Standard': ['normal', 'standard'],
                            'Exponential moving': ['exponential', 'exp', 'ema']
                            }

        found_mech = next(
            (key for key, synonyms in valid_mechanisms.items() if mechanism in synonyms),
            None
        )

        if found_mech is None:
            logging.warning(f"Provided unsupported/unknown mechanism: {mechanism}")
            return None
        else:
            self._mechanism = found_mech
            self._plot_info['Mechanism'] = found_mech + ' ' + 'average'

        self._window_size = window_size
        n_data_all = self._data_all.size

        mse_errors = []
        preds_dates, preds_prices = [], []  # 'preds' == 'predictions'

        if mechanism in valid_mechanisms['Standard']:

            # Block comment - original article's code.
            """
            N = self._data_all.size
            
            std_avg_predictions = []
            std_avg_x = []
            mse_errors = []
            
            for pred_idx in range(window_size, N):

                if pred_idx >= N:
                    last_date = self._dataframe['Date'].iloc[-1]
                    date = last_date.strftime('%Y-%m-%d') + dt.timedelta(days=1)
                else:
                    date = self._dataframe.loc[pred_idx, 'Date']

                std_avg_predictions.append(np.mean(self._data_train[pred_idx - window_size:pred_idx]))
                mse_errors.append((std_avg_predictions[-1] - self._data_train[pred_idx]) ** 2)
                std_avg_x.append(date)
            """

            for idx in range(window_size, n_data_all):
                preds_dates.append(self._dataframe['Date'].iloc[idx])

                window = self._data_all[idx - window_size:idx]
                preds_prices.append(window.mean())

                mse_errors.append((preds_prices[-1] - self._data_all[idx]) ** 2)

            logging.info(f"MSE error for standard averaging: {0.5 * np.mean(mse_errors):.5f}")  # type:ignore

        elif mechanism in valid_mechanisms['Exponential moving']:

            ema_t = 0.0  #: exponential moving average' at current time 't'
            gamma = 0.5  #: weighting factor

            preds_prices.append(ema_t)

            for idx in range(1, n_data_all):
                preds_dates.append(self._dataframe['Date'].iloc[idx])

                # Build EMA from last 'true' value
                ema_t = ema_t * gamma + (1.0 - gamma) * self._data_all[idx - 1]
                preds_prices.append(ema_t)

                mse_errors.append((preds_prices[-1] - self._data_all[idx]) ** 2)

            logging.info(f"MSE error for EMA averaging: {0.5 * np.mean(mse_errors):.5f}")  # type:ignore

        return (np.array(preds_dates, dtype="datetime64"),
                np.array(preds_prices, dtype=float),
                np.array(mse_errors, dtype=float))

    def visualise_data(self, df: Optional[pd.DataFrame] = None) -> None:
        """Visualise the stock data.

        Args:
            df: DataFrame containing the stock data.
        """
        df = self._dataframe if df is None else df

        # Plot
        plt.figure(figsize=(18, 9))

        plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)

        plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Mid Price', fontsize=18)

        plt.show()

    def visualise_averaging_based_predictions(
            self,
            std_avg_predictions: np.ndarray,
            std_avg_predictions_errors: Optional[np.ndarray] = None,
            window_size: Optional[int] = None,
            style: str = 'mpl'
    ) -> None:

        rmse = np.sqrt(std_avg_predictions_errors) if std_avg_predictions_errors else None

        if window_size is None:
            if self._window_size is None:
                logging.warning(f"PredictStockMarket.visualise_predictions_normal_average: No window size specified, "
                                f"and no window_size held by class instance.",
                                )
                raise KeyError("No window size specified.")

            window_size = self._window_size

        else:
            if window_size != self._window_size:
                logging.warning(f"PredictStockMarket.visualise_predictions_normal_average: Method's window_size doesn't"
                                f"match class instance's attribute. Possible user error.")
                raise ValueError("No window size specified.")

        if style == 'mpl':

            fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')

            ax.plot(range(self._data_all.size), self._data_all, color='blue',
                    label='Source')

            xlim_lower = window_size if self._mechanism == 'Standard' else 0
            ax.plot(range(xlim_lower, self._data_all.size), std_avg_predictions, color='orange',
                    label='Predicted')

            ax.set(title=f"Stock Market Predictions | {self.ticker_symbol} | {self._plot_info['Mechanism']}",
                   xlabel="Date", ylabel="Mid Price")

            ax.legend(title='Price data')
            plt.show()

        elif style == 'plotly':

            df_true = pd.DataFrame({
                'Date': self._dataframe['Date'],
                'Price': self._data_all,
                'Type': "Source",
            })

            preds_start = window_size if self._mechanism == 'Standard' else 0
            df_pred = pd.DataFrame({
                'Date': self._dataframe['Date'].iloc[preds_start:self._data_all.size].reset_index(drop=True),
                'Price': std_avg_predictions,
                'Type': 'Predicted',
                'Error': (rmse
                          if std_avg_predictions_errors is not None
                          else None)
            })

            df_plot = pd.concat([df_true, df_pred], ignore_index=True)

            fig = px.line(
                data_frame=df_plot,
                x="Date",
                y="Price",
                title=f"Stock Market Predictions | "
                      f"{self.ticker_symbol}",
                subtitle=f'{self._plot_info.get('Mechanism')}',
                hover_name='Type',
                hover_data={
                    'Date': "|%B %d, %Y",
                    'Price': ':.4f',
                    'Error': False,
                    'Type': False
                    },
                color="Type",
                color_discrete_sequence=px.colors.qualitative.Prism,
            )

            self._apply_plotly_styling(fig, df_plot)

            fig.show()

    @staticmethod
    def _apply_plotly_styling(fig: px.line, df: pd.DataFrame) -> None:
        """
        Private helper to apply final layout tweaks to plotly figures.

        Changes include:
            colors, fonts, axes style, legend placement, rangeselector buttons, etc.

        This method should not handle the means of saving plots
        """

        # Colors and templates
        fig.update_layout(
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        fig.update_layout(
            font=dict(family="Arial, sans‐serif", size=12),
            title_font=dict(family="Arial, sans‐serif", size=18),
            legend_font=dict(family="Arial, sans‐serif", size=11),
        )

        # Axes
        fig.update_xaxes(showgrid=False, showline=False)

        fig.update_yaxes(
            range=[0, df['Price'].max() * 1.02],
            showgrid=False,
            gridcolor="LightGray",
            gridwidth=0.75,
            zeroline=True,
            zerolinecolor="LightGray",
            zerolinewidth=0.75,
        )

        # Extra
        fig.update_layout(
            autosize=True,
            legend=dict(
                title="Price data", orientation="v",
                x=0.98, xanchor="right",
                y=0.95, yanchor="top",
                bgcolor="rgba(0,0,0,0)",
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial, sans‐serif",
            )
        )

        # Widgets
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]),
                    bgcolor="rgba(230, 230, 230, 0.5)",
                    activecolor="Gray",
                    # Widget positioning
                    x=0.5, xanchor="center",
                    y=1.05, yanchor="top",
                    font=dict(size=12),
                ),
                rangeslider=dict(visible=True, thickness=0.1),
                type="date"
            )
        )


class DataGeneratorSeq:
    def __init__(self, batch_size: int, num_unroll: int, prices: Optional[NDArray] = None, ):
        """Docstring.

        Args:
            batch_size: number of parallel sequences per batch
            num_unroll: how many timesteps to unroll per sequence
            prices: 1D array of training-data prices.
        """

        # Training data closing prices
        self._prices: Union[None, NDArray] = None
        self._prices_length: Union[None, int] = None

        # Set by self.initialise()
        self._batch_size: Union[None, int] = None
        self._num_unroll: Union[None, int] = None

        self._segments: Union[None, int] = None  # number of “segments” = floor((_prices_length)/(batch_size))
        self._cursor: Union[None, NDArray] = None  # one cursor position per batch‐slot

        # Currently initialising out with __init__ for debugging purposes.
        self.initialise(prices=prices, batch_size=batch_size, num_unroll=num_unroll)

    def initialise(self, batch_size: int, num_unroll: int, prices: Optional[NDArray] = None) -> None:
        """
        Prepare all internal indices for batch/unroll splitting.

        Args:
            batch_size: number of parallel sequences per batch
            num_unroll: how many timesteps to unroll per sequence
            prices: 1D array of training-data prices.

        Raises:
            ValueError: if prices is None (and parent’s data isn’t loaded), or if batch_size/num_unroll do not fit.
        """
        if prices is None:
            raise ValueError("Cannot initialise - no prices passed and parent's data (dataframe) is empty.")

        self._prices = prices
        self._prices_length = len(prices) - num_unroll

        self._batch_size = batch_size
        self._num_unroll = num_unroll

        self._segments = self._prices_length // self._batch_size
        if self._segments < 1:
            raise ValueError(
                f"batch_size={batch_size} is too large for prices_length={self._prices_length}. "
                f"Each segment must have at least one element."
            )

        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self) -> tuple[NDArray, ...]:
        """Return a single batch of size ``batch_size`` per cursor.

        Returns:
            ``batch_data`` - 1D array of length `batch_size`, where batch_data[i] = prices[cursor[i]].

            ``batch_labels`` - 1D array of length `batch_size`, where
            batch_labels[i] = prices[cursor[i] + random_offset].

        """

        if self._prices is None or self._cursor is None:
            raise RuntimeError("Must call `initialise(...)` before requesting `next_batch()`.")

        batch_data = np.zeros(self._batch_size, dtype=np.float32)
        batch_labels = np.zeros(self._batch_size, dtype=np.float32)

        for i in range(self._batch_size):
            if self._cursor[i] + 1 >= self._prices_length:
                self._cursor[i] = np.random.randint(i * self._segments, (i + 1) * self._segments)

            batch_data[i] = self._prices[self._cursor[i]]
            batch_labels[i] = self._prices[self._cursor[i] + np.random.randint(0, 5)]

            self._cursor[i] = (self._cursor[i] + 1) % self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self) -> tuple[list[NDArray], ...]:
        """Produce ``num_unroll`` steps of consecutive batches.

        Returns:
            ``unroll_data`` - each entry is a 1D array of size ``batch_size``.

            ``unroll_labels`` - corresponding labels at each timestep.
        """

        unroll_data, unroll_labels = [], []

        for _ in range(self._num_unroll):
            batch_data, batch_labels = self.next_batch()
            unroll_data.append(batch_data)
            unroll_labels.append(batch_labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        """Choose new random starting indices for all batch slots.

        Each index is drawn uniformly from the ith segment of length ``self._segments``.
        """

        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b + 1) * self._segments, self._prices_length - 1))


@dataclass
class LSTMHyperParameters:
    """Container for hyperparameters required to execute Long Short-Term Memory (LSTM) model.

    This dataclass replaces ``self._hyperparameters: dict`` which was originally found in `PredictUsingLSTM.__init__`.

    Attributes:
        num_unrollings: Number of timesteps the model looks into the future.
        batch_size: Samples per batch.
        num_nodes: Hidden nodes per layer of deep LSTM stack.
        dropout: Dropout rate between LSTM layers.
        learning_rate: Initial learning rate.
        min_learning_rate: Minimum learning rate.
    """
    dimensionality: int = 1
    num_unrollings: int = 50
    batch_size: int = 500
    num_nodes: list[int] | tuple[int, ...] = (200, 200, 150)
    dropout: float = 0.2
    learning_rate: float = 0.0001
    min_learning_rate: float = 0.000001
    n_layers: int = field(init=False)

    def __post_init__(self):
        self.n_layers = len(self.num_nodes)


class PredictUsingLSTM(PredictStockMarket):
    """A subclass of `PredictStockMarket` that uses Long Short-Term Memory (LSTM) machine learning to predict a
    company's stock price.

    Examples:
        >>> hparams = LSTMHyperParameters(batch_size=256)
        >>> company = 'TSLA'
        >>> model = PredictUsingLSTM(hparams, ticker_symbol=company)

    Todo:
        * Consider whether we could replace any code, including classes, by using `models.Sequential.fit()`.

    """

    def __init__(self, hyperparameters: Optional[LSTMHyperParameters] = LSTMHyperParameters(), **kwargs):
        """Docstring.

        Args:
            hyperparameters: Dataclass to neatly gather the hyperparameters.
            kwargs:
                Keyword arguments passed to the parent `PredictStockMarket.__init__`.
                Options include -``ticker_symbol``.
        """
        super().__init__(**kwargs)

        self.hparams: LSTMHyperParameters = hyperparameters
        self.model = self._build_model()

        self.train_mse_ot, self.test_mse_ot = [], []
        self.prediction_dates_ot, self.predictions_ot = [], []

        self.loss_nondecrease_count, self.loss_nondecrease_threshold = 0, 2

    def _build_model(self):
        """Docstring."""

        model = models.Sequential()
        model.add(
            layers.Input(shape=(self.hparams.num_unrollings, self.hparams.dimensionality))
        )

        for i, units in enumerate(self.hparams.num_nodes):
            model.add(
                layers.LSTM(
                    units=units,
                    return_sequences=(i < self.hparams.n_layers - 1),
                    dropout=self.hparams.dropout,
                )
            )

        model.add(layers.Dense(1))  # Final regression output

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hparams.learning_rate),
            loss='mean_squared_error',
        )

        model.summary()

        return model

    def _predict_autoregressively(self, start_sequence, timesteps):
        """Make predictions autoregressively."""

        input_sequence = start_sequence.copy().reshape(1, -1, 1)
        predictions = []

        for _ in range(timesteps):
            pred = self.model.predict(input_sequence, verbose=0)[0][0]
            predictions.append(pred)
            new_input = np.append(input_sequence[0, 1:], [[pred]], axis=0)
            input_sequence = new_input.reshape(1, -1, 1)

        return predictions

    def _make_predictions(self, data_test, epoch):
        """Docstring."""

        prediction_dates, predictions, mse_test_loss = [], [], []

        for i in data_test:

            # Track prediction date range for plotting
            prediction_dates.extend(self._dataframe['Date'].iloc[i:i + self.hparams.num_unrollings])
            # if epoch == 0:
            #     self.x_axis_seq.append(list(range(i, i + self.hparams.num_unrollings)))

            start = self._data_all[i - self.hparams.num_unrollings:i].reshape(1, -1)
            preds = self._predict_autoregressively(start, self.hparams.num_unrollings)
            predictions.append(preds)

            true = self._data_all[i:i + self.hparams.num_unrollings]
            mse = np.mean(0.5 * (np.array(preds) - true) ** 2)
            mse_test_loss.append(mse)

        self.predictions_ot.append(predictions)
        self.prediction_dates_ot = np.array(prediction_dates, dtype='datetime64')

        current_test_mse = np.mean(mse_test_loss)
        self.train_mse_ot.append(current_test_mse)

        # Note - here there's a `+=` operator and not an assignment `=`
        self.loss_nondecrease_count += (
            1
            if len(self.test_mse_ot) > 0 and current_test_mse > min(self.test_mse_ot)
            else 0
        )

        if self.loss_nondecrease_count > self.loss_nondecrease_threshold:
            # 'lr' alias for 'loss rate'
            old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            new_lr = max(0.5 * old_lr, self.hparams.min_learning_rate)

            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            print(f"\t\tDecreasing learning rate to: {new_lr}")
            self.loss_nondecrease_count = 0

        print(f"\tTest Mean Squared Error: {current_test_mse:.6f}")
        print(f"\tFinished predictions.")

    @staticmethod
    def save_predictions(predictions_: np.ndarray,
                         prediction_dates_ : np.ndarray,
                         filename_: str = 'lstm_predictions.csv') -> None:
        """Save model predictions to a CSV file to avoid expensive recomputations."""

        df = pd.DataFrame({
            'Date': prediction_dates_.astype('datetime64'),
            'Predicted': predictions_,
        })

        df.to_csv(filename_, index=False)
        print(f"Predictions saved to: {filename_}")

    @staticmethod
    def load_predictions(filename_: str = 'lstm_predictions.csv') -> Optional[tuple[NDArray, ...]]:
        """Load previously saved model predictions for plotting.

        Note:
             Invoking this method does not populate the class with many commonly required attributes. Therefore, this
             method is not a viable substitute for skipping the usual startup of ``PredictStockMarket``.

        Args:
            filename_: Filename including path to load predictions from.

        Returns:
            A tuple of the loaded data if successful, else None. The contents of the tuple are as follows.

                - ``Date``
                - ``Predictions``
        """
        if os.path.exists(filename_):
            print(f"Loading predictions from {filename_}")
            df = pd.read_csv(filename_, parse_dates=['Date'])
            return df['Date'].to_numpy(dtype='datetime64'), df.Predicted.to_numpy(dtype=float)
        else:
            print(f"No saved predictions found at {filename_}")
            return None

    def train_lstm_model(self, epochs: int = 30):
        """Docstring."""

        # ELSE values taken directly from tutorial.
        test_points_seq = (
            np.arange(11000, 12000, 50, dtype=int)
            if self._data_test is None
            else np.arange(self._split_point,
                           self._split_point + self.hparams.batch_size,
                           self.hparams.num_unrollings,
                           dtype=int)
        )

        train_seq_len = self._data_train.size
        steps = train_seq_len // self.hparams.batch_size  # TODO. rename variable - 'steps' offers no insight.
        data_gen = DataGeneratorSeq(
            prices=self._data_train,
            batch_size=self.hparams.batch_size,
            num_unroll=self.hparams.num_unrollings
        )

        for ep in range(epochs):
            print(f"Epoch {ep+1}")
            avg_loss = 0

            for step in range(steps):
                data, labels = data_gen.unroll_batches()

                # Using `X` and `y` have standard ML variable meanings
                X = np.stack(data, axis=1)[..., np.newaxis]
                y = np.stack(labels, axis=1)[..., np.newaxis]

                X = X[:, :, 0]
                y = y[:, -1, 0].reshape(-1, 1)

                loss = self.model.train_on_batch(X, y)
                avg_loss += loss

            avg_loss /= steps
            self.train_mse_ot.append(avg_loss)
            print(f"Average training loss: {avg_loss: .6f}")

            self._make_predictions(data_test=test_points_seq, epoch=ep)

    def visualise_lstm_based_predictions(self,
                                         predictions: np.ndarray,
                                         prediction_dates: np.ndarray,
                                         style: str = 'mpl'):
        """Docstring."""

        if style == 'mpl':
            fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')

            ax.plot(range(self._data_all.size), self._data_all, color='blue', label='Source')
            ax.plot(prediction_dates, predictions, color='orange', label='Predicted')

            ax.set(
                title=f"Stock Market Predictions | {self.ticker_symbol} | LSTM",
                xlabel="Date",
                ylabel="Mid. Price",
            )

            ax.legend(title="Price data")
            plt.show()


def initial_run() -> None:
    ps = PredictStockMarket(ticker_symbol='AAL')
    ps.download_data(data_source='yfinance')
    ps.visualise_data()


def std_avg_run() -> None:
    ps = PredictStockMarket(ticker_symbol='AAL')
    ps.download_data(data_source='yfinance')
    ps.prepare_data()
    ps.normalise_data()
    prediction_dates, predictions_prices, mse_errors = ps.predict_by_averaging(mechanism='exp')
    ps.visualise_averaging_based_predictions(std_avg_predictions=predictions_prices, style='plotly')


def lstm_test_run():
    dg = PredictUsingLSTM()
    dg.download_data(data_source='yfinance')
    dg.prepare_data()
    dg.normalise_data()

    dg.train_lstm_model(epochs=2)

    # Plotting predictions across all epochs
    all_mid_data = dg._data_all
    predictions_over_time = dg.predictions_ot
    x_axis_seq = [
        list(range(i, i + dg.hparams.num_unrollings))
        for i in np.arange(
            dg._split_point,
            dg._split_point + dg.hparams.batch_size,
            dg.hparams.num_unrollings,
            dtype=int
        )
    ]

    best_prediction_epoch = 1  # manually selected for now

    plt.figure(figsize=(18, 18))
    plt.subplot(2, 1, 1)
    plt.plot(range(len(all_mid_data)), all_mid_data, color='b')

    start_alpha = 0.25
    step_alpha = (1.0 - start_alpha) / len(predictions_over_time[::1])
    alpha = np.arange(start_alpha, 1.01, step_alpha)

    for p_i, p in enumerate(predictions_over_time[::1]):
        for xval, yval in zip(x_axis_seq, p):
            plt.plot(xval, yval, color='r', alpha=alpha[p_i])

    plt.title('Evolution of Test Predictions Over Time', fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)

    plt.subplot(2, 1, 2)
    plt.plot(range(len(all_mid_data)), all_mid_data, color='b')

    for xval, yval in zip(x_axis_seq, predictions_over_time[best_prediction_epoch]):
        plt.plot(xval, yval, color='r')

    plt.title('Best Test Predictions Over Time', fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.show()

    # dg.save_predictions(
    #     predictions_=flat_predictions,
    #     prediction_dates_=prediction_dates,
    # )
    # dg.visualise_lstm_based_predictions(
    #     predictions=flat_predictions,
    #     prediction_dates=prediction_dates,
    #     style='mpl',
    # )


if __name__ == "__main__":
    # dg = PredictUsingLSTM()
    # predictions, dates = dg.load_predictions()
    # dg.visualise_lstm_based_predictions(
    #     predictions=predictions.reshape(-1),
    #     prediction_dates=dates.reshape(-1),
    # )
    lstm_test_run()
