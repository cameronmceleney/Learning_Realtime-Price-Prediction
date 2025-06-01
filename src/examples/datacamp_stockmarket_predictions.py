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
import json
import logging
import os
from functools import cached_property
from typing import Any, Optional, Union
import urllib.request

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray, ArrayLike
import pandas as pd
# from pandas_datareader import data
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
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
        smoothing_window_size = 1000
        for di in range(0, self._data_test.shape[0] % smoothing_window_size, smoothing_window_size):
            # Added 'window' to guard against empty slices
            window = self._data_train[di:di + smoothing_window_size, :]
            if window.size == 0:
                break
            scaler.fit(window)
            self._data_train[di:di + smoothing_window_size, :] = scaler.transform(window)

        # You normalize the last bit of remaining data
        scaler.fit(self._data_train[di + smoothing_window_size:, :])
        self._data_train[di + smoothing_window_size:, :] = scaler.transform(self._data_train[di + smoothing_window_size:, :])

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

    def predict_by_averaging(self, mechanism='standard', window_size: int = 100) -> None | tuple:
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

            logging.info(f"MSE error for standard averaging: {0.5 * np.mean(mse_errors):.5f}")

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

            logging.info(f"MSE error for EMA averaging: {0.5 * np.mean(mse_errors):.5f}")

        return preds_dates, preds_prices, mse_errors

    def visualise_data(self, df: Optional[pd.DataFrame] = None) -> None:
        """Visualise the stock data.

        Args:
            df: DataFrame containing the stock data.
        """
        if df is None: df = self._dataframe

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
            original_data: Optional[np.ndarray] = None,
            window_size: Optional[int] = None,
            style: str = 'mpl'
    ) -> None:

        if original_data is None: original_data = self._data_all

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
            N = original_data.size

            fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')

            ax.plot(range(N), original_data, color='blue',
                    label='Source')

            xlim_lower = window_size if self._plot_info['Mechanism'] == 'Standard' else 0
            ax.plot(range(xlim_lower, N), std_avg_predictions, color='orange',
                    label='Predicted')

            ax.set(title=f"Stock Market Predictions | {self.ticker_symbol} | {self._plot_info.get('Mechanism')}",
                   xlabel="Date", ylabel="Mid Price")

            ax.legend(title='Price')
            plt.show()

        elif style == 'plotly':
            N = len(original_data)

            # Easier: split into two DataFrames, then concat with a “Type” column:
            df_true = pd.DataFrame({
                # "Index": list(range(N)),
                'Date': self._dataframe['Date'],
                'Price': original_data,
                'Type': "Source",
            })

            preds_start = window_size if self._plot_info['Mechanism'] == 'Standard' else 0
            df_pred = pd.DataFrame({
                # "Index": list(range(preds_start, N)),
                'Date': self._dataframe['Date'].iloc[preds_start:N].reset_index(drop=True),
                'Price': std_avg_predictions,
                'Type': 'Predicted',
                'Error': rmse,
            })

            df_plot = pd.concat([df_true, df_pred], ignore_index=True)
            df_plot["Date"] = pd.to_datetime(df_plot["Date"])

            # Now hand off to Plotly Express:
            fig = px.line(
                df_plot,
                x="Date", y="Price",
                color="Type", color_discrete_sequence=px.colors.qualitative.Prism,
                title=f"Stock Market Predictions | "
                      f"{self.ticker_symbol}",
                subtitle=f'{self._plot_info.get('Mechanism')}',
                hover_name='Type',
                hover_data={
                    'Date': "|%B %d, %Y",
                    'Price': ':.4f',
                    'Error': False,
                    'Type': False,
                }
            )

            # Colors and templates
            fig.update_layout(
                template='plotly_white',
                paper_bgcolor='white',
                plot_bgcolor='white',
                # margin={"l": 40, "r": 30, "t": 60, "b": 40},
            )

            fig.update_layout(
                font=dict(family="Arial, sans‐serif", size=12),
                title_font=dict(family="Arial, sans‐serif", size=18),
                legend_font=dict(family="Arial, sans‐serif", size=11),
            )

            # Axes
            fig.update_xaxes(showgrid=False, showline=False)

            fig.update_yaxes(
                range=[0, df_plot['Price'].max() * 1.02],
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
                    title=None, orientation="v",
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
                        x=0.5, xanchor="center",
                        y=1.02, yanchor="top",
                        font=dict(size=12),
                    ),
                    rangeslider=dict(visible=True, thickness=0.1),
                    type="date"
                )
            )

            fig.show()


def initial_run() -> None:
    ps = PredictStockMarket(ticker_symbol='AAL')
    ps.download_data(data_source='yfinance')
    ps.visualise_data()


def std_avg_run() -> None:
    ps = PredictStockMarket(ticker_symbol='AAL')
    ps.download_data(data_source='yfinance')
    ps.prepare_data()
    ps.normalise_data()
    _, predictions_prices, mse_errors = ps.predict_by_averaging(mechanism='exponential')
    ps.visualise_averaging_based_predictions(std_avg_predictions=predictions_prices,
                                             style='plotly')


if __name__ == "__main__":
    std_avg_run()
