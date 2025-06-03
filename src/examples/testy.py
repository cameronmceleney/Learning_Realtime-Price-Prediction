#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (Same docstring as before)

__all__ = ["PredictStockMarket"]

# Standard library imports
# (Same imports as before)

# [...] All your existing class and function definitions here remain unchanged

# Updated lstm_test_run with correct per-epoch prediction visualisation

def lstm_test_run():
    dg = PredictUsingLSTM()
    dg.download_data(data_source='yfinance')
    dg.prepare_data()
    dg.normalise_data()

    dg.train_lstm_model(epochs=30)

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

    best_prediction_epoch = 28  # manually selected for now

    plt.figure(figsize=(18, 18))
    plt.subplot(2, 1, 1)
    plt.plot(range(len(all_mid_data)), all_mid_data, color='b')

    start_alpha = 0.25
    step_alpha = (1.0 - start_alpha) / len(predictions_over_time[::3])
    alpha = np.arange(start_alpha, 1.01, step_alpha)

    for p_i, p in enumerate(predictions_over_time[::3]):
        for xval, yval in zip(x_axis_seq, p):
            plt.plot(xval, yval, color='r', alpha=alpha[p_i])

    plt.title('Evolution of Test Predictions Over Time', fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.xlim(11000, 12500)

    plt.subplot(2, 1, 2)
    plt.plot(range(len(all_mid_data)), all_mid_data, color='b')

    for xval, yval in zip(x_axis_seq, predictions_over_time[best_prediction_epoch]):
        plt.plot(xval, yval, color='r')

    plt.title('Best Test Predictions Over Time', fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.xlim(11000, 12500)
    plt.show()

if __name__ == "__main__":
    # dg = PredictUsingLSTM()
    # predictions, dates = dg.load_predictions()
    # dg.visualise_lstm_based_predictions(
    #     predictions=predictions.reshape(-1),
    #     prediction_dates=dates.reshape(-1),
    # )
    lstm_test_run()
