import data_loader
import linear_model as lr
import lstm_model
import plotter
import pandas as pd
import numpy as np

def main():
    # --- Configuration ---
    TICKER = 'AAPL'
    START_DATE = '2015-01-01'
    END_DATE = '2024-12-31'
    LSTM_TIME_STEP = 60
    # Use fewer epochs for a quick test, or more (e.g., 50) for better results
    LSTM_EPOCHS = 50 

    # --- 1. Load Data ---
    try:
        data = data_loader.fetch_data(TICKER, START_DATE, END_DATE)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # --- 2. Linear Regression Model ---
    print("\n--- Running Linear Regression Model ---")
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = lr.prepare_data_lr(data)
    
    lr_model = lr.train_lr_model(X_train_lr, y_train_lr)
    
    lr_predictions = lr.evaluate_model(lr_model, X_test_lr, y_test_lr)
    
    # --- 3. LSTM Model ---
    print("\n--- Running LSTM Model ---")
    X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler = \
        lstm_model.prepare_data_lstm(data, LSTM_TIME_STEP)

    lstm_model_instance = lstm_model.build_lstm_model(input_shape=(LSTM_TIME_STEP, 1))
    
    lstm_model_instance = lstm_model.train_lstm_model(
        lstm_model_instance, X_train_lstm, y_train_lstm, epochs=LSTM_EPOCHS
    )
    
    lstm_predictions = lstm_model.evaluate_lstm_model(
        lstm_model_instance, scaler, X_test_lstm, y_test_lstm
    )

    # --- 4. Plotting ---
    print("\nGenerating plots...")
    
    # Plot 1: Linear Regression
    # We need to find the split point for plotting
    train_plot_lr = data.loc[y_train_lr.index]
    test_plot_lr = data.loc[y_test_lr.index]
    
    plotter.plot_results(train_plot_lr, test_plot_lr,
                         f'{TICKER} Stock Price Prediction (Linear Regression)',
                         ('LR Predictions', lr_predictions))

    # Plot 2: LSTM
    # The LSTM test data indices are different due to the time_step
    train_split_idx_lstm = int(len(data) * 0.8)
    train_plot_lstm = data.iloc[:train_split_idx_lstm]
    
    # The test plot data aligns with the predictions
    test_plot_lstm = data.iloc[train_split_idx_lstm + LSTM_TIME_STEP:]

    # Adjust if predictions are not the same length as test_plot_lstm
    if len(lstm_predictions) != len(test_plot_lstm):
        # This is common. The `create_dataset` function results in
        # `len(test_data) - time_step` samples.
        # `test_data` starts at `train_split_idx_lstm - LSTM_TIME_STEP`
        # `len(test_data)` = `len(data) - (train_split_idx_lstm - LSTM_TIME_STEP)`
        # `len(y_test_lstm)` = `len(test_data) - time_step`
        # `len(y_test_lstm)` = `len(data) - train_split_idx_lstm` = `test_size`
        test_plot_lstm = data.iloc[train_split_idx_lstm:]
    
    plotter.plot_results(train_plot_lstm, test_plot_lstm,
                         f'{TICKER} Stock Price Prediction (LSTM)',
                         ('LSTM Predictions', lstm_predictions))

    # Plot 3: Combined
    # We plot both models on the same (LSTM) test set for comparison
    plotter.plot_results(train_plot_m, test_plot_lstm,
                         f'{TICKER} Stock Price Prediction (All Models)',
                         ('LR Predictions (Aligned)', lr_predictions[-len(test_plot_lstm):]),
                         ('LSTM Predictions', lstm_predictions))

    print("\n--- Project Complete ---")
    print("NOTE: These predictions are for educational purposes and are NOT financial advice.")


if __name__ == "__main__":
    main()
