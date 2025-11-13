# Stock Price Prediction Project

This project demonstrates how to build, train, and evaluate two different models for predicting stock prices using historical data:

### 1.Linear Regression: A simple, classical model used as a baseline.

### 2.LSTM (Long Short-Term Memory): A powerful deep learning model (a type of Recurrent Neural Network) specifically designed for sequential data.


## Project File Structure

- main.py: The main entry point. It controls the flow: fetches data, calls the LR model, calls the LSTM model, and plots the results.
- requirements.txt: A list of all necessary Python libraries.
- data_loader.py: A simple module to fetch stock data using the yfinance library.
- linear_regression_model.py: Contains all code for preprocessing, training, and evaluating the Linear Regression model.
- lstm_model.py: Contains all code for preprocessing, building, training, and evaluating the LSTM model.
- plotter.py: A utility module to plot the final results using matplotlib.

