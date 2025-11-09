Stock Price Prediction Project (Linear Regression vs. LSTM)

This project demonstrates how to build, train, and evaluate two different models for predicting stock prices using historical data:

Linear Regression: A simple, classical model used as a baseline.

LSTM (Long Short-Term Memory): A powerful deep learning model (a type of Recurrent Neural Network) specifically designed for sequential data.

DISCLAIMER: This is an educational project, not financial advice. Do not use these models for real-world trading. Stock market prediction is extremely complex, and these models are far too simple to capture all the variables that influence market behavior.

How to Run This Project

Prerequisites: You must have Python 3.8+ installed.

Setup Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install Dependencies:
Save the contents of requirements.txt into a file and run:

pip install -r requirements.txt


Run the Project:
Save all the .py files (main.py, data_loader.py, linear_regression_model.py, lstm_model.py, plotter.py) in the same directory. Then, simply run the main script:

python main.py


The script will:

Fetch data for AAPL stock.

Train and evaluate the Linear Regression model.

Train and evaluate the LSTM model (this will take a few minutes).

Display three plots showing the models' predictions.

Project File Structure

main.py: The main entry point. It controls the flow: fetches data, calls the LR model, calls the LSTM model, and plots the results.

requirements.txt: A list of all necessary Python libraries.

data_loader.py: A simple module to fetch stock data using the yfinance library.

linear_regression_model.py: Contains all code for preprocessing, training, and evaluating the Linear Regression model.

lstm_model.py: Contains all code for preprocessing, building, training, and evaluating the LSTM model.

plotter.py: A utility module to plot the final results using matplotlib.

Step-by-Step Explanation & Theory

Here is a detailed breakdown of each part of the project.

Step 1: Data Fetching (data_loader.py)

What it does: We use the yfinance library to download historical stock data from Yahoo Finance.

Code: The fetch_data function takes a stock ticker (e.g., 'AAPL'), a start_date, and an end_date.

Our Focus: For this project, we are only interested in the Close price of each day. This is the column we will try to predict.

Step 2: Model 1 - Linear Regression (linear_regression_model.py)

This model serves as a simple baseline. It's generally not suitable for stock prediction, but it's important to see why.

Theory: Linear Regression

Linear Regression (LR) finds the best-fitting straight line to a set of data. The formula is y = mx + b.

y is the value we want to predict (the dependent variable).

x is the value we use to make the prediction (the independent variable).

m is the slope of the line.

b is the y-intercept.

For stock prices, a simple LR model like price = m * date + b is useless. Instead, we use an autoregressive approach: we predict tomorrow's price (y) based on today's price (x).

Our Model: Price(tomorrow) = m * Price(today) + b

Limitation: This model is incredibly simple. It assumes the only thing that matters for tomorrow's price is today's price. It has no concept of "momentum" or trends over weeks or months.

Code Explanation:

prepare_data_lr:

We take the Close price data.

We create a new 'Target' column by shift(-1). This moves all prices up by one row, so row[i] has Price(today) and Target(today) has Price(tomorrow).

We drop the last row, which now has a NaN value.

X becomes the Close column (today's price), and y becomes the Target column (tomorrow's price).

We split into training and test sets (80/20 split). Crucially, we set shuffle=False. This is time-series data; we must not shuffle it. We train on the past and test on the most recent data.

train_lr_model:

We use LinearRegression from scikit-learn.

model.fit(X_train, y_train) finds the best m and b values using the training data.

evaluate_model:

model.predict(X_test) uses the trained model to predict prices for the test set.

Root Mean Squared Error (RMSE): This is our main metric. It measures the average "error" (in dollars) of our predictions. A lower RMSE is better.

R-squared (R2): Measures what percentage of the price's movement the model can explain. An R2 of 1.0 is perfect; 0.0 means the model is no better than just predicting the average price.

Step 3: Model 2 - LSTM (lstm_model.py)

This is a much more sophisticated model designed for sequential data.

Theory: RNNs and LSTMs

Recurrent Neural Networks (RNNs): Standard neural networks have no "memory" of past inputs. An RNN has a loop, allowing it to pass information from one step to the next. This "memory" lets it learn from sequences.

Problem: Simple RNNs suffer from the "vanishing gradient" problem. They have trouble learning patterns over long sequences (e.t., remembering a price from 60 days ago).

Long Short-Term Memory (LSTM): LSTMs are a special type of RNN that solve this problem. An LSTM "cell" is more complex than a simple RNN neuron. It has a series of "gates" that control its memory.

Cell State (the "memory"): A "conveyor belt" that runs through the whole sequence.

Forget Gate: Looks at the new input and the previous output, and decides what old information to throw away from the cell state.

Input Gate: Decides what new information from the current input is important enough to add to the cell state.

Output Gate: Looks at the updated cell state and decides what to output for this time step (e.g., the prediction).

Why for Stocks? An LSTM can look at the last 60 days of prices (our time_step) and learn complex patterns and long-term dependencies (like momentum or trends) that the Linear Regression model completely misses.

Code Explanation:

prepare_data_lstm:

Scaling: Neural networks require normalized data to work well. We use MinMaxScaler from scikit-learn to scale all prices to be between 0 and 1.

Train/Test Split: We split the scaled data 80/20.

Important: We make the test data overlap with the training data by time_step. Why? To predict the first day in the test set, the model needs the previous 60 days, which are still part of the training set.

create_dataset:

This is the most important preprocessing step. We convert our flat list of prices into sequences.

If time_step = 60, this function does the following:

X (features): It takes prices from day 0 to 59 and puts them in an array.

y (target): It takes the price from day 60.

It slides this "window" over by one day: X becomes days 1-60, y becomes day 61.

The result is X_train (e.g., [1800 samples, 60 features]) and y_train (e.g., [1800 samples, 1 target]).

Reshaping:

Keras (TensorFlow) LSTMs expect a 3D input shape: [samples, time_steps, features].

Our X_train is [1800, 60]. We reshape it to [1800, 60, 1].

samples: 1800 different 60-day windows.

time_steps: 60 days in each window.

features: 1 feature (the Close price).

build_lstm_model:

Sequential: A linear stack of layers.

LSTM(50, return_sequences=True, ...): Our first LSTM layer. It has 50 "units" (neurons). return_sequences=True means it passes its full sequence output (not just the last output) to the next layer.

Dropout(0.2): A regularization technique. It randomly "drops" 20% of neurons during training to prevent the model from "memorizing" the data (overfitting).

LSTM(50, return_sequences=False): Our second LSTM layer. return_sequences=False (the default) means it only passes the output from the very last time step to the next layer.

Dense(25): A standard, fully-connected neural network layer.

Dense(1): The final output layer with 1 neuron, which will output our single predicted price.

model.compile(optimizer='adam', loss='mean_squared_error'): We tell the model to use the 'adam' optimizer (a popular, effective choice) and to try to minimize the 'mean_squared_error' (which aligns with our RMSE evaluation).

train_lstm_model:

model.fit(...) starts the training process.

epochs=50: The model will go through the entire training dataset 50 times.

batch_size=32: The model updates its weights after looking at 32 samples at a time.

validation_split=0.1: It automatically sets aside 10% of the training data to check its performance as it trains, which helps us see if it's overfitting.

evaluate_lstm_model:

model.predict(X_test): Generates predictions. These predictions will be scaled (between 0 and 1).

scaler.inverse_transform: This is a critical step! We must "un-scale" the predictions and the actual y_test values to get them back into dollars.

We then calculate the RMSE on these unscaled, real-dollar values. You will notice the LSTM's RMSE is significantly lower (better) than the Linear Regression's.

Step 4: Plotting (plotter.py)

plot_results: This function uses matplotlib to create our charts.

It plots the original training data (blue), the actual test prices (orange), and the predicted prices (green/red).

This visual comparison is the best way to see how well each model performed. You'll see the LR model is just a slightly-shifted version of the previous day's price, while the LSTM model's predictions much more closely follow the actual trend of the test data.
