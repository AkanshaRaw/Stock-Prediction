import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_dataset(dataset, time_step=1):
    """
    Converts an array of values into a dataset matrix.
    
    Args:
        dataset (np.array): The data to convert.
        time_step (int): The number of previous time steps to use as input.
        
    Returns:
        tuple: (np.array for X, np.array for y)
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        # `i` to `i + time_step` gives `time_step` elements
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def prepare_data_lstm(data, time_step=60):
    """
    Prepares data for the LSTM model.
    - Scales data
    - Creates train/test split
    - Creates sequences
    - Reshapes data for LSTM
    
    Args:
        data (pd.DataFrame): DataFrame with 'Close' prices.
        time_step (int): The sequence length.
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    print("Preparing data for LSTM model...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split data into training and test sets
    training_size = int(len(scaled_data) * 0.8)
    test_size = len(scaled_data) - training_size
    
    train_data = scaled_data[0:training_size, :]
    
    # We need to include the overlap for the first test prediction
    test_data = scaled_data[training_size - time_step:, :]

    # Create the training and test datasets
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, y_train, X_test, y_test, scaler

def build_lstm_model(input_shape):
    """
    Builds the LSTM model architecture.
    
    Args:
        input_shape (tuple): The input shape for the first layer (time_step, features).
        
    Returns:
        Sequential: The compiled Keras model.
    """
    print("Building LSTM model...")
    model = Sequential()
    
    # Layer 1: LSTM with 50 units
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) # Prevents overfitting
    
    # Layer 2: LSTM with 50 units
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Layer 3: Dense (standard) layer
    model.add(Dense(25))
    
    # Layer 4: Output layer
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary() # Print model summary
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Trains the LSTM model.
    
    Args:
        model (Sequential): The compiled model.
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        
    Returns:
        Sequential: The trained model.
    """
    print("Starting LSTM model training...")
    model.fit(X_train, y_train, 
              validation_split=0.1, 
              epochs=epochs, 
              batch_size=batch_size, 
              verbose=1)
    print("LSTM model training complete.")
    return model

def evaluate_lstm_model(model, scaler, X_test, y_test):
    """
    Evaluates the LSTM model and returns unscaled predictions.
    
    Args:
        model (Sequential): The trained model.
        scaler (MinMaxScaler): The scaler used for preprocessing.
        X_test (np.array): Test features.
        y_test (np.array): Test target (scaled).
        
    Returns:
        np.array: The unscaled predictions.
    """
    print("Evaluating LSTM model...")
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions_unscaled = scaler.inverse_transform(predictions)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions_unscaled))
    print(f"--- LSTM Model Evaluation ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print("---------------------------")
    
    return predictions_unscaled
