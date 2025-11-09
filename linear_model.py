import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def prepare_data_lr(data):
    """
    Prepares data for Linear Regression (autoregressive model).
    We predict tomorrow's price based on today's price.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Close' prices.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Preparing data for Linear Regression...")
    df = data.copy()
    
    # Create the 'Target' column (tomorrow's price)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop the last row as it will have a NaN value for 'Target'
    df = df.dropna()
    
    X = df[['Close']] # Input feature (today's price)
    y = df['Target']  # Target (tomorrow's price)
    
    # Split the data. DO NOT SHUFFLE time-series data.
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def train_lr_model(X_train, y_train):
    """
    Trains a Linear Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        
    Returns:
        LinearRegression: The trained model.
    """
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns predictions and metrics.
    
    Args:
        model (LinearRegression): The trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        
    Returns:
        np.array: The predictions made by the model.
    """
    print("Evaluating Linear Regression model...")
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"--- Linear Regression Evaluation ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    print("------------------------------------")
    
    return predictions
