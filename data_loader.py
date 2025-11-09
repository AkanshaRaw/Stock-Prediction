import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    """
    Fetches stock data from Yahoo Finance.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: A DataFrame with the 'Close' price and a DatetimeIndex.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker} from {start_date} to {end_date}")
        
    # We only need the 'Close' price for this project
    # Returning as a DataFrame to keep the DatetimeIndex
    return data[['Close']]
