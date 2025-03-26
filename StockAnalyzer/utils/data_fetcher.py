import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_stock_data(symbol, period='1mo'):
    """
    Fetch stock data for a given symbol and time period
    
    Parameters:
    symbol (str): Stock ticker symbol
    period (str): Time period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'ytd', 'max')
    
    Returns:
    pd.DataFrame: DataFrame with stock price data
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            return pd.DataFrame()
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_stock_info(symbol):
    """
    Fetch detailed information for a given stock symbol
    
    Parameters:
    symbol (str): Stock ticker symbol
    
    Returns:
    dict: Dictionary with stock information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return info
    
    except Exception as e:
        st.error(f"Error fetching info for {symbol}: {str(e)}")
        return {}

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_fundamentals(symbol):
    """
    Fetch fundamental financial data for a given stock symbol
    
    Parameters:
    symbol (str): Stock ticker symbol
    
    Returns:
    dict: Dictionary with fundamental data
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get various financial data
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        earnings = ticker.earnings
        
        # Extract key metrics
        data = {
            'info': info,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cashflow': cashflow,
            'earnings': earnings
        }
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching fundamentals for {symbol}: {str(e)}")
        return {}

@st.cache_data(ttl=86400)  # Cache data for 1 day
def get_historical_data(symbol, start_date, end_date=None):
    """
    Fetch historical stock data for a specific date range
    
    Parameters:
    symbol (str): Stock ticker symbol
    start_date (str): Start date in format 'YYYY-MM-DD'
    end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
    
    Returns:
    pd.DataFrame: DataFrame with historical stock price data
    """
    try:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            return pd.DataFrame()
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return pd.DataFrame()
