import pandas as pd
import yfinance as yf
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import time

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

@st.cache_data(ttl=86400)  # Cache data for 1 day
def get_market_indices():
    """
    Fetch data for major market indices
    
    Returns:
    pd.DataFrame: DataFrame with index data
    """
    try:
        # List of major indices to track
        indices = [
            '^GSPC',    # S&P 500
            '^DJI',     # Dow Jones
            '^IXIC',    # NASDAQ
            '^FTSE',    # FTSE 100
            '^GDAXI',   # DAX
            '^FCHI',    # CAC 40
            '^N225',    # Nikkei 225
            '^STOXX50E' # Euro Stoxx 50
        ]
        
        # Add descriptive names
        index_names = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^FTSE': 'FTSE 100',
            '^GDAXI': 'DAX',
            '^FCHI': 'CAC 40',
            '^N225': 'Nikkei 225',
            '^STOXX50E': 'Euro Stoxx 50'
        }
        
        # Fetch data with retry mechanism
        max_retries = 3
        retry_delay = 2  # seconds
        
        all_data = {}
        for symbol in indices:
            retries = 0
            while retries < max_retries:
                try:
                    df = yf.download(symbol, period='2d', progress=False)
                    if not df.empty:
                        all_data[symbol] = df
                        break
                except Exception:
                    retries += 1
                    time.sleep(retry_delay)
            
            # Avoid hitting API rate limits
            time.sleep(0.5)
        
        # Process the data
        result = []
        for symbol, df in all_data.items():
            if len(df) >= 2:  # Need at least 2 days for calculating change
                current = df['Close'].iloc[-1]
                previous = df['Close'].iloc[-2]
                change = current - previous
                change_percent = (change / previous) * 100
                
                result.append({
                    'Symbol': symbol,
                    'Name': index_names.get(symbol, symbol),
                    'Price': current,
                    'Change': change,
                    'Change %': change_percent,
                    'Volume': df['Volume'].iloc[-1] if 'Volume' in df else 0
                })
        
        return pd.DataFrame(result)
    
    except Exception as e:
        st.error(f"Error fetching market indices: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)  # Cache data for 1 day
def get_sector_performance():
    """
    Fetch performance data for different market sectors
    
    Returns:
    pd.DataFrame: DataFrame with sector performance data
    """
    try:
        # List of sector ETFs
        sectors = {
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLE': 'Energy',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLK': 'Technology',
            'XLU': 'Utilities',
            'XLC': 'Communication Services'
        }
        
        # Fetch data with retry mechanism
        all_data = {}
        for symbol in sectors.keys():
            try:
                df = yf.download(symbol, period='5d', progress=False)
                if not df.empty:
                    all_data[symbol] = df
            except Exception:
                pass
            
            # Avoid hitting API rate limits
            time.sleep(0.5)
        
        # Process the data
        result = []
        for symbol, df in all_data.items():
            if len(df) >= 2:  # Need at least 2 days for calculating change
                current = df['Close'].iloc[-1]
                previous = df['Close'].iloc[-2]
                change = current - previous
                change_percent = (change / previous) * 100
                
                # Calculate 5-day performance if data is available
                if len(df) >= 5:
                    five_day_prev = df['Close'].iloc[0]
                    five_day_change = ((current - five_day_prev) / five_day_prev) * 100
                else:
                    five_day_change = np.nan
                
                result.append({
                    'Symbol': symbol,
                    'Sector': sectors.get(symbol, symbol),
                    'Price': current,
                    'Daily Change %': change_percent,
                    '5-Day Change %': five_day_change
                })
        
        return pd.DataFrame(result)
    
    except Exception as e:
        st.error(f"Error fetching sector performance: {str(e)}")
        return pd.DataFrame()
