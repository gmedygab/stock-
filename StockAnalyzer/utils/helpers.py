import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

def format_currency(value):
    """
    Format a number as currency with dollar sign
    
    Parameters:
    value (float or int): Value to format
    
    Returns:
    str: Formatted currency string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    return f"${value:,.2f}"

def format_large_number(value):
    """
    Format large numbers with suffixes (K, M, B, T)
    
    Parameters:
    value (float or int): Value to format
    
    Returns:
    str: Formatted number string
    """
    if pd.isna(value) or value is None:
        return "N/A"
        
    if value < 1000:
        return f"{value:.2f}"
    
    if value < 1000000:
        return f"{value/1000:.2f}K"
    
    if value < 1000000000:
        return f"{value/1000000:.2f}M"
    
    if value < 1000000000000:
        return f"{value/1000000000:.2f}B"
    
    return f"{value/1000000000000:.2f}T"

def format_percent(value):
    """
    Format a number as percentage
    
    Parameters:
    value (float): Value to format as percentage
    
    Returns:
    str: Formatted percentage string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    return f"{value:.2f}%"

def format_change(value, include_color=False):
    """
    Format price change with + or - sign
    
    Parameters:
    value (float): Change value
    include_color (bool): Whether to include HTML color formatting
    
    Returns:
    str: Formatted change string, optionally with HTML color
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    formatted = f"{value:+,.2f}"
    
    if include_color:
        color = "green" if value >= 0 else "red"
        return f"<span style='color:{color}'>{formatted}</span>"
    
    return formatted

def format_change_percent(value, include_color=False):
    """
    Format percentage change with + or - sign
    
    Parameters:
    value (float): Percentage change value
    include_color (bool): Whether to include HTML color formatting
    
    Returns:
    str: Formatted percentage change string, optionally with HTML color
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    formatted = f"{value:+,.2f}%"
    
    if include_color:
        color = "green" if value >= 0 else "red"
        return f"<span style='color:{color}'>{formatted}</span>"
    
    return formatted

def calculate_returns(df, periods=[1, 5, 30, 90, 180, 365]):
    """
    Calculate returns over various time periods
    
    Parameters:
    df (pd.DataFrame): DataFrame with stock price data (must include 'Close' column)
    periods (list): List of periods in days to calculate returns for
    
    Returns:
    dict: Dictionary of period returns
    """
    returns = {}
    
    for period in periods:
        if len(df) > period:
            current_price = df['Close'].iloc[-1]
            past_price = df['Close'].iloc[-(period+1)]
            returns[f"{period}d"] = ((current_price - past_price) / past_price) * 100
        else:
            returns[f"{period}d"] = None
    
    return returns

def color_scale(value, min_val=-5, max_val=5, mid_val=0):
    """
    Generate a color based on a value within a scale
    
    Parameters:
    value (float): Value to determine color for
    min_val (float): Minimum value in scale (will be red)
    max_val (float): Maximum value in scale (will be green)
    mid_val (float): Middle value in scale (will be yellow)
    
    Returns:
    str: Hex color code
    """
    if pd.isna(value) or value is None:
        return "#CCCCCC"  # Gray for NA values
    
    if value <= min_val:
        return "#FF0000"  # Red for minimum
    elif value >= max_val:
        return "#00FF00"  # Green for maximum
    elif value < mid_val:
        # Scale from red to yellow
        ratio = (value - min_val) / (mid_val - min_val)
        r = 255
        g = int(255 * ratio)
        b = 0
        return f"#{r:02X}{g:02X}{b:02X}"
    else:
        # Scale from yellow to green
        ratio = (value - mid_val) / (max_val - mid_val)
        r = int(255 * (1 - ratio))
        g = 255
        b = 0
        return f"#{r:02X}{g:02X}{b:02X}"
