import streamlit as st
import yfinance as yf
import pandas as pd

def search_stocks(query):
    """
    Search for stocks based on a query string (symbol or company name)
    """
    try:
        # For simplicity, we'll use a predefined list of popular stocks
        popular_stocks = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com, Inc.',
            'META': 'Meta Platforms, Inc.',
            'TSLA': 'Tesla, Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corporation',
            'DIS': 'The Walt Disney Company',
            'NFLX': 'Netflix, Inc.',
            'V': 'Visa Inc.',
            'JNJ': 'Johnson & Johnson',
            'WMT': 'Walmart Inc.',
            'PG': 'Procter & Gamble Co.',
            'XOM': 'Exxon Mobil Corporation',
            'CVX': 'Chevron Corporation',
            'HD': 'The Home Depot, Inc.',
            'KO': 'The Coca-Cola Company',
            'PEP': 'PepsiCo, Inc.'
        }
        
        # Filter stocks based on query
        if query:
            filtered_stocks = {k: v for k, v in popular_stocks.items() 
                               if query.upper() in k or query.lower() in v.lower()}
            
            if not filtered_stocks and len(query) >= 2:
                # Try to fetch data for the exact symbol
                try:
                    ticker = yf.Ticker(query.upper())
                    info = ticker.info
                    if 'shortName' in info:
                        filtered_stocks = {query.upper(): info['shortName']}
                except:
                    pass
                    
            return filtered_stocks
        return popular_stocks
    except Exception as e:
        st.error(f"Error searching for stocks: {str(e)}")
        return {}

def display_stock_search():
    """
    Display a search box for stocks and return the selected stock symbol
    """
    # Access translation function if available in session state
    t = st.session_state.get('translate', lambda x: x)
    
    query = st.text_input(t("Search for a stock (symbol or name)"), "")
    
    search_results = search_stocks(query)
    
    if not search_results:
        st.warning(t("No stocks found. Try another search term."))
        return st.session_state.selected_stock
    
    options = [f"{k}: {v}" for k, v in search_results.items()]
    
    # Set default index to currently selected stock if it's in the results
    default_index = 0
    for i, option in enumerate(options):
        if option.startswith(st.session_state.selected_stock + ":"):
            default_index = i
            break
    
    selected = st.selectbox(
        t("Select a stock"),
        options=options,
        index=default_index
    )
    
    # Extract the ticker symbol from the selection
    selected_symbol = selected.split(":")[0].strip()
    
    if selected_symbol != st.session_state.selected_stock:
        st.session_state.selected_stock = selected_symbol
    
    return selected_symbol
