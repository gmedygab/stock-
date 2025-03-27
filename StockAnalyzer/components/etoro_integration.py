import streamlit as st
import pandas as pd
import numpy as np
import datetime
import re
import io
from utils.data_fetcher import get_stock_data, get_stock_info

def display_etoro_integration():
    """
    Display eToro integration features
    """
    st.subheader("eToro Portfolio Integration")
    
    with st.expander("About eToro Integration", expanded=True):
        st.write("""
        eToro non fornisce un'API pubblica ufficiale per l'accesso diretto ai dati del tuo account.
        Tuttavia, puoi importare i dati del tuo portfolio eToro utilizzando uno dei seguenti metodi:
        
        1. **Importazione da CSV**: Esporta il tuo portfolio da eToro in formato CSV e caricalo qui.
        2. **Importazione manuale**: Inserisci manualmente le posizioni del tuo portfolio eToro.
        
        I dati importati saranno integrati con l'analisi del portafoglio in FinVision per offrirti insights avanzati.
        """)
    
    tab1, tab2 = st.tabs(["Importa da CSV", "Importazione Manuale"])
    
    with tab1:
        st.subheader("Importa Portfolio da CSV eToro")
        
        st.write("""
        Per esportare il tuo portfolio da eToro:
        1. Accedi al tuo account eToro
        2. Vai alla sezione Portfolio
        3. Clicca sul pulsante di esportazione (solitamente un'icona di download)
        4. Salva il file CSV sul tuo computer
        5. Carica il file qui sotto
        """)
        
        uploaded_file = st.file_uploader("Carica il file CSV del tuo portfolio eToro", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load the CSV data
                df = pd.read_csv(uploaded_file)
                
                # Show the raw data
                st.subheader("Anteprima dati grezzi")
                st.dataframe(df, use_container_width=True)
                
                # Process eToro CSV format
                if process_etoro_csv(df):
                    st.success("Portfolio eToro importato con successo e integrato nel tuo portfolio FinVision!")
                    st.rerun()
            
            except Exception as e:
                st.error(f"Si è verificato un errore durante l'elaborazione del file: {str(e)}")
    
    with tab2:
        st.subheader("Importazione Manuale")
        
        st.write("""
        Se preferisci, puoi inserire manualmente le posizioni del tuo portfolio eToro.
        Compila i campi sottostanti per ogni posizione che desideri aggiungere.
        """)
        
        with st.form("etoro_manual_form"):
            symbol = st.text_input("Simbolo eToro", "").upper()
            
            # Try to auto-match with a standard symbol
            if symbol and len(symbol) > 1:
                matched_symbol = match_etoro_symbol(symbol)
                if matched_symbol and matched_symbol != symbol:
                    st.info(f"Simbolo eToro '{symbol}' mappato automaticamente a '{matched_symbol}'")
                    symbol = matched_symbol
            
            shares = st.number_input("Numero di unità", min_value=0.0001, step=0.01, value=1.0)
            avg_price = st.number_input("Prezzo medio di acquisto ($)", min_value=0.01, step=0.01, value=100.0)
            
            submitted = st.form_submit_button("Aggiungi Posizione")
            
            if submitted and symbol:
                # Add to portfolio
                if add_to_portfolio(symbol, shares, avg_price):
                    st.success(f"Posizione {symbol} aggiunta con successo al tuo portfolio!")
                    st.rerun()
                else:
                    st.error("Non è stato possibile aggiungere la posizione. Verifica che il simbolo sia valido.")


def process_etoro_csv(df):
    """
    Process eToro portfolio CSV export and add positions to the portfolio
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Check if this looks like an eToro CSV
        required_columns = check_etoro_csv_format(df)
        
        if not required_columns:
            st.warning("Il formato del file CSV non sembra corrispondere a un'esportazione di eToro. Verifica il file e riprova.")
            return False
        
        # Map column names to standard format
        column_mapping = map_etoro_columns(df.columns)
        
        if not column_mapping:
            st.warning("Non è stato possibile mappare le colonne del file CSV. Il formato potrebbe essere cambiato.")
            return False
        
        # Create a new dataframe with standard column names
        standardized_df = pd.DataFrame()
        
        for std_col, etoro_col in column_mapping.items():
            if etoro_col in df.columns:
                standardized_df[std_col] = df[etoro_col]
        
        # Process each position
        positions_added = 0
        
        for _, row in standardized_df.iterrows():
            try:
                symbol = str(row.get('Symbol', '')).strip().upper()
                shares = float(row.get('Units', 0))
                avg_price = float(row.get('Open Rate', 0))
                
                if symbol and shares > 0 and avg_price > 0:
                    # Try to map eToro symbol to standard symbol
                    mapped_symbol = match_etoro_symbol(symbol)
                    if mapped_symbol:
                        symbol = mapped_symbol
                    
                    if add_to_portfolio(symbol, shares, avg_price):
                        positions_added += 1
            except Exception as e:
                st.warning(f"Errore durante l'elaborazione della posizione: {str(e)}")
                continue
        
        if positions_added > 0:
            st.success(f"Importate con successo {positions_added} posizioni dal portfolio eToro!")
            return True
        else:
            st.warning("Nessuna posizione valida trovata nel file CSV.")
            return False
    
    except Exception as e:
        st.error(f"Errore durante l'elaborazione del file CSV: {str(e)}")
        return False


def check_etoro_csv_format(df):
    """
    Check if the CSV format looks like an eToro export
    
    Returns:
    bool: True if it's likely an eToro export, False otherwise
    """
    # Common column patterns in eToro exports
    etoro_patterns = [
        'instrument', 'position id', 'open date', 'type', 'amount', 'open rate',
        'market', 'ticker', 'units', 'leverage', 'profit', 'equity'
    ]
    
    # Check if at least some of these patterns are in the column names
    columns_lower = [col.lower() for col in df.columns]
    matches = sum(1 for pattern in etoro_patterns if any(pattern in col for col in columns_lower))
    
    # If at least 3 patterns match, it's likely an eToro export
    return matches >= 3


def map_etoro_columns(columns):
    """
    Map eToro column names to standard names
    
    Returns:
    dict: Mapping from standard column names to eToro column names
    """
    columns_lower = [col.lower() for col in columns]
    column_map = {}
    
    # Map for Symbol
    for pattern in ['instrument', 'ticker', 'market', 'symbol', 'asset']:
        for col in columns:
            if pattern in col.lower():
                column_map['Symbol'] = col
                break
        if 'Symbol' in column_map:
            break
    
    # Map for Units/Shares
    for pattern in ['units', 'amount', 'quantity', 'size', 'position size']:
        for col in columns:
            if pattern in col.lower():
                column_map['Units'] = col
                break
        if 'Units' in column_map:
            break
    
    # Map for Open Rate/Price
    for pattern in ['open rate', 'open price', 'entry price', 'buy price', 'price']:
        for col in columns:
            if pattern in col.lower():
                column_map['Open Rate'] = col
                break
        if 'Open Rate' in column_map:
            break
    
    return column_map if len(column_map) >= 2 else {}


def match_etoro_symbol(etoro_symbol):
    """
    Match eToro symbol to standard market symbol
    
    Parameters:
    etoro_symbol (str): eToro symbol (e.g., 'aapl', 'amzn', 'NFLX', etc.)
    
    Returns:
    str: Standard market symbol if found, original symbol otherwise
    """
    # Common eToro to standard symbol mappings
    etoro_to_standard = {
        # Direct mappings for common symbols
        'aapl': 'AAPL', 'amzn': 'AMZN', 'googl': 'GOOGL', 'msft': 'MSFT', 'nflx': 'NFLX',
        'fb': 'META', 'meta': 'META', 'tsla': 'TSLA', 'nvda': 'NVDA', 'amd': 'AMD',
        'intc': 'INTC', 'baba': 'BABA', 'dis': 'DIS', 'nke': 'NKE', 'ko': 'KO',
        'mcd': 'MCD', 'wmt': 'WMT', 'jpm': 'JPM', 'bac': 'BAC', 'v': 'V',
        'ma': 'MA', 'pypl': 'PYPL', 'adbe': 'ADBE', 'crm': 'CRM', 'csco': 'CSCO',
        
        # Crypto mappings (these would need special handling with a crypto API)
        'btc': 'BTC-USD', 'eth': 'ETH-USD', 'xrp': 'XRP-USD', 'ltc': 'LTC-USD',
        'bitcoin': 'BTC-USD', 'ethereum': 'ETH-USD', 'ripple': 'XRP-USD', 'litecoin': 'LTC-USD',
        
        # Format corrections
        'us500': 'SPY',      # S&P 500 proxy
        'nas100': 'QQQ',     # NASDAQ 100 proxy
        'dj30': 'DIA',       # Dow Jones Industrial Average proxy
        'gold': 'GLD',       # Gold proxy
        'oil': 'USO',        # Oil proxy
    }
    
    # Clean up the symbol
    clean_symbol = etoro_symbol.lower().strip()
    
    # Check direct mappings
    if clean_symbol in etoro_to_standard:
        return etoro_to_standard[clean_symbol]
    
    # Remove special characters and try again
    clean_symbol = re.sub(r'[^a-zA-Z0-9]', '', clean_symbol)
    if clean_symbol in etoro_to_standard:
        return etoro_to_standard[clean_symbol]
    
    # If no mapping found, return the original (but uppercase)
    return etoro_symbol.upper()


def add_to_portfolio(symbol, shares, avg_price):
    """
    Add or update a position in the FinVision portfolio
    
    Parameters:
    symbol (str): Stock symbol
    shares (float): Number of shares
    avg_price (float): Average purchase price
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Verify the symbol exists
        stock_data = get_stock_data(symbol, '1d')
        if stock_data.empty:
            return False
        
        # Initialize portfolio if not exists
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        
        # Add or update the position
        if symbol in st.session_state.portfolio:
            # Calculate new average price when adding more shares
            current_position = st.session_state.portfolio[symbol]
            current_shares = current_position['shares']
            current_avg_price = current_position['avg_price']
            
            # Calculate new average price based on existing and new shares
            total_shares = current_shares + shares
            total_cost = (current_shares * current_avg_price) + (shares * avg_price)
            new_avg_price = total_cost / total_shares
            
            # Update portfolio
            st.session_state.portfolio[symbol] = {
                'shares': total_shares,
                'avg_price': new_avg_price
            }
        else:
            # Add new position
            st.session_state.portfolio[symbol] = {
                'shares': shares,
                'avg_price': avg_price
            }
        
        return True
    
    except Exception as e:
        st.error(f"Errore durante l'aggiunta della posizione: {str(e)}")
        return False