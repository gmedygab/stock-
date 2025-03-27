import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re
from utils.data_fetcher import get_stock_data

def display_advanced_tools():
    """
    Display advanced analysis tools
    """
    tab1, tab2, tab3 = st.tabs(["🔍 Stock Screener", "📊 Correlazione", "🧪 Analisi Comparativa"])
    
    with tab1:
        display_stock_screener()
    
    with tab2:
        display_correlation_analysis()
    
    with tab3:
        display_comparative_analysis()


def display_stock_screener():
    """
    Display stock screener tool
    """
    st.subheader("Stock Screener")
    
    st.markdown("""
    Questo strumento ti permette di trovare azioni che soddisfano specifici criteri di selezione.
    Seleziona i filtri desiderati e premi "Cerca" per trovare le azioni che corrispondono ai tuoi criteri.
    """)
    
    # Market selection
    markets = ["US Market", "Italiane", "Europee"]
    selected_market = st.selectbox("Seleziona Mercato", markets)
    
    market_indices = {
        "US Market": "^GSPC",  # S&P 500
        "Italiane": "FTSEMIB.MI",  # FTSE MIB
        "Europee": "^STOXX50E"  # EURO STOXX 50
    }
    
    # Get index constituents (simplified approach)
    index_stocks = {
        "US Market": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "PG",
            "JNJ", "UNH", "HD", "BAC", "MA", "XOM", "DIS", "CSCO", "ADBE", "CRM"
        ],
        "Italiane": [
            "ENI.MI", "ISP.MI", "UCG.MI", "ENEL.MI", "FCA.MI", "G.MI", "PST.MI", 
            "STLA.MI", "TIT.MI", "RACE.MI", "STM.MI", "PRY.MI", "MB.MI", "CNHI.MI", "ATL.MI"
        ],
        "Europee": [
            "ASML.AS", "MC.PA", "SAP.DE", "SAN.MC", "AIR.PA", "OR.PA", "SIE.DE", 
            "ALV.DE", "BNP.PA", "DTG.DE", "DTE.DE", "CS.PA", "SU.PA", "ABI.BR", "AD.AS"
        ]
    }
    
    # Filter columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Price range filter
        min_price = st.number_input("Prezzo Minimo ($)", min_value=0.0, value=0.0, step=5.0)
        max_price = st.number_input("Prezzo Massimo ($)", min_value=0.0, value=1000.0, step=50.0)
        
        # Market cap filter
        market_cap_options = ["Qualsiasi", "Micro (<300M)", "Piccola (300M-2B)", "Media (2B-10B)", "Grande (10B-100B)", "Mega (>100B)"]
        market_cap_filter = st.selectbox("Capitalizzazione di Mercato", market_cap_options)
        
        # Dividend filter
        dividend_options = ["Qualsiasi", "Solo con dividendo", "Dividendo > 1%", "Dividendo > 3%", "Dividendo > 5%"]
        dividend_filter = st.selectbox("Dividendo", dividend_options)

    with col2:
        # Performance filter
        perf_options = ["Qualsiasi", "Positiva", "Negativa", "> 5%", "> 10%", "> 20%", "< -5%", "< -10%", "< -20%"]
        performance_period = st.selectbox("Periodo Performance", ["1 Giorno", "1 Settimana", "1 Mese", "3 Mesi", "6 Mesi", "1 Anno"])
        performance_filter = st.selectbox("Performance", perf_options)
        
        # Volume filter
        volume_options = ["Qualsiasi", "> 100K", "> 500K", "> 1M", "> 5M", "> 10M"]
        volume_filter = st.selectbox("Volume Giornaliero", volume_options)
        
        # P/E filter
        pe_options = ["Qualsiasi", "< 10", "< 15", "< 20", "< 25", "< 30", "> 30"]
        pe_filter = st.selectbox("Rapporto P/E", pe_options)
    
    # Search button
    if st.button("Cerca", use_container_width=True, type="primary"):
        with st.spinner("Ricerca in corso..."):
            results = screen_stocks(
                index_stocks[selected_market],
                min_price, max_price,
                market_cap_filter,
                dividend_filter,
                performance_period,
                performance_filter,
                volume_filter,
                pe_filter
            )
            
            if results.empty:
                st.warning("Nessun risultato trovato. Prova a modificare i filtri.")
            else:
                st.success(f"Trovate {len(results)} azioni che corrispondono ai criteri.")
                
                # Format the results
                formatted_results = format_screener_results(results)
                
                # Show the results
                st.dataframe(
                    formatted_results,
                    column_config={
                        "Symbol": st.column_config.TextColumn("Simbolo"),
                        "Name": st.column_config.TextColumn("Nome"),
                        "Price": st.column_config.NumberColumn("Prezzo ($)", format="$%.2f"),
                        "Change": st.column_config.NumberColumn("Var %", format="%.2f%%"),
                        "Market Cap": st.column_config.TextColumn("Cap. di Mercato"),
                        "P/E": st.column_config.NumberColumn("P/E", format="%.2f"),
                        "Dividend": st.column_config.NumberColumn("Div %", format="%.2f%%"),
                        "Volume": st.column_config.TextColumn("Volume")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add to portfolio button for each stock
                st.subheader("Aggiungi al Portafoglio")
                selected_symbol = st.selectbox("Seleziona un'azione", options=results['Symbol'].tolist())
                
                if selected_symbol:
                    selected_price = results[results['Symbol'] == selected_symbol]['Price'].values[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        shares = st.number_input("Numero di Azioni", min_value=0.01, step=0.01, value=1.0)
                    with col2:
                        avg_price = st.number_input("Prezzo Medio ($)", min_value=0.01, step=0.01, value=float(selected_price))
                    
                    if st.button("Aggiungi al Portafoglio", use_container_width=True):
                        add_to_portfolio(selected_symbol, shares, avg_price)


def screen_stocks(symbols, min_price, max_price, market_cap_filter, dividend_filter, 
                  performance_period, performance_filter, volume_filter, pe_filter):
    """
    Screen stocks based on the given criteria
    
    Parameters:
    symbols (list): List of stock symbols to screen
    
    Returns:
    pd.DataFrame: DataFrame with screened stocks
    """
    # Initialize results DataFrame
    results = pd.DataFrame(columns=[
        'Symbol', 'Name', 'Price', 'Change', 'Market Cap', 'P/E', 'Dividend', 'Volume'
    ])
    
    # Convert performance period to yfinance format
    period_mapping = {
        "1 Giorno": "1d",
        "1 Settimana": "5d",
        "1 Mese": "1mo",
        "3 Mesi": "3mo",
        "6 Mesi": "6mo",
        "1 Anno": "1y"
    }
    period = period_mapping[performance_period]
    
    # Process each symbol
    for symbol in symbols:
        try:
            # Get basic stock data
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Get price and volume data for the specified period
            hist = stock.history(period=period)
            
            if hist.empty:
                continue
            
            # Calculate current price and change
            current_price = hist['Close'].iloc[-1]
            start_price = hist['Close'].iloc[0]
            change_percent = ((current_price - start_price) / start_price) * 100
            
            # Get market cap
            market_cap = info.get('marketCap', 0)
            
            # Get P/E ratio
            pe_ratio = info.get('trailingPE', None)
            
            # Get dividend yield
            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield:
                dividend_yield = dividend_yield * 100  # Convert to percentage
            
            # Get volume
            volume = hist['Volume'].iloc[-1]
            
            # Get company name
            name = info.get('shortName', symbol)
            
            # Apply filters
            
            # Price filter
            if current_price < min_price or (max_price > 0 and current_price > max_price):
                continue
            
            # Market cap filter
            if market_cap_filter != "Qualsiasi":
                if market_cap_filter == "Micro (<300M)" and market_cap >= 300000000:
                    continue
                elif market_cap_filter == "Piccola (300M-2B)" and (market_cap < 300000000 or market_cap >= 2000000000):
                    continue
                elif market_cap_filter == "Media (2B-10B)" and (market_cap < 2000000000 or market_cap >= 10000000000):
                    continue
                elif market_cap_filter == "Grande (10B-100B)" and (market_cap < 10000000000 or market_cap >= 100000000000):
                    continue
                elif market_cap_filter == "Mega (>100B)" and market_cap < 100000000000:
                    continue
            
            # Dividend filter
            if dividend_filter != "Qualsiasi":
                if dividend_filter == "Solo con dividendo" and (dividend_yield is None or dividend_yield <= 0):
                    continue
                elif dividend_filter == "Dividendo > 1%" and (dividend_yield is None or dividend_yield <= 1):
                    continue
                elif dividend_filter == "Dividendo > 3%" and (dividend_yield is None or dividend_yield <= 3):
                    continue
                elif dividend_filter == "Dividendo > 5%" and (dividend_yield is None or dividend_yield <= 5):
                    continue
            
            # Performance filter
            if performance_filter != "Qualsiasi":
                if performance_filter == "Positiva" and change_percent <= 0:
                    continue
                elif performance_filter == "Negativa" and change_percent >= 0:
                    continue
                elif performance_filter == "> 5%" and change_percent <= 5:
                    continue
                elif performance_filter == "> 10%" and change_percent <= 10:
                    continue
                elif performance_filter == "> 20%" and change_percent <= 20:
                    continue
                elif performance_filter == "< -5%" and change_percent >= -5:
                    continue
                elif performance_filter == "< -10%" and change_percent >= -10:
                    continue
                elif performance_filter == "< -20%" and change_percent >= -20:
                    continue
            
            # Volume filter
            if volume_filter != "Qualsiasi":
                if volume_filter == "> 100K" and volume <= 100000:
                    continue
                elif volume_filter == "> 500K" and volume <= 500000:
                    continue
                elif volume_filter == "> 1M" and volume <= 1000000:
                    continue
                elif volume_filter == "> 5M" and volume <= 5000000:
                    continue
                elif volume_filter == "> 10M" and volume <= 10000000:
                    continue
            
            # P/E filter
            if pe_filter != "Qualsiasi" and pe_ratio is not None:
                if pe_filter == "< 10" and pe_ratio >= 10:
                    continue
                elif pe_filter == "< 15" and pe_ratio >= 15:
                    continue
                elif pe_filter == "< 20" and pe_ratio >= 20:
                    continue
                elif pe_filter == "< 25" and pe_ratio >= 25:
                    continue
                elif pe_filter == "< 30" and pe_ratio >= 30:
                    continue
                elif pe_filter == "> 30" and pe_ratio <= 30:
                    continue
            
            # Add to results
            results = pd.concat([results, pd.DataFrame({
                'Symbol': [symbol],
                'Name': [name],
                'Price': [current_price],
                'Change': [change_percent],
                'Market Cap': [market_cap],
                'P/E': [pe_ratio],
                'Dividend': [dividend_yield if dividend_yield else 0],
                'Volume': [volume]
            })], ignore_index=True)
            
        except Exception as e:
            continue
    
    # Sort by market cap (descending)
    if not results.empty:
        results = results.sort_values(by='Market Cap', ascending=False)
    
    return results


def format_screener_results(results):
    """
    Format screener results for display
    
    Parameters:
    results (pd.DataFrame): DataFrame with screener results
    
    Returns:
    pd.DataFrame: Formatted DataFrame
    """
    formatted = results.copy()
    
    # Format market cap
    def format_market_cap(value):
        if value is None or pd.isna(value):
            return "N/A"
        if value >= 1000000000000:
            return f"${value/1000000000000:.2f}T"
        elif value >= 1000000000:
            return f"${value/1000000000:.2f}B"
        elif value >= 1000000:
            return f"${value/1000000:.2f}M"
        else:
            return f"${value:.2f}"
    
    # Format volume
    def format_volume(value):
        if value is None or pd.isna(value):
            return "N/A"
        if value >= 1000000000:
            return f"{value/1000000000:.2f}B"
        elif value >= 1000000:
            return f"{value/1000000:.2f}M"
        elif value >= 1000:
            return f"{value/1000:.2f}K"
        else:
            return f"{value:.0f}"
    
    # Apply formatting
    formatted['Market Cap'] = formatted['Market Cap'].apply(format_market_cap)
    formatted['Volume'] = formatted['Volume'].apply(format_volume)
    
    # Handle None values
    formatted['P/E'] = formatted['P/E'].apply(lambda x: x if x is not None and not pd.isna(x) else 0)
    
    return formatted


def display_correlation_analysis():
    """
    Display correlation analysis between stocks
    """
    st.subheader("Analisi di Correlazione")
    
    st.markdown("""
    Questo strumento ti permette di analizzare la correlazione tra diverse azioni. Una correlazione bassa 
    indica che le azioni tendono a muoversi in modo indipendente, utile per la diversificazione del portafoglio.
    """)
    
    # Select stocks
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Allow custom input
    custom_input = st.text_input(
        "Inserisci simboli separati da virgola (es: AAPL,MSFT,GOOGL)",
        ",".join(default_stocks)
    )
    
    stocks = [s.strip().upper() for s in custom_input.split(",") if s.strip()]
    
    # Select time period
    period_options = {
        "1 Mese": "1mo",
        "3 Mesi": "3mo",
        "6 Mesi": "6mo",
        "1 Anno": "1y",
        "2 Anni": "2y",
        "5 Anni": "5y"
    }
    
    selected_period = st.selectbox("Periodo", list(period_options.keys()))
    period = period_options[selected_period]
    
    if st.button("Analizza Correlazione", use_container_width=True, type="primary"):
        if len(stocks) < 2:
            st.warning("Inserisci almeno due simboli per l'analisi di correlazione.")
        else:
            # Get price data
            price_data = {}
            valid_stocks = []
            
            with st.spinner("Raccolta dati..."):
                for symbol in stocks:
                    try:
                        data = get_stock_data(symbol, period)
                        if not data.empty:
                            price_data[symbol] = data['Close']
                            valid_stocks.append(symbol)
                    except Exception:
                        continue
            
            if len(valid_stocks) < 2:
                st.warning("Non ci sono abbastanza dati validi per almeno due simboli.")
            else:
                # Create DataFrame with all price data
                df = pd.DataFrame(price_data)
                
                # Calculate correlation matrix
                corr_matrix = df.corr()
                
                # Plot heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    aspect="auto"
                )
                
                fig.update_layout(
                    title=f"Matrice di Correlazione ({selected_period})",
                    height=500,
                    coloraxis_colorbar=dict(
                        title="Correlazione",
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=["-1 (Inversa)", "-0.5", "0 (Nessuna)", "0.5", "1 (Perfetta)"]
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.subheader("Interpretazione")
                
                st.markdown("""
                **Significato dei valori di correlazione:**
                - **1.0**: Correlazione perfetta (le azioni si muovono esattamente nella stessa direzione)
                - **0.7 a 0.9**: Correlazione forte
                - **0.4 a 0.6**: Correlazione moderata
                - **0.1 a 0.3**: Correlazione debole
                - **0.0**: Nessuna correlazione (movimento completamente indipendente)
                - **-0.1 a -0.9**: Correlazione negativa (si muovono in direzioni opposte)
                - **-1.0**: Correlazione negativa perfetta
                
                Per una diversificazione efficace, cerca azioni con **correlazione bassa o negativa** tra loro.
                """)
                
                # Find the lowest correlated pair
                min_corr = 1
                min_pair = None
                
                for i in range(len(valid_stocks)):
                    for j in range(i+1, len(valid_stocks)):
                        corr = corr_matrix.iloc[i, j]
                        if corr < min_corr:
                            min_corr = corr
                            min_pair = (valid_stocks[i], valid_stocks[j])
                
                if min_pair:
                    st.success(f"La coppia di azioni con la correlazione più bassa è **{min_pair[0]}** e **{min_pair[1]}** con correlazione di **{min_corr:.2f}**.")
                
                # Price performance chart
                st.subheader("Andamento Prezzi Normalizzati")
                
                # Normalize price data for comparison
                normalized_df = df.div(df.iloc[0]).mul(100)
                
                fig_prices = go.Figure()
                
                for col in normalized_df.columns:
                    fig_prices.add_trace(go.Scatter(
                        x=normalized_df.index,
                        y=normalized_df[col],
                        name=col,
                        mode='lines'
                    ))
                
                fig_prices.update_layout(
                    title="Andamento Prezzi (Base 100)",
                    xaxis_title="Data",
                    yaxis_title="Prezzo (Base 100)",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_prices, use_container_width=True)


def display_comparative_analysis():
    """
    Display comparative analysis between stocks
    """
    st.subheader("Analisi Comparativa")
    
    st.markdown("""
    Confronta diverse metriche tra azioni selezionate per valutare quale potrebbe essere la migliore per il tuo portafoglio.
    """)
    
    # Select stocks
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    # Allow custom input
    custom_input = st.text_input(
        "Inserisci simboli da confrontare (separati da virgola)",
        ",".join(default_stocks)
    )
    
    stocks = [s.strip().upper() for s in custom_input.split(",") if s.strip()]
    
    if st.button("Confronta", use_container_width=True, type="primary"):
        if len(stocks) < 2:
            st.warning("Inserisci almeno due simboli per il confronto.")
        else:
            with st.spinner("Raccolta dati..."):
                comparison_data = get_comparison_data(stocks)
                
                if not comparison_data:
                    st.warning("Non sono disponibili dati sufficienti per il confronto.")
                else:
                    # Create tabs for different metric categories
                    tab1, tab2, tab3 = st.tabs(["📈 Performance", "📊 Valutazione", "💰 Dividendi e Fondamentali"])
                    
                    with tab1:
                        st.subheader("Confronto Performance")
                        
                        # Create performance comparison chart
                        performance_metrics = ['1d_return', '1mo_return', '3mo_return', '6mo_return', '1y_return', 'ytd_return']
                        performance_labels = ['1 Giorno', '1 Mese', '3 Mesi', '6 Mesi', '1 Anno', 'Anno in Corso']
                        
                        performance_data = []
                        for stock in comparison_data:
                            stock_data = {
                                'Simbolo': stock['Symbol'],
                                'Nome': stock['Name']
                            }
                            
                            for metric, label in zip(performance_metrics, performance_labels):
                                if metric in stock:
                                    stock_data[label] = stock[metric]
                                else:
                                    stock_data[label] = None
                            
                            performance_data.append(stock_data)
                        
                        performance_df = pd.DataFrame(performance_data)
                        
                        # Create a bar chart for each time period
                        for period in performance_labels:
                            if period in performance_df.columns:
                                fig = px.bar(
                                    performance_df,
                                    x='Simbolo',
                                    y=period,
                                    color='Simbolo',
                                    text_auto='.2f',
                                    title=f"Rendimento {period} (%)",
                                    labels={'value': 'Rendimento (%)'},
                                    height=350
                                )
                                
                                fig.update_traces(textposition='outside')
                                fig.update_layout(
                                    xaxis_title=None,
                                    yaxis_title="Rendimento (%)",
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        st.subheader("Metriche di Valutazione")
                        
                        # Create valuation metrics table
                        valuation_metrics = [
                            'Market Cap', 'P/E', 'Forward P/E', 'PEG Ratio', 'P/S', 'P/B', 'EV/EBITDA'
                        ]
                        
                        valuation_data = []
                        for stock in comparison_data:
                            valuation_row = {'Simbolo': stock['Symbol'], 'Nome': stock['Name']}
                            
                            for metric in valuation_metrics:
                                key = metric.lower().replace(' ', '_').replace('/', '_')
                                if key in stock:
                                    valuation_row[metric] = stock[key]
                                else:
                                    valuation_row[metric] = None
                            
                            valuation_data.append(valuation_row)
                        
                        valuation_df = pd.DataFrame(valuation_data)
                        
                        # Format the market cap
                        if 'Market Cap' in valuation_df.columns:
                            valuation_df['Market Cap'] = valuation_df['Market Cap'].apply(
                                lambda x: f"${x/1000000000:.2f}B" if x is not None and not pd.isna(x) else "N/A"
                            )
                        
                        st.dataframe(valuation_df, use_container_width=True)
                        
                        # Create radar chart for valuation comparison
                        radar_metrics = ['P/E', 'P/S', 'P/B', 'EV/EBITDA']
                        radar_df = pd.DataFrame()
                        
                        for stock in comparison_data:
                            stock_data = {}
                            for metric in radar_metrics:
                                key = metric.lower().replace(' ', '_').replace('/', '_')
                                if key in stock and stock[key] is not None and not pd.isna(stock[key]):
                                    stock_data[metric] = stock[key]
                            
                            if stock_data:
                                radar_df[stock['Symbol']] = pd.Series(stock_data)
                        
                        if not radar_df.empty and len(radar_df.columns) >= 2:
                            # Normalize values for the radar chart (lower is better for these metrics)
                            radar_max = radar_df.max(axis=1)
                            normalized_radar = radar_df.div(radar_max, axis=0)
                            
                            # Create radar chart
                            fig = go.Figure()
                            
                            for col in normalized_radar.columns:
                                fig.add_trace(go.Scatterpolar(
                                    r=normalized_radar[col].values.tolist() + [normalized_radar[col].values[0]],
                                    theta=radar_metrics + [radar_metrics[0]],
                                    fill='toself',
                                    name=col
                                ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )
                                ),
                                title="Confronto Multipli (Normalizzato)",
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **Nota:** Nel grafico radar, valori più bassi (più vicini al centro) sono generalmente
                            preferibili per i multipli di valutazione, indicando un titolo potenzialmente più conveniente.
                            """)
                    
                    with tab3:
                        st.subheader("Dividendi e Fondamentali")
                        
                        # Create dividends and fundamentals table
                        fundamental_metrics = [
                            'Dividend Yield', 'Dividend Rate', 'Payout Ratio', 
                            'Profit Margin', 'Operating Margin', 'ROE', 'ROA',
                            'Debt to Equity', 'Current Ratio', 'Quick Ratio'
                        ]
                        
                        fundamental_data = []
                        for stock in comparison_data:
                            fundamental_row = {'Simbolo': stock['Symbol'], 'Nome': stock['Name']}
                            
                            for metric in fundamental_metrics:
                                key = metric.lower().replace(' ', '_').replace('/', '_to_')
                                if key in stock:
                                    # Format percentages
                                    if metric in ['Dividend Yield', 'Payout Ratio', 'Profit Margin', 'Operating Margin', 'ROE', 'ROA']:
                                        value = stock[key]
                                        if value is not None and not pd.isna(value):
                                            fundamental_row[metric] = f"{value*100:.2f}%" if metric == 'Dividend Yield' else f"{value:.2f}%"
                                        else:
                                            fundamental_row[metric] = "N/A"
                                    else:
                                        fundamental_row[metric] = stock[key]
                                else:
                                    fundamental_row[metric] = "N/A"
                            
                            fundamental_data.append(fundamental_row)
                        
                        fundamental_df = pd.DataFrame(fundamental_data)
                        
                        st.dataframe(fundamental_df, use_container_width=True)
                        
                        # Create dividend comparison chart
                        dividend_data = []
                        for stock in comparison_data:
                            if 'dividend_yield' in stock and stock['dividend_yield'] is not None:
                                dividend_data.append({
                                    'Simbolo': stock['Symbol'],
                                    'Rendimento Dividendo (%)': stock['dividend_yield'] * 100
                                })
                        
                        if dividend_data:
                            dividend_df = pd.DataFrame(dividend_data)
                            
                            fig = px.bar(
                                dividend_df,
                                x='Simbolo',
                                y='Rendimento Dividendo (%)',
                                color='Simbolo',
                                text_auto='.2f',
                                title="Confronto Rendimento Dividendi",
                                height=350
                            )
                            
                            fig.update_traces(textposition='outside')
                            fig.update_layout(
                                xaxis_title=None,
                                yaxis_title="Rendimento (%)",
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Nessun dato sui dividendi disponibile per le azioni selezionate.")


def get_comparison_data(symbols):
    """
    Get comparison data for the given symbols
    
    Parameters:
    symbols (list): List of stock symbols to compare
    
    Returns:
    list: List of dictionaries with comparison data for each symbol
    """
    comparison_data = []
    
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info or 'regularMarketPrice' not in info:
                continue
            
            # Get stock data for different periods
            hist_1d = stock.history(period='2d')
            hist_1mo = stock.history(period='1mo')
            hist_3mo = stock.history(period='3mo')
            hist_6mo = stock.history(period='6mo')
            hist_1y = stock.history(period='1y')
            hist_ytd = stock.history(period='ytd')
            
            # Calculate returns
            if len(hist_1d) >= 2:
                return_1d = ((hist_1d['Close'].iloc[-1] / hist_1d['Close'].iloc[-2]) - 1) * 100
            else:
                return_1d = None
                
            if len(hist_1mo) >= 2:
                return_1mo = ((hist_1mo['Close'].iloc[-1] / hist_1mo['Close'].iloc[0]) - 1) * 100
            else:
                return_1mo = None
                
            if len(hist_3mo) >= 2:
                return_3mo = ((hist_3mo['Close'].iloc[-1] / hist_3mo['Close'].iloc[0]) - 1) * 100
            else:
                return_3mo = None
                
            if len(hist_6mo) >= 2:
                return_6mo = ((hist_6mo['Close'].iloc[-1] / hist_6mo['Close'].iloc[0]) - 1) * 100
            else:
                return_6mo = None
                
            if len(hist_1y) >= 2:
                return_1y = ((hist_1y['Close'].iloc[-1] / hist_1y['Close'].iloc[0]) - 1) * 100
            else:
                return_1y = None
                
            if len(hist_ytd) >= 2:
                return_ytd = ((hist_ytd['Close'].iloc[-1] / hist_ytd['Close'].iloc[0]) - 1) * 100
            else:
                return_ytd = None
            
            # Build comparison data
            stock_data = {
                'Symbol': symbol,
                'Name': info.get('shortName', symbol),
                'market_cap': info.get('marketCap'),
                'p_e': info.get('trailingPE'),
                'forward_p_e': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'p_s': info.get('priceToSalesTrailing12Months'),
                'p_b': info.get('priceToBook'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'dividend_yield': info.get('dividendYield'),
                'dividend_rate': info.get('dividendRate'),
                'payout_ratio': info.get('payoutRatio'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                '1d_return': return_1d,
                '1mo_return': return_1mo,
                '3mo_return': return_3mo,
                '6mo_return': return_6mo,
                '1y_return': return_1y,
                'ytd_return': return_ytd
            }
            
            comparison_data.append(stock_data)
            
        except Exception as e:
            continue
    
    return comparison_data


def add_to_portfolio(symbol, shares, avg_price):
    """
    Add a stock to the portfolio
    
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
            st.error(f"Impossibile verificare il simbolo: {symbol}")
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
            st.success(f"Aggiunte {shares} azioni di {symbol} al tuo portafoglio")
        else:
            # Add new position
            st.session_state.portfolio[symbol] = {
                'shares': shares,
                'avg_price': avg_price
            }
            st.success(f"Aggiunto {symbol} al tuo portafoglio")
        
        return True
    
    except Exception as e:
        st.error(f"Errore nell'aggiunta dell'azione: {str(e)}")
        return False