import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

def display_advanced_charts():
    """
    Display advanced charts with multiple visualization options
    """
    # Create tabs for different chart types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Candlestick Avanzato", 
        "Multi-indicatori", 
        "Multi-azioni", 
        "Seasonal Analysis"
    ])
    
    with tab1:
        display_advanced_candlestick()
    
    with tab2:
        display_multi_indicator_chart()
    
    with tab3:
        display_multi_stock_chart()
    
    with tab4:
        display_seasonal_analysis()

def display_advanced_candlestick():
    """
    Display an advanced candlestick chart with customizable indicators and volume
    """
    st.subheader("Grafico Candlestick Avanzato")
    
    st.markdown("""
    Questo grafico candlestick avanzato permette di visualizzare i dati di prezzo con ulteriori indicatori.
    Scegli un simbolo e personalizza la visualizzazione.
    """)
    
    # Symbol input
    symbol = st.text_input("Simbolo Azione", value="AAPL")
    
    # Time period selection
    periods = {
        "1mo": "1 Mese",
        "3mo": "3 Mesi",
        "6mo": "6 Mesi",
        "1y": "1 Anno",
        "2y": "2 Anni",
        "5y": "5 Anni",
        "max": "Massimo"
    }
    
    period = st.select_slider(
        "Periodo",
        options=list(periods.keys()),
        format_func=lambda x: periods[x],
        value="6mo"
    )
    
    # Chart options
    col1, col2 = st.columns(2)
    
    with col1:
        show_volume = st.checkbox("Mostra Volume", value=True)
        show_ma = st.checkbox("Mostra Medie Mobili", value=True)
    
    with col2:
        show_bollinger = st.checkbox("Mostra Bande di Bollinger", value=False)
        show_ema_fast = st.checkbox("Mostra EMA Veloce", value=True)
    
    # Advanced options for moving averages
    if show_ma:
        col1, col2, col3 = st.columns(3)
        with col1:
            ma1 = st.number_input("MA1 Periodo", value=20, min_value=1, max_value=200)
        with col2:
            ma2 = st.number_input("MA2 Periodo", value=50, min_value=1, max_value=200)
        with col3:
            ma3 = st.number_input("MA3 Periodo", value=200, min_value=1, max_value=500)
    else:
        ma1, ma2, ma3 = 20, 50, 200
    
    # Advanced options for Bollinger Bands
    if show_bollinger:
        col1, col2 = st.columns(2)
        with col1:
            bb_period = st.number_input("Bollinger Periodo", value=20, min_value=1, max_value=100)
        with col2:
            bb_std = st.number_input("Bollinger Deviazioni Standard", value=2.0, min_value=0.1, max_value=5.0, step=0.1)
    else:
        bb_period, bb_std = 20, 2.0
    
    # Advanced options for EMA
    if show_ema_fast:
        ema_fast = st.number_input("EMA Veloce Periodo", value=9, min_value=1, max_value=50)
    else:
        ema_fast = 9
    
    # Chart style
    chart_style = st.selectbox(
        "Stile Grafico",
        options=["Default", "Dark", "Light", "Financial"],
        index=0
    )
    
    # Generate chart
    if st.button("Genera Grafico", use_container_width=True, type="primary"):
        with st.spinner("Caricamento dati e generazione grafico..."):
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                df = stock.history(period=period)
                
                if df.empty:
                    st.error(f"Nessun dato disponibile per il simbolo: {symbol}")
                else:
                    # Create figure
                    fig = create_advanced_candlestick(
                        df, 
                        symbol, 
                        show_volume, 
                        show_ma, 
                        [ma1, ma2, ma3], 
                        show_bollinger, 
                        bb_period, 
                        bb_std,
                        show_ema_fast,
                        ema_fast,
                        chart_style
                    )
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add simple summary stats
                    display_price_stats(df)
            
            except Exception as e:
                st.error(f"Si è verificato un errore: {str(e)}")

def create_advanced_candlestick(df, symbol, show_volume=True, show_ma=True, ma_periods=[20, 50, 200], 
                               show_bollinger=False, bb_period=20, bb_std=2.0, 
                               show_ema_fast=True, ema_fast=9, chart_style="Default"):
    """
    Create an advanced candlestick chart with multiple indicators
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLC data and volume
    symbol (str): Stock symbol to display
    show_volume (bool): Whether to show volume
    show_ma (bool): Whether to show moving averages
    ma_periods (list): Periods for moving averages
    show_bollinger (bool): Whether to show Bollinger Bands
    bb_period (int): Period for Bollinger Bands
    bb_std (float): Standard deviation for Bollinger Bands
    show_ema_fast (bool): Whether to show fast EMA
    ema_fast (int): Period for fast EMA
    chart_style (str): Style for the chart
    
    Returns:
    go.Figure: Plotly figure object
    """
    # Set up row heights and spacing
    row_heights = [0.7, 0.3] if show_volume else [1]
    row_count = 2 if show_volume else 1
    
    # Create figure with secondary y-axis (for volume)
    fig = make_subplots(
        rows=row_count, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=(symbol, "Volume") if show_volume else (symbol,)
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC",
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add volume
    if show_volume:
        colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors,
                name="Volume",
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Add moving averages
    if show_ma:
        for period in ma_periods:
            ma_name = f"MA-{period}"
            df[ma_name] = df['Close'].rolling(window=period).mean()
            
            # Skip if we don't have enough data
            if df[ma_name].isna().all():
                continue
                
            ma_colors = {
                ma_periods[0]: 'rgba(144, 202, 249, 0.9)',  # Light blue
                ma_periods[1]: 'rgba(30, 136, 229, 0.9)',   # Blue
                ma_periods[2]: 'rgba(13, 71, 161, 0.9)'     # Dark blue
            }
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ma_name],
                    name=ma_name,
                    line=dict(width=1.2, color=ma_colors.get(period, 'blue'))
                ),
                row=1, col=1
            )
    
    # Add fast EMA
    if show_ema_fast:
        ema_name = f"EMA-{ema_fast}"
        df[ema_name] = df['Close'].ewm(span=ema_fast, adjust=False).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[ema_name],
                name=ema_name,
                line=dict(width=1.5, color='rgba(255, 183, 77, 0.9)', dash='dot')  # Orange
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands
    if show_bollinger:
        # Calculate Bollinger Bands
        df['MA'] = df['Close'].rolling(window=bb_period).mean()
        df['STD'] = df['Close'].rolling(window=bb_period).std()
        
        df['Upper'] = df['MA'] + (df['STD'] * bb_std)
        df['Lower'] = df['MA'] - (df['STD'] * bb_std)
        
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Upper'],
                name=f'Upper BB ({bb_period}, {bb_std})',
                line=dict(width=1, color='rgba(153, 102, 255, 0.6)'),  # Purple
                fill=None
            ),
            row=1, col=1
        )
        
        # Lower band
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Lower'],
                name=f'Lower BB ({bb_period}, {bb_std})',
                line=dict(width=1, color='rgba(153, 102, 255, 0.6)'),  # Purple
                fill='tonexty',  # Fill between upper and lower bands
                fillcolor='rgba(153, 102, 255, 0.1)'
            ),
            row=1, col=1
        )
    
    # Set chart style
    if chart_style == "Dark":
        bg_color = 'rgb(26, 26, 26)'
        text_color = 'white'
        grid_color = 'rgba(70, 70, 70, 0.3)'
    elif chart_style == "Light":
        bg_color = 'rgb(255, 255, 255)'
        text_color = 'black'
        grid_color = 'rgba(220, 220, 220, 0.5)'
    elif chart_style == "Financial":
        bg_color = 'rgb(240, 252, 255)'
        text_color = 'rgb(41, 56, 73)'
        grid_color = 'rgba(220, 220, 220, 0.5)'
    else:
        # Default style
        bg_color = 'rgb(247, 247, 249)'
        text_color = 'rgb(68, 68, 68)'
        grid_color = 'rgba(230, 230, 230, 0.5)'
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Grafico Candlestick",
        xaxis_title="Data",
        yaxis_title="Prezzo",
        height=800 if show_volume else 600,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        autosize=True,
    )
    
    # Update axis styles
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=grid_color,
        zeroline=False,
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=grid_color,
        zeroline=False,
    )
    
    return fig

def display_price_stats(df):
    """
    Display price statistics for the selected stock
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLC data
    """
    # Recent price data
    latest_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else df['Open'].iloc[-1]
    change = latest_close - prev_close
    change_pct = (change / prev_close) * 100
    
    # Period stats
    period_high = df['High'].max()
    period_low = df['Low'].min()
    period_avg = df['Close'].mean()
    
    # Layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Prezzo Attuale",
            value=f"${latest_close:.2f}",
            delta=f"{change:.2f} ({change_pct:.2f}%)"
        )
    
    with col2:
        st.metric(
            label="Massimo Periodo",
            value=f"${period_high:.2f}",
            delta=f"{(period_high - latest_close):.2f} ({((period_high - latest_close)/latest_close*100):.2f}%)"
        )
    
    with col3:
        st.metric(
            label="Minimo Periodo",
            value=f"${period_low:.2f}",
            delta=f"{(latest_close - period_low):.2f} ({((latest_close - period_low)/period_low*100):.2f}%)"
        )

def display_multi_indicator_chart():
    """
    Display a chart with multiple technical indicators
    """
    st.subheader("Grafico Multi-Indicatori")
    
    st.markdown("""
    Visualizza un grafico con molteplici indicatori tecnici in un'unica visuale.
    Seleziona gli indicatori che vuoi visualizzare.
    """)
    
    # Symbol input
    symbol = st.text_input("Simbolo Azione", value="MSFT", key="multi_indicator_symbol")
    
    # Time period selection
    periods = {
        "1mo": "1 Mese",
        "3mo": "3 Mesi",
        "6mo": "6 Mesi",
        "1y": "1 Anno",
        "2y": "2 Anni"
    }
    
    period = st.select_slider(
        "Periodo",
        options=list(periods.keys()),
        format_func=lambda x: periods[x],
        value="3mo",
        key="multi_indicator_period"
    )
    
    # Indicator selection
    st.subheader("Seleziona Indicatori")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_sma = st.checkbox("Medie Mobili (SMA)", value=True)
        show_volume = st.checkbox("Volume", value=True)
        show_macd = st.checkbox("MACD", value=True)
    
    with col2:
        show_rsi = st.checkbox("RSI", value=True)
        show_bollinger = st.checkbox("Bande di Bollinger", value=True)
        show_atr = st.checkbox("ATR", value=False)
    
    with col3:
        show_stochastic = st.checkbox("Stochastic", value=False)
        show_obv = st.checkbox("On-Balance Volume", value=False)
        show_cci = st.checkbox("CCI", value=False)
    
    # Generate chart
    if st.button("Genera Grafico Multi-Indicatori", use_container_width=True, type="primary"):
        with st.spinner("Caricamento dati e generazione grafico..."):
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                df = stock.history(period=period)
                
                if df.empty:
                    st.error(f"Nessun dato disponibile per il simbolo: {symbol}")
                else:
                    # Create multi-indicator chart
                    fig = create_multi_indicator_chart(
                        df, 
                        symbol,
                        show_sma,
                        show_volume,
                        show_macd,
                        show_rsi,
                        show_bollinger,
                        show_atr,
                        show_stochastic,
                        show_obv,
                        show_cci
                    )
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True, height=900)
            
            except Exception as e:
                st.error(f"Si è verificato un errore: {str(e)}")

def create_multi_indicator_chart(df, symbol, show_sma=True, show_volume=True, 
                               show_macd=True, show_rsi=True, show_bollinger=True,
                               show_atr=False, show_stochastic=False, show_obv=False,
                               show_cci=False):
    """
    Create a chart with multiple technical indicators
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLC data
    symbol (str): Stock symbol
    show_* (bool): Boolean flags for different indicators
    
    Returns:
    go.Figure: Plotly figure with multiple indicators
    """
    # Count how many rows we need
    row_count = 1  # Main price chart
    
    if show_volume:
        row_count += 1
    if show_macd:
        row_count += 1
    if show_rsi:
        row_count += 1
    if show_stochastic:
        row_count += 1
    if show_obv:
        row_count += 1
    if show_atr:
        row_count += 1
    if show_cci:
        row_count += 1
    
    # Calculate row heights
    price_height = 0.4
    other_height = (1 - price_height) / (row_count - 1) if row_count > 1 else 0
    
    row_heights = [price_height]
    for _ in range(row_count - 1):
        row_heights.append(other_height)
    
    # Create subplot titles
    subplot_titles = [f"{symbol} - Prezzo"]
    
    if show_volume:
        subplot_titles.append("Volume")
    if show_macd:
        subplot_titles.append("MACD")
    if show_rsi:
        subplot_titles.append("RSI")
    if show_stochastic:
        subplot_titles.append("Stochastic")
    if show_obv:
        subplot_titles.append("On-Balance Volume")
    if show_atr:
        subplot_titles.append("Average True Range")
    if show_cci:
        subplot_titles.append("Commodity Channel Index")
    
    # Create figure
    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC",
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    current_row = 1
    
    # Add SMA
    if show_sma:
        # Calculate SMAs
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # Add SMA traces
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA20'],
                name="SMA(20)",
                line=dict(width=1.2, color='rgba(144, 202, 249, 0.9)')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA50'],
                name="SMA(50)",
                line=dict(width=1.2, color='rgba(30, 136, 229, 0.9)')
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands
    if show_bollinger:
        # Calculate Bollinger Bands
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        
        df['Upper'] = df['SMA20'] + (df['STD'] * 2)
        df['Lower'] = df['SMA20'] - (df['STD'] * 2)
        
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Upper'],
                name='Upper BB(20,2)',
                line=dict(width=1, color='rgba(153, 102, 255, 0.6)'),
                fill=None
            ),
            row=1, col=1
        )
        
        # Lower band
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Lower'],
                name='Lower BB(20,2)',
                line=dict(width=1, color='rgba(153, 102, 255, 0.6)'),
                fill='tonexty',
                fillcolor='rgba(153, 102, 255, 0.1)'
            ),
            row=1, col=1
        )
    
    # Add Volume
    if show_volume:
        current_row += 1
        colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors,
                name="Volume",
                opacity=0.7
            ),
            row=current_row, col=1
        )
    
    # Add MACD
    if show_macd:
        current_row += 1
        
        # Calculate MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        # Add MACD line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name="MACD",
                line=dict(width=1.5, color='rgba(30, 136, 229, 0.9)')
            ),
            row=current_row, col=1
        )
        
        # Add signal line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Signal'],
                name="Signal",
                line=dict(width=1.5, color='rgba(255, 183, 77, 0.9)')
            ),
            row=current_row, col=1
        )
        
        # Add histogram
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Histogram'],
                name="Histogram",
                marker_color=['#26a69a' if val >= 0 else '#ef5350' for val in df['Histogram']],
                opacity=0.7
            ),
            row=current_row, col=1
        )
    
    # Add RSI
    if show_rsi:
        current_row += 1
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name="RSI(14)",
                line=dict(width=1.5, color='rgba(103, 58, 183, 0.9)')
            ),
            row=current_row, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[70] * len(df),
                name="Overbought",
                line=dict(width=1, color='rgba(255, 82, 82, 0.7)', dash='dash')
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[30] * len(df),
                name="Oversold",
                line=dict(width=1, color='rgba(38, 166, 154, 0.7)', dash='dash')
            ),
            row=current_row, col=1
        )
    
    # Add Stochastic
    if show_stochastic:
        current_row += 1
        
        # Calculate Stochastic
        high_14 = df['High'].rolling(window=14).max()
        low_14 = df['Low'].rolling(window=14).min()
        
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Add %K line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['%K'],
                name="%K",
                line=dict(width=1.5, color='rgba(30, 136, 229, 0.9)')
            ),
            row=current_row, col=1
        )
        
        # Add %D line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['%D'],
                name="%D",
                line=dict(width=1.5, color='rgba(255, 183, 77, 0.9)')
            ),
            row=current_row, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[80] * len(df),
                name="Overbought",
                line=dict(width=1, color='rgba(255, 82, 82, 0.7)', dash='dash')
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[20] * len(df),
                name="Oversold",
                line=dict(width=1, color='rgba(38, 166, 154, 0.7)', dash='dash')
            ),
            row=current_row, col=1
        )
    
    # Add OBV
    if show_obv:
        current_row += 1
        
        # Calculate OBV
        df['OBV'] = 0
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['Volume'].iloc[i]
            else:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1]
        
        # Add OBV line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['OBV'],
                name="OBV",
                line=dict(width=1.5, color='rgba(255, 112, 67, 0.9)')
            ),
            row=current_row, col=1
        )
    
    # Add ATR
    if show_atr:
        current_row += 1
        
        # Calculate ATR
        df['TR'] = df.apply(
            lambda row: max(
                row['High'] - row['Low'],
                abs(row['High'] - df['Close'].shift(1).loc[row.name] if not pd.isna(df['Close'].shift(1).loc[row.name]) else 0),
                abs(row['Low'] - df['Close'].shift(1).loc[row.name] if not pd.isna(df['Close'].shift(1).loc[row.name]) else 0)
            ),
            axis=1
        )
        
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Add ATR line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ATR'],
                name="ATR(14)",
                line=dict(width=1.5, color='rgba(156, 39, 176, 0.9)')
            ),
            row=current_row, col=1
        )
    
    # Add CCI
    if show_cci:
        current_row += 1
        
        # Calculate CCI
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['SMA_TP'] = df['TP'].rolling(window=20).mean()
        df['MAD'] = df['TP'].rolling(window=20).apply(lambda x: pd.Series(x).mad())
        df['CCI'] = (df['TP'] - df['SMA_TP']) / (0.015 * df['MAD'])
        
        # Add CCI line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['CCI'],
                name="CCI(20)",
                line=dict(width=1.5, color='rgba(0, 121, 107, 0.9)')
            ),
            row=current_row, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[100] * len(df),
                name="Overbought",
                line=dict(width=1, color='rgba(255, 82, 82, 0.7)', dash='dash')
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[-100] * len(df),
                name="Oversold",
                line=dict(width=1, color='rgba(38, 166, 154, 0.7)', dash='dash')
            ),
            row=current_row, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Grafico Multi-Indicatori",
        xaxis_title="Data",
        height=300 * row_count,  # Adjust height based on number of rows
        hovermode="x unified",
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        autosize=True,
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    # Update Y axis labels
    axis_num = 1
    for i in range(1, row_count+1):
        yaxis_key = f"yaxis{i if i > 1 else ''}"
        
        # Set specific Y-axis formatting for indicators
        if i == 1:  # Price chart
            title = "Prezzo"
        elif i == axis_num and show_volume:
            title = "Volume"
            axis_num += 1
        elif i == axis_num and show_macd:
            title = "MACD"
            axis_num += 1
        elif i == axis_num and show_rsi:
            title = "RSI"
            # Set fixed range for RSI
            fig.update_layout(**{f"{yaxis_key}.range": [0, 100]})
            axis_num += 1
        elif i == axis_num and show_stochastic:
            title = "Stochastic"
            # Set fixed range for Stochastic
            fig.update_layout(**{f"{yaxis_key}.range": [0, 100]})
            axis_num += 1
        elif i == axis_num and show_obv:
            title = "OBV"
            axis_num += 1
        elif i == axis_num and show_atr:
            title = "ATR"
            axis_num += 1
        elif i == axis_num and show_cci:
            title = "CCI"
            axis_num += 1
        else:
            title = ""
        
        # Update axis title and formatting
        fig.update_layout(**{f"{yaxis_key}.title": title})
    
    return fig

def display_multi_stock_chart():
    """
    Display a chart comparing multiple stocks
    """
    st.subheader("Grafico Comparativo Multi-Azioni")
    
    st.markdown("""
    Confronta l'andamento di più azioni su un unico grafico.
    Inserisci fino a 5 simboli separati da virgola.
    """)
    
    # Symbols input
    symbols_input = st.text_input(
        "Simboli Azioni (separati da virgola)",
        value="AAPL,MSFT,GOOGL"
    )
    
    symbols = [s.strip().upper() for s in symbols_input.split(",")]
    
    # Time period selection
    periods = {
        "1mo": "1 Mese",
        "3mo": "3 Mesi",
        "6mo": "6 Mesi",
        "1y": "1 Anno",
        "2y": "2 Anni",
        "5y": "5 Anni"
    }
    
    period = st.select_slider(
        "Periodo",
        options=list(periods.keys()),
        format_func=lambda x: periods[x],
        value="1y",
        key="multi_stock_period"
    )
    
    # Display options
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Tipo di Grafico",
            options=["Line", "Area", "Candlestick"],
            index=0
        )
        
        normalize = st.checkbox("Normalizza Prezzi (Base 100)", value=True)
    
    with col2:
        include_volume = st.checkbox("Includi Volume", value=False)
        log_scale = st.checkbox("Scala Logaritmica", value=False)
    
    # Generate chart
    if st.button("Genera Grafico Comparativo", use_container_width=True, type="primary"):
        with st.spinner("Caricamento dati e generazione grafico..."):
            if len(symbols) > 5:
                st.warning("Inseriti troppi simboli. Verranno utilizzati solo i primi 5.")
                symbols = symbols[:5]
            
            try:
                # Get stock data for all symbols
                all_data = {}
                valid_symbols = []
                
                for symbol in symbols:
                    stock = yf.Ticker(symbol)
                    df = stock.history(period=period)
                    
                    if not df.empty:
                        all_data[symbol] = df
                        valid_symbols.append(symbol)
                
                if not valid_symbols:
                    st.error("Nessun dato disponibile per i simboli inseriti.")
                else:
                    # Create comparison chart
                    fig = create_multi_stock_chart(
                        all_data,
                        valid_symbols,
                        chart_type,
                        normalize,
                        include_volume,
                        log_scale
                    )
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Si è verificato un errore: {str(e)}")

def create_multi_stock_chart(data_dict, symbols, chart_type="Line", normalize=True, 
                            include_volume=False, log_scale=False):
    """
    Create a chart comparing multiple stocks
    
    Parameters:
    data_dict (dict): Dictionary mapping symbols to their respective DataFrames
    symbols (list): List of stock symbols to include
    chart_type (str): Type of chart ("Line", "Area", or "Candlestick")
    normalize (bool): Whether to normalize prices to a base of 100
    include_volume (bool): Whether to include volume subplot
    log_scale (bool): Whether to use logarithmic scale for price axis
    
    Returns:
    go.Figure: Plotly figure with multiple stocks
    """
    # Determine number of rows
    row_count = 2 if include_volume else 1
    row_heights = [0.8, 0.2] if include_volume else [1]
    
    # Create figure
    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=["Confronto Prezzi", "Volume"] if include_volume else ["Confronto Prezzi"]
    )
    
    # Set color for each symbol
    colors = {
        symbols[0]: 'rgb(0, 121, 107)',      # Teal
        symbols[1] if len(symbols) > 1 else '': 'rgb(30, 136, 229)',  # Blue
        symbols[2] if len(symbols) > 2 else '': 'rgb(156, 39, 176)',  # Purple
        symbols[3] if len(symbols) > 3 else '': 'rgb(214, 61, 57)',   # Red
        symbols[4] if len(symbols) > 4 else '': 'rgb(255, 183, 77)'   # Orange
    }
    
    # Process each symbol's data
    for symbol in symbols:
        df = data_dict[symbol]
        
        # Normalize if requested
        if normalize:
            first_close = df['Close'].iloc[0]
            df = df.copy()
            df['Close'] = (df['Close'] / first_close) * 100
            df['Open'] = (df['Open'] / first_close) * 100
            df['High'] = (df['High'] / first_close) * 100
            df['Low'] = (df['Low'] / first_close) * 100
        
        # Add traces based on chart type
        if chart_type == "Line":
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name=symbol,
                    mode='lines',
                    line=dict(width=2, color=colors.get(symbol, 'blue'))
                ),
                row=1, col=1
            )
        
        elif chart_type == "Area":
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name=symbol,
                    mode='lines',
                    line=dict(width=2, color=colors.get(symbol, 'blue')),
                    fill='tozeroy',
                    fillcolor=f"rgba({colors.get(symbol, 'rgb(30, 136, 229)').replace('rgb(', '').replace(')', '')}, 0.2)"
                ),
                row=1, col=1
            )
        
        elif chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=symbol,
                    increasing_line_color=colors.get(symbol, '#26a69a'),
                    decreasing_line_color=f"rgba({colors.get(symbol, 'rgb(30, 136, 229)').replace('rgb(', '').replace(')', '')}, 0.7)"
                ),
                row=1, col=1
            )
        
        # Add volume
        if include_volume:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name=f"{symbol} Volume",
                    marker_color=f"rgba({colors.get(symbol, 'rgb(30, 136, 229)').replace('rgb(', '').replace(')', '')}, 0.5)"
                ),
                row=2, col=1
            )
    
    # Update layout
    yaxis_title = "Prezzo (Base 100)" if normalize else "Prezzo"
    
    fig.update_layout(
        title=f"Confronto {'Normalizzato ' if normalize else ''}tra {', '.join(symbols)}",
        xaxis_title="Data",
        yaxis_title=yaxis_title,
        height=800 if include_volume else 600,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    # Set log scale if requested
    if log_scale:
        fig.update_layout(yaxis_type="log")
    
    return fig

def display_seasonal_analysis():
    """
    Display seasonal analysis for a stock
    """
    st.subheader("Analisi Stagionale")
    
    st.markdown("""
    Visualizza l'andamento stagionale di un'azione per identificare pattern ricorrenti.
    L'analisi stagionale può aiutare a identificare periodi dell'anno in cui un titolo tende a performare meglio o peggio.
    """)
    
    # Symbol input
    symbol = st.text_input("Simbolo Azione", value="AAPL", key="seasonal_symbol")
    
    # Period options
    col1, col2 = st.columns(2)
    
    with col1:
        years_back = st.slider(
            "Anni di Storia",
            min_value=3,
            max_value=10,
            value=5,
            help="Quanti anni di dati storici utilizzare per l'analisi"
        )
    
    with col2:
        seasonality_type = st.selectbox(
            "Tipo di Stagionalità",
            options=["Mensile", "Trimestrale", "Giorno della Settimana"],
            index=0
        )
    
    # Generate chart
    if st.button("Analizza Stagionalità", use_container_width=True, type="primary"):
        with st.spinner("Analisi della stagionalità in corso..."):
            try:
                # End date (today)
                end_date = datetime.now()
                # Start date (years back)
                start_date = end_date - timedelta(days=365 * years_back)
                
                # Get stock data
                stock = yf.Ticker(symbol)
                df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
                
                if df.empty:
                    st.error(f"Nessun dato disponibile per il simbolo: {symbol}")
                else:
                    # Create seasonal analysis
                    if seasonality_type == "Mensile":
                        fig, monthly_returns = create_monthly_seasonality(df, symbol, years_back)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display monthly returns table
                        st.subheader("Rendimenti Medi Mensili")
                        st.dataframe(monthly_returns, use_container_width=True)
                        
                    elif seasonality_type == "Trimestrale":
                        fig, quarterly_returns = create_quarterly_seasonality(df, symbol, years_back)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display quarterly returns table
                        st.subheader("Rendimenti Medi Trimestrali")
                        st.dataframe(quarterly_returns, use_container_width=True)
                        
                    elif seasonality_type == "Giorno della Settimana":
                        fig, day_returns = create_day_of_week_seasonality(df, symbol, years_back)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display day of week returns table
                        st.subheader("Rendimenti Medi per Giorno della Settimana")
                        st.dataframe(day_returns, use_container_width=True)
            
            except Exception as e:
                st.error(f"Si è verificato un errore: {str(e)}")

def create_monthly_seasonality(df, symbol, years_back):
    """
    Create monthly seasonality analysis
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLC data
    symbol (str): Stock symbol
    years_back (int): Number of years to analyze
    
    Returns:
    tuple: (go.Figure, pd.DataFrame) with the chart and monthly returns
    """
    # Calculate monthly returns
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['MonthName'] = df.index.month_name()
    
    # Order months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Group by year and month, calculate returns
    monthly_returns = df.groupby(['Year', 'Month', 'MonthName'])['Close'].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 else 0
    ).reset_index()
    
    # Create pivot table for heatmap
    pivot_table = monthly_returns.pivot_table(
        index='Year', 
        columns='Month',
        values='Close'
    ).fillna(0)
    
    # Calculate average monthly returns
    avg_monthly_returns = monthly_returns.groupby(['Month', 'MonthName'])['Close'].mean().reset_index()
    avg_monthly_returns = avg_monthly_returns.sort_values('Month')
    
    # Create bar chart for average monthly returns
    fig1 = go.Figure()
    
    # Add bars
    fig1.add_trace(
        go.Bar(
            x=avg_monthly_returns['MonthName'],
            y=avg_monthly_returns['Close'],
            marker_color=[
                'rgb(0, 153, 51)' if val >= 0 else 'rgb(204, 51, 51)' 
                for val in avg_monthly_returns['Close']
            ],
            text=[f"{val:.2f}%" for val in avg_monthly_returns['Close']],
            textposition='auto',
            name="Rendimento Medio"
        )
    )
    
    # Update layout
    fig1.update_layout(
        title=f"{symbol} - Rendimenti Medi Mensili (Ultimi {years_back} Anni)",
        xaxis_title="Mese",
        yaxis_title="Rendimento Medio (%)",
        xaxis=dict(categoryorder='array', categoryarray=month_order),
        height=500,
        template='plotly_white',
        hovermode="x unified",
        yaxis=dict(tickformat=".2f")
    )
    
    # Create heatmap data
    years = sorted(pivot_table.index.unique())
    months = list(range(1, 13))
    
    heatmap_data = []
    for year in years:
        for month in months:
            if year in pivot_table.index and month in pivot_table.columns:
                value = pivot_table.loc[year, month]
                heatmap_data.append([year, month, value])
    
    heatmap_df = pd.DataFrame(heatmap_data, columns=['Year', 'Month', 'Return'])
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=(
            f"{symbol} - Rendimenti Medi Mensili (Ultimi {years_back} Anni)",
            f"{symbol} - Heatmap Rendimenti Mensili per Anno"
        ),
        row_heights=[0.5, 0.5],
        vertical_spacing=0.1
    )
    
    # Add bar chart to first subplot
    fig.add_trace(
        go.Bar(
            x=avg_monthly_returns['MonthName'],
            y=avg_monthly_returns['Close'],
            marker_color=[
                'rgb(0, 153, 51)' if val >= 0 else 'rgb(204, 51, 51)' 
                for val in avg_monthly_returns['Close']
            ],
            text=[f"{val:.2f}%" for val in avg_monthly_returns['Close']],
            textposition='auto',
            name="Rendimento Medio"
        ),
        row=1, col=1
    )
    
    # Add heatmap to second subplot
    fig.add_trace(
        go.Heatmap(
            z=heatmap_df['Return'],
            x=heatmap_df['Month'],
            y=heatmap_df['Year'],
            colorscale=[
                [0, 'rgb(204, 51, 51)'],      # Red for negative
                [0.5, 'rgb(255, 255, 255)'],  # White for zero
                [1, 'rgb(0, 153, 51)']        # Green for positive
            ],
            zmid=0,
            text=[[f"{val:.2f}%" for val in heatmap_df['Return']]],
            hovertemplate='Anno: %{y}<br>Mese: %{x}<br>Rendimento: %{z:.2f}%<extra></extra>',
            colorbar=dict(title="Rendimento (%)")
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        template='plotly_white',
        hovermode="closest",
        xaxis=dict(
            categoryorder='array', 
            categoryarray=month_order,
            title="Mese"
        ),
        yaxis=dict(
            title="Rendimento Medio (%)",
            tickformat=".2f"
        ),
        xaxis2=dict(
            title="Mese",
            tickvals=list(range(1, 13)),
            ticktext=['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
        ),
        yaxis2=dict(
            title="Anno",
            autorange="reversed"  # To have most recent years at the top
        )
    )
    
    # Create a nice table of monthly returns
    table_data = avg_monthly_returns.copy()
    table_data['Mese'] = table_data['MonthName'].apply(lambda x: x[:3])
    table_data['Rendimento Medio (%)'] = table_data['Close'].round(2)
    table_data['Positivo'] = table_data['Close'] >= 0
    table_data = table_data[['Mese', 'Rendimento Medio (%)', 'Positivo']]
    table_data = table_data.sort_values('Mese')
    
    return fig, table_data

def create_quarterly_seasonality(df, symbol, years_back):
    """
    Create quarterly seasonality analysis
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLC data
    symbol (str): Stock symbol
    years_back (int): Number of years to analyze
    
    Returns:
    tuple: (go.Figure, pd.DataFrame) with the chart and quarterly returns
    """
    # Calculate quarterly returns
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year
    
    # Create quarter names
    quarter_names = {
        1: 'Q1 (Gen-Mar)',
        2: 'Q2 (Apr-Giu)',
        3: 'Q3 (Lug-Set)',
        4: 'Q4 (Ott-Dic)'
    }
    
    # Group by year and quarter, calculate returns
    quarterly_returns = df.groupby(['Year', 'Quarter'])['Close'].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 else 0
    ).reset_index()
    
    # Add quarter names
    quarterly_returns['QuarterName'] = quarterly_returns['Quarter'].map(quarter_names)
    
    # Create pivot table for heatmap
    pivot_table = quarterly_returns.pivot_table(
        index='Year', 
        columns='Quarter',
        values='Close'
    ).fillna(0)
    
    # Calculate average quarterly returns
    avg_quarterly_returns = quarterly_returns.groupby(['Quarter', 'QuarterName'])['Close'].mean().reset_index()
    avg_quarterly_returns = avg_quarterly_returns.sort_values('Quarter')
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=(
            f"{symbol} - Rendimenti Medi Trimestrali (Ultimi {years_back} Anni)",
            f"{symbol} - Heatmap Rendimenti Trimestrali per Anno"
        ),
        row_heights=[0.5, 0.5],
        vertical_spacing=0.1
    )
    
    # Add bar chart to first subplot
    fig.add_trace(
        go.Bar(
            x=avg_quarterly_returns['QuarterName'],
            y=avg_quarterly_returns['Close'],
            marker_color=[
                'rgb(0, 153, 51)' if val >= 0 else 'rgb(204, 51, 51)' 
                for val in avg_quarterly_returns['Close']
            ],
            text=[f"{val:.2f}%" for val in avg_quarterly_returns['Close']],
            textposition='auto',
            name="Rendimento Medio"
        ),
        row=1, col=1
    )
    
    # Create heatmap data
    years = sorted(pivot_table.index.unique())
    quarters = list(range(1, 5))
    
    heatmap_data = []
    for year in years:
        for quarter in quarters:
            if year in pivot_table.index and quarter in pivot_table.columns:
                value = pivot_table.loc[year, quarter]
                heatmap_data.append([year, quarter, value])
    
    heatmap_df = pd.DataFrame(heatmap_data, columns=['Year', 'Quarter', 'Return'])
    
    # Add heatmap to second subplot
    fig.add_trace(
        go.Heatmap(
            z=heatmap_df['Return'],
            x=heatmap_df['Quarter'],
            y=heatmap_df['Year'],
            colorscale=[
                [0, 'rgb(204, 51, 51)'],      # Red for negative
                [0.5, 'rgb(255, 255, 255)'],  # White for zero
                [1, 'rgb(0, 153, 51)']        # Green for positive
            ],
            zmid=0,
            text=[[f"{val:.2f}%" for val in heatmap_df['Return']]],
            hovertemplate='Anno: %{y}<br>Trimestre: %{x}<br>Rendimento: %{z:.2f}%<extra></extra>',
            colorbar=dict(title="Rendimento (%)")
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        template='plotly_white',
        hovermode="closest",
        xaxis=dict(
            title="Trimestre",
            tickvals=list(range(4)),
            ticktext=list(quarter_names.values())
        ),
        yaxis=dict(
            title="Rendimento Medio (%)",
            tickformat=".2f"
        ),
        xaxis2=dict(
            title="Trimestre",
            tickvals=[1, 2, 3, 4],
            ticktext=list(quarter_names.values())
        ),
        yaxis2=dict(
            title="Anno",
            autorange="reversed"  # To have most recent years at the top
        )
    )
    
    # Create a nice table of quarterly returns
    table_data = avg_quarterly_returns.copy()
    table_data['Trimestre'] = table_data['QuarterName']
    table_data['Rendimento Medio (%)'] = table_data['Close'].round(2)
    table_data['Positivo'] = table_data['Close'] >= 0
    table_data = table_data[['Trimestre', 'Rendimento Medio (%)', 'Positivo']]
    
    return fig, table_data

def create_day_of_week_seasonality(df, symbol, years_back):
    """
    Create day of week seasonality analysis
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLC data
    symbol (str): Stock symbol
    years_back (int): Number of years to analyze
    
    Returns:
    tuple: (go.Figure, pd.DataFrame) with the chart and day of week returns
    """
    # Calculate daily returns
    df['Return'] = df['Close'].pct_change() * 100
    df = df.dropna()
    
    # Add day of week
    df['DayOfWeek'] = df.index.dayofweek
    df['DayName'] = df.index.day_name()
    
    # Define Italian day names
    italian_days = {
        'Monday': 'Lunedì',
        'Tuesday': 'Martedì',
        'Wednesday': 'Mercoledì',
        'Thursday': 'Giovedì',
        'Friday': 'Venerdì',
        'Saturday': 'Sabato',
        'Sunday': 'Domenica'
    }
    
    # Map to Italian day names
    df['DayNameIT'] = df['DayName'].map(italian_days)
    
    # Order days correctly
    day_order = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì']
    
    # Calculate average returns by day of week
    day_returns = df.groupby(['DayOfWeek', 'DayNameIT'])['Return'].agg(['mean', 'count']).reset_index()
    day_returns = day_returns.sort_values('DayOfWeek')
    
    # Filter out weekends (usually no trading)
    day_returns = day_returns[day_returns['DayNameIT'].isin(day_order)]
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(
        go.Bar(
            x=day_returns['DayNameIT'],
            y=day_returns['mean'],
            marker_color=[
                'rgb(0, 153, 51)' if val >= 0 else 'rgb(204, 51, 51)' 
                for val in day_returns['mean']
            ],
            text=[f"{val:.3f}%" for val in day_returns['mean']],
            textposition='auto',
            name="Rendimento Medio"
        )
    )
    
    # Add sample size as text
    for i, row in day_returns.iterrows():
        fig.add_annotation(
            x=row['DayNameIT'],
            y=row['mean'] + (0.02 if row['mean'] >= 0 else -0.02),
            text=f"n={row['count']}",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Rendimenti Medi per Giorno della Settimana (Ultimi {years_back} Anni)",
        xaxis_title="Giorno della Settimana",
        yaxis_title="Rendimento Medio Giornaliero (%)",
        xaxis=dict(categoryorder='array', categoryarray=day_order),
        height=500,
        template='plotly_white',
        hovermode="x unified",
        yaxis=dict(tickformat=".3f")
    )
    
    # Create a nice table of day returns
    table_data = day_returns.copy()
    table_data['Giorno'] = table_data['DayNameIT']
    table_data['Rendimento Medio (%)'] = table_data['mean'].round(3)
    table_data['Numero Osservazioni'] = table_data['count']
    table_data['Positivo'] = table_data['mean'] >= 0
    table_data = table_data[['Giorno', 'Rendimento Medio (%)', 'Numero Osservazioni', 'Positivo']]
    
    return fig, table_data