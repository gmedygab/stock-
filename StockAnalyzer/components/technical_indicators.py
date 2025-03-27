import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_fetcher import get_stock_data

def calculate_rsi(df, window=14):
    """Calculate the Relative Strength Index (RSI)"""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)"""
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def calculate_bollinger_bands(df, window=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    
    return upper_band, sma, lower_band

def calculate_moving_averages(df):
    """Calculate various moving averages"""
    sma_20 = df['Close'].rolling(window=20).mean()
    sma_50 = df['Close'].rolling(window=50).mean()
    sma_200 = df['Close'].rolling(window=200).mean()
    ema_20 = df['Close'].ewm(span=20, adjust=False).mean()
    
    return sma_20, sma_50, sma_200, ema_20

def calculate_stochastic_oscillator(df, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    # Calculate %K
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    
    # Fast %K
    k_fast = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    
    # Slow %K (i.e., %K smoothed)
    k_slow = k_fast.rolling(window=d_window).mean()
    
    # %D (i.e., %D smoothed)
    d_slow = k_slow.rolling(window=d_window).mean()
    
    return k_slow, d_slow

def calculate_atr(df, window=14):
    """Calculate Average True Range (ATR)"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_obv(df):
    """Calculate On-Balance Volume (OBV)"""
    obv = np.zeros(len(df))
    obv[0] = df['Volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['Volume'].iloc[i]
        else:
            obv[i] = obv[i-1]
    
    return pd.Series(obv, index=df.index)

def calculate_fibonacci_levels(df, trend='downtrend'):
    """Calculate Fibonacci Retracement Levels"""
    if trend == 'downtrend':
        high = df['High'].max()
        low = df['Low'].min()
    else:  # uptrend
        high = df['High'].iloc[-int(len(df) * 0.2):].max()  # Last 20% of data
        low = df['Low'].iloc[-int(len(df) * 0.2):].min()    # Last 20% of data
    
    diff = high - low
    
    levels = {
        '0.0': low,
        '0.236': low + 0.236 * diff,
        '0.382': low + 0.382 * diff,
        '0.5': low + 0.5 * diff,
        '0.618': low + 0.618 * diff,
        '0.786': low + 0.786 * diff,
        '1.0': high
    }
    
    return levels

def display_technical_indicators(symbol, timeframe, detailed=False):
    """Display technical indicators for a stock"""
    try:
        # Fetch stock data
        df = get_stock_data(symbol, timeframe)
        
        if df.empty:
            st.error(f"No data available for {symbol}")
            return
        
        # Calculate standard indicators
        rsi = calculate_rsi(df)
        macd, signal_line, histogram = calculate_macd(df)
        upper_band, middle_band, lower_band = calculate_bollinger_bands(df)
        sma_20, sma_50, sma_200, ema_20 = calculate_moving_averages(df)
        
        # Calculate additional indicators
        k_slow, d_slow = calculate_stochastic_oscillator(df)
        atr = calculate_atr(df)
        obv = calculate_obv(df)
        fibonacci_levels = calculate_fibonacci_levels(df)
        
        # Create a tab layout for indicators
        standard_tab, advanced_tab = st.tabs(["Standard Indicators", "Advanced Indicators"])
        
        with standard_tab:
            # Create indicators data frame for display
            indicators_data = {
                'Indicator': ['RSI (14)', 'MACD', 'Signal Line', 'BB Upper', 'BB Middle', 'BB Lower', 'SMA (20)', 'SMA (50)', 'SMA (200)', 'EMA (20)'],
                'Value': [
                    f"{rsi.iloc[-1]:.2f}",
                    f"{macd.iloc[-1]:.2f}",
                    f"{signal_line.iloc[-1]:.2f}",
                    f"${upper_band.iloc[-1]:.2f}",
                    f"${middle_band.iloc[-1]:.2f}",
                    f"${lower_band.iloc[-1]:.2f}",
                    f"${sma_20.iloc[-1]:.2f}",
                    f"${sma_50.iloc[-1]:.2f}",
                    f"${sma_200.iloc[-1]:.2f}",
                    f"${ema_20.iloc[-1]:.2f}"
                ]
            }
            
            indicators_df = pd.DataFrame(indicators_data)
            
            # Display standard indicators in a table
            st.subheader("Standard Technical Indicators")
            st.dataframe(indicators_df, use_container_width=True)
        
        with advanced_tab:
            # Create advanced indicators data frame
            advanced_data = {
                'Indicator': [
                    'Stochastic %K', 
                    'Stochastic %D', 
                    'ATR (14)', 
                    'OBV'
                ],
                'Value': [
                    f"{k_slow.iloc[-1]:.2f}",
                    f"{d_slow.iloc[-1]:.2f}",
                    f"{atr.iloc[-1]:.2f}",
                    f"{obv.iloc[-1]:.0f}"
                ]
            }
            
            advanced_df = pd.DataFrame(advanced_data)
            
            # Display advanced indicators in a table
            st.subheader("Advanced Technical Indicators")
            st.dataframe(advanced_df, use_container_width=True)
            
            # Display Fibonacci levels
            st.subheader("Fibonacci Retracement Levels")
            fib_data = {
                'Level': list(fibonacci_levels.keys()),
                'Price': [f"${v:.2f}" for v in fibonacci_levels.values()]
            }
            fib_df = pd.DataFrame(fib_data)
            st.dataframe(fib_df, use_container_width=True)
            
            # Allow user to customize indicators
            st.subheader("Customize Indicators")
            col1, col2 = st.columns(2)
            
            with col1:
                rsi_period = st.slider("RSI Period", min_value=5, max_value=30, value=14, step=1)
                macd_fast = st.slider("MACD Fast Period", min_value=5, max_value=20, value=12, step=1)
                macd_slow = st.slider("MACD Slow Period", min_value=15, max_value=40, value=26, step=1)
            
            with col2:
                bb_period = st.slider("Bollinger Period", min_value=5, max_value=30, value=20, step=1)
                bb_std = st.slider("Bollinger StdDev", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
                stoch_period = st.slider("Stochastic Period", min_value=5, max_value=30, value=14, step=1)
            
            # Recalculate with custom parameters if user clicks the button
            if st.button("Recalculate Indicators", use_container_width=True):
                # Recalculate with custom parameters
                rsi_custom = calculate_rsi(df, window=rsi_period)
                macd_custom, signal_line_custom, histogram_custom = calculate_macd(df, fast=macd_fast, slow=macd_slow)
                upper_band_custom, middle_band_custom, lower_band_custom = calculate_bollinger_bands(df, window=bb_period, std_dev=bb_std)
                k_slow_custom, d_slow_custom = calculate_stochastic_oscillator(df, k_window=stoch_period)
                
                # Show custom results
                custom_data = {
                    'Indicator': [
                        f'RSI ({rsi_period})', 
                        f'MACD ({macd_fast}/{macd_slow})', 
                        'Signal Line', 
                        f'BB Upper ({bb_period}, {bb_std})', 
                        f'BB Middle ({bb_period})',
                        f'BB Lower ({bb_period}, {bb_std})',
                        f'Stochastic %K ({stoch_period})',
                        f'Stochastic %D ({stoch_period})'
                    ],
                    'Value': [
                        f"{rsi_custom.iloc[-1]:.2f}",
                        f"{macd_custom.iloc[-1]:.2f}",
                        f"{signal_line_custom.iloc[-1]:.2f}",
                        f"${upper_band_custom.iloc[-1]:.2f}",
                        f"${middle_band_custom.iloc[-1]:.2f}",
                        f"${lower_band_custom.iloc[-1]:.2f}",
                        f"{k_slow_custom.iloc[-1]:.2f}",
                        f"{d_slow_custom.iloc[-1]:.2f}"
                    ]
                }
                
                custom_df = pd.DataFrame(custom_data)
                st.subheader("Custom Parameters Results")
                st.dataframe(custom_df, use_container_width=True)
        
        # Display interpretation if detailed view
        if detailed:
            # Create subplot for RSI and MACD
            fig = make_subplots(
                rows=2, 
                cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('RSI (14)', 'MACD')
            )
            
            # Add RSI
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rsi,
                    name="RSI",
                    line=dict(color='purple', width=1.5)
                ),
                row=1, col=1
            )
            
            # Add RSI reference lines at 70 and 30
            fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=1, col=1)
            
            # Add MACD
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=macd,
                    name="MACD",
                    line=dict(color='blue', width=1.5)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=signal_line,
                    name="Signal Line",
                    line=dict(color='red', width=1.5)
                ),
                row=2, col=1
            )
            
            # Add MACD histogram
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=histogram,
                    name="Histogram",
                    marker=dict(
                        color=np.where(histogram > 0, 'green', 'red'),
                        opacity=0.5
                    )
                ),
                row=2, col=1
            )
            
            # Configure layout
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=50, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Analysis Interpretation
            st.subheader("Technical Analysis Interpretation")
            
            # RSI Interpretation
            rsi_value = rsi.iloc[-1]
            if rsi_value > 70:
                rsi_status = "📈 Overbought - The stock may be overvalued and could see a price correction."
                rsi_color = "red"
            elif rsi_value < 30:
                rsi_status = "📉 Oversold - The stock may be undervalued and could see a price increase."
                rsi_color = "green"
            else:
                rsi_status = "✅ Neutral - The stock is neither overbought nor oversold."
                rsi_color = "gray"
            
            # MACD Interpretation
            macd_value = macd.iloc[-1]
            signal_value = signal_line.iloc[-1]
            
            if macd_value > signal_value and macd.iloc[-2] <= signal_line.iloc[-2]:
                macd_status = "📈 Bullish Crossover - MACD crossed above signal line, suggesting potential uptrend."
                macd_color = "green"
            elif macd_value < signal_value and macd.iloc[-2] >= signal_line.iloc[-2]:
                macd_status = "📉 Bearish Crossover - MACD crossed below signal line, suggesting potential downtrend."
                macd_color = "red"
            elif macd_value > signal_value:
                macd_status = "🔼 Bullish - MACD is above signal line, suggesting upward momentum."
                macd_color = "green"
            elif macd_value < signal_value:
                macd_status = "🔽 Bearish - MACD is below signal line, suggesting downward momentum."
                macd_color = "red"
            else:
                macd_status = "⚖️ Neutral - MACD and signal line are equal."
                macd_color = "gray"
            
            # Moving Averages Interpretation
            current_price = df['Close'].iloc[-1]
            
            if current_price > sma_20.iloc[-1] and current_price > sma_50.iloc[-1] and current_price > sma_200.iloc[-1]:
                ma_status = "📈 Strong Uptrend - Price is above all major moving averages."
                ma_color = "green"
            elif current_price < sma_20.iloc[-1] and current_price < sma_50.iloc[-1] and current_price < sma_200.iloc[-1]:
                ma_status = "📉 Strong Downtrend - Price is below all major moving averages."
                ma_color = "red"
            elif current_price > sma_200.iloc[-1]:
                ma_status = "📊 Long-term Bullish - Price is above 200-day SMA."
                ma_color = "green"
            elif current_price < sma_200.iloc[-1]:
                ma_status = "📊 Long-term Bearish - Price is below 200-day SMA."
                ma_color = "red"
            else:
                ma_status = "⚖️ Mixed Signals - Price is showing mixed signals against moving averages."
                ma_color = "gray"
            
            # Bollinger Bands Interpretation
            if current_price > upper_band.iloc[-1]:
                bb_status = "📈 Overbought - Price is above the upper Bollinger Band."
                bb_color = "red"
            elif current_price < lower_band.iloc[-1]:
                bb_status = "📉 Oversold - Price is below the lower Bollinger Band."
                bb_color = "green"
            else:
                bb_status = "✅ Normal Volatility - Price is within the Bollinger Bands."
                bb_color = "gray"
            
            # Display interpretations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"<h4 style='color:{rsi_color}'>RSI</h4>", unsafe_allow_html=True)
                st.markdown(rsi_status)
                
                st.markdown(f"<h4 style='color:{macd_color}'>MACD</h4>", unsafe_allow_html=True)
                st.markdown(macd_status)
            
            with col2:
                st.markdown(f"<h4 style='color:{ma_color}'>Moving Averages</h4>", unsafe_allow_html=True)
                st.markdown(ma_status)
                
                st.markdown(f"<h4 style='color:{bb_color}'>Bollinger Bands</h4>", unsafe_allow_html=True)
                st.markdown(bb_status)
            
            # Overall sentiment
            signals = [
                1 if rsi_value < 30 else (-1 if rsi_value > 70 else 0),
                1 if macd_value > signal_value else -1,
                1 if current_price > sma_50.iloc[-1] else -1,
                1 if current_price > sma_200.iloc[-1] else -1,
                1 if current_price < upper_band.iloc[-1] and current_price > lower_band.iloc[-1] else 0
            ]
            
            signal_sum = sum(signals)
            
            if signal_sum >= 3:
                sentiment = "Strong Buy"
                sentiment_color = "green"
            elif signal_sum >= 1:
                sentiment = "Buy"
                sentiment_color = "lightgreen"
            elif signal_sum <= -3:
                sentiment = "Strong Sell"
                sentiment_color = "red"
            elif signal_sum <= -1:
                sentiment = "Sell"
                sentiment_color = "lightcoral"
            else:
                sentiment = "Neutral"
                sentiment_color = "gray"
            
            st.markdown("---")
            st.markdown(f"<h3>Overall Sentiment: <span style='color:{sentiment_color}'>{sentiment}</span></h3>", unsafe_allow_html=True)
            st.caption("Note: This is not financial advice. Always do your own research before making investment decisions.")
        
    except Exception as e:
        st.error(f"Error displaying technical indicators: {str(e)}")
