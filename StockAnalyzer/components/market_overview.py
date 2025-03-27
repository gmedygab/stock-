import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.data_fetcher import get_market_indices, get_sector_performance

def display_market_overview():
    """
    Display an overview of the market, including major indices, sectors, and top gainers/losers
    """
    # Market Overview Tabs
    overview_tab, indices_tab, sectors_tab, gainers_losers_tab, heat_map_tab, world_markets_tab = st.tabs([
        "Overview 📊", "Indices 📈", "Sectors 🏭", "Gainers/Losers 💰", "Heat Map 🔥", "World Markets 🌎"
    ])
    
    with overview_tab:
        # Major market indices summary
        st.subheader("Major Market Indices")
        
        # Get market indices data from data_fetcher
        df_indices = get_market_indices()
        
        if not df_indices.empty:
            # Save the original numeric data for calculations
            df_numeric = df_indices.copy()
            
            # Create a display copy for formatting
            df_display = df_indices.copy()
            
            # Format price and change values using string formatting for each element individually
            df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
            df_display['Change'] = df_display['Change'].apply(lambda x: f"{x:+.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
            df_display['Change %'] = df_display['Change %'].apply(lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
            
            # Style the dataframe
            def color_change(val):
                if '+' in str(val):
                    return 'background-color: rgba(0, 255, 0, 0.2)'
                elif '-' in str(val):
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                return ''
            
            styled_df = df_display.style.map(color_change, subset=['Change', 'Change %'])
            
            # Display the dataframe
            st.dataframe(styled_df, use_container_width=True)
            
            # Market health/sentiment gauge
            st.subheader("Market Health Indicator")
            
            # Calculate an overall market health score based on index performances
            # Ensure we're using numeric values for calculation
            try:
                # Convert 'Change %' to float explicitly to ensure numeric calculation
                numeric_changes = pd.to_numeric(df_numeric['Change %'], errors='coerce')
                health_score = numeric_changes.mean()
                if pd.isna(health_score):  # If mean returns NaN, use a default value
                    health_score = 0.0
            except Exception as e:
                st.error(f"Error calculating market health: {str(e)}")
                health_score = 0.0  # Use a default value in case of error
            
            # Create gauge chart
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Market Health", 'font': {'size': 24}},
                delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [-3, 3], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [-3, -2], 'color': 'red'},
                        {'range': [-2, -1], 'color': 'lightcoral'},
                        {'range': [-1, 0], 'color': 'lightyellow'},
                        {'range': [0, 1], 'color': 'lightgreen'},
                        {'range': [1, 2], 'color': 'palegreen'},
                        {'range': [2, 3], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            
            gauge_fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Add interpretation of market health
            if health_score < -2:
                health_text = "🔴 **Market under significant pressure**: Consider defensive strategies or staying on sidelines"
                health_color = "red"
            elif health_score < -1:
                health_text = "🟠 **Market showing weakness**: Caution advised, potential opportunity for strategic entries"
                health_color = "orange"
            elif health_score < 0:
                health_text = "🟡 **Market slightly negative**: Mixed signals, maintain diversification"
                health_color = "gold"
            elif health_score < 1:
                health_text = "🟢 **Market slightly positive**: Favorable conditions, watch for breakouts"
                health_color = "lightgreen"
            elif health_score < 2:
                health_text = "🟢 **Market showing strength**: Bullish conditions, momentum strategies may be effective"
                health_color = "green"
            else:
                health_text = "🟢 **Market strongly bullish**: Potential for overbought conditions, consider taking profits"
                health_color = "darkgreen"
            
            st.markdown(f"<div style='text-align: center; color: {health_color};'>{health_text}</div>", unsafe_allow_html=True)
            
            # Get sector performance
            df_sectors = get_sector_performance()
            
            if not df_sectors.empty:
                st.subheader("Sector Overview")
                
                # Horizontal bar chart for sectors - ensure we're sorting with numeric values
                # First ensure Daily Change % is numeric
                df_sectors_plot = df_sectors.copy()
                df_sectors_plot['Daily Change %'] = pd.to_numeric(df_sectors_plot['Daily Change %'], errors='coerce')
                
                # Then sort and create the plot with the numeric values
                sectors_fig = px.bar(
                    df_sectors_plot.sort_values('Daily Change %', ascending=True),
                    y='Sector',
                    x='Daily Change %',
                    orientation='h',
                    color='Daily Change %',
                    color_continuous_scale=['red', 'lightyellow', 'green'],
                    range_color=[-2, 2],
                    labels={'Daily Change %': 'Daily Change (%)', 'Sector': 'Sector'},
                    title="Sector Performance (Daily Change %)"
                )
                
                sectors_fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                    yaxis={'categoryorder': 'total ascending'},
                    coloraxis_colorbar=dict(
                        title='Change %',
                        tickvals=[-2, -1, 0, 1, 2],
                        ticktext=['-2%', '-1%', '0%', '1%', '2%']
                    )
                )
                
                # Add reference line at 0
                sectors_fig.add_shape(
                    type="line",
                    x0=0, y0=-0.5,
                    x1=0, y1=len(df_sectors) - 0.5,
                    line=dict(color="gray", width=1, dash="dash")
                )
                
                st.plotly_chart(sectors_fig, use_container_width=True)
            
            # Define indices dictionary for the next sections
            indices = {
                '^GSPC': 'S&P 500',
                '^DJI': 'Dow Jones',
                '^IXIC': 'NASDAQ',
                '^RUT': 'Russell 2000',
                '^VIX': 'Volatility Index (VIX)'
            }
            
            # Chart the indices
            st.subheader("Market Performance (Last 6 Months)")
            
            # Get 6 month data for indices
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            fig = go.Figure()
            
            for symbol, name in indices.items():
                if symbol == '^VIX':  # Skip VIX for this chart
                    continue
                    
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Normalize to percentage change from first day
                    first_price = hist['Close'].iloc[0]
                    normalized = ((hist['Close'] - first_price) / first_price) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        name=name,
                        mode='lines'
                    ))
            
            fig.update_layout(
                title="Major Indices Performance (% Change)",
                xaxis_title="Date",
                yaxis_title="% Change",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig.add_shape(
                type="line",
                x0=start_date,
                y0=0,
                x1=end_date,
                y1=0,
                line=dict(
                    color="Gray",
                    width=1,
                    dash="dash",
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Could not retrieve market index data.")
    
    with indices_tab:
        st.subheader("Major Market Indices Detail")
        
        # Market Indices Analysis
        try:
            # Chart the indices performance
            st.subheader("Market Performance (Last 6 Months)")
            
            # Get 6 month data for indices
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            # Define indices to chart
            indices = {
                '^GSPC': 'S&P 500',
                '^DJI': 'Dow Jones',
                '^IXIC': 'NASDAQ',
                '^RUT': 'Russell 2000',
                '^VIX': 'Volatility Index (VIX)'
            }
            
            fig = go.Figure()
            
            for symbol, name in indices.items():
                if symbol == '^VIX':  # Skip VIX for this chart
                    continue
                    
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Normalize to percentage change from first day
                    first_price = hist['Close'].iloc[0]
                    normalized = ((hist['Close'] - first_price) / first_price) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        name=name,
                        mode='lines'
                    ))
            
            fig.update_layout(
                title="Major Indices Performance (% Change)",
                xaxis_title="Date",
                yaxis_title="% Change",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig.add_shape(
                type="line",
                x0=start_date,
                y0=0,
                x1=end_date,
                y1=0,
                line=dict(
                    color="Gray",
                    width=1,
                    dash="dash",
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add comparison option
            st.subheader("Compare Specific Indices")
            selected_indices = st.multiselect(
                "Select indices to compare:",
                options=list(indices.keys()),
                default=['^GSPC', '^IXIC'],
                format_func=lambda x: indices.get(x, x)
            )
            
            timeframe = st.select_slider(
                "Select timeframe:",
                options=["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"],
                value="1Y"
            )
            
            chart_type = st.radio(
                "Chart type:",
                options=["Line", "Area", "Candlestick"],
                horizontal=True
            )
            
            if st.button("Compare Indices", use_container_width=True):
                if selected_indices:
                    # Map timeframe to days
                    timeframe_map = {
                        "1M": 30, "3M": 90, "6M": 180,
                        "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
                        "1Y": 365, "3Y": 1095, "5Y": 1825, "10Y": 3650
                    }
                    
                    comp_end_date = datetime.now()
                    comp_start_date = comp_end_date - timedelta(days=timeframe_map[timeframe])
                    
                    if chart_type == "Line" or chart_type == "Area":
                        comp_fig = go.Figure()
                        
                        for symbol in selected_indices:
                            ticker = yf.Ticker(symbol)
                            hist = ticker.history(start=comp_start_date, end=comp_end_date)
                            
                            if not hist.empty:
                                if chart_type == "Line":
                                    comp_fig.add_trace(go.Scatter(
                                        x=hist.index,
                                        y=hist['Close'],
                                        name=indices.get(symbol, symbol),
                                        mode='lines'
                                    ))
                                else:  # Area
                                    comp_fig.add_trace(go.Scatter(
                                        x=hist.index,
                                        y=hist['Close'],
                                        name=indices.get(symbol, symbol),
                                        fill='tozeroy',
                                        mode='lines'
                                    ))
                        
                    else:  # Candlestick
                        # Create subplots, one for each selected index
                        comp_fig = make_subplots(
                            rows=len(selected_indices),
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=[indices.get(s, s) for s in selected_indices]
                        )
                        
                        for i, symbol in enumerate(selected_indices, 1):
                            ticker = yf.Ticker(symbol)
                            hist = ticker.history(start=comp_start_date, end=comp_end_date)
                            
                            if not hist.empty:
                                comp_fig.add_trace(
                                    go.Candlestick(
                                        x=hist.index,
                                        open=hist['Open'],
                                        high=hist['High'],
                                        low=hist['Low'],
                                        close=hist['Close'],
                                        name=indices.get(symbol, symbol)
                                    ),
                                    row=i, col=1
                                )
                                
                                # Add volume as bar chart
                                comp_fig.add_trace(
                                    go.Bar(
                                        x=hist.index,
                                        y=hist['Volume'],
                                        name='Volume',
                                        marker=dict(color='rgba(100, 100, 200, 0.3)'),
                                        showlegend=False
                                    ),
                                    row=i, col=1
                                )
                    
                    # Update layout
                    height = 400 if chart_type != "Candlestick" else 300 * len(selected_indices)
                    comp_fig.update_layout(
                        title=f"Comparison of Selected Indices ({timeframe})",
                        height=height,
                        xaxis_title="Date",
                        yaxis_title="Price",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(comp_fig, use_container_width=True)
                else:
                    st.warning("Please select at least one index to compare.")
                    
        except Exception as e:
            st.error(f"Error generating index comparisons: {str(e)}")
    
    with sectors_tab:
        st.subheader("Sector Performance Analysis")
        
        # Get sector performance data
        df_sectors = get_sector_performance()
        
        try:
            if not df_sectors.empty:
                # Display sector performance data
                st.subheader("Sector Performance Overview")
                
                # Format for display
                df_display = df_sectors.copy()
                df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
                df_display['Daily Change %'] = df_display['Daily Change %'].apply(lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
                df_display['5-Day Change %'] = df_display['5-Day Change %'].apply(lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
                
                # Style the dataframe
                def color_change(val):
                    if '+' in str(val):
                        return 'background-color: rgba(0, 255, 0, 0.2)'
                    elif '-' in str(val):
                        return 'background-color: rgba(255, 0, 0, 0.2)'
                    return ''
                
                styled_df = df_display.style.map(color_change, subset=['Daily Change %', '5-Day Change %'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Create horizontal bar chart for daily change - convert to numeric first
                daily_df = df_sectors.copy()
                daily_df['Daily Change %'] = pd.to_numeric(daily_df['Daily Change %'], errors='coerce')
                
                daily_fig = px.bar(
                    daily_df.sort_values('Daily Change %'),
                    y='Sector',
                    x='Daily Change %',
                    orientation='h',
                    color='Daily Change %',
                    color_continuous_scale=['red', 'lightyellow', 'green'],
                    range_color=[-2, 2],
                    labels={'Daily Change %': 'Daily Change (%)', 'Sector': 'Sector'},
                    title="Sector Performance (Daily Change %)"
                )
                
                daily_fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                # Add reference line at 0
                daily_fig.add_shape(
                    type="line",
                    x0=0, y0=-0.5,
                    x1=0, y1=len(df_sectors) - 0.5,
                    line=dict(color="gray", width=1, dash="dash")
                )
                
                st.plotly_chart(daily_fig, use_container_width=True)
                
                # Create horizontal bar chart for 5-day change - convert to numeric first
                five_day_df = df_sectors.copy()
                five_day_df['5-Day Change %'] = pd.to_numeric(five_day_df['5-Day Change %'], errors='coerce')
                
                five_day_fig = px.bar(
                    five_day_df.sort_values('5-Day Change %'),
                    y='Sector',
                    x='5-Day Change %',
                    orientation='h',
                    color='5-Day Change %',
                    color_continuous_scale=['red', 'lightyellow', 'green'],
                    range_color=[-5, 5],
                    labels={'5-Day Change %': '5-Day Change (%)', 'Sector': 'Sector'},
                    title="Sector Performance (5-Day Change %)"
                )
                
                five_day_fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                # Add reference line at 0
                five_day_fig.add_shape(
                    type="line",
                    x0=0, y0=-0.5,
                    x1=0, y1=len(df_sectors) - 0.5,
                    line=dict(color="gray", width=1, dash="dash")
                )
                
                st.plotly_chart(five_day_fig, use_container_width=True)
                
                # Sector Rotation Analysis
                st.subheader("Sector Rotation Analysis")
                st.markdown("""
                Sector rotation is the movement of money from one industry to another as investors anticipate the next stage of the economic cycle.
                """)
                
                # Create a polar chart to show sector performance
                theta = df_sectors['Sector'].tolist()
                r = df_sectors['Daily Change %'].tolist()
                
                polar_fig = go.Figure(go.Barpolar(
                    r=r,
                    theta=theta,
                    marker_color=["red" if x < 0 else "green" for x in r],
                    opacity=0.8
                ))
                
                polar_fig.update_layout(
                    title="Sector Rotation (Daily Change %)",
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[-max(abs(min(r)), abs(max(r)))*1.2, max(abs(min(r)), abs(max(r)))*1.2]
                        )
                    ),
                    height=500
                )
                
                st.plotly_chart(polar_fig, use_container_width=True)
            
            else:
                st.warning("Could not retrieve sector performance data.")
        
        except Exception as e:
            st.error(f"Error analyzing sector performance: {str(e)}")
    
    with gainers_losers_tab:
        st.subheader("Top Gainers and Losers")
        
        try:
            # For simplicity, use a predefined list of S&P 500 stocks
            # In a real application, you would fetch the actual components
            sp500_sample = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG',
                'UNH', 'MA', 'HD', 'BAC', 'XOM', 'DIS', 'ADBE', 'CRM', 'NFLX', 'CMCSA',
                'VZ', 'KO', 'PEP', 'INTC', 'CSCO', 'ABT', 'MRK', 'WMT', 'PFE', 'TMO'
            ]
            
            # Get performance data for each stock
            performance_data = []
            
            for symbol in sp500_sample:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    
                    if len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        change = current - previous
                        change_pct = (change / previous) * 100
                        
                        # Get company name
                        info = ticker.info
                        name = info.get('shortName', symbol)
                        
                        performance_data.append({
                            'Symbol': symbol,
                            'Name': name,
                            'Price': current,
                            'Change': change,
                            'Change %': change_pct
                        })
                except:
                    continue
            
            if performance_data:
                df = pd.DataFrame(performance_data)
                
                # Sort for gainers and losers - ensure data is numeric first
                df_numeric = df.copy()
                df_numeric['Change %'] = pd.to_numeric(df_numeric['Change %'], errors='coerce')
                
                gainers = df_numeric.sort_values('Change %', ascending=False).head(5)
                losers = df_numeric.sort_values('Change %').head(5)
                
                # Format the dataframes for display
                for frame in [gainers, losers]:
                    frame['Price'] = frame['Price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
                    frame['Change'] = frame['Change'].apply(lambda x: f"{x:+.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
                    frame['Change %'] = frame['Change %'].apply(lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
                
                # Create columns for display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Gainers")
                    
                    # Style the dataframe
                    def color_green(val):
                        return 'background-color: rgba(0, 255, 0, 0.2)'
                    
                    styled_gainers = gainers.style.map(color_green, subset=['Change', 'Change %'])
                    st.dataframe(styled_gainers, use_container_width=True)
                
                with col2:
                    st.subheader("Top Losers")
                    
                    # Style the dataframe
                    def color_red(val):
                        return 'background-color: rgba(255, 0, 0, 0.2)'
                    
                    styled_losers = losers.style.map(color_red, subset=['Change', 'Change %'])
                    st.dataframe(styled_losers, use_container_width=True)
            
            else:
                st.warning("Could not retrieve top gainers and losers data.")
        
        except Exception as e:
            st.error(f"Error loading top gainers and losers: {str(e)}")
    
    with heat_map_tab:
        st.subheader("Market Heat Map")
        
        try:
            # For simplicity, we'll use the same S&P 500 sample
            # In a real application, you would group by sector
            
            # Group companies by sector
            sector_mapping = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'ADBE', 'CRM', 'INTC', 'CSCO'],
                'Communication': ['NFLX', 'CMCSA', 'VZ', 'DIS'],
                'Consumer': ['AMZN', 'TSLA', 'HD', 'KO', 'PEP', 'WMT', 'PG'],
                'Financial': ['JPM', 'V', 'MA', 'BAC'],
                'Healthcare': ['UNH', 'ABT', 'MRK', 'PFE', 'TMO'],
                'Energy': ['XOM']
            }
            
            # Get performance data for each stock
            heat_map_data = []
            
            for sector, symbols in sector_mapping.items():
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="2d")
                        
                        if len(hist) >= 2:
                            current = hist['Close'].iloc[-1]
                            previous = hist['Close'].iloc[-2]
                            change_pct = ((current - previous) / previous) * 100
                            
                            # Get market cap
                            info = ticker.info
                            market_cap = info.get('marketCap', 1000000000)  # Default to 1B if not available
                            
                            heat_map_data.append({
                                'Symbol': symbol,
                                'Sector': sector,
                                'Change %': change_pct,
                                'Market Cap': market_cap
                            })
                    except:
                        continue
            
            if heat_map_data:
                df = pd.DataFrame(heat_map_data)
                
                # Create treemap chart
                fig = go.Figure(go.Treemap(
                    labels=df['Symbol'],
                    parents=df['Sector'],
                    values=df['Market Cap'],
                    textinfo="label+percent parent",
                    marker=dict(
                        colors=[
                            'rgb(255, 0, 0)' if x < -2 else
                            'rgb(255, 100, 100)' if x < -1 else
                            'rgb(255, 200, 200)' if x < 0 else
                            'rgb(200, 255, 200)' if x < 1 else
                            'rgb(100, 255, 100)' if x < 2 else
                            'rgb(0, 255, 0)'
                            for x in df['Change %']
                        ],
                        line=dict(width=0.5, color='black')
                    ),
                    hovertemplate='<b>%{label}</b><br>Sector: %{parent}<br>Change: %{customdata:.2f}%<extra></extra>',
                    customdata=df['Change %']
                ))
                
                fig.update_layout(
                    title="Market Heat Map by Sector and Market Cap",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add legend for colors
                st.markdown("""
                <div style="display: flex; justify-content: center; align-items: center; margin-top: -15px;">
                    <div style="display: flex; align-items: center; margin: 0 10px;">
                        <div style="width: 20px; height: 20px; background-color: rgb(255, 0, 0); margin-right: 5px;"></div>
                        <span>Below -2%</span>
                    </div>
                    <div style="display: flex; align-items: center; margin: 0 10px;">
                        <div style="width: 20px; height: 20px; background-color: rgb(255, 200, 200); margin-right: 5px;"></div>
                        <span>-2% to 0%</span>
                    </div>
                    <div style="display: flex; align-items: center; margin: 0 10px;">
                        <div style="width: 20px; height: 20px; background-color: rgb(200, 255, 200); margin-right: 5px;"></div>
                        <span>0% to 2%</span>
                    </div>
                    <div style="display: flex; align-items: center; margin: 0 10px;">
                        <div style="width: 20px; height: 20px; background-color: rgb(0, 255, 0); margin-right: 5px;"></div>
                        <span>Above 2%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.warning("Could not generate market heat map.")
        
        except Exception as e:
            st.error(f"Error generating market heat map: {str(e)}")
    
    with world_markets_tab:
        st.subheader("Global Markets")
        
        try:
            # Define global markets indices
            global_indices = {
                '^GSPC': 'S&P 500 (USA)',
                '^IXIC': 'NASDAQ (USA)',
                '^DJI': 'Dow Jones (USA)',
                '^FTSE': 'FTSE 100 (UK)',
                '^GDAXI': 'DAX (Germany)',
                '^FCHI': 'CAC 40 (France)',
                '^STOXX50E': 'Euro Stoxx 50',
                '^N225': 'Nikkei 225 (Japan)',
                '^HSI': 'Hang Seng (Hong Kong)',
                '000001.SS': 'Shanghai Composite',
                '^BSESN': 'BSE Sensex (India)',
                '^BVSP': 'Bovespa (Brazil)',
                '^MXX': 'IPC Mexico'
            }
            
            # Get data for each index
            global_data = []
            
            for symbol, name in global_indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    
                    if len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        change = current - previous
                        change_pct = (change / previous) * 100
                        
                        # Determine region
                        if 'USA' in name:
                            region = 'North America'
                        elif any(x in name for x in ['UK', 'Germany', 'France', 'Euro']):
                            region = 'Europe'
                        elif any(x in name for x in ['Japan', 'Hong Kong', 'India', 'Shanghai']):
                            region = 'Asia-Pacific'
                        elif any(x in name for x in ['Brazil', 'Mexico']):
                            region = 'Latin America'
                        else:
                            region = 'Other'
                        
                        global_data.append({
                            'Symbol': symbol,
                            'Index': name,
                            'Region': region,
                            'Price': current,
                            'Change': change,
                            'Change %': change_pct
                        })
                except:
                    continue
            
            if global_data:
                df = pd.DataFrame(global_data)
                
                # Group by region
                regions = df['Region'].unique()
                
                # Create tabs for regions
                region_tabs = st.tabs(regions)
                
                for i, region in enumerate(regions):
                    with region_tabs[i]:
                        region_df = df[df['Region'] == region].copy()
                        
                        # Format for display
                        region_df['Price'] = region_df['Price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
                        region_df['Change'] = region_df['Change'].apply(lambda x: f"{x:+.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
                        region_df['Change %'] = region_df['Change %'].apply(lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
                        
                        # Style the dataframe
                        def color_change(val):
                            if '+' in str(val):
                                return 'background-color: rgba(0, 255, 0, 0.2)'
                            elif '-' in str(val):
                                return 'background-color: rgba(255, 0, 0, 0.2)'
                            return ''
                        
                        styled_df = region_df.style.map(color_change, subset=['Change', 'Change %'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Create a bar chart for performance
                        # Convert Change % to numeric for sorting
                        region_plot = region_df.copy()
                        # Extract numeric values from formatted strings like "+1.23%"
                        region_plot['Change %'] = pd.to_numeric(region_plot['Change %'].str.replace('%', '').str.replace('+', ''), errors='coerce')
                        
                        performance_fig = px.bar(
                            region_plot.sort_values('Change %'),
                            y='Index',
                            x='Change %',
                            orientation='h',
                            color='Change %',
                            color_continuous_scale=['red', 'lightyellow', 'green'],
                            range_color=[-3, 3],
                            labels={'Change %': 'Change (%)', 'Index': 'Index'},
                            title=f"{region} Markets Performance"
                        )
                        
                        performance_fig.update_layout(
                            height=400,
                            margin=dict(l=10, r=10, t=50, b=10),
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        # Add reference line at 0
                        performance_fig.add_shape(
                            type="line",
                            x0=0, y0=-0.5,
                            x1=0, y1=len(region_df) - 0.5,
                            line=dict(color="gray", width=1, dash="dash")
                        )
                        
                        st.plotly_chart(performance_fig, use_container_width=True)
                
                # Global heat map
                st.subheader("Global Markets Heat Map")
                
                # Create treemap for global markets
                heat_fig = go.Figure(go.Treemap(
                    labels=df['Index'],
                    parents=df['Region'],
                    values=np.ones(len(df)),  # Equal size for all indices
                    textinfo="label",
                    marker=dict(
                        colors=[
                            'rgb(255, 0, 0)' if x < -2 else
                            'rgb(255, 100, 100)' if x < -1 else
                            'rgb(255, 200, 200)' if x < 0 else
                            'rgb(200, 255, 200)' if x < 1 else
                            'rgb(100, 255, 100)' if x < 2 else
                            'rgb(0, 255, 0)'
                            for x in df['Change %']
                        ],
                        line=dict(width=0.5, color='black')
                    ),
                    hovertemplate='<b>%{label}</b><br>Region: %{parent}<br>Change: %{customdata:.2f}%<extra></extra>',
                    customdata=df['Change %']
                ))
                
                heat_fig.update_layout(
                    title="Global Markets Heat Map",
                    height=500
                )
                
                st.plotly_chart(heat_fig, use_container_width=True)
            
            else:
                st.warning("Could not retrieve global markets data.")
        
        except Exception as e:
            st.error(f"Error generating global markets view: {str(e)}")