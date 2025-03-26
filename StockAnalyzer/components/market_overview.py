import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def display_market_overview():
    """
    Display an overview of the market, including major indices, sectors, and top gainers/losers
    """
    # Major market indices
    st.subheader("Major Market Indices")
    
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^RUT': 'Russell 2000',
        '^VIX': 'Volatility Index (VIX)'
    }
    
    try:
        # Create a dataframe to hold index data
        index_data = []
        
        for symbol, name in indices.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change = current - previous
                change_pct = (change / previous) * 100
                
                index_data.append({
                    'Index': name,
                    'Symbol': symbol,
                    'Price': current,
                    'Change': change,
                    'Change %': change_pct
                })
        
        if index_data:
            df = pd.DataFrame(index_data)
            
            # Format the dataframe for display
            df['Price'] = df['Price'].map('${:,.2f}'.format)
            df['Change'] = df['Change'].map('{:+,.2f}'.format)
            df['Change %'] = df['Change %'].map('{:+,.2f}%'.format)
            
            # Style the dataframe
            def color_change(val):
                if '+' in str(val):
                    return 'background-color: rgba(0, 255, 0, 0.2)'
                elif '-' in str(val):
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                return ''
            
            styled_df = df.style.applymap(color_change, subset=['Change', 'Change %'])
            
            # Display the dataframe
            st.dataframe(styled_df, use_container_width=True)
            
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
    
    except Exception as e:
        st.error(f"Error loading market index data: {str(e)}")
    
    # Market Sectors Performance
    st.subheader("Sector Performance (1 Day)")
    
    sectors = {
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLE': 'Energy',
        'XLF': 'Financials',
        'XLV': 'Health Care',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLK': 'Technology',
        'XLU': 'Utilities',
        'XLC': 'Communication Services'
    }
    
    try:
        # Create a dataframe to hold sector data
        sector_data = []
        
        for symbol, name in sectors.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change = current - previous
                change_pct = (change / previous) * 100
                
                sector_data.append({
                    'Sector': name,
                    'Change %': change_pct
                })
        
        if sector_data:
            # Create dataframe and sort by performance
            df = pd.DataFrame(sector_data)
            df = df.sort_values('Change %', ascending=False)
            
            # Create a horizontal bar chart
            fig = go.Figure()
            
            # Define colors based on performance
            colors = []
            for change in df['Change %']:
                if change >= 0:
                    colors.append('rgba(0, 204, 102, 0.8)')
                else:
                    colors.append('rgba(255, 51, 51, 0.8)')
            
            fig.add_trace(go.Bar(
                x=df['Change %'],
                y=df['Sector'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
                ),
                text=[f"{x:.2f}%" for x in df['Change %']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Sector Performance (1 Day)",
                xaxis_title="% Change",
                yaxis_title="",
                height=500,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            # Add reference line at 0
            fig.add_shape(
                type="line",
                x0=0,
                y0=-0.5,
                x1=0,
                y1=len(df) - 0.5,
                line=dict(
                    color="Gray",
                    width=1,
                    dash="dash",
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Could not retrieve sector performance data.")
    
    except Exception as e:
        st.error(f"Error loading sector performance data: {str(e)}")
    
    # Top Gainers and Losers
    st.subheader("Top Gainers and Losers (S&P 500)")
    
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
            
            # Sort for gainers and losers
            gainers = df.sort_values('Change %', ascending=False).head(5)
            losers = df.sort_values('Change %').head(5)
            
            # Format the dataframes for display
            for frame in [gainers, losers]:
                frame['Price'] = frame['Price'].map('${:,.2f}'.format)
                frame['Change'] = frame['Change'].map('{:+,.2f}'.format)
                frame['Change %'] = frame['Change %'].map('{:+,.2f}%'.format)
            
            # Create columns for display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Gainers")
                
                # Style the dataframe
                def color_green(val):
                    return 'background-color: rgba(0, 255, 0, 0.2)'
                
                styled_gainers = gainers.style.applymap(color_green, subset=['Change', 'Change %'])
                st.dataframe(styled_gainers, use_container_width=True)
            
            with col2:
                st.subheader("Top Losers")
                
                # Style the dataframe
                def color_red(val):
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                
                styled_losers = losers.style.applymap(color_red, subset=['Change', 'Change %'])
                st.dataframe(styled_losers, use_container_width=True)
        
        else:
            st.warning("Could not retrieve top gainers and losers data.")
    
    except Exception as e:
        st.error(f"Error loading top gainers and losers: {str(e)}")
    
    # Heat Map
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
