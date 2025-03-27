"""
Portfolio Balance Component for FinVision application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

def display_portfolio_balance():
    """
    Display portfolio balancing tools based on multiple dimensions: 
    fundamentals, sentiment, geopolitical factors, and sector rotation.
    """
    # Get translation function from session state
    t = st.session_state.translate if "translate" in st.session_state else lambda x: x
    
    st.title(t("Portfolio Balance"))
    
    # Check if portfolio is empty
    if not st.session_state.portfolio:
        st.warning(t("portfolio_empty"))
        return
    
    # Side tabs for different balancing aspects
    balance_tab = st.selectbox(
        t("Select Balancing Method"),
        [
            t("Comprehensive Balance"), 
            t("Fundamental-Based"), 
            t("Sentiment-Based"), 
            t("Geopolitical Factors"), 
            t("Sector Rotation")
        ]
    )
    
    # Get current portfolio data
    portfolio_data = get_portfolio_data()
    
    # Display the selected balancing method
    if balance_tab == t("Comprehensive Balance"):
        display_comprehensive_balance(portfolio_data)
    elif balance_tab == t("Fundamental-Based"):
        display_fundamental_balance(portfolio_data)
    elif balance_tab == t("Sentiment-Based"):
        display_sentiment_balance(portfolio_data)
    elif balance_tab == t("Geopolitical Factors"):
        display_geopolitical_balance(portfolio_data)
    elif balance_tab == t("Sector Rotation"):
        display_sector_rotation(portfolio_data)


def get_portfolio_data():
    """
    Get detailed data for the current portfolio including fundamentals.
    
    Returns:
        pd.DataFrame: DataFrame with portfolio stock data
    """
    portfolio_data = pd.DataFrame()
    
    for symbol, position in st.session_state.portfolio.items():
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            stock_info = stock.info
            
            # Create a row for this stock
            stock_data = {
                'symbol': symbol,
                'shares': position['shares'],
                'avg_price': position['avg_price'],
                'current_price': stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 0)),
                'company_name': stock_info.get('shortName', symbol),
                'sector': stock_info.get('sector', 'Unknown'),
                'industry': stock_info.get('industry', 'Unknown'),
                'country': stock_info.get('country', 'Unknown'),
                'market_cap': stock_info.get('marketCap', 0),
                'pe_ratio': stock_info.get('forwardPE', stock_info.get('trailingPE', 0)),
                'pb_ratio': stock_info.get('priceToBook', 0),
                'roe': stock_info.get('returnOnEquity', 0) * 100 if stock_info.get('returnOnEquity') is not None else 0,
                'dividend_yield': stock_info.get('dividendYield', 0) * 100 if stock_info.get('dividendYield') is not None else 0,
                'debt_to_equity': stock_info.get('debtToEquity', 0) / 100 if stock_info.get('debtToEquity') is not None else 0,
                'beta': stock_info.get('beta', 1),
                'revenue_growth': stock_info.get('revenueGrowth', 0) * 100 if stock_info.get('revenueGrowth') is not None else 0,
                'sentiment_score': 0  # Will be filled later
            }
            
            # Calculate position value
            stock_data['position_value'] = stock_data['current_price'] * position['shares']
            stock_data['weight'] = 0  # Will be calculated after all stocks are processed
            stock_data['return_pct'] = ((stock_data['current_price'] - position['avg_price']) / position['avg_price']) * 100
            
            # Append to the dataframe
            portfolio_data = pd.concat([portfolio_data, pd.DataFrame([stock_data])], ignore_index=True)
            
        except Exception as e:
            st.error(f"Error getting data for {symbol}: {str(e)}")
    
    # Calculate portfolio weights
    if not portfolio_data.empty:
        total_value = portfolio_data['position_value'].sum()
        portfolio_data['weight'] = (portfolio_data['position_value'] / total_value) * 100
    
    return portfolio_data


def calculate_fundamental_score(portfolio_df):
    """
    Calculate a fundamental score for each stock based on key metrics.
    
    Args:
        portfolio_df (pd.DataFrame): Portfolio data
        
    Returns:
        pd.DataFrame: Updated DataFrame with fundamental scores
    """
    # Create a copy to avoid modifying the original
    df = portfolio_df.copy()
    
    # Define weights for each metric
    pe_weight = 0.25
    roe_weight = 0.2
    debt_equity_weight = 0.15
    revenue_growth_weight = 0.2
    dividend_yield_weight = 0.1
    pb_weight = 0.1
    
    # Min-max scaling for PE ratio (lower is better)
    if 'pe_ratio' in df.columns and not df['pe_ratio'].isnull().all() and not (df['pe_ratio'] <= 0).all():
        valid_pe = df[df['pe_ratio'] > 0]['pe_ratio']
        if not valid_pe.empty:
            max_pe = valid_pe.max()
            min_pe = valid_pe.min()
            df['pe_score'] = df['pe_ratio'].apply(lambda x: 1 - ((x - min_pe) / (max_pe - min_pe)) if x > 0 else 0.5)
        else:
            df['pe_score'] = 0.5
    else:
        df['pe_score'] = 0.5  # Neutral score if no valid data
    
    # ROE (higher is better)
    if 'roe' in df.columns and not df['roe'].isnull().all():
        max_roe = df['roe'].max()
        min_roe = df['roe'].min()
        if max_roe > min_roe:
            df['roe_score'] = df['roe'].apply(lambda x: (x - min_roe) / (max_roe - min_roe))
        else:
            df['roe_score'] = 0.5
    else:
        df['roe_score'] = 0.5
    
    # Debt to equity (lower is better)
    if 'debt_to_equity' in df.columns and not df['debt_to_equity'].isnull().all():
        max_de = df['debt_to_equity'].max()
        min_de = df['debt_to_equity'].min()
        if max_de > min_de:
            df['debt_score'] = df['debt_to_equity'].apply(lambda x: 1 - ((x - min_de) / (max_de - min_de)))
        else:
            df['debt_score'] = 0.5
    else:
        df['debt_score'] = 0.5
    
    # Revenue growth (higher is better)
    if 'revenue_growth' in df.columns and not df['revenue_growth'].isnull().all():
        max_growth = df['revenue_growth'].max()
        min_growth = df['revenue_growth'].min()
        if max_growth > min_growth:
            df['growth_score'] = df['revenue_growth'].apply(lambda x: (x - min_growth) / (max_growth - min_growth))
        else:
            df['growth_score'] = 0.5
    else:
        df['growth_score'] = 0.5
    
    # Dividend yield (higher is better)
    if 'dividend_yield' in df.columns and not df['dividend_yield'].isnull().all():
        max_yield = df['dividend_yield'].max()
        min_yield = df['dividend_yield'].min()
        if max_yield > min_yield:
            df['yield_score'] = df['dividend_yield'].apply(lambda x: (x - min_yield) / (max_yield - min_yield))
        else:
            df['yield_score'] = 0.5
    else:
        df['yield_score'] = 0.5
    
    # P/B ratio (lower is better)
    if 'pb_ratio' in df.columns and not df['pb_ratio'].isnull().all() and not (df['pb_ratio'] <= 0).all():
        valid_pb = df[df['pb_ratio'] > 0]['pb_ratio']
        if not valid_pb.empty:
            max_pb = valid_pb.max()
            min_pb = valid_pb.min()
            df['pb_score'] = df['pb_ratio'].apply(lambda x: 1 - ((x - min_pb) / (max_pb - min_pb)) if x > 0 else 0.5)
        else:
            df['pb_score'] = 0.5
    else:
        df['pb_score'] = 0.5
    
    # Calculate total fundamental score
    df['fundamental_score'] = (
        pe_weight * df['pe_score'] +
        roe_weight * df['roe_score'] +
        debt_equity_weight * df['debt_score'] +
        revenue_growth_weight * df['growth_score'] +
        dividend_yield_weight * df['yield_score'] +
        pb_weight * df['pb_score']
    )
    
    # Normalize scores to sum up to 100%
    total_score = df['fundamental_score'].sum()
    if total_score > 0:
        df['fundamental_weight'] = (df['fundamental_score'] / total_score) * 100
    else:
        df['fundamental_weight'] = 100 / len(df)  # Equal weight if no valid scores
    
    return df


def display_fundamental_balance(portfolio_df):
    """
    Display fundamental-based portfolio balancing analysis.
    
    Args:
        portfolio_df (pd.DataFrame): Portfolio data
    """
    t = st.session_state.translate if "translate" in st.session_state else lambda x: x
    
    st.header(t("Fundamental-Based Portfolio Balance"))
    
    # Calculate fundamental scores
    df_with_scores = calculate_fundamental_score(portfolio_df)
    
    # Display current vs recommended weights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t("Current Allocation"))
        current_pie = px.pie(
            df_with_scores, 
            values='weight', 
            names='symbol',
            title=t("Current Portfolio Weights"),
            hover_data=['company_name', 'sector', 'pe_ratio', 'roe'],
            labels={'weight': 'Weight (%)', 'symbol': 'Symbol', 'company_name': 'Company', 
                   'sector': 'Sector', 'pe_ratio': 'P/E', 'roe': 'ROE (%)'}
        )
        current_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(current_pie, use_container_width=True)
    
    with col2:
        st.subheader(t("Recommended Allocation"))
        recommended_pie = px.pie(
            df_with_scores, 
            values='fundamental_weight', 
            names='symbol',
            title=t("Recommended Weights Based on Fundamentals"),
            hover_data=['company_name', 'sector', 'pe_ratio', 'roe'],
            labels={'fundamental_weight': 'Weight (%)', 'symbol': 'Symbol', 'company_name': 'Company', 
                   'sector': 'Sector', 'pe_ratio': 'P/E', 'roe': 'ROE (%)'}
        )
        recommended_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(recommended_pie, use_container_width=True)
    
    # Display detailed metrics
    st.subheader(t("Fundamental Analysis"))
    
    # Format the metrics table
    metrics_df = df_with_scores[['symbol', 'company_name', 'weight', 'fundamental_weight', 
                                'pe_ratio', 'pb_ratio', 'roe', 'dividend_yield', 
                                'debt_to_equity', 'revenue_growth']].copy()
    
    # Format columns
    metrics_df['weight'] = metrics_df['weight'].round(2).astype(str) + '%'
    metrics_df['fundamental_weight'] = metrics_df['fundamental_weight'].round(2).astype(str) + '%'
    metrics_df['pe_ratio'] = metrics_df['pe_ratio'].round(2)
    metrics_df['pb_ratio'] = metrics_df['pb_ratio'].round(2)
    metrics_df['roe'] = metrics_df['roe'].round(2).astype(str) + '%'
    metrics_df['dividend_yield'] = metrics_df['dividend_yield'].round(2).astype(str) + '%'
    metrics_df['debt_to_equity'] = metrics_df['debt_to_equity'].round(2)
    metrics_df['revenue_growth'] = metrics_df['revenue_growth'].round(2).astype(str) + '%'
    
    # Rename columns for display
    metrics_df.columns = [
        t('Symbol'), 
        t('Company'),
        t('Current Weight'),
        t('Recommended Weight'),
        t('P/E Ratio'),
        t('P/B Ratio'),
        t('ROE'),
        t('Dividend Yield'),
        t('Debt/Equity'),
        t('Revenue Growth')
    ]
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Display rebalancing actions
    st.subheader(t("Rebalancing Actions"))
    
    # Calculate the difference between current and recommended weights
    rebalance_df = df_with_scores[['symbol', 'company_name', 'weight', 'fundamental_weight', 'position_value', 'shares']].copy()
    rebalance_df['weight_difference'] = rebalance_df['fundamental_weight'] - rebalance_df['weight']
    rebalance_df['action'] = rebalance_df['weight_difference'].apply(
        lambda x: t('Buy') if x > 1 else (t('Sell') if x < -1 else t('Hold'))
    )
    
    # Calculate the amount to buy/sell for each stock
    total_portfolio_value = rebalance_df['position_value'].sum()
    rebalance_df['target_value'] = (rebalance_df['fundamental_weight'] / 100) * total_portfolio_value
    rebalance_df['value_change'] = rebalance_df['target_value'] - rebalance_df['position_value']
    # Get current prices from the original dataframe to calculate shares change
    rebalance_df = pd.merge(
        rebalance_df,
        df_with_scores[['symbol', 'current_price']],
        on='symbol',
        how='left'
    )
    
    # Calculate shares change based on value change and current price
    rebalance_df['shares_change'] = rebalance_df.apply(
        lambda row: row['value_change'] / row['current_price'] if pd.notna(row['current_price']) and row['current_price'] > 0 else 0, 
        axis=1
    )
    
    # Format for display
    actions_df = rebalance_df[['symbol', 'company_name', 'action', 'weight_difference', 'value_change', 'shares_change']].copy()
    actions_df['weight_difference'] = actions_df['weight_difference'].round(2).astype(str) + '%'
    actions_df['value_change'] = actions_df['value_change'].round(2).apply(lambda x: f"${x:,.2f}")
    actions_df['shares_change'] = actions_df['shares_change'].round(2)
    
    # Rename columns for display
    actions_df.columns = [
        t('Symbol'), 
        t('Company'),
        t('Action'),
        t('Weight Change'),
        t('Value Change'),
        t('Shares to Buy/Sell')
    ]
    
    st.dataframe(actions_df, use_container_width=True)
    
    st.markdown("""
    ### Understanding Fundamental Balance
    
    The fundamental balance approach weights stocks based on their financial strength. 
    Stocks with better fundamental metrics receive higher allocations. This aims to enhance 
    long-term performance while reducing risk.
    
    **Key considerations:**
    - Low P/E ratios suggest stocks may be undervalued
    - High ROE indicates efficient use of equity capital
    - Low debt-to-equity ratio suggests financial stability
    - Strong revenue growth shows business momentum
    """)


def get_sentiment_data(portfolio_df):
    """
    Get sentiment data for stocks in the portfolio.
    
    Args:
        portfolio_df (pd.DataFrame): Portfolio data
        
    Returns:
        pd.DataFrame: Updated DataFrame with sentiment scores
    """
    # Create a copy to avoid modifying the original
    df = portfolio_df.copy()
    
    # Get the news sentiment from the news component if available
    from components.news import get_news_from_api, analyze_sentiment
    
    # Add sentiment scores
    for idx, row in df.iterrows():
        symbol = row['symbol']
        
        try:
            # Try to get news from API
            news = get_news_from_api(symbol, max_news=5)
            
            # Calculate average sentiment
            if news and len(news) > 0:
                sentiments = []
                for article in news:
                    if 'description' in article and article['description']:
                        sentiment_score, _, _ = analyze_sentiment(article['description'])
                        sentiments.append(sentiment_score)
                
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    df.at[idx, 'sentiment_score'] = avg_sentiment
            
        except Exception as e:
            st.warning(f"Could not retrieve sentiment data for {symbol}: {str(e)}")
    
    # Normalize sentiment scores to calculate weights
    if 'sentiment_score' in df.columns and not df['sentiment_score'].isnull().all():
        min_sentiment = df['sentiment_score'].min()
        max_sentiment = df['sentiment_score'].max()
        
        if max_sentiment > min_sentiment:
            df['sentiment_normalized'] = df['sentiment_score'].apply(
                lambda x: (x - min_sentiment) / (max_sentiment - min_sentiment) if pd.notnull(x) else 0.5
            )
        else:
            df['sentiment_normalized'] = 0.5
        
        # Calculate sentiment-based weights
        total_sentiment = df['sentiment_normalized'].sum()
        if total_sentiment > 0:
            df['sentiment_weight'] = (df['sentiment_normalized'] / total_sentiment) * 100
        else:
            df['sentiment_weight'] = 100 / len(df)  # Equal weight if no valid scores
    else:
        df['sentiment_normalized'] = 0.5
        df['sentiment_weight'] = 100 / len(df)  # Equal weight if no sentiment data
    
    return df


def display_sentiment_balance(portfolio_df):
    """
    Display sentiment-based portfolio balancing analysis.
    
    Args:
        portfolio_df (pd.DataFrame): Portfolio data
    """
    t = st.session_state.translate if "translate" in st.session_state else lambda x: x
    
    st.header(t("Sentiment-Based Portfolio Balance"))
    
    with st.spinner(t("Analyzing market sentiment...")):
        # Get sentiment data
        df_with_sentiment = get_sentiment_data(portfolio_df)
    
    # Display sentiment scores
    st.subheader(t("Market Sentiment Analysis"))
    
    # Format sentiment scores
    sentiment_df = df_with_sentiment[['symbol', 'company_name', 'sentiment_score']].copy()
    sentiment_df['sentiment_label'] = sentiment_df['sentiment_score'].apply(
        lambda x: t('Positive') if x > 0.05 else (t('Negative') if x < -0.05 else t('Neutral'))
    )
    sentiment_df['sentiment_color'] = sentiment_df['sentiment_score'].apply(
        lambda x: 'green' if x > 0.05 else ('red' if x < -0.05 else 'grey')
    )
    
    # Create sentiment indicators
    for idx, row in sentiment_df.iterrows():
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.write(f"**{row['symbol']}**")
        
        with col2:
            # Create a progress bar-like indicator for sentiment
            score = row['sentiment_score']
            normalized_score = (score + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            # Create the sentiment bar with appropriate color
            st.markdown(
                f"""
                <div style="width:100%; background-color:#f0f0f0; height:20px; border-radius:10px; overflow:hidden;">
                    <div style="width:{normalized_score*100}%; background-color:{row['sentiment_color']}; height:20px;"></div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            st.write(f"**{row['sentiment_label']}** ({score:.2f})")
    
    # Display allocation charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t("Current Allocation"))
        current_pie = px.pie(
            df_with_sentiment, 
            values='weight', 
            names='symbol',
            title=t("Current Portfolio Weights"),
            hover_data=['company_name', 'sentiment_score'],
            labels={'weight': 'Weight (%)', 'symbol': 'Symbol', 'company_name': 'Company', 
                   'sentiment_score': 'Sentiment Score'}
        )
        current_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(current_pie, use_container_width=True)
    
    with col2:
        st.subheader(t("Sentiment-Based Allocation"))
        sentiment_pie = px.pie(
            df_with_sentiment, 
            values='sentiment_weight', 
            names='symbol',
            title=t("Recommended Weights Based on Sentiment"),
            hover_data=['company_name', 'sentiment_score'],
            labels={'sentiment_weight': 'Weight (%)', 'symbol': 'Symbol', 'company_name': 'Company', 
                   'sentiment_score': 'Sentiment Score'}
        )
        sentiment_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(sentiment_pie, use_container_width=True)
    
    st.markdown("""
    ### Understanding Sentiment Balance
    
    The sentiment balance approach weights stocks based on market sentiment and news analysis. 
    Stocks with more positive sentiment receive higher allocations, while those with negative 
    sentiment are reduced.
    
    **Key considerations:**
    - Higher weightings for stocks with positive news coverage
    - Reduced exposure to stocks with negative sentiment
    - Sentiment can change rapidly, requiring more frequent rebalancing
    - This approach is more responsive to market dynamics
    """)


def get_geopolitical_risk(country):
    """
    Return a geopolitical risk score for a given country.
    
    Args:
        country (str): Country name
        
    Returns:
        float: Risk score between 0 (low risk) and 1 (high risk)
    """
    # Simulated geopolitical risk scores
    geopolitical_risk_db = {
        "United States": 0.2,
        "USA": 0.2,
        "US": 0.2,
        "China": 0.7,
        "Russia": 0.8,
        "United Kingdom": 0.3,
        "UK": 0.3,
        "Germany": 0.3,
        "France": 0.3,
        "Italy": 0.4,
        "Spain": 0.4,
        "Japan": 0.3,
        "South Korea": 0.5,
        "Taiwan": 0.7,
        "India": 0.5,
        "Brazil": 0.5,
        "Mexico": 0.6,
        "Canada": 0.2,
        "Australia": 0.2,
        "Netherlands": 0.3,
        "Switzerland": 0.2,
        "Israel": 0.7,
        "Saudi Arabia": 0.6,
        "United Arab Emirates": 0.5,
        "South Africa": 0.6,
        "Turkey": 0.7
    }
    
    # Default risk score if country is unknown
    return geopolitical_risk_db.get(country, 0.5)


def predict_economic_phase():
    """
    Predict the current economic phase based on market indicators.
    
    Returns:
        str: Economic phase ('expansion', 'recession', 'recovery', or 'stagflation')
    """
    # Get SPY (S&P 500) data for overall market trend
    try:
        spy = yf.Ticker("SPY")
        spy_data = spy.history(period="6mo")
        
        # Calculate 50-day vs 200-day moving averages
        spy_data['MA50'] = spy_data['Close'].rolling(window=50).mean()
        spy_data['MA200'] = spy_data['Close'].rolling(window=200).mean()
        
        # Check if 50-day MA is above 200-day MA (golden cross - bullish)
        ma_bullish = spy_data['MA50'].iloc[-1] > spy_data['MA200'].iloc[-1]
        
        # Check if price is in an uptrend (last price > 50 days ago)
        price_uptrend = spy_data['Close'].iloc[-1] > spy_data['Close'].iloc[-50]
        
        # Get inflation data from TIP vs IEF ratio
        tip = yf.Ticker("TIP")
        ief = yf.Ticker("IEF")
        
        tip_data = tip.history(period="1mo")
        ief_data = ief.history(period="1mo")
        
        # Calculate TIP/IEF ratio (inflation expectations)
        tip_price = tip_data['Close'].iloc[-1]
        ief_price = ief_data['Close'].iloc[-1]
        inflation_ratio = tip_price / ief_price
        
        # Get previous month ratio
        tip_price_prev = tip_data['Close'].iloc[0]
        ief_price_prev = ief_data['Close'].iloc[0]
        inflation_ratio_prev = tip_price_prev / ief_price_prev
        
        # Check if inflation expectations are rising
        inflation_rising = inflation_ratio > inflation_ratio_prev
        
        # Determine economic phase
        if price_uptrend and ma_bullish:
            if inflation_rising:
                return "expansion"  # Growth with inflation
            else:
                return "recovery"   # Growth with moderate inflation
        else:
            if inflation_rising:
                return "stagflation"  # Slow growth with inflation
            else:
                return "recession"    # Slow growth with low inflation
                
    except Exception as e:
        # Default to expansion if we can't determine
        return "expansion"


def display_geopolitical_balance(portfolio_df):
    """
    Display geopolitical-based portfolio balancing analysis.
    
    Args:
        portfolio_df (pd.DataFrame): Portfolio data
    """
    t = st.session_state.translate if "translate" in st.session_state else lambda x: x
    
    st.header(t("Geopolitical Risk Analysis"))
    
    # Add geopolitical risk scores to the dataframe
    df = portfolio_df.copy()
    df['geopolitical_risk'] = df['country'].apply(get_geopolitical_risk)
    
    # Calculate geopolitical-adjusted weights
    df['geo_safe_score'] = 1 - df['geopolitical_risk']  # Invert so higher is better
    total_score = df['geo_safe_score'].sum()
    df['geo_weight'] = (df['geo_safe_score'] / total_score) * 100 if total_score > 0 else 100 / len(df)
    
    # Create a world map of geopolitical risk
    countries_df = df[['country', 'symbol', 'geopolitical_risk', 'weight']].copy()
    countries_df = countries_df.dropna(subset=['country'])
    
    # Create a choropleth map
    try:
        fig = px.choropleth(
            countries_df,
            locations='country',
            locationmode='country names',
            color='geopolitical_risk',
            hover_name='country',
            hover_data=['symbol', 'weight'],
            color_continuous_scale='RdYlGn_r',  # Red for high risk, green for low risk
            labels={'geopolitical_risk': 'Risk Level', 'weight': 'Portfolio Weight (%)'},
            title=t('Geopolitical Risk Exposure')
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(t('Could not create map visualization. Make sure country data is available.'))
        
        # Create a table instead
        country_risk_df = df[['symbol', 'company_name', 'country', 'geopolitical_risk', 'weight']].copy()
        country_risk_df['geopolitical_risk'] = country_risk_df['geopolitical_risk'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A"
        )
        country_risk_df['weight'] = country_risk_df['weight'].apply(lambda x: f"{x:.1f}%")
        
        country_risk_df.columns = [
            t('Symbol'), t('Company'), t('Country'), 
            t('Geopolitical Risk'), t('Current Weight')
        ]
        
        st.dataframe(country_risk_df, use_container_width=True)
    
    # Display allocation comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t("Current Allocation"))
        current_pie = px.pie(
            df, 
            values='weight', 
            names='symbol',
            title=t("Current Portfolio Weights"),
            hover_data=['company_name', 'country', 'geopolitical_risk'],
            labels={'weight': 'Weight (%)', 'symbol': 'Symbol', 'company_name': 'Company', 
                   'country': 'Country', 'geopolitical_risk': 'Risk Level'}
        )
        current_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(current_pie, use_container_width=True)
    
    with col2:
        st.subheader(t("Geopolitical Risk-Adjusted Allocation"))
        geo_pie = px.pie(
            df, 
            values='geo_weight', 
            names='symbol',
            title=t("Recommended Weights Based on Geopolitical Risk"),
            hover_data=['company_name', 'country', 'geopolitical_risk'],
            labels={'geo_weight': 'Weight (%)', 'symbol': 'Symbol', 'company_name': 'Company', 
                   'country': 'Country', 'geopolitical_risk': 'Risk Level'}
        )
        geo_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(geo_pie, use_container_width=True)
    
    st.markdown("""
    ### Understanding Geopolitical Balance
    
    The geopolitical balance approach adjusts portfolio weights based on the geopolitical risk 
    of each company's primary market. Companies with lower geopolitical risk receive higher 
    allocations to enhance portfolio stability during global uncertainty.
    
    **Key considerations:**
    - Lower allocation to companies with high geopolitical risk exposure
    - Higher allocation to companies in stable regions
    - Protection against sanctions, trade wars, and political instability
    - Reduced vulnerability to supply chain disruptions
    """)


def display_sector_rotation(portfolio_df):
    """
    Display sector rotation-based portfolio balancing analysis.
    
    Args:
        portfolio_df (pd.DataFrame): Portfolio data
    """
    t = st.session_state.translate if "translate" in st.session_state else lambda x: x
    
    st.header(t("Sector Rotation Strategy"))
    
    # Determine current economic phase
    economic_phase = predict_economic_phase()
    
    # Define sector preferences for different economic phases
    sector_rotation_model = {
        "expansion": {
            "preferred": ["Technology", "Consumer Discretionary", "Industrials", "Financials"],
            "neutral": ["Communication Services", "Materials", "Real Estate"],
            "reduced": ["Consumer Staples", "Utilities", "Health Care", "Energy"]
        },
        "recession": {
            "preferred": ["Consumer Staples", "Utilities", "Health Care"],
            "neutral": ["Communication Services", "Energy"],
            "reduced": ["Technology", "Consumer Discretionary", "Industrials", "Financials", "Materials", "Real Estate"]
        },
        "recovery": {
            "preferred": ["Financials", "Materials", "Industrials", "Real Estate"],
            "neutral": ["Technology", "Energy", "Consumer Discretionary"],
            "reduced": ["Utilities", "Consumer Staples", "Health Care", "Communication Services"]
        },
        "stagflation": {
            "preferred": ["Energy", "Materials", "Utilities", "Consumer Staples"],
            "neutral": ["Health Care", "Real Estate"],
            "reduced": ["Technology", "Consumer Discretionary", "Financials", "Industrials", "Communication Services"]
        }
    }
    
    # Display current economic phase
    phase_labels = {
        "expansion": t("Expansion (Growth with Inflation)"),
        "recession": t("Recession (Economic Contraction)"),
        "recovery": t("Recovery (Early Economic Cycle)"),
        "stagflation": t("Stagflation (Slow Growth with Inflation)")
    }
    
    st.subheader(t("Current Economic Phase: ") + phase_labels.get(economic_phase, economic_phase))
    
    # Display phase description
    phase_descriptions = {
        "expansion": """
            **Characteristics:** Strong economic growth, rising interest rates, moderate to high inflation.
            **Preferred Sectors:** Technology, Consumer Discretionary, Industrials, Financials.
            **Sectors to Reduce:** Consumer Staples, Utilities, Health Care.
        """,
        "recession": """
            **Characteristics:** Economic contraction, falling interest rates, low inflation.
            **Preferred Sectors:** Consumer Staples, Utilities, Health Care.
            **Sectors to Reduce:** Technology, Consumer Discretionary, Industrials, Financials.
        """,
        "recovery": """
            **Characteristics:** Early cycle growth, low but rising interest rates, low inflation.
            **Preferred Sectors:** Financials, Materials, Industrials, Real Estate.
            **Sectors to Reduce:** Utilities, Consumer Staples.
        """,
        "stagflation": """
            **Characteristics:** Slow economic growth, high inflation, rising interest rates.
            **Preferred Sectors:** Energy, Materials, Utilities, Consumer Staples.
            **Sectors to Reduce:** Technology, Consumer Discretionary, Financials.
        """
    }
    
    st.markdown(phase_descriptions.get(economic_phase, ""))
    
    # Calculate sector-based weights
    df = portfolio_df.copy()
    
    # Determine sector category based on current economic phase
    def get_sector_category(sector, model):
        if sector in model['preferred']:
            return "preferred"
        elif sector in model['neutral']:
            return "neutral"
        else:
            return "reduced"
    
    # Assign sector multipliers based on category
    sector_multipliers = {
        "preferred": 1.5,  # Increase weight by 50%
        "neutral": 1.0,    # Keep weight the same
        "reduced": 0.5     # Reduce weight by 50%
    }
    
    # Apply sector rotation model
    current_model = sector_rotation_model[economic_phase]
    df['sector_category'] = df['sector'].apply(lambda x: get_sector_category(x, current_model))
    df['sector_multiplier'] = df['sector_category'].map(sector_multipliers)
    
    # Calculate sector-adjusted weights
    df['sector_weight_raw'] = df['weight'] * df['sector_multiplier']
    total_adjusted_weight = df['sector_weight_raw'].sum()
    df['sector_weight'] = (df['sector_weight_raw'] / total_adjusted_weight) * 100
    
    # Create sector analysis visualization
    sector_summary = df.groupby('sector').agg({
        'weight': 'sum',
        'sector_weight': 'sum',
        'sector_category': 'first'
    }).reset_index()
    
    # Make a color map for the categories
    color_map = {
        "preferred": "#4CAF50",  # Green for preferred
        "neutral": "#FFC107",    # Yellow for neutral
        "reduced": "#F44336"     # Red for reduced
    }
    
    sector_summary['color'] = sector_summary['sector_category'].map(color_map)
    
    # Sort by current weight
    sector_summary = sector_summary.sort_values('weight', ascending=False)
    
    # Create the sector comparison chart
    fig = go.Figure()
    
    # Add current weights
    fig.add_trace(go.Bar(
        x=sector_summary['sector'],
        y=sector_summary['weight'],
        name=t('Current Weight'),
        marker_color='rgba(55, 83, 109, 0.7)',
        offsetgroup=0
    ))
    
    # Add recommended sector weights
    fig.add_trace(go.Bar(
        x=sector_summary['sector'],
        y=sector_summary['sector_weight'],
        name=t('Recommended Weight'),
        marker_color=sector_summary['color'],
        offsetgroup=1
    ))
    
    # Customize layout
    fig.update_layout(
        title=t('Sector Allocation: Current vs Recommended'),
        xaxis_title=t('Sector'),
        yaxis_title=t('Weight (%)'),
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display allocation comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t("Current Allocation"))
        current_pie = px.pie(
            df, 
            values='weight', 
            names='symbol',
            title=t("Current Portfolio Weights"),
            hover_data=['company_name', 'sector', 'sector_category'],
            labels={'weight': 'Weight (%)', 'symbol': 'Symbol', 'company_name': 'Company', 
                   'sector': 'Sector', 'sector_category': 'Category'}
        )
        current_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(current_pie, use_container_width=True)
    
    with col2:
        st.subheader(t("Sector Rotation-Based Allocation"))
        sector_pie = px.pie(
            df, 
            values='sector_weight', 
            names='symbol',
            title=t("Recommended Weights Based on Sector Rotation"),
            hover_data=['company_name', 'sector', 'sector_category'],
            labels={'sector_weight': 'Weight (%)', 'symbol': 'Symbol', 'company_name': 'Company', 
                   'sector': 'Sector', 'sector_category': 'Category'}
        )
        sector_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(sector_pie, use_container_width=True)
    
    st.markdown("""
    ### Understanding Sector Rotation
    
    The sector rotation approach adjusts portfolio weights based on the current economic phase.
    Different sectors tend to outperform at different points in the economic cycle.
    
    **Key considerations:**
    - Overweight sectors that historically perform well in the current economic phase
    - Underweight sectors that typically underperform in the current environment
    - Regular rebalancing as economic conditions evolve
    - Combines both tactical and strategic allocation methods
    """)


def display_comprehensive_balance(portfolio_df):
    """
    Display comprehensive portfolio balancing analysis that integrates all approaches.
    
    Args:
        portfolio_df (pd.DataFrame): Portfolio data
    """
    t = st.session_state.translate if "translate" in st.session_state else lambda x: x
    
    st.header(t("Comprehensive Portfolio Balance"))
    
    st.markdown("""
    This comprehensive analysis integrates multiple dimensions to provide a balanced portfolio recommendation:
    
    1. **Fundamental Analysis** - Financial metrics including P/E, ROE, debt levels
    2. **Sentiment Analysis** - News sentiment and market perception
    3. **Geopolitical Risk** - Country and region-specific risk factors
    4. **Sector Rotation** - Economic cycle positioning
    """)
    
    with st.spinner(t("Calculating comprehensive balance...")):
        # Calculate scores from each dimension
        
        # 1. Fundamental scores
        fundamental_df = calculate_fundamental_score(portfolio_df)
        
        # 2. Sentiment scores
        sentiment_df = get_sentiment_data(portfolio_df)
        
        # 3. Geopolitical scores
        geo_df = portfolio_df.copy()
        geo_df['geopolitical_risk'] = geo_df['country'].apply(get_geopolitical_risk)
        geo_df['geo_safe_score'] = 1 - geo_df['geopolitical_risk']  # Invert so higher is better
        
        # 4. Sector rotation scores
        economic_phase = predict_economic_phase()
        sector_rotation_model = {
            "expansion": {
                "preferred": ["Technology", "Consumer Discretionary", "Industrials", "Financials"],
                "neutral": ["Communication Services", "Materials", "Real Estate"],
                "reduced": ["Consumer Staples", "Utilities", "Health Care", "Energy"]
            },
            "recession": {
                "preferred": ["Consumer Staples", "Utilities", "Health Care"],
                "neutral": ["Communication Services", "Energy"],
                "reduced": ["Technology", "Consumer Discretionary", "Industrials", "Financials", "Materials", "Real Estate"]
            },
            "recovery": {
                "preferred": ["Financials", "Materials", "Industrials", "Real Estate"],
                "neutral": ["Technology", "Energy", "Consumer Discretionary"],
                "reduced": ["Utilities", "Consumer Staples", "Health Care", "Communication Services"]
            },
            "stagflation": {
                "preferred": ["Energy", "Materials", "Utilities", "Consumer Staples"],
                "neutral": ["Health Care", "Real Estate"],
                "reduced": ["Technology", "Consumer Discretionary", "Financials", "Industrials", "Communication Services"]
            }
        }
        
        # Sector rotation multiplier
        def get_sector_multiplier(sector, model):
            current_model = model[economic_phase]
            if sector in current_model['preferred']:
                return 1.5
            elif sector in current_model['neutral']:
                return 1.0
            else:
                return 0.5
        
        sector_df = portfolio_df.copy()
        sector_df['sector_multiplier'] = sector_df['sector'].apply(
            lambda x: get_sector_multiplier(x, sector_rotation_model)
        )
        
        # Combine all scores with configurable weights
        combined_df = portfolio_df.copy()
        
        # Allow user to adjust dimension weights
        st.subheader(t("Balancing Dimension Weights"))
        st.write(t("Adjust the importance of each dimension in the final recommendation:"))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fundamental_weight = st.slider(t("Fundamental Weight"), 0.0, 1.0, 0.4, 0.1)
        with col2:
            sentiment_weight = st.slider(t("Sentiment Weight"), 0.0, 1.0, 0.2, 0.1)
        with col3:
            geo_weight = st.slider(t("Geopolitical Weight"), 0.0, 1.0, 0.2, 0.1)
        with col4:
            sector_weight = st.slider(t("Sector Weight"), 0.0, 1.0, 0.2, 0.1)
            
        # Normalize weights to sum to 1
        total_weight = fundamental_weight + sentiment_weight + geo_weight + sector_weight
        if total_weight > 0:
            fundamental_weight /= total_weight
            sentiment_weight /= total_weight
            geo_weight /= total_weight
            sector_weight /= total_weight
        
        # Calculate weighted scores
        for idx, row in combined_df.iterrows():
            symbol = row['symbol']
            
            # Get scores from each dimension
            f_score = fundamental_df.loc[fundamental_df['symbol'] == symbol, 'fundamental_score'].values[0] if not fundamental_df.empty else 0.5
            s_score = sentiment_df.loc[sentiment_df['symbol'] == symbol, 'sentiment_normalized'].values[0] if not sentiment_df.empty else 0.5
            g_score = geo_df.loc[geo_df['symbol'] == symbol, 'geo_safe_score'].values[0] if not geo_df.empty else 0.5
            r_score = sector_df.loc[sector_df['symbol'] == symbol, 'sector_multiplier'].values[0] if not sector_df.empty else 1.0
            
            # Calculate weighted score
            combined_score = (
                fundamental_weight * f_score +
                sentiment_weight * s_score +
                geo_weight * g_score +
                sector_weight * r_score
            )
            
            combined_df.at[idx, 'combined_score'] = combined_score
        
        # Calculate recommended weights based on combined score
        total_score = combined_df['combined_score'].sum()
        if total_score > 0:
            combined_df['recommended_weight'] = (combined_df['combined_score'] / total_score) * 100
        else:
            combined_df['recommended_weight'] = 100 / len(combined_df)
    
    # Display current vs recommended allocation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t("Current Allocation"))
        current_pie = px.pie(
            combined_df, 
            values='weight', 
            names='symbol',
            title=t("Current Portfolio Weights"),
            hover_data=['company_name', 'sector'],
            labels={'weight': 'Weight (%)', 'symbol': 'Symbol', 'company_name': 'Company', 'sector': 'Sector'}
        )
        current_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(current_pie, use_container_width=True)
    
    with col2:
        st.subheader(t("Recommended Allocation"))
        recommended_pie = px.pie(
            combined_df, 
            values='recommended_weight', 
            names='symbol',
            title=t("Recommended Portfolio Weights"),
            hover_data=['company_name', 'sector'],
            labels={'recommended_weight': 'Weight (%)', 'symbol': 'Symbol', 'company_name': 'Company', 'sector': 'Sector'}
        )
        recommended_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(recommended_pie, use_container_width=True)
    
    # Display rebalancing actions
    st.subheader(t("Rebalancing Actions"))
    
    # Calculate the difference between current and recommended weights
    rebalance_df = combined_df[['symbol', 'company_name', 'weight', 'recommended_weight', 'position_value', 'shares', 'current_price']].copy()
    rebalance_df['weight_difference'] = rebalance_df['recommended_weight'] - rebalance_df['weight']
    rebalance_df['action'] = rebalance_df['weight_difference'].apply(
        lambda x: t('Buy') if x > 1 else (t('Sell') if x < -1 else t('Hold'))
    )
    
    # Calculate the amount to buy/sell for each stock
    total_portfolio_value = rebalance_df['position_value'].sum()
    rebalance_df['target_value'] = (rebalance_df['recommended_weight'] / 100) * total_portfolio_value
    rebalance_df['value_change'] = rebalance_df['target_value'] - rebalance_df['position_value']
    rebalance_df['shares_change'] = rebalance_df.apply(
        lambda row: row['value_change'] / row['current_price'], axis=1
    )
    
    # Format for display
    actions_df = rebalance_df[['symbol', 'company_name', 'action', 'weight_difference', 'value_change', 'shares_change']].copy()
    actions_df['weight_difference'] = actions_df['weight_difference'].round(2).astype(str) + '%'
    actions_df['value_change'] = actions_df['value_change'].round(2).apply(lambda x: f"${x:,.2f}")
    actions_df['shares_change'] = actions_df['shares_change'].round(2)
    
    # Rename columns for display
    actions_df.columns = [
        t('Symbol'), 
        t('Company'),
        t('Action'),
        t('Weight Change'),
        t('Value Change'),
        t('Shares to Buy/Sell')
    ]
    
    st.dataframe(actions_df, use_container_width=True)
    
    # Display dimension impact
    st.subheader(t("Dimension Impact Analysis"))
    
    # Prepare data for radar chart
    dimensions = [t('Fundamentals'), t('Sentiment'), t('Geopolitical'), t('Sector')]
    
    fig = go.Figure()
    
    for idx, row in combined_df.iterrows():
        symbol = row['symbol']
        
        # Get scores from each dimension (normalized to 0-1)
        f_score = fundamental_df.loc[fundamental_df['symbol'] == symbol, 'fundamental_score'].values[0] if not fundamental_df.empty else 0.5
        s_score = sentiment_df.loc[sentiment_df['symbol'] == symbol, 'sentiment_normalized'].values[0] if not sentiment_df.empty else 0.5
        g_score = geo_df.loc[geo_df['symbol'] == symbol, 'geo_safe_score'].values[0] if not geo_df.empty else 0.5
        r_score = sector_df.loc[sector_df['symbol'] == symbol, 'sector_multiplier'].values[0] / 1.5 if not sector_df.empty else 0.67
        
        # Add radar chart trace for each stock
        fig.add_trace(go.Scatterpolar(
            r=[f_score, s_score, g_score, r_score],
            theta=dimensions,
            fill='toself',
            name=symbol
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=t("Stock Performance by Dimension")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Benefits of Multi-Dimensional Balancing
    
    This comprehensive approach provides several advantages:
    
    1. **Reduced blind spots** - No single factor dominates allocation decisions
    2. **Better risk management** - Multiple risk dimensions are considered
    3. **Adaptive to changing conditions** - Different factors can be emphasized as needed
    4. **More robust performance** - Less vulnerable to failures in any single approach
    
    Regular rebalancing (quarterly recommended) helps maintain optimal allocation as market conditions evolve.
    """)