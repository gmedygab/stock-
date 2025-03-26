import streamlit as st
import pandas as pd
import numpy as np
from components.stock_search import display_stock_search
from components.chart import display_stock_chart
from components.technical_indicators import display_technical_indicators
from components.news import display_stock_news
from components.portfolio import display_portfolio
from components.market_overview import display_market_overview
from utils.data_fetcher import get_stock_data
from utils.translations import get_translation, languages

# Page configuration
st.set_page_config(
    page_title="FinVision - Stock Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = 'AAPL'
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = '6mo'
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # Default language is English
if 'portfolio' not in st.session_state:
    # Portfolio structure: {symbol: {'shares': number, 'avg_price': number}}
    st.session_state.portfolio = {
        'AAPL': {'shares': 10, 'avg_price': 170.50}, 
        'MSFT': {'shares': 5, 'avg_price': 320.75}, 
        'GOOGL': {'shares': 2, 'avg_price': 140.25}
    }

# Create translation function shortcut
def t(key):
    return get_translation(key, st.session_state.language)

# Language selector in the top left
# Create custom CSS for flag buttons
st.markdown("""
<style>
.flag-btn {
    width: 40px;
    height: 30px;
    margin: 3px;
    background-color: transparent;
    border-radius: 5px;
    border: 2px solid transparent;
    cursor: pointer;
    transition: all 0.3s;
    font-size: 24px;
    line-height: 20px;
    padding: 0;
}
.flag-btn:hover {
    border: 2px solid #4285f4;
    transform: scale(1.1);
}
.flag-active {
    border: 2px solid #4285f4;
    box-shadow: 0 0 8px rgba(66, 133, 244, 0.8);
}
.flag-container {
    display: flex;
    align-items: center;
    padding: 5px;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    margin-bottom: 10px;
}
.language-title {
    margin-right: 10px;
    color: #888;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Create the language buttons with improved styling
st.markdown(
    f"""
    <div class="flag-container">
        <div class="language-title">🌐 Lingua:</div>
        <button onclick="changeLang('en')" class="flag-btn {'flag-active' if st.session_state.language == 'en' else ''}">🇬🇧</button>
        <button onclick="changeLang('it')" class="flag-btn {'flag-active' if st.session_state.language == 'it' else ''}">🇮🇹</button>
        <button onclick="changeLang('es')" class="flag-btn {'flag-active' if st.session_state.language == 'es' else ''}">🇪🇸</button>
        <button onclick="changeLang('fr')" class="flag-btn {'flag-active' if st.session_state.language == 'fr' else ''}">🇫🇷</button>
    </div>
    
    <script>
    function changeLang(lang) {{
        window.parent.postMessage({{
            type: "streamlit:setComponentValue",
            value: lang
        }}, "*");
    }}
    </script>
    """,
    unsafe_allow_html=True
)

# Create hidden component to receive language selection
language_selection = st.empty()
selected_lang = language_selection.text_input("", key="lang_selector", label_visibility="collapsed")

# Update language if selection changes
if selected_lang and selected_lang in ['en', 'it', 'es', 'fr'] and selected_lang != st.session_state.language:
    st.session_state.language = selected_lang
    st.rerun()

# Sidebar for navigation
st.sidebar.title(f"FinVision 📊")
st.sidebar.markdown(t("app_subtitle"))

# Navigation
page = st.sidebar.radio(
    t("nav_dashboard"),
    [t("nav_dashboard"), t("nav_stock_analysis"), t("nav_portfolio"), t("nav_portfolio_analysis"), t("nav_market_overview"), t("nav_backtest")]
)

# Sidebar - Stock Search
with st.sidebar:
    selected_stock = display_stock_search()
    
    # Timeframe selection
    timeframe_options = {
        '1d': '1 Day', 
        '5d': '5 Days', 
        '1mo': '1 Month', 
        '3mo': '3 Months',
        '6mo': '6 Months', 
        '1y': '1 Year', 
        '2y': '2 Years',
        '5y': '5 Years'
    }
    
    selected_timeframe = st.selectbox(
        "Select Timeframe",
        options=list(timeframe_options.keys()),
        format_func=lambda x: timeframe_options[x],
        index=list(timeframe_options.keys()).index(st.session_state.timeframe)
    )
    
    if selected_timeframe != st.session_state.timeframe:
        st.session_state.timeframe = selected_timeframe

# Main content area
if page == t("nav_dashboard"):
    st.title(t("dashboard_title"))
    
    # Top row - Overview metrics
    col1, col2, col3 = st.columns(3)
    
    try:
        stock_data = get_stock_data(selected_stock, '1d')
        current_price = stock_data['Close'].iloc[-1]
        previous_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else stock_data['Open'].iloc[-1]
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100
        
        # Color coding based on performance
        price_color = "green" if price_change >= 0 else "red"
        
        with col1:
            st.metric(
                label=f"{selected_stock} Price",
                value=f"${current_price:.2f}",
                delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
            )
        
        with col2:
            st.metric(
                label="Market Cap",
                value=f"${stock_data['Volume'].iloc[-1] * current_price/1000000000:.2f}B"
            )
        
        with col3:
            st.metric(
                label="Volume",
                value=f"{stock_data['Volume'].iloc[-1]/1000000:.1f}M"
            )
    except Exception as e:
        st.error(f"Error loading overview metrics: {str(e)}")
    
    # Second row - Chart and indicators
    chart_col, indicators_col = st.columns([2, 1])
    
    with chart_col:
        display_stock_chart(selected_stock, st.session_state.timeframe)
    
    with indicators_col:
        display_technical_indicators(selected_stock, st.session_state.timeframe)
    
    # Third row - News
    st.subheader(f"Latest News for {selected_stock}")
    display_stock_news(selected_stock)

elif page == t("nav_stock_analysis"):
    st.title(f"{t('stock_analysis_title')}: {selected_stock}")
    
    # Detailed chart with all indicators
    display_stock_chart(selected_stock, st.session_state.timeframe, detailed=True)
    
    # Technical analysis
    st.subheader("Technical Analysis")
    display_technical_indicators(selected_stock, st.session_state.timeframe, detailed=True)
    
    # Fundamentals and News in tabs
    tab1, tab2 = st.tabs(["Fundamentals", "News"])
    
    with tab1:
        try:
            import yfinance as yf
            stock = yf.Ticker(selected_stock)
            info = stock.info
            
            # Company profile
            st.subheader("Company Profile")
            if 'longBusinessSummary' in info:
                st.write(info['longBusinessSummary'])
            
            # Key stats in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Market Cap", f"${info.get('marketCap', 'N/A')/1000000000:.2f}B" if isinstance(info.get('marketCap'), (int, float)) else "N/A")
                st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else "N/A")
                st.metric("EPS", f"${info.get('trailingEps', 'N/A'):.2f}" if isinstance(info.get('trailingEps'), (int, float)) else "N/A")
            
            with col2:
                st.metric("Dividend Yield", f"{info.get('dividendYield', 'N/A')*100:.2f}%" if isinstance(info.get('dividendYield'), (int, float)) else "N/A")
                st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if isinstance(info.get('fiftyTwoWeekHigh'), (int, float)) else "N/A")
                st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if isinstance(info.get('fiftyTwoWeekLow'), (int, float)) else "N/A")
            
            with col3:
                st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else "N/A")
                st.metric("Avg Volume", f"{info.get('averageVolume', 'N/A')/1000000:.1f}M" if isinstance(info.get('averageVolume'), (int, float)) else "N/A")
                st.metric("Target Price", f"${info.get('targetMeanPrice', 'N/A'):.2f}" if isinstance(info.get('targetMeanPrice'), (int, float)) else "N/A")
        
        except Exception as e:
            st.error(f"Error loading fundamental data: {str(e)}")
    
    with tab2:
        display_stock_news(selected_stock, max_news=10)

elif page == t("nav_portfolio"):
    st.title(t("portfolio_title"))
    display_portfolio()

elif page == t("nav_portfolio_analysis"):
    st.title(t("portfolio_analysis_title"))
    
    # Initialize portfolio analysis session state if needed
    if 'future_date' not in st.session_state:
        st.session_state.future_date = 30  # Default 30 days in the future
    
    # Ensure we have portfolio data
    if not st.session_state.portfolio:
        st.warning("Please add stocks to your portfolio before running analysis.")
    else:
        # Portfolio header stats
        st.subheader("Portfolio Overview")
        col1, col2, col3 = st.columns(3)
        
        # Calculate portfolio metrics
        total_value = 0
        total_investment = 0
        positions = len(st.session_state.portfolio)
        
        # For volatility calculation
        all_returns = []
        
        for symbol, position in st.session_state.portfolio.items():
            try:
                shares = position['shares']
                avg_cost = position['avg_price']
                
                # Get current stock data
                stock_data = get_stock_data(symbol, '1mo')  # Get 1 month of data for volatility
                
                if not stock_data.empty:
                    current_price = stock_data['Close'].iloc[-1]
                    
                    # Calculate position value and investment
                    position_value = current_price * shares
                    investment = avg_cost * shares
                    
                    total_value += position_value
                    total_investment += investment
                    
                    # Calculate daily returns for volatility
                    daily_returns = stock_data['Close'].pct_change().dropna() * 100
                    all_returns.extend(daily_returns.tolist())
            
            except Exception as e:
                st.error(f"Error processing {symbol}: {str(e)}")
        
        # Calculate metrics
        total_return = ((total_value - total_investment) / total_investment) * 100
        
        # Calculate volatility (standard deviation of returns)
        import numpy as np
        volatility = np.std(all_returns) if all_returns else 0
        
        # Create delta color format for returns
        if total_return < 0:
            delta_color = "inverse"  # Red for negative
        else:
            delta_color = "normal"   # Green for positive
        
        # Display metrics
        with col1:
            st.metric("Portfolio Value", f"${total_value:.2f}")
        
        with col2:
            st.metric(
                "Total Return", 
                f"{total_return:.2f}%", 
                f"${total_value - total_investment:.2f}", 
                delta_color=delta_color
            )
        
        with col3:
            st.metric("Portfolio Volatility", f"{volatility:.2f}%")
        
        # Risk assessment
        st.subheader("AI Risk Assessment")
        
        # Calculate risk score (simple model)
        risk_score = volatility * 0.7 + (1 if total_return < 0 else 0) * 30
        
        # Stability rating
        if risk_score < 10:
            stability = "Very Stable"
            stability_color = "green"
        elif risk_score < 20:
            stability = "Stable"
            stability_color = "lightgreen"
        elif risk_score < 30:
            stability = "Moderately Stable"
            stability_color = "orange"
        elif risk_score < 40:
            stability = "Volatile"
            stability_color = "darkorange"
        else:
            stability = "Highly Volatile"
            stability_color = "red"
        
        # Display stability assessment
        st.markdown(f"<h3 style='color: {stability_color}'>Portfolio Stability: {stability}</h3>", unsafe_allow_html=True)
        
        # Portfolio improvement suggestions
        st.subheader("AI Improvement Suggestions")
        
        # Generate improvement suggestions based on portfolio analysis
        suggestions = []
        
        # Check if portfolio is too concentrated
        symbols = list(st.session_state.portfolio.keys())
        if len(symbols) < 5:
            suggestions.append("📊 **Diversification Needed**: Your portfolio has fewer than 5 stocks. Consider adding more diverse assets to reduce risk.")
        
        # Check if volatility is high
        if volatility > 3:
            suggestions.append("📉 **High Volatility**: Consider adding more stable assets like blue-chip stocks or ETFs like SPY or QQQ.")
        
        # Check if portfolio is doing poorly
        if total_return < 0:
            suggestions.append("💰 **Negative Returns**: Review underperforming positions and consider rebalancing or averaging down on quality stocks.")
        
        # Check portfolio sectors for diversification (simplified)
        tech_stocks = sum(1 for s in symbols if s in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'])
        if tech_stocks / len(symbols) > 0.5:
            suggestions.append("🔄 **Sector Imbalance**: Your portfolio appears heavily weighted in technology. Consider diversifying with other sectors like healthcare, finance, or consumer staples.")
        
        # Add some default suggestions if we don't have any
        if not suggestions:
            suggestions.append("✅ **Well Balanced**: Your portfolio appears to be well diversified and stable.")
            suggestions.append("💡 **Consider Regular Rebalancing**: Even good portfolios benefit from periodic review and rebalancing.")
        
        # Display suggestions
        for suggestion in suggestions:
            st.markdown(suggestion)
        
        # Future performance prediction
        st.subheader("Future Performance Prediction")
        
        # Date selector for prediction
        prediction_days = st.slider("Select days for prediction", 
                                   min_value=7, 
                                   max_value=180, 
                                   value=st.session_state.future_date, 
                                   step=1)
        
        if prediction_days != st.session_state.future_date:
            st.session_state.future_date = prediction_days
        
        # Create future performance prediction
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        
        # Calculate expected future value (simple model)
        # This uses historical volatility and return to simulate possible futures
        
        # Get current and future dates
        current_date = datetime.now().date()
        future_date = current_date + timedelta(days=prediction_days)
        
        # Create date range for prediction
        date_range = pd.date_range(start=current_date, end=future_date)
        
        # Simple prediction model with random variations based on volatility
        np.random.seed(42)  # For reproducibility
        
        # Calculate daily return rate from historical performance
        daily_return_rate = (1 + total_return/100) ** (1/30) - 1  # Assuming 1 month of data
        
        # Generate random daily returns with historical volatility
        random_daily_returns = np.random.normal(
            loc=daily_return_rate,
            scale=volatility/100,  # Convert percentage to decimal
            size=len(date_range)
        )
        
        # Calculate cumulative returns
        cumulative_returns = (1 + random_daily_returns).cumprod()
        
        # Calculate predicted values
        predicted_values = total_value * cumulative_returns
        
        # Create uncertainty band (upper and lower bounds)
        upper_band = predicted_values * (1 + volatility/100 * np.sqrt(np.arange(len(date_range))/252))
        lower_band = predicted_values * (1 - volatility/100 * np.sqrt(np.arange(len(date_range))/252))
        
        # Create forecast plot
        fig = go.Figure()
        
        # Add the main prediction line
        fig.add_trace(go.Scatter(
            x=date_range,
            y=predicted_values,
            name='Predicted Value',
            line=dict(color='rgb(0, 153, 204)', width=3)
        ))
        
        # Add uncertainty band
        fig.add_trace(go.Scatter(
            x=date_range,
            y=upper_band,
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=date_range,
            y=lower_band,
            name='Lower Bound',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 153, 204, 0.2)',
            showlegend=False
        ))
        
        # Add horizontal line for current value
        fig.add_trace(go.Scatter(
            x=[date_range[0], date_range[-1]],
            y=[total_value, total_value],
            name='Current Value',
            line=dict(color='rgba(200, 200, 200, 0.8)', width=2, dash='dash')
        ))
        
        # Format the plot
        fig.update_layout(
            title=f"Portfolio Value Prediction: Next {prediction_days} Days",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        # Add prediction metrics
        final_predicted_value = predicted_values[-1]
        predicted_return = ((final_predicted_value - total_value) / total_value) * 100
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction summary
        st.markdown(f"""
        **Prediction Summary (based on historical performance):**
        - Current portfolio value: **${total_value:.2f}**
        - Predicted value on {future_date.strftime('%B %d, %Y')}: **${final_predicted_value:.2f}**
        - Predicted return over {prediction_days} days: **{predicted_return:.2f}%**
        - Prediction confidence interval: **±{volatility:.2f}%**
        
        *Note: This prediction is based on historical volatility and returns. Actual market performance may vary significantly.*
        """)
        
        # Portfolio Optimization Suggestions (simplified)
        st.subheader("Portfolio Optimization")
        
        optimization_tab1, optimization_tab2 = st.tabs(["Risk Reduction", "Return Maximization"])
        
        with optimization_tab1:
            st.markdown("""
            ### Risk Reduction Strategy
            
            Based on your current portfolio volatility, here are some suggestions to reduce risk:
            
            1. **Add Stable Assets**: Consider allocating 20-30% of your portfolio to stable assets like:
               - **Treasury Bonds (TLT, IEF)**
               - **Low Volatility ETFs (SPLV, USMV)**
               - **Consumer Staples (XLP, KO, PG)**
               
            2. **Diversify Across Sectors**:
               - Aim for exposure across at least 6-8 different market sectors
               - No single sector should represent more than 25% of your portfolio
               
            3. **Position Sizing**:
               - No single stock should represent more than 10% of your portfolio
               - Consider trimming overweight positions
            """)
        
        with optimization_tab2:
            st.markdown("""
            ### Return Maximization Strategy
            
            If you're looking to optimize for higher returns while accepting additional risk:
            
            1. **Growth Opportunities**:
               - Consider allocating 5-10% to high-growth sectors like AI, clean energy, or biotechnology
               - Look for companies with strong growth forecasts and defensible market positions
            
            2. **Momentum Strategy**:
               - Identify stocks showing consistent upward price movement
               - Consider ETFs like MTUM that focus on momentum stocks
               
            3. **Regular Rebalancing**:
               - Set calendar-based reminders to review and rebalance your portfolio
               - Consider rebalancing every quarter to capture gains and maintain your target allocation
            """)

elif page == t("nav_market_overview"):
    st.title(t("market_overview_title"))
    display_market_overview()

elif page == t("nav_backtest"):
    st.title(t("backtest_title"))
    
    # Backtest parameters
    st.sidebar.header("Backtest Parameters")
    
    # Set date range
    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.to_datetime("2023-01-01")
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=pd.to_datetime("today")
    )
    
    if start_date >= end_date:
        st.error("End date must be after start date.")
    else:
        # Check if portfolio exists
        if not st.session_state.portfolio:
            st.warning("Please add stocks to your portfolio before running a backtest.")
        else:
            st.subheader("Portfolio Backtest Performance")
            
            # Get historical data for all stocks in portfolio
            import yfinance as yf
            import pandas as pd
            import plotly.graph_objects as go
            from datetime import datetime
            
            try:
                # Get data for each stock in portfolio
                portfolio_data = {}
                for symbol, position in st.session_state.portfolio.items():
                    try:
                        # Get historical data
                        stock_data = yf.download(symbol, start=start_date, end=end_date)
                        
                        # Make sure the data has all required columns
                        if not stock_data.empty:
                            # Check if 'Adj Close' column exists, if not, use 'Close'
                            if 'Adj Close' not in stock_data.columns:
                                stock_data['Adj Close'] = stock_data['Close']
                            portfolio_data[symbol] = stock_data
                    except Exception as e:
                        st.error(f"Error fetching data for {symbol}: {str(e)}")
                
                if portfolio_data:
                    # Create dataframe with adjusted close prices
                    prices_df = pd.DataFrame()
                    
                    for symbol, data in portfolio_data.items():
                        prices_df[symbol] = data['Adj Close']
                    
                    # Calculate daily returns
                    returns_df = prices_df.pct_change().dropna()
                    
                    # Calculate portfolio returns based on current weights
                    portfolio_weights = {}
                    total_value = 0
                    
                    # Calculate total portfolio value first
                    for symbol, position in st.session_state.portfolio.items():
                        if symbol in portfolio_data:
                            shares = position['shares']
                            current_price = portfolio_data[symbol]['Adj Close'][-1]
                            position_value = current_price * shares
                            total_value += position_value
                    
                    # Then calculate weights
                    for symbol, position in st.session_state.portfolio.items():
                        if symbol in portfolio_data:
                            shares = position['shares']
                            current_price = portfolio_data[symbol]['Adj Close'][-1]
                            position_value = current_price * shares
                            portfolio_weights[symbol] = position_value / total_value
                    
                    # Calculate portfolio returns
                    portfolio_returns = pd.Series(0, index=returns_df.index)
                    
                    for symbol, weight in portfolio_weights.items():
                        if symbol in returns_df.columns:
                            portfolio_returns += returns_df[symbol] * weight
                    
                    # Calculate cumulative returns
                    cumulative_returns = (1 + portfolio_returns).cumprod()
                    
                    # Calculate benchmark returns (S&P 500)
                    benchmark_data = yf.download("SPY", start=start_date, end=end_date)
                    # Check if 'Adj Close' column exists in benchmark data
                    if 'Adj Close' not in benchmark_data.columns:
                        benchmark_data['Adj Close'] = benchmark_data['Close']
                    benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()
                    benchmark_cumulative = (1 + benchmark_returns).cumprod()
                    
                    # Create a plot
                    fig = go.Figure()
                    
                    # Add portfolio performance line
                    fig.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns * 100 - 100,  # Convert to percentage
                        name='Portfolio',
                        line=dict(color='rgb(0, 153, 204)', width=3)
                    ))
                    
                    # Add benchmark line
                    fig.add_trace(go.Scatter(
                        x=benchmark_cumulative.index,
                        y=benchmark_cumulative * 100 - 100,  # Convert to percentage
                        name='S&P 500 (SPY)',
                        line=dict(color='rgba(200, 100, 100, 0.8)', width=2)
                    ))
                    
                    # Format the plot
                    fig.update_layout(
                        title="Portfolio vs S&P 500 Performance",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return (%)",
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode="x unified"
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate performance metrics
                    if not portfolio_returns.empty and not benchmark_returns.empty:
                        # Calculate metrics
                        portfolio_total_return = (cumulative_returns.iloc[-1] - 1) * 100
                        benchmark_total_return = (benchmark_cumulative.iloc[-1] - 1) * 100
                        
                        # Annualized returns
                        days = (returns_df.index[-1] - returns_df.index[0]).days
                        portfolio_annual_return = ((1 + portfolio_total_return/100) ** (365/days) - 1) * 100
                        benchmark_annual_return = ((1 + benchmark_total_return/100) ** (365/days) - 1) * 100
                        
                        # Volatility
                        portfolio_volatility = portfolio_returns.std() * (252 ** 0.5) * 100  # Annualized
                        benchmark_volatility = benchmark_returns.std() * (252 ** 0.5) * 100  # Annualized
                        
                        # Sharpe Ratio (simplified, using 0% risk-free rate)
                        portfolio_sharpe = portfolio_annual_return / portfolio_volatility if portfolio_volatility > 0 else 0
                        benchmark_sharpe = benchmark_annual_return / benchmark_volatility if benchmark_volatility > 0 else 0
                        
                        # Maximum Drawdown
                        portfolio_cummax = cumulative_returns.cummax()
                        portfolio_drawdown = ((cumulative_returns - portfolio_cummax) / portfolio_cummax) * 100
                        portfolio_max_drawdown = portfolio_drawdown.min()
                        
                        benchmark_cummax = benchmark_cumulative.cummax()
                        benchmark_drawdown = ((benchmark_cumulative - benchmark_cummax) / benchmark_cummax) * 100
                        benchmark_max_drawdown = benchmark_drawdown.min()
                        
                        # Display metrics in table format
                        metrics_data = {
                            'Metric': ['Total Return (%)', 'Annualized Return (%)', 'Annualized Volatility (%)', 
                                       'Sharpe Ratio', 'Maximum Drawdown (%)'],
                            'Portfolio': [f"{portfolio_total_return:.2f}", f"{portfolio_annual_return:.2f}", 
                                         f"{portfolio_volatility:.2f}", f"{portfolio_sharpe:.2f}", 
                                         f"{portfolio_max_drawdown:.2f}"],
                            'S&P 500': [f"{benchmark_total_return:.2f}", f"{benchmark_annual_return:.2f}", 
                                       f"{benchmark_volatility:.2f}", f"{benchmark_sharpe:.2f}", 
                                       f"{benchmark_max_drawdown:.2f}"]
                        }
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.table(metrics_df)
                        
                        # Portfolio performance analysis
                        st.subheader("Performance Analysis")
                        
                        if portfolio_total_return > benchmark_total_return:
                            st.success(f"Your portfolio outperformed the S&P 500 by {portfolio_total_return - benchmark_total_return:.2f}%")
                        else:
                            st.warning(f"Your portfolio underperformed the S&P 500 by {benchmark_total_return - portfolio_total_return:.2f}%")
                        
                        # Risk analysis
                        st.subheader("Risk Analysis")
                        
                        # Drawdown chart
                        drawdown_fig = go.Figure()
                        
                        drawdown_fig.add_trace(go.Scatter(
                            x=portfolio_drawdown.index,
                            y=portfolio_drawdown,
                            name='Portfolio Drawdown',
                            line=dict(color='rgb(0, 153, 204)', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(0, 153, 204, 0.2)'
                        ))
                        
                        drawdown_fig.add_trace(go.Scatter(
                            x=benchmark_drawdown.index,
                            y=benchmark_drawdown,
                            name='S&P 500 Drawdown',
                            line=dict(color='rgba(200, 100, 100, 0.8)', width=2),
                            visible='legendonly'  # Hidden by default
                        ))
                        
                        drawdown_fig.update_layout(
                            title="Portfolio Drawdown Analysis",
                            xaxis_title="Date",
                            yaxis_title="Drawdown (%)",
                            height=400,
                            yaxis=dict(tickformat=".2f"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        
                        st.plotly_chart(drawdown_fig, use_container_width=True)
                        
                        # Stock contribution to portfolio
                        st.subheader("Stock Contribution")
                        
                        # Calculate individual stock returns
                        stock_returns = {}
                        for symbol in portfolio_weights.keys():
                            if symbol in returns_df.columns:
                                stock_cumulative = (1 + returns_df[symbol]).cumprod()
                                stock_returns[symbol] = ((stock_cumulative.iloc[-1] - 1) * 100).round(2)
                        
                        # Create contribution chart
                        contribution_data = {
                            'Symbol': list(stock_returns.keys()),
                            'Return (%)': list(stock_returns.values()),
                            'Weight (%)': [(w * 100).round(2) for w in portfolio_weights.values()],
                            'Contribution (%)': [(stock_returns[s] * portfolio_weights[s]).round(2) for s in stock_returns.keys()]
                        }
                        
                        contribution_df = pd.DataFrame(contribution_data)
                        contribution_df = contribution_df.sort_values('Contribution (%)', ascending=False)
                        
                        # Show contribution table
                        st.dataframe(contribution_df, use_container_width=True)
                        
                        # Chart of returns by stock
                        returns_fig = go.Figure()
                        
                        for symbol in stock_returns.keys():
                            stock_cumulative = (1 + returns_df[symbol]).cumprod() * 100 - 100
                            returns_fig.add_trace(go.Scatter(
                                x=stock_cumulative.index,
                                y=stock_cumulative,
                                name=symbol,
                            ))
                        
                        returns_fig.update_layout(
                            title="Individual Stock Performance",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Return (%)",
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        
                        st.plotly_chart(returns_fig, use_container_width=True)
                        
                else:
                    st.error("Could not retrieve data for any stocks in your portfolio for the selected date range.")
            
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align: center;'>{t('footer')}</div>", unsafe_allow_html=True)
