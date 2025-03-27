import streamlit as st
import pandas as pd
import numpy as np
from components.stock_search import display_stock_search
from components.chart import display_stock_chart
from components.technical_indicators import display_technical_indicators
from components.news import display_stock_news
from components.portfolio import display_portfolio
from components.market_overview import display_market_overview
from components.advanced_tools import display_advanced_tools
from components.advanced_charts import display_advanced_charts
from components.portfolio_analyzer import display_portfolio_analyzer
from components.backtesting import display_backtesting
from components.portfolio_balance import display_portfolio_balance
from utils.data_fetcher import get_stock_data
from utils.translations import get_translation, languages, translations, translate_ui
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="FinVision - Stock Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve layout
st.markdown("""
    <style>
    .stApp {
        width: 100%;
        max-width: 100vw;
        margin: 0 auto;
        background: linear-gradient(to bottom, #f8f9fa, #ffffff);
    }
    
    @media screen and (max-width: 768px) {
        .main > div {
            padding: 1rem !important;
        }
        .stMetric {
            padding: 0.75rem !important;
        }
    }
    
    /* Make tables responsive */
    .stDataFrame {
        width: 100% !important;
        overflow-x: auto !important;
    }
    
    /* Make charts responsive */
    .plot-container {
        width: 100% !important;
        height: auto !important;
        min-height: 300px;
    }
    
    /* Adjust metrics for smaller screens */
    .row-widget.stMetric {
        width: 100% !important;
        flex: 1 1 auto !important;
    }
    
    /* Make columns more responsive */
    div[data-testid="stHorizontalBlock"] > div {
        min-width: 250px !important;
        flex: 1 1 auto !important;
    }
    .language-selector {
        position: fixed;
        top: 0.5rem;
        right: 1rem;
        z-index: 1000;
        background: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
    .section-header {
        margin: 2rem 0 1rem 0;
        padding: 0.5rem;
        background: rgba(0,0,0,0.02);
        border-radius: 0.5rem;
    }
    h1, h2, h3 {
        margin-top: 2rem !important;
        margin-bottom: 1.5rem !important;
        clear: both;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stMetric {
        margin-bottom: 1.5rem !important;
    }
    .main > div {
        padding: 2rem;
    }
    h1 {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        text-align: center;
        color: #1E88E5;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h2 {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        color: #0D47A1;
        font-size: 2rem;
        border-bottom: 2px solid rgba(13,71,161,0.1);
        margin-bottom: 1rem;
    }
    h3 {
        padding-top: 1rem;
        padding-bottom: 1rem;
        color: #1565C0;
        font-size: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(to right, #1E88E5, #1976D2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .row-widget.stSelectbox {
        padding: 0.75rem 0;
    }
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
    }
    .stMetric {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid rgba(28, 131, 225, 0.1);
        transition: transform 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
    }
    div.stDataFrame {
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        background: white;
    }
    .plot-container {
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-radius: 12px;
        padding: 1rem;
        background: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #1E88E5, #1976D2);
        color: white;
    }
    .sidebar .sidebar-content .stRadio > label {
        color: white;
    }
    div[data-testid="stMarkdownContainer"] > p {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

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
def t(key_or_text):
    """
    Enhanced translation function that handles both predefined keys and direct text translation.
    
    Args:
        key_or_text (str): Either a translation key or a text to translate
        
    Returns:
        str: The translated text
    """
    lang = st.session_state.language
    
    # First try treating the input as a key
    result = get_translation(key_or_text, lang)
    
    # If the result is the same as the input and not in our translation dict keys, 
    # it might be a dynamic text that needs translation
    if result == key_or_text and key_or_text not in translations.get("en", {}):
        result = translate_ui(key_or_text, lang)
        
    return result

# Make translation function available in session state for components to use
st.session_state.translate = t

# Language selector in the top right
st.markdown("<div class='language-selector'>", unsafe_allow_html=True)
flag_cols = st.columns([1,1,1,1])

# Add language selection buttons with clear flag emojis and colored backgrounds when selected
with flag_cols[0]:
    if st.button("🇬🇧 English", 
                key="en_lang",
                help="Switch to English",
                use_container_width=True,
                type="primary" if st.session_state.language == "en" else "secondary"):
        st.session_state.language = "en"
        st.rerun()

with flag_cols[1]:
    if st.button("🇮🇹 Italiano", 
                key="it_lang",
                help="Passa all'italiano",
                use_container_width=True,
                type="primary" if st.session_state.language == "it" else "secondary"):
        st.session_state.language = "it"
        st.rerun()

with flag_cols[2]:
    if st.button("🇪🇸 Español", 
                key="es_lang",
                help="Cambiar a Español",
                use_container_width=True,
                type="primary" if st.session_state.language == "es" else "secondary"):
        st.session_state.language = "es"
        st.rerun()

with flag_cols[3]:
    if st.button("🇫🇷 Français", 
                key="fr_lang",
                help="Passer au français",
                use_container_width=True,
                type="primary" if st.session_state.language == "fr" else "secondary"):
        st.session_state.language = "fr"
        st.rerun()

# Divider after language selection
st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title(f"FinVision 📊")
st.sidebar.markdown(t("app_subtitle"))

# Navigation
page = st.sidebar.radio(
    t("nav_dashboard"),
    [t("nav_dashboard"), t("nav_stock_analysis"), t("nav_portfolio"), t("nav_portfolio_analysis"), 
     t("nav_market_overview"), t("nav_advanced_tools"), t("nav_charts"), t("nav_portfolio_tool"), 
     t("nav_portfolio_balance"), t("nav_real_time_prices"), t("nav_backtest"), "Guida"]
)

# Sidebar - Stock Search
with st.sidebar:
    selected_stock = display_stock_search()

    # Timeframe selection with translations
    timeframe_options = {
        '1d': t('1 Day'), 
        '5d': t('5 Days'), 
        '1mo': t('1 Month'), 
        '3mo': t('3 Months'),
        '6mo': t('6 Months'), 
        '1y': t('1 Year'), 
        '2y': t('2 Years'),
        '5y': t('5 Years')
    }

    selected_timeframe = st.selectbox(
        t("Select Timeframe"),
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
                label=t(f"{selected_stock} Price"),
                value=f"${current_price:.2f}",
                delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
            )

        with col2:
            st.metric(
                label=t("Market Cap"),
                value=f"${stock_data['Volume'].iloc[-1] * current_price/1000000000:.2f}B"
            )

        with col3:
            st.metric(
                label=t("Volume"),
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
    st.subheader(t(f"Latest News for {selected_stock}"))
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
        prediction_days = st.slider(t("select_days"), 
                                   min_value=1, 
                                   max_value=30, 
                                   value=min(st.session_state.future_date, 30), 
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

elif page == t("nav_advanced_tools"):
    st.title("Strumenti Avanzati")
    display_advanced_tools()
    
elif page == t("nav_charts"):
    st.title("Grafici Avanzati")
    display_advanced_charts()
    
elif page == t("nav_portfolio_tool"):
    st.title("Analizzatore del Portafoglio")
    display_portfolio_analyzer()

elif page == "Guida":
    st.title("Guida Completa a FinVision")
    from components.guida import display_guida
    display_guida()
    
elif page == t("nav_portfolio_balance"):
    # Use the portfolio balance component
    st.title(t("portfolio_balance_title"))
    display_portfolio_balance()

elif page == t("nav_real_time_prices"):
    # Use the real-time prices component
    from components.real_time_prices import display_real_time_prices
    display_real_time_prices()

elif page == t("nav_backtest"):
    # Use the backtesting component
    display_backtesting()

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align: center;'>{t('footer')}</div>", unsafe_allow_html=True)