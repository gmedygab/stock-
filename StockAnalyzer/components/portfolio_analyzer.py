import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import math

def display_portfolio_analyzer():
    """
    Display advanced portfolio analysis tools
    """
    st.subheader("Analizzatore Avanzato del Portafoglio")
    
    # Check if portfolio is empty
    if not st.session_state.portfolio:
        st.warning("Il tuo portafoglio è vuoto. Aggiungi alcune azioni per iniziare l'analisi.")
        return
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Composizione", 
        "Analisi del Rischio", 
        "Correlazione Titoli", 
        "Simulazione Monte Carlo",
        "Calcolatore Interesse Composto"
    ])
    
    with tab1:
        display_portfolio_composition()
    
    with tab2:
        display_risk_analysis()
    
    with tab3:
        display_portfolio_correlation()
    
    with tab4:
        display_monte_carlo_simulation()
        
    with tab5:
        display_compound_interest_calculator()


def display_portfolio_composition():
    """
    Display detailed portfolio composition analysis
    """
    st.subheader("Composizione Dettagliata del Portafoglio")
    
    # Gather data about the portfolio
    portfolio_data = {
        'Symbol': [],
        'Name': [],
        'Shares': [],
        'Avg Price': [],
        'Current Price': [],
        'Value': [],
        'Cost': [],
        'Return': [],
        'Return %': [],
        'Weight': [],
        'Sector': [],
        'Industry': []
    }
    
    total_value = 0
    total_cost = 0
    
    # Collect data for each position
    for symbol, position in st.session_state.portfolio.items():
        try:
            # Get basic position data
            shares = position['shares']
            avg_price = position['avg_price']
            cost = shares * avg_price
            
            # Get current price and company info
            stock = yf.Ticker(symbol)
            info = stock.info
            current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
            name = info.get('shortName', symbol)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            
            # Calculate returns
            value = shares * current_price
            return_value = value - cost
            return_pct = (return_value / cost) * 100
            
            # Add to total value and cost
            total_value += value
            total_cost += cost
            
            # Add to portfolio data
            portfolio_data['Symbol'].append(symbol)
            portfolio_data['Name'].append(name)
            portfolio_data['Shares'].append(shares)
            portfolio_data['Avg Price'].append(avg_price)
            portfolio_data['Current Price'].append(current_price)
            portfolio_data['Value'].append(value)
            portfolio_data['Cost'].append(cost)
            portfolio_data['Return'].append(return_value)
            portfolio_data['Return %'].append(return_pct)
            portfolio_data['Weight'].append(0)  # Will calculate after all data is collected
            portfolio_data['Sector'].append(sector)
            portfolio_data['Industry'].append(industry)
            
        except Exception as e:
            st.warning(f"Error processing {symbol}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(portfolio_data)
    
    # Calculate weights
    df['Weight'] = (df['Value'] / total_value) * 100
    
    # Display portfolio summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Valore Totale",
            value=f"${total_value:,.2f}"
        )
    
    with col2:
        total_return = total_value - total_cost
        total_return_pct = (total_return / total_cost) * 100
        
        st.metric(
            label="Rendimento Totale",
            value=f"${total_return:,.2f}",
            delta=f"{total_return_pct:.2f}%"
        )
    
    with col3:
        num_positions = len(df)
        avg_position_size = total_value / num_positions if num_positions > 0 else 0
        
        st.metric(
            label="Posizioni",
            value=f"{num_positions}",
            delta=f"Dim. Media: ${avg_position_size:,.2f}"
        )
    
    # Display portfolio breakdown charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Create pie chart of portfolio allocation
        fig = px.pie(
            df, 
            values='Value', 
            names='Symbol',
            title='Composizione del Portafoglio',
            hover_data=['Name', 'Value', 'Weight'],
            labels={'Value': 'Valore ($)', 'Weight': 'Peso (%)'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Filter out positions with missing sector data
        sector_df = df[df['Sector'] != 'N/A'].copy()
        
        if not sector_df.empty:
            # Group by sector
            sector_breakdown = sector_df.groupby('Sector')['Value'].sum().reset_index()
            sector_breakdown['Weight'] = (sector_breakdown['Value'] / sector_breakdown['Value'].sum()) * 100
            
            # Create sector allocation pie chart
            fig = px.pie(
                sector_breakdown, 
                values='Value', 
                names='Sector',
                title='Allocazione Settoriale',
                hover_data=['Weight'],
                labels={'Value': 'Valore ($)', 'Weight': 'Peso (%)'},
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dati settoriali non disponibili per le azioni nel tuo portafoglio.")
    
    # Create performance comparison chart
    st.subheader("Confronto delle Performance")
    
    # Sort by return percentage
    performance_df = df.sort_values('Return %', ascending=False)
    
    # Create performance bar chart
    fig = px.bar(
        performance_df,
        x='Symbol',
        y='Return %',
        title='Rendimento % per Posizione',
        color='Return %',
        color_continuous_scale='RdYlGn',
        text='Return %',
        hover_data=['Name', 'Value', 'Return']
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed table
    st.subheader("Dettaglio delle Posizioni")
    
    # Format DataFrame for display
    display_df = df.copy()
    display_df['Avg Price'] = display_df['Avg Price'].map('${:,.2f}'.format)
    display_df['Current Price'] = display_df['Current Price'].map('${:,.2f}'.format)
    display_df['Value'] = display_df['Value'].map('${:,.2f}'.format)
    display_df['Cost'] = display_df['Cost'].map('${:,.2f}'.format)
    display_df['Return'] = display_df['Return'].map('${:,.2f}'.format)
    display_df['Return %'] = display_df['Return %'].map('{:,.2f}%'.format)
    display_df['Weight'] = display_df['Weight'].map('{:,.2f}%'.format)
    
    # Rename columns to Italian
    display_df = display_df.rename(columns={
        'Symbol': 'Simbolo',
        'Name': 'Nome',
        'Shares': 'Azioni',
        'Avg Price': 'Prezzo Medio',
        'Current Price': 'Prezzo Attuale',
        'Value': 'Valore',
        'Cost': 'Costo',
        'Return': 'Rendimento',
        'Return %': 'Rendimento %',
        'Weight': 'Peso',
        'Sector': 'Settore',
        'Industry': 'Industria'
    })
    
    st.dataframe(display_df, use_container_width=True)


def display_risk_analysis():
    """
    Display risk analysis of the portfolio
    """
    st.subheader("Analisi del Rischio")
    
    # Time period selection
    periods = {
        "1m": "1 Mese",
        "3m": "3 Mesi",
        "6m": "6 Mesi",
        "1y": "1 Anno",
        "2y": "2 Anni"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        period = st.select_slider(
            "Periodo per Analisi del Rischio",
            options=list(periods.keys()),
            format_func=lambda x: periods[x],
            value="1y"
        )
    
    with col2:
        benchmark = st.selectbox(
            "Benchmark",
            options=["^GSPC", "^DJI", "^IXIC", "^FTSE"],
            format_func=lambda x: {
                "^GSPC": "S&P 500",
                "^DJI": "Dow Jones",
                "^IXIC": "NASDAQ",
                "^FTSE": "FTSE MIB"
            }.get(x, x),
            index=0
        )
    
    # Calculate portfolio returns
    try:
        # Get portfolio data
        portfolio_data = {}
        portfolio_weights = {}
        total_value = 0
        
        for symbol, position in st.session_state.portfolio.items():
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    portfolio_data[symbol] = hist
                    
                    # Calculate position value
                    shares = position['shares']
                    current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                    position_value = shares * current_price
                    
                    total_value += position_value
                    portfolio_weights[symbol] = position_value
            except:
                st.warning(f"Could not retrieve data for {symbol}")
        
        # Normalize weights
        portfolio_weights = {k: v/total_value for k, v in portfolio_weights.items()}
        
        if not portfolio_data:
            st.error("No data available for any stocks in the portfolio")
            return
        
        # Get benchmark data
        benchmark_data = yf.Ticker(benchmark).history(period=period)
        
        if benchmark_data.empty:
            st.error(f"No data available for benchmark {benchmark}")
            return
        
        # Create a combined DataFrame with daily returns
        returns_df = pd.DataFrame()
        
        # Add portfolio stock returns
        for symbol, hist in portfolio_data.items():
            if 'Close' in hist.columns and not hist.empty:
                returns_df[symbol] = hist['Close'].pct_change()
        
        # Add benchmark returns
        returns_df['Benchmark'] = benchmark_data['Close'].pct_change()
        
        # Drop NAs
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            st.error("Insufficient data for analysis")
            return
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0, index=returns_df.index)
        for symbol, weight in portfolio_weights.items():
            if symbol in returns_df.columns:
                portfolio_returns += returns_df[symbol] * weight
        
        # Add portfolio returns to DataFrame
        returns_df['Portfolio'] = portfolio_returns
        
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(returns_df, portfolio_returns, returns_df['Benchmark'])
        
        # Display risk metrics
        st.subheader("Metriche di Rischio")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Volatilità Annualizzata",
                value=f"{risk_metrics['portfolio_volatility']:.2f}%",
                delta=f"{risk_metrics['portfolio_volatility'] - risk_metrics['benchmark_volatility']:.2f}% vs Benchmark"
            )
        
        with col2:
            st.metric(
                label="Sharpe Ratio",
                value=f"{risk_metrics['portfolio_sharpe']:.2f}",
                delta=f"{risk_metrics['portfolio_sharpe'] - risk_metrics['benchmark_sharpe']:.2f} vs Benchmark"
            )
        
        with col3:
            st.metric(
                label="Max Drawdown",
                value=f"{risk_metrics['portfolio_max_drawdown']:.2f}%",
                delta=f"{risk_metrics['portfolio_max_drawdown'] - risk_metrics['benchmark_max_drawdown']:.2f}% vs Benchmark",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                label="Beta vs Benchmark",
                value=f"{risk_metrics['portfolio_beta']:.2f}"
            )
        
        # Display risk charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Create volatility comparison chart
            volatility_data = []
            
            # Add individual stocks
            for symbol in portfolio_data:
                if symbol in returns_df.columns:
                    vol = returns_df[symbol].std() * np.sqrt(252) * 100
                    volatility_data.append({
                        'Asset': symbol,
                        'Volatilità (%)': vol,
                        'Tipo': 'Azione'
                    })
            
            # Add portfolio
            volatility_data.append({
                'Asset': 'Portafoglio',
                'Volatilità (%)': risk_metrics['portfolio_volatility'],
                'Tipo': 'Portafoglio'
            })
            
            # Add benchmark
            volatility_data.append({
                'Asset': 'Benchmark',
                'Volatilità (%)': risk_metrics['benchmark_volatility'],
                'Tipo': 'Benchmark'
            })
            
            # Create DataFrame
            vol_df = pd.DataFrame(volatility_data)
            
            # Create chart
            fig = px.bar(
                vol_df,
                x='Asset',
                y='Volatilità (%)',
                color='Tipo',
                title='Confronto Volatilità Annualizzata',
                color_discrete_map={
                    'Azione': 'lightblue',
                    'Portafoglio': 'darkblue',
                    'Benchmark': 'gray'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create drawdown chart
            fig = go.Figure()
            
            # Add portfolio drawdown
            portfolio_cum_returns = (1 + portfolio_returns).cumprod()
            portfolio_running_max = portfolio_cum_returns.cummax()
            portfolio_drawdown = ((portfolio_cum_returns - portfolio_running_max) / portfolio_running_max) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_drawdown.index,
                    y=portfolio_drawdown,
                    name='Portafoglio',
                    line=dict(color='blue', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 0, 255, 0.1)'
                )
            )
            
            # Add benchmark drawdown
            benchmark_returns = returns_df['Benchmark']
            benchmark_cum_returns = (1 + benchmark_returns).cumprod()
            benchmark_running_max = benchmark_cum_returns.cummax()
            benchmark_drawdown = ((benchmark_cum_returns - benchmark_running_max) / benchmark_running_max) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=benchmark_drawdown.index,
                    y=benchmark_drawdown,
                    name='Benchmark',
                    line=dict(color='gray', width=2, dash='dash')
                )
            )
            
            # Update layout
            fig.update_layout(
                title='Confronto Drawdown',
                xaxis_title='Data',
                yaxis_title='Drawdown (%)',
                template='plotly_white',
                yaxis=dict(tickformat=".2f"),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display risk contribution chart
        st.subheader("Contributo al Rischio")
        
        # Calculate risk contribution
        risk_contribution = calculate_risk_contribution(returns_df.iloc[:, :-2], portfolio_weights)
        
        # Create risk contribution chart
        fig = px.pie(
            risk_contribution,
            values='Risk Contribution',
            names='Symbol',
            title='Contributo al Rischio di Portafoglio',
            hover_data=['Risk Contribution %'],
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display efficiency frontier
        st.subheader("Efficienza del Portafoglio")
        
        # Create efficiency frontier
        fig = create_efficiency_frontier(returns_df.iloc[:, :-2], portfolio_weights, risk_metrics)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Si è verificato un errore nell'analisi del rischio: {str(e)}")

def calculate_risk_metrics(returns_df, portfolio_returns, benchmark_returns):
    """
    Calculate risk metrics for portfolio and benchmark
    
    Parameters:
    returns_df (pd.DataFrame): DataFrame with returns for portfolio constituents
    portfolio_returns (pd.Series): Series with portfolio returns
    benchmark_returns (pd.Series): Series with benchmark returns
    
    Returns:
    dict: Dictionary with risk metrics
    """
    # Annualized return
    portfolio_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
    benchmark_return = (1 + benchmark_returns).prod() ** (252 / len(benchmark_returns)) - 1
    
    # Annualized volatility
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252) * 100
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100
    
    # Sharpe ratio (assuming 0% risk-free rate)
    portfolio_sharpe = (portfolio_return * 100) / portfolio_volatility if portfolio_volatility > 0 else 0
    benchmark_sharpe = (benchmark_return * 100) / benchmark_volatility if benchmark_volatility > 0 else 0
    
    # Maximum drawdown
    portfolio_cum_returns = (1 + portfolio_returns).cumprod()
    portfolio_running_max = portfolio_cum_returns.cummax()
    portfolio_drawdown = (portfolio_cum_returns - portfolio_running_max) / portfolio_running_max
    portfolio_max_drawdown = portfolio_drawdown.min() * 100
    
    benchmark_cum_returns = (1 + benchmark_returns).cumprod()
    benchmark_running_max = benchmark_cum_returns.cummax()
    benchmark_drawdown = (benchmark_cum_returns - benchmark_running_max) / benchmark_running_max
    benchmark_max_drawdown = benchmark_drawdown.min() * 100
    
    # Beta
    cov_matrix = returns_df[['Portfolio', 'Benchmark']].cov()
    portfolio_beta = cov_matrix.loc['Portfolio', 'Benchmark'] / returns_df['Benchmark'].var()
    
    # Alpha (Jensen's Alpha)
    portfolio_alpha = (portfolio_return - (0 + portfolio_beta * (benchmark_return - 0))) * 100
    
    # Information ratio
    tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252) * 100
    information_ratio = ((portfolio_return - benchmark_return) * 100) / tracking_error if tracking_error > 0 else 0
    
    return {
        'portfolio_return': portfolio_return * 100,
        'benchmark_return': benchmark_return * 100,
        'portfolio_volatility': portfolio_volatility,
        'benchmark_volatility': benchmark_volatility,
        'portfolio_sharpe': portfolio_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'portfolio_max_drawdown': portfolio_max_drawdown,
        'benchmark_max_drawdown': benchmark_max_drawdown,
        'portfolio_beta': portfolio_beta,
        'portfolio_alpha': portfolio_alpha,
        'information_ratio': information_ratio
    }

def calculate_risk_contribution(returns_df, weights):
    """
    Calculate the risk contribution of each asset to portfolio risk
    
    Parameters:
    returns_df (pd.DataFrame): DataFrame with returns for portfolio constituents
    weights (dict): Dictionary with portfolio weights
    
    Returns:
    pd.DataFrame: DataFrame with risk contribution
    """
    # Convert weights to Series
    weight_series = pd.Series(weights)
    
    # Align weights with returns_df columns
    common_assets = set(weight_series.index).intersection(set(returns_df.columns))
    
    if not common_assets:
        return pd.DataFrame({
            'Symbol': ['N/A'],
            'Risk Contribution': [0],
            'Risk Contribution %': [0]
        })
    
    # Filter returns and weights
    returns_filtered = returns_df[list(common_assets)]
    weights_filtered = weight_series[list(common_assets)]
    weights_filtered = weights_filtered / weights_filtered.sum()  # Re-normalize
    
    # Calculate covariance matrix
    cov_matrix = returns_filtered.cov() * 252  # Annualized
    
    # Convert weights to numpy array
    weight_array = weights_filtered.values
    
    # Calculate portfolio variance
    portfolio_variance = weight_array.T @ cov_matrix @ weight_array
    
    # Calculate marginal risk contribution
    marginal_contribution = cov_matrix @ weight_array
    
    # Calculate risk contribution
    risk_contribution = weight_array * marginal_contribution
    
    # Normalize to percentage
    risk_contribution_pct = risk_contribution / portfolio_variance * 100
    
    # Create DataFrame
    result = pd.DataFrame({
        'Symbol': weights_filtered.index,
        'Risk Contribution': risk_contribution,
        'Risk Contribution %': risk_contribution_pct
    })
    
    return result

def create_efficiency_frontier(returns_df, actual_weights, risk_metrics):
    """
    Create an efficient frontier and plot the current portfolio
    
    Parameters:
    returns_df (pd.DataFrame): DataFrame with returns for portfolio constituents
    actual_weights (dict): Dictionary with current portfolio weights
    risk_metrics (dict): Dictionary with risk metrics
    
    Returns:
    go.Figure: Plotly figure with efficient frontier
    """
    # Align weights with returns_df columns
    weight_series = pd.Series(actual_weights)
    common_assets = set(weight_series.index).intersection(set(returns_df.columns))
    
    if not common_assets:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Dati insufficienti per l'analisi dell'efficienza",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=15)
        )
        return fig
    
    # Filter returns and weights
    returns_filtered = returns_df[list(common_assets)]
    weights_filtered = weight_series[list(common_assets)]
    weights_filtered = weights_filtered / weights_filtered.sum()  # Re-normalize
    
    # Calculate mean returns and covariance
    mean_returns = returns_filtered.mean() * 252  # Annualized
    cov_matrix = returns_filtered.cov() * 252  # Annualized
    
    # Create efficient frontier (Monte Carlo approach)
    num_portfolios = 5000
    results = []
    
    for _ in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(len(common_assets))
        weights = weights / np.sum(weights)
        
        # Calculate returns and volatility
        returns = np.sum(mean_returns * weights) * 100  # In percent
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 100  # In percent
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe = returns / volatility if volatility > 0 else 0
        
        # Store results
        results.append({
            'Returns': returns,
            'Volatility': volatility,
            'Sharpe': sharpe,
            'Weights': weights
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate current portfolio values
    weights_array = np.array([weights_filtered[asset] for asset in common_assets])
    current_return = np.sum(mean_returns * weights_array) * 100
    current_volatility = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array))) * 100
    current_sharpe = current_return / current_volatility if current_volatility > 0 else 0
    
    # Create figure
    fig = go.Figure()
    
    # Add random portfolios
    fig.add_trace(
        go.Scatter(
            x=results_df['Volatility'],
            y=results_df['Returns'],
            mode='markers',
            marker=dict(
                size=5,
                color=results_df['Sharpe'],
                colorscale='Viridis',
                colorbar=dict(title='Sharpe Ratio'),
                showscale=True
            ),
            text=[f"Sharpe: {s:.2f}" for s in results_df['Sharpe']],
            name='Portafogli Simulati'
        )
    )
    
    # Add current portfolio
    fig.add_trace(
        go.Scatter(
            x=[current_volatility],
            y=[current_return],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            ),
            text=[f"Portafoglio Attuale<br>Rendimento: {current_return:.2f}%<br>Volatilità: {current_volatility:.2f}%<br>Sharpe: {current_sharpe:.2f}"],
            name='Portafoglio Attuale'
        )
    )
    
    # Add benchmark if available
    if 'benchmark_volatility' in risk_metrics and 'benchmark_return' in risk_metrics:
        fig.add_trace(
            go.Scatter(
                x=[risk_metrics['benchmark_volatility']],
                y=[risk_metrics['benchmark_return']],
                mode='markers',
                marker=dict(
                    size=15,
                    color='gray',
                    symbol='diamond'
                ),
                text=[f"Benchmark<br>Rendimento: {risk_metrics['benchmark_return']:.2f}%<br>Volatilità: {risk_metrics['benchmark_volatility']:.2f}%<br>Sharpe: {risk_metrics['benchmark_sharpe']:.2f}"],
                name='Benchmark'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Frontiera Efficiente (Simulata)',
        xaxis_title='Volatilità Annualizzata (%)',
        yaxis_title='Rendimento Annualizzato (%)',
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


def display_portfolio_correlation():
    """
    Display correlation analysis between portfolio assets
    """
    st.subheader("Analisi di Correlazione del Portafoglio")
    
    # Time period selection
    periods = {
        "1m": "1 Mese",
        "3m": "3 Mesi",
        "6m": "6 Mesi",
        "1y": "1 Anno",
        "2y": "2 Anni"
    }
    
    period = st.select_slider(
        "Periodo per Analisi di Correlazione",
        options=list(periods.keys()),
        format_func=lambda x: periods[x],
        value="1y",
        key="corr_period"
    )
    
    # Calculate correlations
    try:
        # Get portfolio data
        portfolio_data = {}
        
        for symbol in st.session_state.portfolio:
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    portfolio_data[symbol] = hist['Close']
            except:
                st.warning(f"Could not retrieve data for {symbol}")
        
        if not portfolio_data:
            st.error("No data available for any stocks in the portfolio")
            return
        
        # Create a combined DataFrame with daily returns
        prices_df = pd.DataFrame(portfolio_data)
        returns_df = prices_df.pct_change().dropna()
        
        if returns_df.empty:
            st.error("Insufficient data for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Plot correlation heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title="Matrice di Correlazione"
        )
        
        fig.update_layout(
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        st.subheader("Insight sulla Diversificazione")
        
        # Calculate average correlation
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        st.write(f"**Correlazione Media del Portafoglio: {avg_corr:.2f}**")
        
        if avg_corr > 0.7:
            st.warning("""
            **Alta Correlazione Rilevata**
            
            Il tuo portafoglio mostra un'alta correlazione media tra le diverse azioni. Ciò può ridurre l'efficacia della diversificazione e aumentare il rischio in caso di mercati in calo.
            
            **Suggerimenti:**
            - Considera l'aggiunta di asset con correlazione negativa o bassa rispetto al tuo portafoglio attuale
            - Esplora opportunità in settori diversi o classi di attività alternative
            - Valuta l'inclusione di ETF su obbligazioni o materie prime per una migliore diversificazione
            """)
        elif avg_corr > 0.3:
            st.info("""
            **Correlazione Moderata Rilevata**
            
            Il tuo portafoglio mostra una correlazione moderata tra le diverse azioni, offrendo un certo livello di diversificazione.
            
            **Suggerimenti:**
            - Per migliorare ulteriormente la diversificazione, considera asset con correlazione più bassa
            - Valuta se alcune coppie di azioni altamente correlate possono essere gestite per ridurre la ridondanza
            """)
        else:
            st.success("""
            **Buona Diversificazione Rilevata**
            
            Il tuo portafoglio mostra una bassa correlazione media tra le diverse azioni, indicando una buona diversificazione.
            
            **Suggerimenti:**
            - Continua a monitorare le correlazioni nel tempo, poiché possono cambiare in base alle condizioni di mercato
            - Il tuo approccio alla diversificazione sembra efficace, mantienilo nelle future decisioni di investimento
            """)
        
        # Find highest and lowest correlations
        if len(corr_matrix) > 1:
            # Get upper triangle of correlation matrix (excluding diagonal)
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find highest correlation
            highest_corr = upper_tri.max().max()
            highest_pair = np.unravel_index(upper_tri.values.argmax(), upper_tri.shape)
            highest_symbols = (corr_matrix.index[highest_pair[0]], corr_matrix.columns[highest_pair[1]])
            
            # Find lowest correlation
            lowest_corr = upper_tri.min().min()
            lowest_pair = np.unravel_index(upper_tri.values.argmin(), upper_tri.shape)
            lowest_symbols = (corr_matrix.index[lowest_pair[0]], corr_matrix.columns[lowest_pair[1]])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Coppia più correlata:**  \n{highest_symbols[0]} e {highest_symbols[1]} ({highest_corr:.2f})")
            
            with col2:
                st.info(f"**Coppia meno correlata:**  \n{lowest_symbols[0]} e {lowest_symbols[1]} ({lowest_corr:.2f})")
        
        # Plot price movements
        st.subheader("Movimenti di Prezzo Normalizzati")
        
        # Normalize price data
        norm_prices = prices_df.div(prices_df.iloc[0]).mul(100)
        
        # Create chart
        fig = px.line(
            norm_prices,
            title="Andamento Prezzi Normalizzati (Base 100)",
            labels={'value': 'Prezzo (Base 100)', 'index': 'Data'}
        )
        
        fig.update_layout(
            legend_title="Simbolo",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Si è verificato un errore nell'analisi di correlazione: {str(e)}")


def display_monte_carlo_simulation():
    """
    Display Monte Carlo simulation for the portfolio
    """
    st.subheader("Simulazione Monte Carlo")
    
    st.markdown("""
    La simulazione Monte Carlo genera migliaia di scenari possibili per l'andamento futuro del tuo portafoglio
    in base alla volatilità storica e alla correlazione tra gli asset.
    """)
    
    # Simulation parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days = st.slider(
            "Giorni di Simulazione",
            min_value=30,
            max_value=365 * 3,
            value=365,
            step=30
        )
    
    with col2:
        sims = st.slider(
            "Numero di Simulazioni",
            min_value=100,
            max_value=1000,
            value=500,
            step=100
        )
    
    with col3:
        confidence = st.slider(
            "Livello di Confidenza (%)",
            min_value=75,
            max_value=99,
            value=95,
            step=5
        )
    
    # Time period for historical data
    hist_period = st.select_slider(
        "Dati Storici da Utilizzare",
        options=["3m", "6m", "1y", "2y", "3y"],
        format_func=lambda x: {
            "3m": "3 Mesi",
            "6m": "6 Mesi",
            "1y": "1 Anno",
            "2y": "2 Anni",
            "3y": "3 Anni"
        }.get(x, x),
        value="1y"
    )
    
    # Run simulation
    if st.button("Esegui Simulazione Monte Carlo", use_container_width=True, type="primary"):
        with st.spinner("Esecuzione della simulazione in corso..."):
            try:
                # Get portfolio data
                portfolio_data = {}
                portfolio_weights = {}
                total_value = 0
                
                for symbol, position in st.session_state.portfolio.items():
                    try:
                        # Get stock data
                        stock = yf.Ticker(symbol)
                        hist = stock.history(period=hist_period)
                        
                        if not hist.empty:
                            portfolio_data[symbol] = hist
                            
                            # Calculate position value
                            shares = position['shares']
                            current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                            position_value = shares * current_price
                            
                            total_value += position_value
                            portfolio_weights[symbol] = position_value
                    except:
                        st.warning(f"Could not retrieve data for {symbol}")
                
                # Normalize weights
                portfolio_weights = {k: v/total_value for k, v in portfolio_weights.items()}
                
                if not portfolio_data:
                    st.error("No data available for any stocks in the portfolio")
                    return
                
                # Run Monte Carlo simulation
                simulation_results = run_monte_carlo(portfolio_data, portfolio_weights, days, sims)
                
                # Plot results
                fig = plot_monte_carlo_results(simulation_results, total_value, days, confidence)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics
                display_monte_carlo_stats(simulation_results, total_value, confidence)
                
            except Exception as e:
                st.error(f"Si è verificato un errore nella simulazione: {str(e)}")


def run_monte_carlo(portfolio_data, weights, days, sims):
    """
    Run Monte Carlo simulation for a portfolio
    
    Parameters:
    portfolio_data (dict): Dictionary of historical price data for each asset
    weights (dict): Dictionary of portfolio weights
    days (int): Number of days to simulate
    sims (int): Number of simulations to run
    
    Returns:
    np.array: Array of simulation results
    """
    # Create returns DataFrame
    returns_data = {}
    
    for symbol, hist in portfolio_data.items():
        if 'Close' in hist.columns and len(hist) > 1:
            returns_data[symbol] = hist['Close'].pct_change().dropna()
    
    # Create combined returns DataFrame
    returns_df = pd.DataFrame(returns_data)
    
    # Filter weights to match available returns
    weights = {k: v for k, v in weights.items() if k in returns_df.columns}
    
    # Re-normalize weights
    weight_sum = sum(weights.values())
    weights = {k: v/weight_sum for k, v in weights.items()}
    
    # Convert weights to array in same order as returns_df columns
    weights_arr = np.array([weights.get(col, 0) for col in returns_df.columns])
    
    # Calculate portfolio mean return and covariance
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    
    # Generate correlated random returns
    np.random.seed(42)  # For reproducibility
    
    # Perform Cholesky decomposition
    chol_matrix = np.linalg.cholesky(cov_matrix)
    
    # Initialize simulation results array
    results = np.zeros((sims, days+1))
    results[:, 0] = 1  # Start with $1
    
    # Run simulations
    for i in range(sims):
        for j in range(1, days+1):
            # Generate random normal samples
            random_returns = np.random.normal(0, 1, len(mean_returns))
            
            # Generate correlated returns
            correlated_returns = mean_returns + chol_matrix.dot(random_returns)
            
            # Calculate portfolio return
            portfolio_return = weights_arr.dot(correlated_returns)
            
            # Update portfolio value
            results[i, j] = results[i, j-1] * (1 + portfolio_return)
    
    return results


def plot_monte_carlo_results(results, initial_value, days, confidence):
    """
    Plot Monte Carlo simulation results
    
    Parameters:
    results (np.array): Array of simulation results
    initial_value (float): Initial portfolio value
    days (int): Number of days simulated
    confidence (int): Confidence level for prediction interval
    
    Returns:
    go.Figure: Plotly figure with simulation results
    """
    # Calculate dates
    dates = pd.date_range(start=pd.Timestamp.today(), periods=days+1)
    
    # Scale results to initial value
    scaled_results = results * initial_value
    
    # Create figure
    fig = go.Figure()
    
    # Add median line
    median_line = np.median(scaled_results, axis=0)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=median_line,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Valore Mediano'
        )
    )
    
    # Add confidence interval
    lower_percentile = (100 - confidence) / 2
    upper_percentile = 100 - lower_percentile
    
    lower_bound = np.percentile(scaled_results, lower_percentile, axis=0)
    upper_bound = np.percentile(scaled_results, upper_percentile, axis=0)
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            name=f'Intervallo di Confidenza {confidence}%'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.1)',
            name=f'Intervallo di Confidenza {confidence}%'
        )
    )
    
    # Add a few sample paths
    for i in range(min(10, results.shape[0])):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=scaled_results[i, :],
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.2)'),
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f'Simulazione Monte Carlo del Portafoglio ({days} giorni)',
        xaxis_title='Data',
        yaxis_title='Valore del Portafoglio ($)',
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(tickprefix='$')
    )
    
    return fig


def display_monte_carlo_stats(results, initial_value, confidence):
    """
    Display statistics from Monte Carlo simulation
    
    Parameters:
    results (np.array): Array of simulation results
    initial_value (float): Initial portfolio value
    confidence (int): Confidence level for prediction interval
    """
    # Scale results to initial value
    scaled_results = results * initial_value
    
    # Get final values
    final_values = scaled_results[:, -1]
    
    # Calculate statistics
    median_value = np.median(final_values)
    mean_value = np.mean(final_values)
    std_dev = np.std(final_values)
    min_value = np.min(final_values)
    max_value = np.max(final_values)
    
    # Calculate confidence interval
    lower_percentile = (100 - confidence) / 2
    upper_percentile = 100 - lower_percentile
    
    lower_bound = np.percentile(final_values, lower_percentile)
    upper_bound = np.percentile(final_values, upper_percentile)
    
    # Calculate probability of profit/loss
    prob_profit = (final_values > initial_value).mean() * 100
    prob_loss = 100 - prob_profit
    
    # Potential returns
    median_return = (median_value / initial_value - 1) * 100
    mean_return = (mean_value / initial_value - 1) * 100
    lower_return = (lower_bound / initial_value - 1) * 100
    upper_return = (upper_bound / initial_value - 1) * 100
    
    # Display statistics
    st.subheader("Statistiche della Simulazione")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Valore Iniziale",
            value=f"${initial_value:,.2f}"
        )
    
    with col2:
        st.metric(
            label="Valore Mediano Simulato",
            value=f"${median_value:,.2f}",
            delta=f"{median_return:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Intervallo di Confidenza",
            value=f"${lower_bound:,.2f} - ${upper_bound:,.2f}",
            delta=f"{lower_return:.2f}% - {upper_return:.2f}%"
        )
    
    # Additional stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Probabilità di Guadagno",
            value=f"{prob_profit:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Probabilità di Perdita",
            value=f"{prob_loss:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Valore Minimo Simulato",
            value=f"${min_value:,.2f}",
            delta=f"{(min_value / initial_value - 1) * 100:.2f}%"
        )
    
    with col4:
        st.metric(
            label="Valore Massimo Simulato",
            value=f"${max_value:,.2f}",
            delta=f"{(max_value / initial_value - 1) * 100:.2f}%"
        )
    
    # Display return distribution
    st.subheader("Distribuzione dei Rendimenti Simulati")
    
    # Calculate returns
    returns = (final_values / initial_value - 1) * 100
    
    # Create histogram
    fig = px.histogram(
        returns,
        nbins=50,
        labels={'value': 'Rendimento (%)'},
        title='Distribuzione dei Rendimenti Simulati',
        color_discrete_sequence=['blue']
    )
    
    # Add vertical lines for key statistics
    fig.add_vline(
        x=median_return,
        line_dash="solid",
        line_color="green",
        annotation_text=f"Mediana: {median_return:.2f}%",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=lower_return,
        line_dash="dash",
        line_color="red",
        annotation_text=f"{lower_percentile}%: {lower_return:.2f}%",
        annotation_position="top left"
    )
    
    fig.add_vline(
        x=upper_return,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"{upper_percentile}%: {upper_return:.2f}%",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="black",
        annotation_text="Break-even",
        annotation_position="bottom right"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Rendimento (%)",
        yaxis_title="Frequenza",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display advice based on simulation
    st.subheader("Analisi del Rischio")
    
    if prob_profit >= 80:
        st.success(f"""
        **Prospettiva Molto Positiva**
        
        La simulazione indica un'alta probabilità di guadagno ({prob_profit:.1f}%), con un rendimento mediano previsto del {median_return:.2f}%.
        
        Il portafoglio mostra una buona resistenza al rischio, con un intervallo di confidenza al {confidence}% tra {lower_return:.2f}% e {upper_return:.2f}%.
        """)
    elif prob_profit >= 60:
        st.info(f"""
        **Prospettiva Moderatamente Positiva**
        
        La simulazione indica una probabilità di guadagno del {prob_profit:.1f}%, con un rendimento mediano previsto del {median_return:.2f}%.
        
        Il portafoglio mostra una moderata esposizione al rischio, con un intervallo di confidenza al {confidence}% tra {lower_return:.2f}% e {upper_return:.2f}%.
        """)
    elif prob_profit >= 40:
        st.warning(f"""
        **Prospettiva Incerta**
        
        La simulazione indica una probabilità di guadagno del {prob_profit:.1f}%, con un rendimento mediano previsto del {median_return:.2f}%.
        
        Il portafoglio mostra una significativa esposizione al rischio, con un intervallo di confidenza al {confidence}% tra {lower_return:.2f}% e {upper_return:.2f}%.
        
        Potresti considerare di rivedere la composizione del portafoglio per migliorare il profilo rischio-rendimento.
        """)
    else:
        st.error(f"""
        **Prospettiva Sfavorevole**
        
        La simulazione indica una bassa probabilità di guadagno ({prob_profit:.1f}%), con un rendimento mediano previsto del {median_return:.2f}%.
        
        Il portafoglio mostra un'alta esposizione al rischio, con un intervallo di confidenza al {confidence}% tra {lower_return:.2f}% e {upper_return:.2f}%.
        
        Sarebbe consigliabile rivedere la strategia di investimento e la composizione del portafoglio.
        """)
def display_compound_interest_calculator():
    """
    Display a compound interest calculator integrated with portfolio data
    """
    st.subheader("Calcolatore di Interesse Composto")
    
    st.markdown("""
    Questo strumento ti permette di calcolare la crescita potenziale del tuo investimento nel tempo, 
    tenendo conto dell'interesse composto. Puoi partire dal valore attuale del tuo portafoglio
    o inserire un valore personalizzato.
    """)
    
    # Get current portfolio value if available
    total_portfolio_value = 0
    if st.session_state.portfolio:
        for symbol, position in st.session_state.portfolio.items():
            try:
                shares = position['shares']
                stock = yf.Ticker(symbol)
                info = stock.info
                current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
                total_portfolio_value += shares * current_price
            except Exception as e:
                st.warning(f"Errore nel recuperare il prezzo attuale per {symbol}: {str(e)}")
    
    # Create columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dati Iniziali")
        
        # Option to use current portfolio value or custom value
        use_portfolio_value = False
        if total_portfolio_value > 0:
            use_portfolio_value = st.checkbox(
                "Usa il valore attuale del portafoglio", 
                value=True,
                help="Deseleziona per inserire manualmente un capitale iniziale diverso"
            )
            
        if use_portfolio_value and total_portfolio_value > 0:
            initial_investment = total_portfolio_value
            st.info(f"Valore attuale del portafoglio: €{total_portfolio_value:,.2f}")
        else:
            initial_investment = st.number_input(
                "Investimento iniziale (€)",
                min_value=0.0,
                value=10000.0,
                step=1000.0,
                format="%.2f",
                help="Inserisci l'importo iniziale con cui vuoi iniziare l'investimento"
            )
        
        monthly_contribution = st.number_input(
            "Contributo mensile (€)",
            min_value=0.0,
            value=500.0,
            step=100.0,
            format="%.2f"
        )
        
        years = st.slider(
            "Durata (anni)",
            min_value=1,
            max_value=50,
            value=20,
            step=1
        )
        
        interest_rate = st.slider(
            "Tasso di interesse annuo (%)",
            min_value=0.0,
            max_value=15.0,
            value=7.0,
            step=0.1,
            format="%.1f"
        )
        
        # Aggiungiamo l'opzione per i dividendi
        dividend_yield = st.slider(
            "Percentuale dividendi annui (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.1f",
            help="Percentuale media dei dividendi annui distribuiti dalle azioni nel portafoglio"
        )
        
        # Opzione per reinvestire i dividendi
        reinvest_dividends = st.checkbox("Reinvestire i dividendi", value=True,
            help="Se selezionato, i dividendi vengono reinvestiti automaticamente aumentando il capitale")
        
        compound_frequency = st.selectbox(
            "Frequenza di capitalizzazione",
            options=["Mensile", "Trimestrale", "Semestrale", "Annuale"],
            index=0
        )
        
        frequency_map = {
            "Mensile": 12,
            "Trimestrale": 4,
            "Semestrale": 2,
            "Annuale": 1
        }
        n = frequency_map[compound_frequency]
    
    with col2:
        st.subheader("Risultati")
        
        # Calculate compound interest with dividends
        def compound_interest_with_dividends(P, PMT, r, d, t, n, reinvest=True):
            """
            Calculate compound interest with regular contributions and dividends
            
            Parameters:
            P (float): Initial principal
            PMT (float): Regular monthly contribution
            r (float): Annual interest rate (%)
            d (float): Annual dividend yield (%)
            t (int): Time in years
            n (int): Compounding frequency per year
            reinvest (bool): Whether to reinvest dividends
            
            Returns:
            tuple: (total future value, future value of initial investment, future value of contributions, total dividends)
            """
            r_decimal = r / 100
            d_decimal = d / 100
            
            # Total effective rate if dividends are reinvested
            effective_rate = r_decimal
            if reinvest:
                effective_rate += d_decimal
            
            # Calculate future value of initial investment
            FV_initial = P * (1 + effective_rate/n)**(n*t)
            
            # Handle the case where effective rate is close to zero
            if abs(effective_rate) < 0.0001:
                FV_contributions = PMT * n * t
            else:
                FV_contributions = PMT * (((1 + effective_rate/n)**(n*t) - 1) / (effective_rate/n))
            
            total = FV_initial + FV_contributions
            
            # Calculate dividends if not reinvested
            dividends = 0
            if not reinvest:
                # Simple approximation of dividends (not exact but reasonable)
                avg_balance = (P + total) / 2  # Average balance over time
                dividends = avg_balance * d_decimal * t
                total += dividends
            
            return round(total, 2), round(FV_initial, 2), round(FV_contributions, 2), round(dividends if not reinvest else 0, 2)
        
        future_value, fv_initial, fv_contributions, dividends_not_reinvested = compound_interest_with_dividends(
            initial_investment, 
            monthly_contribution, 
            interest_rate,
            dividend_yield,
            years, 
            n,
            reinvest_dividends
        )
        
        # Calculate values for each year for the chart
        years_range = list(range(years + 1))
        values = []
        values_without_dividends = []
        
        # Calculate values with and without dividends for comparison
        for t in years_range:
            if t == 0:
                # At year 0, it's just the initial investment
                values.append(initial_investment)
                values_without_dividends.append(initial_investment)
                continue
                
            # With dividends
            fv, _, _, _ = compound_interest_with_dividends(
                initial_investment, 
                monthly_contribution, 
                interest_rate,
                dividend_yield,
                t, 
                n,
                reinvest_dividends
            )
            values.append(fv)
            
            # Without dividends (for comparison)
            fv_no_div, _, _, _ = compound_interest_with_dividends(
                initial_investment, 
                monthly_contribution, 
                interest_rate,
                0,  # No dividends
                t, 
                n,
                True
            )
            values_without_dividends.append(fv_no_div)
        
        # Calculate total growth from dividends
        growth_from_dividends = future_value - values_without_dividends[-1]
        
        # Calculate total investment
        total_investment = initial_investment + (monthly_contribution * 12 * years)
        
        # Calculate total returns
        total_returns = future_value - total_investment
        
        # Display metrics
        st.metric(
            label="Valore Finale dell'Investimento",
            value=f"€{future_value:,.2f}",
            delta=f"€{total_returns:,.2f}"
        )
        
        # Display breakdown
        dividend_text = ""
        if reinvest_dividends:
            dividend_text = f"""
            - Crescita aggiuntiva da dividendi reinvestiti: €{growth_from_dividends:,.2f} ({(growth_from_dividends/future_value*100):.1f}% del totale)
            """
        else:
            dividend_text = f"""
            - Dividendi accumulati (non reinvestiti): €{dividends_not_reinvested:,.2f}
            """
        
        st.markdown(f"""
        **Dettaglio del Valore Finale:**
        - Capitale iniziale investito: €{initial_investment:,.2f}
        - Totale contributi mensili: €{monthly_contribution * 12 * years:,.2f}
        - Rendimento totale: €{total_returns:,.2f} ({(total_returns/total_investment*100):.1f}%)
        {dividend_text}
        
        **Crescita del Capitale Iniziale:** €{fv_initial:,.2f} (x{fv_initial/initial_investment:.2f})
        """)
        
        # Add a download button for the calculation results
        result_df = pd.DataFrame({
            'Anno': years_range,
            'Valore con Dividendi': values,
            'Valore senza Dividendi': values_without_dividends,
            'Impatto Dividendi': [v1 - v2 for v1, v2 in zip(values, values_without_dividends)]
        })
        
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Scarica i risultati (CSV)",
            data=csv,
            file_name="risultati_interesse_composto.csv",
            mime="text/csv"
        )
    
    # Create a chart showing growth over time
    st.subheader("Crescita dell'Investimento nel Tempo")
    
    # Prepare data for the chart
    chart_data = pd.DataFrame({
        'Anno': years_range,
        'Con Dividendi': values,
        'Senza Dividendi': values_without_dividends
    })
    
    # Create a line chart showing both scenarios
    fig = px.line(
        chart_data,
        x='Anno',
        y=['Con Dividendi', 'Senza Dividendi'],
        title='Impatto dei Dividendi sulla Crescita del Capitale',
        labels={'value': 'Valore (€)', 'Anno': 'Anno', 'variable': 'Scenario'},
        markers=True,
        color_discrete_map={
            'Con Dividendi': 'green',
            'Senza Dividendi': 'blue'
        }
    )
    
    # Add initial investment as horizontal line
    fig.add_shape(
        type="line",
        x0=0,
        y0=initial_investment,
        x1=years,
        y1=initial_investment,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add annotation for initial investment
    fig.add_annotation(
        x=0,
        y=initial_investment,
        text=f"Investimento iniziale: €{initial_investment:,.0f}",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-30
    )
    
    # Calculate total contributions
    total_contributions = initial_investment + (monthly_contribution * 12 * years)
    
    # Add total contributions as horizontal line
    fig.add_shape(
        type="line",
        x0=0,
        y0=total_contributions,
        x1=years,
        y1=total_contributions,
        line=dict(color="green", width=2, dash="dash"),
    )
    
    # Add annotation for total contributions
    fig.add_annotation(
        x=years/2,
        y=total_contributions,
        text=f"Capitale investito totale: €{total_contributions:,.0f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Anni",
        yaxis_title="Valore (€)",
        yaxis_tickformat=",.0f",
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add an explanation section
    st.markdown("""
    ### Come Funziona l'Interesse Composto
    
    L'interesse composto è il concetto per cui gli interessi generati da un capitale iniziale vengono reinvestiti
    e generano a loro volta ulteriori interessi. Questo effetto, nel tempo, crea una crescita esponenziale del capitale.
    
    #### Formula Utilizzata
    
    La formula generale per calcolare il valore futuro (FV) è:
    
    FV = P × (1 + (r+d)/n)^(n×t) + PMT × [(1 + (r+d)/n)^(n×t) - 1] / ((r+d)/n)
    
    Dove:
    - P = Investimento iniziale
    - r = Tasso di interesse annuo (in decimale)
    - d = Rendimento dividendi annuo (in decimale)
    - n = Frequenza di capitalizzazione all'anno
    - t = Durata in anni
    - PMT = Contributo mensile
    
    #### Impatto dei Dividendi
    
    I dividendi hanno un impatto significativo sulla crescita di lungo termine di un investimento:
    
    - **Dividendi reinvestiti**: Quando i dividendi vengono reinvestiti, entrano immediatamente nel ciclo dell'interesse composto,
      accelerando la crescita del capitale (come mostrato nel grafico).
    
    - **Dividendi non reinvestiti**: Se i dividendi non vengono reinvestiti, possono comunque fornire un flusso di reddito
      regolare, ma non contribuiscono alla crescita esponenziale del capitale principale.
    
    #### Fattori Critici per Massimizzare i Rendimenti
    
    1. **Iniziare presto**: Il tempo è il fattore più importante nell'interesse composto.
    2. **Contribuire regolarmente**: Anche piccole somme aggiunte regolarmente fanno una grande differenza.
    3. **Reinvestire i dividendi**: Reinvestire automaticamente aumenta l'effetto dell'interesse composto.
    4. **Minimizzare le commissioni**: Le commissioni riducono il capitale che può generare interessi futuri.
    5. **Ottimizzare il tasso di rendimento**: Anche piccole differenze nei tassi hanno impatti enormi nel lungo periodo.
    6. **Selezionare titoli con dividendi**: Investire in azioni che pagano dividendi può aumentare significativamente
       il rendimento complessivo, specialmente su orizzonti temporali lunghi.
    """)