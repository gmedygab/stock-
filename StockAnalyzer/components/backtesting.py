import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import math

def display_backtesting():
    """
    Display backtesting functionality for testing trading strategies
    """
    # Access translation function if available in session state
    t = st.session_state.get('translate', lambda x: x)
    
    st.title(t("Backtesting delle Strategie di Trading"))
    
    # Description
    st.write(t("Testa la tua strategia di trading su dati storici per valutarne le prestazioni. " +
               "Il backtesting ti permette di simulare una strategia di trading su dati storici " +
               "per vedere come avrebbe performato se fosse stata utilizzata in passato."))
    
    # Sidebar for parameters
    st.sidebar.header(t("Parametri di Backtesting"))
    
    # Select ticker
    ticker = st.sidebar.text_input(t("Simbolo"), "AAPL").upper()
    
    # Time period
    start_date = st.sidebar.date_input(t("Data di inizio"), datetime.now() - timedelta(days=365*3))
    end_date = st.sidebar.date_input(t("Data di fine"), datetime.now())
    
    # Initial capital
    initial_capital = st.sidebar.number_input(t("Capitale iniziale ($)"), value=10000, min_value=1000, step=1000)
    
    # Commission per trade
    commission = st.sidebar.number_input(t("Commissione per operazione ($)"), value=5.0, min_value=0.0, step=1.0)
    
    # Strategy selection
    strategy_options = {
        "sma_crossover": t("Incrocio delle medie mobili (SMA)"),
        "rsi_strategy": t("Strategia RSI (Ipercomprato/Ipervenduto)"),
        "bollinger_bands": t("Bande di Bollinger"),
        "macd_strategy": t("Strategia MACD"),
        "dual_ma_strategy": t("Doppia Media Mobile"),
    }
    
    selected_strategy = st.sidebar.selectbox(
        t("Seleziona strategia"),
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x]
    )
    
    # Initialize strategy_params to avoid undefined variable
    strategy_params = {}
    
    # Strategy-specific parameters
    with st.sidebar.expander(t("Parametri della strategia")):
        if selected_strategy == "sma_crossover":
            fast_period = st.number_input(t("Periodo SMA veloce"), value=20, min_value=1, max_value=100)
            slow_period = st.number_input(t("Periodo SMA lenta"), value=50, min_value=10, max_value=200)
            strategy_params = {"fast_period": fast_period, "slow_period": slow_period}
            
        elif selected_strategy == "rsi_strategy":
            rsi_period = st.number_input(t("Periodo RSI"), value=14, min_value=1, max_value=30)
            rsi_overbought = st.number_input(t("Livello di ipercomprato"), value=70, min_value=60, max_value=90)
            rsi_oversold = st.number_input(t("Livello di ipervenduto"), value=30, min_value=10, max_value=40)
            strategy_params = {"rsi_period": rsi_period, "rsi_overbought": rsi_overbought, "rsi_oversold": rsi_oversold}
            
        elif selected_strategy == "bollinger_bands":
            bb_period = st.number_input(t("Periodo BB"), value=20, min_value=5, max_value=50)
            bb_std = st.number_input(t("Deviazione standard"), value=2.0, min_value=1.0, max_value=4.0, step=0.1)
            strategy_params = {"bb_period": bb_period, "bb_std": bb_std}
            
        elif selected_strategy == "macd_strategy":
            macd_fast = st.number_input(t("MACD Periodo veloce"), value=12, min_value=5, max_value=30)
            macd_slow = st.number_input(t("MACD Periodo lento"), value=26, min_value=15, max_value=50)
            macd_signal = st.number_input(t("MACD Periodo segnale"), value=9, min_value=5, max_value=20)
            strategy_params = {"macd_fast": macd_fast, "macd_slow": macd_slow, "macd_signal": macd_signal}
            
        elif selected_strategy == "dual_ma_strategy":
            short_ma = st.number_input(t("MA corta (giorni)"), value=5, min_value=1, max_value=30)
            long_ma = st.number_input(t("MA lunga (giorni)"), value=20, min_value=10, max_value=50)
            strategy_params = {"short_ma": short_ma, "long_ma": long_ma}
    
    # Button to run backtest
    if st.sidebar.button(t("Esegui Backtest")):
        # Show spinner during data loading
        with st.spinner(t("Caricamento dei dati e calcolo del backtest...")):
            # Get stock data
            try:
                data = get_stock_data(ticker, start_date, end_date)
                
                if data.empty:
                    st.error(t(f"Nessun dato trovato per {ticker}. Controlla il simbolo e riprova."))
                    return
                
                # Run the selected strategy
                if selected_strategy == "sma_crossover":
                    result = backtest_sma_crossover(data, initial_capital, commission, strategy_params)
                elif selected_strategy == "rsi_strategy":
                    result = backtest_rsi_strategy(data, initial_capital, commission, strategy_params)
                elif selected_strategy == "bollinger_bands":
                    result = backtest_bollinger_bands(data, initial_capital, commission, strategy_params)
                elif selected_strategy == "macd_strategy":
                    result = backtest_macd_strategy(data, initial_capital, commission, strategy_params)
                elif selected_strategy == "dual_ma_strategy":
                    result = backtest_dual_ma_strategy(data, initial_capital, commission, strategy_params)
                
                # Display the results
                display_backtest_results(ticker, data, result, strategy_options[selected_strategy])
                
            except Exception as e:
                st.error(t(f"Errore durante l'esecuzione del backtest: {str(e)}"))
                st.info(t("Suggerimento: verifica che il simbolo sia corretto e che ci siano dati disponibili per il periodo selezionato."))
    
    # Help section
    with st.expander(t("Aiuto: Come funziona il backtesting?")):
        st.markdown(f"""
        ### {t("Come funziona il backtesting?")}
        
        {t("Il backtesting è un metodo per testare una strategia di trading su dati storici per valutare come avrebbe performato in passato. Sebbene i risultati passati non garantiscano performance future, il backtesting può aiutare a validare le strategie prima di utilizzarle con denaro reale.")}
        
        ### {t("Come interpretare i risultati?")}
        
        {t("I risultati del backtesting includono:")}
        - **{t("Rendimento totale")}**: {t("Il ritorno percentuale dell'investimento")}
        - **{t("Rendimento annualizzato")}**: {t("Il ritorno annualizzato")}
        - **{t("Sharpe ratio")}**: {t("Una misura del rendimento aggiustato per il rischio")}
        - **{t("Drawdown massimo")}**: {t("La perdita massima dal picco al minimo")}
        - **{t("Numero di operazioni")}**: {t("Il numero totale di operazioni eseguite")}
        - **{t("Percentuale di operazioni vincenti")}**: {t("La percentuale di operazioni che hanno generato profitto")}
        
        ### {t("Limitazioni del backtesting")}
        
        {t("È importante considerare queste limitazioni:")}
        - {t("Il backtesting si basa su dati storici e non garantisce risultati futuri")}
        - {t("Non tiene conto di tutti i costi di transazione o slippage")}
        - {t("Può essere soggetto a overfitting (adattarsi troppo ai dati storici)")}
        - {t("Ogni strategia dovrebbe essere verificata in diversi scenari di mercato")}
        """)

def get_stock_data(symbol, start_date, end_date):
    """
    Fetch historical stock data for the given symbol and date range
    """
    # Format dates as strings
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Download data from yfinance
    data = yf.download(symbol, start=start_str, end=end_str)
    
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Ensure we have the required columns
    if 'Date' not in data.columns or 'Open' not in data.columns or 'Close' not in data.columns:
        return pd.DataFrame()
    
    return data

def calculate_metrics(backtest_result):
    """
    Calculate performance metrics for the backtest
    """
    # Access translation function if available in session state
    t = st.session_state.get('translate', lambda x: x)
    
    equity = backtest_result['equity']
    trades = backtest_result['trades']
    
    # Skip metrics calculation if no trades were made
    if len(trades) == 0:
        return {
            "total_return": 0,
            "annual_return": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
        }
    
    # Total return
    initial_capital = equity[0]
    final_capital = equity[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    
    # Annual return
    trading_days = len(equity)
    if trading_days > 1:
        annual_return = ((final_capital / initial_capital) ** (252 / trading_days) - 1) * 100
    else:
        annual_return = 0
    
    # Daily returns for Sharpe ratio
    daily_returns = []
    for i in range(1, len(equity)):
        daily_return = (equity[i] / equity[i-1]) - 1
        daily_returns.append(daily_return)
    
    # Sharpe ratio (assuming risk-free rate of 0% for simplicity)
    if len(daily_returns) > 0 and np.std(daily_returns) > 0:
        sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    max_dd = 0
    peak = equity[0]
    
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    
    max_drawdown = max_dd * 100
    
    # Trade statistics
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Profit factor (sum of profits / sum of losses)
    gross_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
    gross_loss = abs(sum(trade['profit'] for trade in trades if trade['profit'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }

def display_backtest_results(ticker, data, result, strategy_name):
    """
    Display the backtest results including charts and metrics
    """
    # Access translation function if available in session state
    t = st.session_state.get('translate', lambda x: x)
    
    # Calculate performance metrics
    metrics = calculate_metrics(result)
    
    # Display the results
    st.header(t("Risultati del Backtesting"))
    st.subheader(f"{ticker}: {strategy_name}")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Format metrics for display
    with col1:
        st.metric(t("Rendimento Totale"), f"{metrics['total_return']:.2f}%")
        st.metric(t("Sharpe Ratio"), f"{metrics['sharpe_ratio']:.2f}")
    
    with col2:
        st.metric(t("Rendimento Annualizzato"), f"{metrics['annual_return']:.2f}%")
        st.metric(t("Drawdown Massimo"), f"{metrics['max_drawdown']:.2f}%")
    
    with col3:
        st.metric(t("Numero di Operazioni"), f"{metrics['total_trades']}")
        st.metric(t("Operazioni Vincenti"), f"{metrics['win_rate']:.1f}%")
    
    with col4:
        st.metric(t("Profit Factor"), f"{metrics['profit_factor']:.2f}")
        initial_capital = result['equity'][0]
        final_capital = result['equity'][-1]
        st.metric(t("Capitale Finale"), f"${final_capital:.2f}", delta=f"{final_capital-initial_capital:.2f}")
    
    # Create equity curve chart
    st.subheader(t("Curva di Equity"))
    
    fig_equity = go.Figure()
    
    # Add equity curve
    fig_equity.add_trace(go.Scatter(
        x=data['Date'],
        y=result['equity'],
        mode='lines',
        name=t('Equity'),
        line=dict(color='blue', width=2)
    ))
    
    # Add buy and sell markers
    buy_dates = [trade['entry_date'] for trade in result['trades'] if 'entry_date' in trade]
    buy_prices = [trade['entry_price'] * trade['size'] for trade in result['trades'] if 'entry_price' in trade and 'size' in trade]
    
    sell_dates = [trade['exit_date'] for trade in result['trades'] if 'exit_date' in trade]
    sell_prices = [trade['exit_price'] * trade['size'] for trade in result['trades'] if 'exit_price' in trade and 'size' in trade]
    
    if buy_dates:
        fig_equity.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            name=t('Acquisto'),
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    if sell_dates:
        fig_equity.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            name=t('Vendita'),
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    # Update layout
    fig_equity.update_layout(
        title=t('Curva di Equity'),
        xaxis_title=t('Data'),
        yaxis_title=t('Valore ($)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Create price chart with buy/sell signals
    st.subheader(t("Grafico Prezzi con Segnali"))
    
    # Create figure with secondary y-axis
    fig_signals = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, 
                               row_heights=[0.7, 0.3])
    
    # Add price data to the first subplot
    fig_signals.add_trace(
        go.Candlestick(
            x=data['Date'],
            open=data['Open'], 
            high=data['High'],
            low=data['Low'], 
            close=data['Close'],
            name=t('Prezzo')
        ),
        row=1, col=1
    )
    
    # Add volume to the second subplot
    fig_signals.add_trace(
        go.Bar(
            x=data['Date'], 
            y=data['Volume'],
            name=t('Volume'),
            marker=dict(color='rgba(0,0,0,0.3)')
        ),
        row=2, col=1
    )
    
    # Add buy signals
    buy_dates = [trade['entry_date'] for trade in result['trades'] if 'entry_date' in trade]
    buy_prices = [trade['entry_price'] for trade in result['trades'] if 'entry_price' in trade]
    
    if buy_dates:
        fig_signals.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name=t('Acquisto'),
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
            row=1, col=1
        )
    
    # Add sell signals
    sell_dates = [trade['exit_date'] for trade in result['trades'] if 'exit_date' in trade]
    sell_prices = [trade['exit_price'] for trade in result['trades'] if 'exit_price' in trade]
    
    if sell_dates:
        fig_signals.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name=t('Vendita'),
                marker=dict(color='red', size=10, symbol='triangle-down')
            ),
            row=1, col=1
        )
    
    # Update layout
    fig_signals.update_layout(
        title=t('Grafico Prezzi con Segnali di Trading'),
        xaxis_title=t('Data'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    # Update y-axes
    fig_signals.update_yaxes(title_text=t("Prezzo ($)"), row=1, col=1)
    fig_signals.update_yaxes(title_text=t("Volume"), row=2, col=1)
    
    st.plotly_chart(fig_signals, use_container_width=True)
    
    # Display trade log
    st.subheader(t("Registro delle Operazioni"))
    
    if result['trades']:
        # Convert trades to DataFrame for display
        trades_df = pd.DataFrame(result['trades'])
        
        # Format dates if they exist
        date_columns = ['entry_date', 'exit_date']
        for col in date_columns:
            if col in trades_df.columns:
                trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime('%Y-%m-%d')
        
        # Translate column names
        column_translations = {
            'entry_date': t('Data di Entrata'),
            'entry_price': t('Prezzo di Entrata'),
            'exit_date': t('Data di Uscita'),
            'exit_price': t('Prezzo di Uscita'),
            'size': t('Quantità'),
            'profit': t('Profitto'),
            'pnl_pct': t('Rendimento %')
        }
        
        # Rename columns if they exist
        rename_dict = {col: column_translations[col] for col in trades_df.columns if col in column_translations}
        trades_df = trades_df.rename(columns=rename_dict)
        
        # Add percentage return column if not already present
        if t('Rendimento %') not in trades_df.columns and t('Prezzo di Entrata') in trades_df.columns and t('Prezzo di Uscita') in trades_df.columns:
            trades_df[t('Rendimento %')] = ((trades_df[t('Prezzo di Uscita')] / trades_df[t('Prezzo di Entrata')]) - 1) * 100
        
        # Format numeric columns
        for col in trades_df.columns:
            if trades_df[col].dtype in [np.float64, np.int64]:
                if col == t('Profitto') or col == t('Rendimento %'):
                    # Format with 2 decimal places and add % for percentage
                    if col == t('Rendimento %'):
                        trades_df[col] = trades_df[col].map('{:.2f}%'.format)
                    else:
                        trades_df[col] = trades_df[col].map('${:.2f}'.format)
        
        # Display the DataFrame
        st.dataframe(trades_df, use_container_width=True)
        
        # Display trades summary
        winning_trades = sum(1 for trade in result['trades'] if trade['profit'] > 0)
        losing_trades = sum(1 for trade in result['trades'] if trade['profit'] < 0)
        break_even_trades = sum(1 for trade in result['trades'] if trade['profit'] == 0)
        
        st.write(f"{t('Operazioni vincenti')}: {winning_trades}, {t('Operazioni perdenti')}: {losing_trades}, {t('Operazioni in pareggio')}: {break_even_trades}")
        
    else:
        st.info(t("Nessuna operazione eseguita durante il periodo di backtest."))

def backtest_sma_crossover(data, initial_capital, commission, params):
    """
    Backtest a simple moving average crossover strategy
    
    Parameters:
    data (pd.DataFrame): OHLC price data
    initial_capital (float): Starting capital
    commission (float): Commission per trade
    params (dict): Strategy parameters - fast_period and slow_period
    
    Returns:
    dict: Results of the backtest including equity curve and trades
    """
    # Extract parameters
    fast_period = params['fast_period']
    slow_period = params['slow_period']
    
    # Copy data to avoid modifying the original DataFrame
    df = data.copy()
    
    # Calculate moving averages
    df['fast_ma'] = df['Close'].rolling(window=fast_period).mean()
    df['slow_ma'] = df['Close'].rolling(window=slow_period).mean()
    
    # Drop NaN values (at the beginning due to rolling window)
    df = df.dropna()
    
    # Initialize variables
    position = 0  # 0: no position, 1: long position
    entry_price = 0
    entry_date = None
    equity = [initial_capital]
    current_capital = initial_capital
    trades = []
    
    # Iterate through data points
    for i in range(1, len(df)):
        current_date = df['Date'].iloc[i]
        current_price = df['Close'].iloc[i]
        
        # Check for buy signal (fast MA crosses above slow MA)
        if df['fast_ma'].iloc[i-1] < df['slow_ma'].iloc[i-1] and df['fast_ma'].iloc[i] >= df['slow_ma'].iloc[i] and position == 0:
            # Calculate position size (invest all capital)
            position_size = current_capital / current_price
            position_size = math.floor(position_size)  # Whole number of shares
            
            # Only take the trade if we can buy at least one share
            if position_size > 0:
                # Enter long position
                position = 1
                entry_price = current_price
                entry_date = current_date
                
                # Deduct commission
                current_capital -= commission
        
        # Check for sell signal (fast MA crosses below slow MA)
        elif df['fast_ma'].iloc[i-1] > df['slow_ma'].iloc[i-1] and df['fast_ma'].iloc[i] <= df['slow_ma'].iloc[i] and position == 1:
            # Exit long position
            exit_price = current_price
            
            # Calculate profit
            profit = (exit_price - entry_price) * position_size
            current_capital = (position_size * exit_price) - commission
            
            # Record the trade
            trade = {
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': current_date,
                'exit_price': exit_price,
                'size': position_size,
                'profit': profit,
                'pnl_pct': (exit_price / entry_price - 1) * 100
            }
            trades.append(trade)
            
            # Reset position
            position = 0
            entry_price = 0
            entry_date = None
        
        # Update equity curve
        if position == 0:
            equity.append(current_capital)
        else:
            # Mark-to-market - current value of position plus remaining cash
            equity.append((position_size * current_price) - commission)
    
    # If we still have an open position at the end, close it at the last price
    if position == 1:
        exit_price = df['Close'].iloc[-1]
        exit_date = df['Date'].iloc[-1]
        
        # Calculate profit
        profit = (exit_price - entry_price) * position_size
        current_capital = (position_size * exit_price) - commission
        
        # Record the trade
        trade = {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'size': position_size,
            'profit': profit,
            'pnl_pct': (exit_price / entry_price - 1) * 100
        }
        trades.append(trade)
        
        # Update equity for the last day
        equity[-1] = current_capital
    
    return {
        'equity': equity,
        'trades': trades,
    }

def backtest_rsi_strategy(data, initial_capital, commission, params):
    """
    Backtest a Relative Strength Index (RSI) strategy
    
    Parameters:
    data (pd.DataFrame): OHLC price data
    initial_capital (float): Starting capital
    commission (float): Commission per trade
    params (dict): Strategy parameters - rsi_period, rsi_overbought, rsi_oversold
    
    Returns:
    dict: Results of the backtest including equity curve and trades
    """
    # Extract parameters
    rsi_period = params['rsi_period']
    rsi_overbought = params['rsi_overbought']
    rsi_oversold = params['rsi_oversold']
    
    # Copy data to avoid modifying the original DataFrame
    df = data.copy()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values (at the beginning due to rolling window)
    df = df.dropna()
    
    # Initialize variables
    position = 0  # 0: no position, 1: long position
    entry_price = 0
    entry_date = None
    equity = [initial_capital]
    current_capital = initial_capital
    trades = []
    
    # Iterate through data points
    for i in range(1, len(df)):
        current_date = df['Date'].iloc[i]
        current_price = df['Close'].iloc[i]
        
        # Check for buy signal (RSI crosses above oversold level)
        if df['RSI'].iloc[i-1] < rsi_oversold and df['RSI'].iloc[i] >= rsi_oversold and position == 0:
            # Calculate position size (invest all capital)
            position_size = current_capital / current_price
            position_size = math.floor(position_size)  # Whole number of shares
            
            # Only take the trade if we can buy at least one share
            if position_size > 0:
                # Enter long position
                position = 1
                entry_price = current_price
                entry_date = current_date
                
                # Deduct commission
                current_capital -= commission
        
        # Check for sell signal (RSI crosses above overbought level)
        elif df['RSI'].iloc[i-1] < rsi_overbought and df['RSI'].iloc[i] >= rsi_overbought and position == 1:
            # Exit long position
            exit_price = current_price
            
            # Calculate profit
            profit = (exit_price - entry_price) * position_size
            current_capital = (position_size * exit_price) - commission
            
            # Record the trade
            trade = {
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': current_date,
                'exit_price': exit_price,
                'size': position_size,
                'profit': profit,
                'pnl_pct': (exit_price / entry_price - 1) * 100
            }
            trades.append(trade)
            
            # Reset position
            position = 0
            entry_price = 0
            entry_date = None
        
        # Update equity curve
        if position == 0:
            equity.append(current_capital)
        else:
            # Mark-to-market - current value of position plus remaining cash
            equity.append((position_size * current_price) - commission)
    
    # If we still have an open position at the end, close it at the last price
    if position == 1:
        exit_price = df['Close'].iloc[-1]
        exit_date = df['Date'].iloc[-1]
        
        # Calculate profit
        profit = (exit_price - entry_price) * position_size
        current_capital = (position_size * exit_price) - commission
        
        # Record the trade
        trade = {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'size': position_size,
            'profit': profit,
            'pnl_pct': (exit_price / entry_price - 1) * 100
        }
        trades.append(trade)
        
        # Update equity for the last day
        equity[-1] = current_capital
    
    return {
        'equity': equity,
        'trades': trades,
    }

def backtest_bollinger_bands(data, initial_capital, commission, params):
    """
    Backtest a Bollinger Bands strategy
    
    Parameters:
    data (pd.DataFrame): OHLC price data
    initial_capital (float): Starting capital
    commission (float): Commission per trade
    params (dict): Strategy parameters - bb_period, bb_std
    
    Returns:
    dict: Results of the backtest including equity curve and trades
    """
    # Extract parameters
    bb_period = params['bb_period']
    bb_std = params['bb_std']
    
    # Copy data to avoid modifying the original DataFrame
    df = data.copy()
    
    # Calculate Bollinger Bands
    df['MA'] = df['Close'].rolling(window=bb_period).mean()
    df['StdDev'] = df['Close'].rolling(window=bb_period).std()
    df['UpperBB'] = df['MA'] + (df['StdDev'] * bb_std)
    df['LowerBB'] = df['MA'] - (df['StdDev'] * bb_std)
    
    # Drop NaN values (at the beginning due to rolling window)
    df = df.dropna()
    
    # Initialize variables
    position = 0  # 0: no position, 1: long position
    entry_price = 0
    entry_date = None
    equity = [initial_capital]
    current_capital = initial_capital
    trades = []
    
    # Iterate through data points
    for i in range(1, len(df)):
        current_date = df['Date'].iloc[i]
        current_price = df['Close'].iloc[i]
        
        # Check for buy signal (price crosses below lower band)
        if df['Close'].iloc[i-1] > df['LowerBB'].iloc[i-1] and df['Close'].iloc[i] <= df['LowerBB'].iloc[i] and position == 0:
            # Calculate position size (invest all capital)
            position_size = current_capital / current_price
            position_size = math.floor(position_size)  # Whole number of shares
            
            # Only take the trade if we can buy at least one share
            if position_size > 0:
                # Enter long position
                position = 1
                entry_price = current_price
                entry_date = current_date
                
                # Deduct commission
                current_capital -= commission
        
        # Check for sell signal (price crosses above upper band)
        elif df['Close'].iloc[i-1] < df['UpperBB'].iloc[i-1] and df['Close'].iloc[i] >= df['UpperBB'].iloc[i] and position == 1:
            # Exit long position
            exit_price = current_price
            
            # Calculate profit
            profit = (exit_price - entry_price) * position_size
            current_capital = (position_size * exit_price) - commission
            
            # Record the trade
            trade = {
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': current_date,
                'exit_price': exit_price,
                'size': position_size,
                'profit': profit,
                'pnl_pct': (exit_price / entry_price - 1) * 100
            }
            trades.append(trade)
            
            # Reset position
            position = 0
            entry_price = 0
            entry_date = None
        
        # Update equity curve
        if position == 0:
            equity.append(current_capital)
        else:
            # Mark-to-market - current value of position plus remaining cash
            equity.append((position_size * current_price) - commission)
    
    # If we still have an open position at the end, close it at the last price
    if position == 1:
        exit_price = df['Close'].iloc[-1]
        exit_date = df['Date'].iloc[-1]
        
        # Calculate profit
        profit = (exit_price - entry_price) * position_size
        current_capital = (position_size * exit_price) - commission
        
        # Record the trade
        trade = {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'size': position_size,
            'profit': profit,
            'pnl_pct': (exit_price / entry_price - 1) * 100
        }
        trades.append(trade)
        
        # Update equity for the last day
        equity[-1] = current_capital
    
    return {
        'equity': equity,
        'trades': trades,
    }

def backtest_macd_strategy(data, initial_capital, commission, params):
    """
    Backtest a MACD (Moving Average Convergence Divergence) strategy
    
    Parameters:
    data (pd.DataFrame): OHLC price data
    initial_capital (float): Starting capital
    commission (float): Commission per trade
    params (dict): Strategy parameters - macd_fast, macd_slow, macd_signal
    
    Returns:
    dict: Results of the backtest including equity curve and trades
    """
    # Extract parameters
    macd_fast = params['macd_fast']
    macd_slow = params['macd_slow']
    macd_signal = params['macd_signal']
    
    # Copy data to avoid modifying the original DataFrame
    df = data.copy()
    
    # Calculate MACD
    df['ema_fast'] = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    df['ema_slow'] = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']
    
    # Drop NaN values (at the beginning due to EMA calculation)
    df = df.dropna()
    
    # Initialize variables
    position = 0  # 0: no position, 1: long position
    entry_price = 0
    entry_date = None
    equity = [initial_capital]
    current_capital = initial_capital
    trades = []
    
    # Iterate through data points
    for i in range(1, len(df)):
        current_date = df['Date'].iloc[i]
        current_price = df['Close'].iloc[i]
        
        # Check for buy signal (MACD crosses above signal line)
        if df['macd'].iloc[i-1] < df['signal'].iloc[i-1] and df['macd'].iloc[i] >= df['signal'].iloc[i] and position == 0:
            # Calculate position size (invest all capital)
            position_size = current_capital / current_price
            position_size = math.floor(position_size)  # Whole number of shares
            
            # Only take the trade if we can buy at least one share
            if position_size > 0:
                # Enter long position
                position = 1
                entry_price = current_price
                entry_date = current_date
                
                # Deduct commission
                current_capital -= commission
        
        # Check for sell signal (MACD crosses below signal line)
        elif df['macd'].iloc[i-1] > df['signal'].iloc[i-1] and df['macd'].iloc[i] <= df['signal'].iloc[i] and position == 1:
            # Exit long position
            exit_price = current_price
            
            # Calculate profit
            profit = (exit_price - entry_price) * position_size
            current_capital = (position_size * exit_price) - commission
            
            # Record the trade
            trade = {
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': current_date,
                'exit_price': exit_price,
                'size': position_size,
                'profit': profit,
                'pnl_pct': (exit_price / entry_price - 1) * 100
            }
            trades.append(trade)
            
            # Reset position
            position = 0
            entry_price = 0
            entry_date = None
        
        # Update equity curve
        if position == 0:
            equity.append(current_capital)
        else:
            # Mark-to-market - current value of position plus remaining cash
            equity.append((position_size * current_price) - commission)
    
    # If we still have an open position at the end, close it at the last price
    if position == 1:
        exit_price = df['Close'].iloc[-1]
        exit_date = df['Date'].iloc[-1]
        
        # Calculate profit
        profit = (exit_price - entry_price) * position_size
        current_capital = (position_size * exit_price) - commission
        
        # Record the trade
        trade = {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'size': position_size,
            'profit': profit,
            'pnl_pct': (exit_price / entry_price - 1) * 100
        }
        trades.append(trade)
        
        # Update equity for the last day
        equity[-1] = current_capital
    
    return {
        'equity': equity,
        'trades': trades,
    }

def backtest_dual_ma_strategy(data, initial_capital, commission, params):
    """
    Backtest a Dual Moving Average strategy with different calculation method from SMA
    
    Parameters:
    data (pd.DataFrame): OHLC price data
    initial_capital (float): Starting capital
    commission (float): Commission per trade
    params (dict): Strategy parameters - short_ma, long_ma
    
    Returns:
    dict: Results of the backtest including equity curve and trades
    """
    # Extract parameters
    short_ma = params['short_ma']
    long_ma = params['long_ma']
    
    # Copy data to avoid modifying the original DataFrame
    df = data.copy()
    
    # Calculate moving averages - using exponential moving average instead of simple
    df['short_ema'] = df['Close'].ewm(span=short_ma, adjust=False).mean()
    df['long_ema'] = df['Close'].ewm(span=long_ma, adjust=False).mean()
    
    # Drop NaN values (at the beginning due to rolling window)
    df = df.dropna()
    
    # Initialize variables
    position = 0  # 0: no position, 1: long position
    entry_price = 0
    entry_date = None
    equity = [initial_capital]
    current_capital = initial_capital
    trades = []
    
    # Iterate through data points
    for i in range(1, len(df)):
        current_date = df['Date'].iloc[i]
        current_price = df['Close'].iloc[i]
        
        # Check for buy signal (short MA crosses above long MA)
        if df['short_ema'].iloc[i-1] < df['long_ema'].iloc[i-1] and df['short_ema'].iloc[i] >= df['long_ema'].iloc[i] and position == 0:
            # Calculate position size (invest all capital)
            position_size = current_capital / current_price
            position_size = math.floor(position_size)  # Whole number of shares
            
            # Only take the trade if we can buy at least one share
            if position_size > 0:
                # Enter long position
                position = 1
                entry_price = current_price
                entry_date = current_date
                
                # Deduct commission
                current_capital -= commission
        
        # Check for sell signal (short MA crosses below long MA)
        elif df['short_ema'].iloc[i-1] > df['long_ema'].iloc[i-1] and df['short_ema'].iloc[i] <= df['long_ema'].iloc[i] and position == 1:
            # Exit long position
            exit_price = current_price
            
            # Calculate profit
            profit = (exit_price - entry_price) * position_size
            current_capital = (position_size * exit_price) - commission
            
            # Record the trade
            trade = {
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': current_date,
                'exit_price': exit_price,
                'size': position_size,
                'profit': profit,
                'pnl_pct': (exit_price / entry_price - 1) * 100
            }
            trades.append(trade)
            
            # Reset position
            position = 0
            entry_price = 0
            entry_date = None
        
        # Update equity curve
        if position == 0:
            equity.append(current_capital)
        else:
            # Mark-to-market - current value of position plus remaining cash
            equity.append((position_size * current_price) - commission)
    
    # If we still have an open position at the end, close it at the last price
    if position == 1:
        exit_price = df['Close'].iloc[-1]
        exit_date = df['Date'].iloc[-1]
        
        # Calculate profit
        profit = (exit_price - entry_price) * position_size
        current_capital = (position_size * exit_price) - commission
        
        # Record the trade
        trade = {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'size': position_size,
            'profit': profit,
            'pnl_pct': (exit_price / entry_price - 1) * 100
        }
        trades.append(trade)
        
        # Update equity for the last day
        equity[-1] = current_capital
    
    return {
        'equity': equity,
        'trades': trades,
    }