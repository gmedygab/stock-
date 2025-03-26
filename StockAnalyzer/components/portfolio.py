import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from utils.data_fetcher import get_stock_data

def display_portfolio():
    """
    Display and manage the user's stock portfolio
    """
    # Initialize portfolio in session state if not exists
    if 'portfolio' not in st.session_state:
        # Portfolio structure: {symbol: {'shares': number, 'avg_price': number}}
        st.session_state.portfolio = {
            'AAPL': {'shares': 10, 'avg_price': 170.50}, 
            'MSFT': {'shares': 5, 'avg_price': 320.75}, 
            'GOOGL': {'shares': 2, 'avg_price': 140.25}
        }
    
    # CSV Upload Section
    st.subheader("Import Portfolio from CSV")
    with st.expander("Upload CSV File"):
        st.write("Upload a CSV file with your portfolio data. The file should include columns for Symbol, Shares, and Average Purchase Price.")
        
        # Sample format
        st.markdown("""
        **Expected CSV format:**
        ```
        Symbol,Shares,Average Purchase Price ($)
        AAPL,10,170.50
        MSFT,5,320.75
        GOOGL,2,140.25
        ```
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load and process the CSV data
                df = pd.read_csv(uploaded_file)
                
                # Check for required columns
                standard_columns = ['Symbol', 'Shares', 'Average Purchase Price ($)']
                # Also check for column names from the exported CSV format
                exported_format_columns = ['Symbol', 'Shares', 'Average Purchase Price ($)']
                
                # First see if we have an index column to skip
                if any('Unnamed' in col for col in df.columns):
                    # Skip index column if it exists
                    if df.columns[0].startswith('Unnamed'):
                        df = df.iloc[:, 1:].copy()  # Skip the first column which is an index
                
                # Check if Symbol and Shares exist directly
                has_symbol = 'Symbol' in df.columns
                has_shares = 'Shares' in df.columns
                
                # Look for average price column in various formats
                avg_price_col = None
                for col in df.columns:
                    if col == 'Average Purchase Price ($)':
                        avg_price_col = col
                        break
                    elif 'Average Purchase Price' in col:
                        avg_price_col = col
                        break
                    elif 'avg' in col.lower() and 'price' in col.lower() and '$' in col:
                        avg_price_col = col
                        break
                
                # If we don't have all required columns, try to map them
                if not (has_symbol and has_shares and avg_price_col):
                    # Try finding columns with similar names
                    column_mapping = {}
                    for col in df.columns:
                        # Look for symbol column
                        if not has_symbol and ('symbol' in col.lower() or 'ticker' in col.lower()):
                            column_mapping[col] = 'Symbol'
                            has_symbol = True
                        # Look for shares column
                        elif not has_shares and ('share' in col.lower() or 'quantity' in col.lower()):
                            column_mapping[col] = 'Shares'
                            has_shares = True
                        # Look for price column if we haven't found it yet
                        elif not avg_price_col and ('price' in col.lower() or 'cost' in col.lower()) and ('avg' in col.lower() or 'average' in col.lower() or 'purchase' in col.lower()):
                            column_mapping[col] = 'Average Purchase Price ($)'
                            avg_price_col = 'Average Purchase Price ($)'
                    
                    # Apply the mapping if we found any matches
                    if column_mapping:
                        df.rename(columns=column_mapping, inplace=True)
                
                # Final check for required columns
                required_columns = ['Symbol', 'Shares', 'Average Purchase Price ($)']
                
                # Check if we have columns for the exact format from the sample CSV
                if 'Symbol' in df.columns and 'Shares' in df.columns:
                    # Look for average price column
                    price_col = None
                    for col in df.columns:
                        if 'Average Purchase Price' in col or 'avg' in col.lower() and 'price' in col.lower():
                            price_col = col
                            break
                    
                    # If found, map it to our standard name
                    if price_col and price_col != 'Average Purchase Price ($)':
                        df.rename(columns={price_col: 'Average Purchase Price ($)'}, inplace=True)
                
                if not all(col in df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in df.columns]
                    st.error(f"CSV is missing required columns: {', '.join(missing)}. Please check your file format.")
                else:
                    # Clean up the data
                    for col in df.columns:
                        if 'Price' in col or 'price' in col:
                            # Remove $ and convert to float
                            if df[col].dtype == object:  # Only if it's a string
                                df[col] = df[col].astype(str).str.replace('$', '', regex=False).astype(float)
                    
                    # Create a new portfolio
                    new_portfolio = {}
                    for _, row in df.iterrows():
                        symbol = row['Symbol'].strip().upper()
                        shares = float(row['Shares'])
                        avg_price = float(row['Average Purchase Price ($)'])
                        
                        if shares > 0:
                            new_portfolio[symbol] = {
                                'shares': shares,
                                'avg_price': avg_price
                            }
                    
                    if new_portfolio:
                        # Ask user what to do with existing portfolio
                        action = st.radio(
                            "How would you like to import the data?",
                            ["Replace my entire portfolio", "Add to my existing portfolio"],
                            index=0
                        )
                        
                        if st.button("Import Portfolio"):
                            if action == "Replace my entire portfolio":
                                st.session_state.portfolio = new_portfolio
                                st.success(f"Portfolio replaced with {len(new_portfolio)} imported positions.")
                            else:
                                # Merge with existing portfolio
                                for symbol, position in new_portfolio.items():
                                    if symbol in st.session_state.portfolio:
                                        # Calculate new average price
                                        current = st.session_state.portfolio[symbol]
                                        current_shares = current['shares']
                                        current_avg_price = current['avg_price']
                                        new_shares = position['shares']
                                        new_avg_price = position['avg_price']
                                        
                                        # Calculate weighted average price
                                        total_shares = current_shares + new_shares
                                        total_cost = (current_shares * current_avg_price) + (new_shares * new_avg_price)
                                        avg_price = total_cost / total_shares
                                        
                                        st.session_state.portfolio[symbol] = {
                                            'shares': total_shares,
                                            'avg_price': avg_price
                                        }
                                    else:
                                        st.session_state.portfolio[symbol] = position
                                
                                st.success(f"Added {len(new_portfolio)} positions to your portfolio.")
                            
                            st.rerun()
                    else:
                        st.warning("No valid portfolio positions found in the uploaded file.")
            
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
                st.write("Please make sure your CSV file is properly formatted with the required columns.")
    
    # Portfolio management section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Your Portfolio")
        
        # Calculate portfolio value and performance
        portfolio_data = []
        total_value = 0
        total_investment = 0
        
        for symbol, position in st.session_state.portfolio.items():
            try:
                shares = position['shares']
                avg_cost = position['avg_price']
                
                # Get current stock data
                stock_data = get_stock_data(symbol, '5d')
                
                if stock_data.empty:
                    continue
                
                # Get current price and previous day price
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else stock_data['Open'].iloc[-1]
                
                # Calculate values
                position_value = current_price * shares
                total_value += position_value
                
                # Calculate investment cost
                investment = avg_cost * shares
                total_investment += investment
                
                # Calculate daily change
                daily_change = ((current_price - prev_price) / prev_price) * 100
                
                # Calculate total return
                total_return = ((current_price - avg_cost) / avg_cost) * 100
                
                portfolio_data.append({
                    'Symbol': symbol,
                    'Shares': shares,
                    'Avg Price': f"${avg_cost:.2f}",
                    'Current Price': f"${current_price:.2f}",
                    'Position Value': f"${position_value:.2f}",
                    'Daily Change %': f"{daily_change:.2f}%",
                    'Total Return %': f"{total_return:.2f}%"
                })
            
            except Exception as e:
                st.error(f"Error loading data for {symbol}: {str(e)}")
        
        # Create portfolio dataframe
        if portfolio_data:
            df = pd.DataFrame(portfolio_data)
            
            # Style the dataframe
            def highlight_performance(val):
                if isinstance(val, str) and '%' in val:
                    val_num = float(val.replace('%', ''))
                    if val_num > 0:
                        return 'background-color: rgba(0, 255, 0, 0.2)'
                    elif val_num < 0:
                        return 'background-color: rgba(255, 0, 0, 0.2)'
                return ''
            
            styled_df = df.style.applymap(highlight_performance, subset=['Daily Change %', 'Total Return %'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Display portfolio summary
            total_return_pct = ((total_value - total_investment) / total_investment) * 100
            total_return_value = total_value - total_investment
            
            # Create delta color format
            if total_return_pct < 0:
                delta_color = "inverse"  # Red for negative
            else:
                delta_color = "normal"   # Green for positive
            
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                st.metric("Total Portfolio Value", f"${total_value:.2f}")
            
            with col1b:
                st.metric("Total Investment", f"${total_investment:.2f}")
            
            with col1c:
                st.metric(
                    "Total Return", 
                    f"{total_return_pct:.2f}%", 
                    f"${total_return_value:.2f}", 
                    delta_color=delta_color
                )
        
        else:
            st.warning("Your portfolio is empty. Add some stocks to get started.")
    
    with col2:
        st.subheader("Manage Portfolio")
        
        # Add new stock to portfolio
        with st.form("add_stock_form"):
            st.subheader("Add Stock")
            new_symbol = st.text_input("Stock Symbol", "").upper()
            shares = st.number_input("Number of Shares", min_value=0.01, step=0.01, value=1.0)
            avg_price = st.number_input("Average Purchase Price ($)", min_value=0.01, step=0.01, value=100.0)
            
            add_submitted = st.form_submit_button("Add/Update Stock")
            
            if add_submitted and new_symbol:
                try:
                    # Verify the stock symbol exists
                    stock = yf.Ticker(new_symbol)
                    info = stock.info
                    
                    if 'regularMarketPrice' in info or 'currentPrice' in info:
                        if new_symbol in st.session_state.portfolio:
                            # Calculate new average price when adding more shares
                            current_position = st.session_state.portfolio[new_symbol]
                            current_shares = current_position['shares']
                            current_avg_price = current_position['avg_price']
                            
                            # Calculate new average price based on existing and new shares
                            total_shares = current_shares + shares
                            total_cost = (current_shares * current_avg_price) + (shares * avg_price)
                            new_avg_price = total_cost / total_shares
                            
                            # Update portfolio
                            st.session_state.portfolio[new_symbol] = {
                                'shares': total_shares,
                                'avg_price': new_avg_price
                            }
                            st.success(f"Added {shares} more shares of {new_symbol} to your portfolio")
                        else:
                            # Add new position
                            st.session_state.portfolio[new_symbol] = {
                                'shares': shares,
                                'avg_price': avg_price
                            }
                            st.success(f"Added {new_symbol} to your portfolio")
                        
                        st.rerun()
                    else:
                        st.error(f"Could not verify stock symbol: {new_symbol}")
                
                except Exception as e:
                    st.error(f"Error adding stock: {str(e)}")
        
        # Remove stock from portfolio
        with st.form("remove_stock_form"):
            st.subheader("Remove Stock")
            remove_symbol = st.selectbox(
                "Select Stock to Remove",
                options=list(st.session_state.portfolio.keys())
            )
            
            if remove_symbol:
                current_shares = st.session_state.portfolio[remove_symbol]['shares']
                
                remove_all = st.checkbox("Remove all shares")
                
                if not remove_all:
                    remove_shares = st.number_input(
                        f"Number of Shares to Remove (max: {current_shares})",
                        min_value=0.01,
                        max_value=float(current_shares),
                        step=0.01,
                        value=float(current_shares)
                    )
                
                remove_submitted = st.form_submit_button("Remove Stock")
                
                if remove_submitted:
                    if remove_all or remove_shares >= current_shares:
                        # Remove the entire position
                        del st.session_state.portfolio[remove_symbol]
                        st.success(f"Removed {remove_symbol} from your portfolio")
                    else:
                        # Update the position with remaining shares
                        st.session_state.portfolio[remove_symbol]['shares'] -= remove_shares
                        st.success(f"Removed {remove_shares} shares of {remove_symbol} from your portfolio")
                    
                    st.rerun()
    
    # Portfolio allocation chart
    st.subheader("Portfolio Allocation")
    
    if st.session_state.portfolio:
        # Get data for pie chart
        labels = []
        values = []
        colors = []
        
        color_map = {
            'AAPL': '#999999',
            'MSFT': '#00a2ed',
            'GOOGL': '#34a853',
            'AMZN': '#ff9900',
            'META': '#1877f2',
            'TSLA': '#cc0000',
            'NVDA': '#76b900',
        }
        
        for symbol, position in st.session_state.portfolio.items():
            try:
                shares = position['shares']
                
                # Get current stock data
                stock_data = get_stock_data(symbol, '1d')
                
                if stock_data.empty:
                    continue
                
                # Get current price
                current_price = stock_data['Close'].iloc[-1]
                
                # Calculate position value
                position_value = current_price * shares
                
                labels.append(symbol)
                values.append(position_value)
                
                # Assign color from map or generate random color
                if symbol in color_map:
                    colors.append(color_map[symbol])
                else:
                    import random
                    random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                    colors.append(random_color)
            
            except Exception as e:
                st.error(f"Error loading data for {symbol}: {str(e)}")
        
        # Create pie chart
        if values:
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=colors
            )])
            
            fig.update_layout(
                title="Portfolio Allocation by Value",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Could not generate portfolio allocation chart.")
    
    else:
        st.info("Add stocks to your portfolio to see your allocation.")
    
    # Portfolio performance chart (simplified)
    st.subheader("Portfolio Performance (Last Month)")
    
    if st.session_state.portfolio:
        try:
            # Get data for line chart
            # For simplicity, we'll aggregate the close prices weighted by shares
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=30)
            
            portfolio_value = pd.DataFrame()
            
            for symbol, position in st.session_state.portfolio.items():
                try:
                    shares = position['shares']
                    
                    # Get historical stock data
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if not hist.empty:
                        # Calculate daily position value
                        position_df = pd.DataFrame()
                        position_df[symbol] = hist['Close'] * shares
                        
                        if portfolio_value.empty:
                            portfolio_value = position_df
                        else:
                            portfolio_value = portfolio_value.join(position_df, how='outer')
                
                except Exception as e:
                    st.error(f"Error loading historical data for {symbol}: {str(e)}")
            
            if not portfolio_value.empty:
                # Sum all positions for total portfolio value
                portfolio_value['Total'] = portfolio_value.sum(axis=1)
                
                # Create line chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=portfolio_value.index,
                    y=portfolio_value['Total'],
                    name='Portfolio Value',
                    line=dict(color='rgb(0, 153, 204)', width=3)
                ))
                
                # Calculate initial value for reference line
                initial_value = portfolio_value['Total'].iloc[0] if len(portfolio_value) > 0 else 0
                
                fig.add_trace(go.Scatter(
                    x=[portfolio_value.index[0], portfolio_value.index[-1]],
                    y=[initial_value, initial_value],
                    name='Initial Value',
                    line=dict(color='rgba(200, 200, 200, 0.5)', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Portfolio Performance (Last 30 Days)",
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Add percentage change annotation
                if len(portfolio_value) > 0:
                    first_value = portfolio_value['Total'].iloc[0]
                    last_value = portfolio_value['Total'].iloc[-1]
                    pct_change = ((last_value - first_value) / first_value) * 100
                    
                    fig.add_annotation(
                        x=portfolio_value.index[-1],
                        y=last_value,
                        text=f"{pct_change:.2f}%",
                        showarrow=True,
                        arrowhead=1,
                        ax=40,
                        ay=-40,
                        font=dict(
                            size=16,
                            color="white"
                        ),
                        bgcolor="rgba(0, 153, 204, 0.8)",
                        bordercolor="rgba(0, 153, 204, 1)",
                        borderwidth=2,
                        borderpad=4,
                        opacity=0.8
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("Could not generate portfolio performance chart.")
        
        except Exception as e:
            st.error(f"Error generating portfolio performance chart: {str(e)}")
    
    else:
        st.info("Add stocks to your portfolio to see your performance over time.")
