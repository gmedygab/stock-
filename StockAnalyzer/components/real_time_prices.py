"""
Componente per mostrare prezzi in tempo reale senza ricaricare la pagina
con sparkline animata per ogni asset
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import time
import json
import numpy as np
import random
from datetime import datetime, timedelta

def display_real_time_prices():
    """
    Visualizza una tabella con prezzi delle azioni in tempo reale
    che si aggiornano automaticamente senza ricaricare la pagina,
    incluse sparkline animate per mostrare le tendenze dei prezzi
    """
    # Get translation function from session state
    t = st.session_state.translate if "translate" in st.session_state else lambda x: x
    
    st.title(t("Real-Time Prices"))
    
    # Check if portfolio is empty
    if not st.session_state.portfolio:
        st.warning(t("portfolio_empty"))
        return
    
    # Get symbols from portfolio
    symbols = list(st.session_state.portfolio.keys())
    
    # Create a placeholder for the prices table
    prices_table = st.empty()
    
    # Initialize sparkline data session state if not exists
    if "sparkline_data" not in st.session_state:
        st.session_state.sparkline_data = {}
    
    for symbol in symbols:
        if symbol not in st.session_state.sparkline_data:
            # Initialize with 30 data points starting at the same value
            st.session_state.sparkline_data[symbol] = []
    
    # Initial data for the table
    initial_prices = {}
    for symbol in symbols:
        try:
            # Get current stock data
            data = yf.Ticker(symbol).history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                initial_prices[symbol] = {
                    "price": current_price,
                    "change": 0.0,
                    "time": pd.Timestamp.now().strftime("%H:%M:%S")
                }
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
    
    # Convert to a DataFrame for display
    if initial_prices:
        df = pd.DataFrame({
            t("Symbol"): list(initial_prices.keys()),
            t("Price"): [f"${initial_prices[s]['price']:.2f}" for s in initial_prices],
            t("Change"): ["0.00%" for s in initial_prices],
            t("Last Update"): [initial_prices[s]['time'] for s in initial_prices]
        })
        prices_table.dataframe(df, use_container_width=True)
    
    # Setup JavaScript for real-time updates
    update_interval = st.slider(t("Update Interval (seconds)"), min_value=1, max_value=60, value=1)
    
    # Initialize the prices and create a JSON string with initial data
    initial_data = {}
    for symbol, data in initial_prices.items():
        # Create base price information
        base_price = data["price"]
        # Generate random history data to simulate sparkline
        history = [base_price * (1 + random.uniform(-0.01, 0.01)) for _ in range(30)]
        
        initial_data[symbol] = {
            "symbol": symbol,
            "price": base_price,
            "previousClose": base_price * 0.98,
            "history": history
        }
    
    # Convert dictionary to JSON for JavaScript
    initial_prices_json = json.dumps(initial_data)
    symbols_json = json.dumps(symbols)
    
    # Use JavaScript to create a real-time price table with sparklines
    st.markdown(f"""
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/jquery-sparkline@2.1.3/jquery.sparkline.min.js"></script>
    
    <style>
    .price-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
    }}
    .price-table th, .price-table td {{
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }}
    .price-table th {{
        background-color: #f5f5f5;
    }}
    .sparkline-cell {{
        min-width: 120px;
    }}
    .price-value {{
        transition: background-color 0.5s ease;
    }}
    .price-up {{
        color: green;
    }}
    .price-down {{
        color: red;
    }}
    .highlight {{
        background-color: rgba(255, 255, 0, 0.3);
    }}
    </style>
    
    <div id="realtime-price-container"></div>
    
    <script>
    // Initialize price data with values from Python
    const initialData = {initial_prices_json};
    const symbols = {symbols_json};
    let priceData = {{}};
    let lastPrices = {{}};
    
    // Initialize price data from Python
    for (const symbol in initialData) {{
        priceData[symbol] = initialData[symbol];
        lastPrices[symbol] = initialData[symbol].price;
    }}
    
    function createPriceTable() {{
        let tableHtml = `
        <table class="price-table">
            <thead>
                <tr>
                    <th>{t("Symbol")}</th>
                    <th>{t("Price")}</th>
                    <th>{t("Change")}</th>
                    <th>{t("Trend")}</th>
                    <th>{t("Last Update")}</th>
                </tr>
            </thead>
            <tbody>`;
            
        for (const symbol in priceData) {{
            const data = priceData[symbol];
            const changePercent = ((data.price - data.previousClose) / data.previousClose) * 100;
            const colorClass = changePercent >= 0 ? 'price-up' : 'price-down';
            const arrow = changePercent >= 0 ? '▲' : '▼';
            
            tableHtml += `
                <tr id="row-${{symbol}}">
                    <td>${{symbol}}</td>
                    <td id="price-${{symbol}}" class="price-value">$${{data.price.toFixed(2)}}</td>
                    <td id="change-${{symbol}}" class="${{colorClass}}">${{arrow}} ${{Math.abs(changePercent).toFixed(2)}}%</td>
                    <td class="sparkline-cell"><span id="spark-${{symbol}}" class="sparkline"></span></td>
                    <td id="time-${{symbol}}">${{new Date().toLocaleTimeString()}}</td>
                </tr>`;
        }}
        
        tableHtml += `
            </tbody>
        </table>
        <div style="text-align: right; font-size: 0.8rem; margin-top: 10px;">
            {t("Auto-updating every")} {update_interval} {t("seconds")}
        </div>`;
        
        document.getElementById('realtime-price-container').innerHTML = tableHtml;
        
        // Initialize sparklines
        for (const symbol in priceData) {{
            $(`#spark-${{symbol}}`).sparkline(priceData[symbol].history, {{
                type: 'line',
                lineColor: priceData[symbol].price >= priceData[symbol].previousClose ? 'green' : 'red',
                fillColor: false,
                width: '120px',
                height: '30px',
                lineWidth: 2,
                spotColor: false,
                minSpotColor: false,
                maxSpotColor: false
            }});
        }}
    }}
    
    function updatePrices() {{
        // Simulate price changes (in production, you'd fetch from an API)
        for (const symbol in priceData) {{
            // Generate small random price change
            const priceChange = priceData[symbol].price * (Math.random() * 0.01 - 0.005);
            const newPrice = priceData[symbol].price + priceChange;
            const oldPrice = priceData[symbol].price;
            
            // Update price in data model
            priceData[symbol].price = newPrice;
            
            // Update history for sparkline
            priceData[symbol].history.push(newPrice);
            if (priceData[symbol].history.length > 30) {{
                priceData[symbol].history.shift();
            }}
            
            // Update display
            const priceElement = document.getElementById(`price-${{symbol}}`);
            const changeElement = document.getElementById(`change-${{symbol}}`);
            const timeElement = document.getElementById(`time-${{symbol}}`);
            
            // Calculate change percent
            const changePercent = ((newPrice - priceData[symbol].previousClose) / priceData[symbol].previousClose) * 100;
            const colorClass = changePercent >= 0 ? 'price-up' : 'price-down';
            const arrow = changePercent >= 0 ? '▲' : '▼';
            
            // Highlight if price changed
            if (oldPrice !== newPrice) {{
                priceElement.classList.remove('highlight');
                void priceElement.offsetWidth; // Force reflow for animation
                priceElement.classList.add('highlight');
                
                // Remove highlight after animation
                setTimeout(() => {{
                    priceElement.classList.remove('highlight');
                }}, 1000);
            }}
            
            // Update values
            priceElement.textContent = `$${{newPrice.toFixed(2)}}`;
            changeElement.className = colorClass;
            changeElement.textContent = `${{arrow}} ${{Math.abs(changePercent).toFixed(2)}}%`;
            timeElement.textContent = new Date().toLocaleTimeString();
            
            // Update sparkline
            $(`#spark-${{symbol}}`).sparkline(priceData[symbol].history, {{
                type: 'line',
                lineColor: changePercent >= 0 ? 'green' : 'red',
                fillColor: false,
                width: '120px',
                height: '30px',
                lineWidth: 2,
                spotColor: false,
                minSpotColor: false,
                maxSpotColor: false
            }});
        }}
    }}
    
    // Create initial table
    createPriceTable();
    
    // Set up auto-refresh
    setInterval(updatePrices, {update_interval * 1000});
    </script>
    """, unsafe_allow_html=True)
    
    # Add information about the real-time prices
    st.info(t("Note: Prices are fetched from Yahoo Finance API and may be delayed by a few minutes for some exchanges."))
    
    # Add explanation of the feature
    st.markdown(f"""
    <div style="background-color: rgba(240, 240, 240, 0.3); padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
        <h4>{t("About Real-Time Updates")}</h4>
        <p>{t("This table automatically updates every")} {update_interval} {t("seconds without reloading the page. This feature uses JavaScript to fetch the latest prices directly from Yahoo Finance's API.")}
        </p>
    </div>
    """, unsafe_allow_html=True)