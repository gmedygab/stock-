import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.data_fetcher import get_stock_data
from components.technical_indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_moving_averages

def display_stock_chart(symbol, timeframe, detailed=False):
    """
    Display an interactive stock chart with selected technical indicators, including potential pre/after-hours data.
    """
    try:
        # Fetch stock data (assuming get_stock_data now handles pre/post market data)
        df = get_stock_data(symbol, timeframe)

        if df.empty:
            st.error(f"No data available for {symbol}")
            return

        # Identify pre/after-hours data (assuming data is already correctly labeled)
        df['RegularHours'] = (df.index.time >= pd.to_datetime('09:30:00').time()) & (df.index.time <= pd.to_datetime('16:00:00').time())

        # Create a subplot with 2 rows (price chart and volume)
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price', 'Volume'),
            row_heights=[0.8, 0.2]
        )

        # Add candlestick chart with improved colors, separating regular and extended hours
        fig.add_trace(
            go.Candlestick(
                x=df[df['RegularHours']].index,
                open=df[df['RegularHours']]['Open'],
                high=df[df['RegularHours']]['High'],
                low=df[df['RegularHours']]['Low'],
                close=df[df['RegularHours']]['Close'],
                name="Regular Hours",
                increasing=dict(line=dict(color='rgba(0, 180, 0, 1)'), fillcolor='rgba(0, 180, 0, 0.7)'),
                decreasing=dict(line=dict(color='rgba(220, 0, 0, 1)'), fillcolor='rgba(220, 0, 0, 0.7)'),
                showlegend=False
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Candlestick(
                x=df[~df['RegularHours']].index,
                open=df[~df['RegularHours']]['Open'],
                high=df[~df['RegularHours']]['High'],
                low=df[~df['RegularHours']]['Low'],
                close=df[~df['RegularHours']]['Close'],
                name="Extended Hours",
                increasing=dict(line=dict(color='rgba(0, 180, 0, 0.5)'), fillcolor='rgba(0, 180, 0, 0.3)'),
                decreasing=dict(line=dict(color='rgba(220, 0, 0, 0.5)'), fillcolor='rgba(220, 0, 0, 0.3)'),
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=1
        )


        # Add volume bars with color based on price movement
        volume_colors = []
        for i in range(len(df)):
            if i > 0:
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    volume_colors.append('rgba(0, 180, 0, 0.5)')  # Green for up days
                else:
                    volume_colors.append('rgba(220, 0, 0, 0.5)')  # Red for down days
            else:
                volume_colors.append('rgba(100, 100, 255, 0.5)')  # Default color for first day

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker=dict(color=volume_colors),
                showlegend=False
            ),
            row=2, col=1
        )

        # Technical indicators (either based on user selection or show most important ones if detailed)
        if detailed:
            # Add all indicators

            # Moving averages with improved colors and width
            sma_20, sma_50, sma_200, ema_20 = calculate_moving_averages(df)

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=sma_20,
                    name="SMA (20)",
                    line=dict(color='rgba(30, 144, 255, 0.9)', width=1.5)  # Dodger Blue
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=sma_50,
                    name="SMA (50)",
                    line=dict(color='rgba(255, 140, 0, 0.9)', width=1.5)  # Dark Orange
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=sma_200,
                    name="SMA (200)",
                    line=dict(color='rgba(148, 0, 211, 0.9)', width=1.5)  # Dark Violet
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ema_20,
                    name="EMA (20)",
                    line=dict(color='rgba(50, 205, 50, 0.9)', width=1.8, dash='dot')  # Lime Green
                ),
                row=1, col=1
            )

            # Bollinger Bands with improved colors
            upper_band, middle_band, lower_band = calculate_bollinger_bands(df)

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=upper_band,
                    name="Upper BB",
                    line=dict(color='rgba(220, 20, 60, 0.8)', width=1.5),  # Crimson
                    showlegend=True
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=middle_band,
                    name="Middle BB",
                    line=dict(color='rgba(75, 0, 130, 0.8)', width=1.5),  # Indigo
                    showlegend=True
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=lower_band,
                    name="Lower BB",
                    line=dict(color='rgba(220, 20, 60, 0.8)', width=1.5),  # Crimson
                    showlegend=True,
                    fill='tonexty',
                    fillcolor='rgba(173, 204, 255, 0.2)'
                ),
                row=1, col=1
            )

        else:
            # Add only the most important indicators for the basic view
            sma_20, _, _, _ = calculate_moving_averages(df)

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=sma_20,
                    name="SMA (20)",
                    line=dict(color='rgba(30, 144, 255, 0.9)', width=2)  # Dodger Blue
                ),
                row=1, col=1
            )

        # Configure layout with improved styling
        fig.update_layout(
            title=f"{symbol} Stock Price ({timeframe})",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )

        # Configure y-axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Update background colors
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            plot_bgcolor='rgba(240, 240, 240, 0.1)',  # Very light gray
            paper_bgcolor='rgba(0, 0, 0, 0)',
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

        # Add chart legend/explanation if detailed view
        if detailed:
            legend_html = """
            <div style="background-color: rgba(240, 240, 240, 0.7); padding: 10px; border-radius: 5px; margin-top: 10px; margin-bottom: 15px; border: 1px solid #ddd;">
                <h4 style="margin-top: 0; margin-bottom: 8px;">📈 Indicatori del grafico</h4>
                <ul style="margin-bottom: 0; padding-left: 20px;">
                    <li><b>Candele:</b> Verde = Aumento prezzo, Rosso = Diminuzione prezzo</li>
                    <li><b>SMA (20):</b> Media mobile semplice a 20 giorni - Tendenza a breve termine</li>
                    <li><b>SMA (50):</b> Media mobile semplice a 50 giorni - Tendenza a medio termine</li>
                    <li><b>SMA (200):</b> Media mobile semplice a 200 giorni - Tendenza a lungo termine</li>
                    <li><b>EMA (20):</b> Media mobile esponenziale a 20 giorni - Dà più peso ai prezzi recenti</li>
                    <li><b>Bande di Bollinger:</b> Canali di volatilità (deviazione standard dalla media mobile)</li>
                </ul>
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying chart: {str(e)}")