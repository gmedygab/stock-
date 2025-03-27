import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import os
import re
from bs4 import BeautifulSoup
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data (only needs to run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def get_news_from_api(symbol, max_news=5):
    """
    Get news articles for a specific stock symbol from a news API
    """
    # Access translation function if available in session state
    t = st.session_state.get('translate', lambda x: x)
    
    try:
        # Try to get news API key from environment variables
        api_key = os.getenv("NEWS_API_KEY")
        
        if api_key:
            # Calculate date range (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Format dates for API
            from_date = start_date.strftime("%Y-%m-%d")
            to_date = end_date.strftime("%Y-%m-%d")
            
            # Make API request
            url = f"https://newsapi.org/v2/everything?q={symbol}+stock&apiKey={api_key}&from={from_date}&to={to_date}&language=en&sortBy=publishedAt"
            response = requests.get(url)
            
            if response.status_code == 200:
                news_data = response.json()
                
                if news_data["status"] == "ok" and news_data["totalResults"] > 0:
                    # Format articles
                    articles = news_data["articles"][:max_news]
                    return [
                        {
                            "title": article["title"],
                            "description": article["description"],
                            "url": article["url"],
                            "source": article["source"]["name"],
                            "published_at": article["publishedAt"]
                        }
                        for article in articles
                    ]
        
        # If API key is not available or request failed, use fallback method
        return get_news_fallback(symbol, max_news)
    
    except Exception as e:
        st.warning(f"Unable to fetch news from API: {str(e)}")
        return get_news_fallback(symbol, max_news)

def get_news_fallback(symbol, max_news=5):
    """
    Fallback method to get financial news by scraping Yahoo Finance
    """
    # Access translation function if available in session state
    t = st.session_state.get('translate', lambda x: x)
    
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles
            articles = []
            news_items = soup.find_all('div', {'class': 'Ov(h)'})
            
            for item in news_items[:max_news]:
                try:
                    # Extract headline
                    headline_elem = item.find('h3')
                    if not headline_elem:
                        continue
                    
                    title = headline_elem.text.strip()
                    
                    # Extract link
                    link_elem = item.find('a')
                    url = "https://finance.yahoo.com" + link_elem['href'] if link_elem else ""
                    
                    # Extract source and time
                    source_elem = item.find('div', {'class': 'C(#959595)'})
                    source_text = source_elem.text.strip() if source_elem else ""
                    
                    # Parse source and time
                    source_match = re.search(r'^(.*?)\s+·', source_text)
                    source = source_match.group(1) if source_match else t("Yahoo Finance")
                    
                    time_match = re.search(r'·\s+(.*?)$', source_text)
                    published_at = time_match.group(1) if time_match else t("Recent")
                    
                    articles.append({
                        "title": title,
                        "description": "",
                        "url": url,
                        "source": source,
                        "published_at": published_at
                    })
                    
                except Exception as e:
                    continue
            
            return articles
        
        # If scraping fails, return mock data
        return get_mock_news(symbol, max_news)
    
    except Exception as e:
        st.warning(t(f"Unable to fetch news: {str(e)}"))
        return get_mock_news(symbol, max_news)

def get_mock_news(symbol, max_news=5):
    """
    Generate empty news template
    """
    # Access translation function if available in session state
    t = st.session_state.get('translate', lambda x: x)
    
    return [
        {
            "title": t(f"Unable to load {symbol} news"),
            "description": t("Please check your internet connection or try again later."),
            "url": "#",
            "source": t("System"),
            "published_at": datetime.now().strftime("%Y-%m-%d")
        }
    ]

def analyze_sentiment(text):
    """
    Analyze the sentiment of a piece of text
    
    Returns:
    - sentiment_score: A float between -1 (very negative) and 1 (very positive)
    - sentiment_label: A string label ('positive', 'neutral', or 'negative')
    - sentiment_color: A color to represent the sentiment visually
    """
    if not text:
        return 0, "neutral", "gray"
    
    # Get sentiment scores
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    
    # Determine sentiment label and color
    if compound_score >= 0.05:
        return compound_score, "positive", "green"
    elif compound_score <= -0.05:
        return compound_score, "negative", "red"
    else:
        return compound_score, "neutral", "gray"

def display_sentiment_badge(sentiment_label, sentiment_score, sentiment_color):
    """
    Display a sentiment badge with appropriate color and score
    """
    # Access translation function if available in session state
    t = st.session_state.get('translate', lambda x: x)
    
    # Map sentiment labels for translation
    sentiment_labels = {
        "positive": t("positive"),
        "neutral": t("neutral"),
        "negative": t("negative")
    }
    
    # Get the translated sentiment label
    translated_label = sentiment_labels.get(sentiment_label, sentiment_label)
    
    # Format the score to 2 decimal places
    score_formatted = f"{sentiment_score:.2f}"
    
    # Create the badge HTML
    badge_html = f"""
        <div style="
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background-color: {sentiment_color};
            color: white;
            font-weight: bold;
            font-size: 0.8em;
            margin-top: 4px;
        ">
            {translated_label.upper()}: {score_formatted}
        </div>
    """
    
    return st.markdown(badge_html, unsafe_allow_html=True)

def display_stock_news(symbol, max_news=5):
    """
    Display news related to a stock symbol with sentiment analysis
    """
    # Access translation function if available in session state
    t = st.session_state.get('translate', lambda x: x)
    
    news_articles = get_news_from_api(symbol, max_news)
    
    if not news_articles:
        st.info(t(f"No recent news found for {symbol}"))
        return
    
    # Calculate overall sentiment for the news
    sentiment_scores = []
    
    for article in news_articles:
        # Analyze sentiment of title and description
        title_text = article['title']
        desc_text = article.get('description', '')
        
        # Combine texts for sentiment analysis
        full_text = f"{title_text} {desc_text}".strip()
        
        # Get sentiment score
        sentiment_score, sentiment_label, sentiment_color = analyze_sentiment(full_text)
        
        # Add sentiment to article
        article['sentiment_score'] = sentiment_score
        article['sentiment_label'] = sentiment_label
        article['sentiment_color'] = sentiment_color
        
        # Add to list of scores for overall sentiment
        sentiment_scores.append(sentiment_score)
    
    # Calculate overall sentiment
    if sentiment_scores:
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        overall_sentiment, overall_label, overall_color = analyze_sentiment(None)
        
        # Determine overall sentiment based on average score
        if avg_sentiment >= 0.05:
            overall_label = "positive"
            overall_color = "green"
        elif avg_sentiment <= -0.05:
            overall_label = "negative"
            overall_color = "red"
        else:
            overall_label = "neutral"
            overall_color = "gray"
        
        # Map sentiment labels for translation
        sentiment_labels = {
            "positive": t("positive"),
            "neutral": t("neutral"),
            "negative": t("negative")
        }
        
        # Display overall sentiment
        st.markdown(f"### {t('News Sentiment Analysis')}")
        
        # Create columns for visualization
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display overall sentiment with gauge-like visualization
            st.markdown(f"""
                <div style='
                    text-align: center;
                    padding: 10px;
                    border-radius: 5px;
                    background-color: rgba(0,0,0,0.05);
                '>
                    <div style='font-size: 1.2em; font-weight: bold;'>{t('Overall Sentiment')}</div>
                    <div style='
                        font-size: 2.2em;
                        font-weight: bold;
                        color: {overall_color};
                        margin: 10px 0;
                    '>{avg_sentiment:.2f}</div>
                    <div style='
                        display: inline-block;
                        padding: 5px 10px;
                        background-color: {overall_color};
                        color: white;
                        border-radius: 4px;
                        font-weight: bold;
                    '>{sentiment_labels[overall_label].upper()}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create a sentiment distribution chart
            positive_count = sum(1 for article in news_articles if article['sentiment_label'] == 'positive')
            neutral_count = sum(1 for article in news_articles if article['sentiment_label'] == 'neutral')
            negative_count = sum(1 for article in news_articles if article['sentiment_label'] == 'negative')
            
            # Create progress bars for each sentiment
            st.markdown(f"<div style='margin-top: 10px;'><b>{t('Sentiment Distribution')}</b></div>", unsafe_allow_html=True)
            
            # Positive
            st.markdown(f"""
                <div style='margin-bottom: 10px;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='width: 90px;'>{t('Positive')}:</div>
                        <div style='flex-grow: 1; background-color: #f0f0f0; border-radius: 4px; height: 20px;'>
                            <div style='
                                width: {(positive_count / len(news_articles)) * 100}%;
                                background-color: green;
                                height: 20px;
                                border-radius: 4px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                color: white;
                                font-size: 0.8em;
                                font-weight: bold;
                            '>{positive_count}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Neutral
            st.markdown(f"""
                <div style='margin-bottom: 10px;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='width: 90px;'>{t('Neutral')}:</div>
                        <div style='flex-grow: 1; background-color: #f0f0f0; border-radius: 4px; height: 20px;'>
                            <div style='
                                width: {(neutral_count / len(news_articles)) * 100}%;
                                background-color: gray;
                                height: 20px;
                                border-radius: 4px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                color: white;
                                font-size: 0.8em;
                                font-weight: bold;
                            '>{neutral_count}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Negative
            st.markdown(f"""
                <div style='margin-bottom: 10px;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='width: 90px;'>{t('Negative')}:</div>
                        <div style='flex-grow: 1; background-color: #f0f0f0; border-radius: 4px; height: 20px;'>
                            <div style='
                                width: {(negative_count / len(news_articles)) * 100}%;
                                background-color: red;
                                height: 20px;
                                border-radius: 4px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                color: white;
                                font-size: 0.8em;
                                font-weight: bold;
                            '>{negative_count}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Display divider
    st.markdown("---")
    
    # Display individual news articles with sentiment
    st.markdown(f"### {t('Latest News')}")
    
    for article in news_articles:
        with st.container():
            # Main columns for article display
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Display title with sentiment indicator
                sentiment_label = article.get('sentiment_label', 'neutral')
                sentiment_color = article.get('sentiment_color', 'gray')
                
                # Create title with sentiment indicator
                icon = "📈" if sentiment_label == "positive" else "📉" if sentiment_label == "negative" else "➖"
                st.markdown(f"#### {icon} [{article['title']}]({article['url']})")
                
                # Display description if available
                if article.get('description'):
                    st.markdown(article['description'])
                
                # Display sentiment badge
                sentiment_score = article.get('sentiment_score', 0)
                display_sentiment_badge(sentiment_label, sentiment_score, sentiment_color)
                
            with col2:
                st.text(article['source'])
                st.text(article['published_at'])
            
            st.markdown("---")
