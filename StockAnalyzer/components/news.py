import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import os
import re
from bs4 import BeautifulSoup

def get_news_from_api(symbol, max_news=5):
    """
    Get news articles for a specific stock symbol from a news API
    """
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
                    source = source_match.group(1) if source_match else "Yahoo Finance"
                    
                    time_match = re.search(r'·\s+(.*?)$', source_text)
                    published_at = time_match.group(1) if time_match else "Recent"
                    
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
        st.warning(f"Unable to fetch news: {str(e)}")
        return get_mock_news(symbol, max_news)

def get_mock_news(symbol, max_news=5):
    """
    Generate empty news template
    """
    return [
        {
            "title": f"Unable to load {symbol} news",
            "description": "Please check your internet connection or try again later.",
            "url": "#",
            "source": "System",
            "published_at": datetime.now().strftime("%Y-%m-%d")
        }
    ]

def display_stock_news(symbol, max_news=5):
    """
    Display news related to a stock symbol
    """
    news_articles = get_news_from_api(symbol, max_news)
    
    if not news_articles:
        st.info(f"No recent news found for {symbol}")
        return
    
    for article in news_articles:
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"#### [{article['title']}]({article['url']})")
                
                if article['description']:
                    st.markdown(article['description'])
                
            with col2:
                st.text(article['source'])
                st.text(article['published_at'])
            
            st.markdown("---")
