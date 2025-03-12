import os
import logging
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import time
import queue
import threading

class MarketNewsAPI:
    def __init__(self, config, news_queue):
        """
        Initialize MarketNewsAPI with configuration for The 5%ers MT5 trading.

        Args:
            config (dict): Configuration dictionary with news settings.
            news_queue (queue.Queue): Queue for news updates to other modules.
        """
        self.config = config
        self.news_queue = news_queue
        self.logger = logging.getLogger(__name__)

        # Validate API keys
        self.news_api_key = config.get('market_news', {}).get('news_api_key')
        self.twitter_bearer_token = config.get('market_news', {}).get('twitter_bearer_token')
        if not self.news_api_key or not self.twitter_bearer_token:
            raise ValueError("Missing API keys for NewsAPI or Twitter.")

        # Set up API endpoints
        self.news_endpoint = "https://newsapi.org/v2/everything"
        self.twitter_endpoint = "https://api.twitter.com/2/tweets/search/recent"
        self.truthsocial_endpoint = "https://truthsocial.com/api/v1/posts"  # Placeholder, not implemented

        # Initialize last-checked timestamps for each category
        self.last_checked = {
            'trump': datetime.utcnow() - timedelta(days=1),
            'nasdaq': datetime.utcnow() - timedelta(days=1),  # Kept for general market context, not specific to NASDAQ futures
            'currency': datetime.utcnow() - timedelta(days=1),
            'macroeconomic': datetime.utcnow() - timedelta(days=1),
            'geopolitical': datetime.utcnow() - timedelta(days=1),
            'global_trends': datetime.utcnow() - timedelta(days=1),
            'investor_sentiment': datetime.utcnow() - timedelta(days=1),
            'market_liquidity': datetime.utcnow() - timedelta(days=1),
            'fiscal_policy': datetime.utcnow() - timedelta(days=1)
        }

        # Risk adjustment tracking
        self.risk_adjusted_until = None
        self.risk_factor = 1.0  # Default risk factor

        # Initialize VADER sentiment analyzer
        self.sid = SentimentIntensityAnalyzer()

    def _fetch_news(self, query, from_date):
        """
        Fetch news articles from NewsAPI.

        Args:
            query (str): Search query for news articles.
            from_date (datetime): Start date for news search.

        Returns:
            list: List of news articles, or empty list if failed.
        """
        try:
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'apiKey': self.news_api_key
            }
            response = requests.get(self.news_endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('articles', [])
        except requests.RequestException as e:
            self.logger.error(f"Error fetching news for {query}: {e}")
            return []

    def _check_news_category(self, category, query, handler):
        """
        Check news for a specific category and process results.

        Args:
            category (str): News category (e.g., 'currency').
            query (str): Search query for the category.
            handler (callable): Function to process the results.
        """
        from_date = self.last_checked[category]
        articles = self._fetch_news(query, from_date)
        if not articles:
            return
        sentiments = []
        for article in articles:
            text = article.get('title', '') + ' ' + article.get('description', '')
            sentiment = self.sid.polarity_scores(text)['compound']
            sentiments.append(sentiment)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        self.last_checked[category] = datetime.utcnow()
        handler(articles, avg_sentiment)

    def check_trump_news(self):
        """Check news related to Trump for market impact."""
        self._check_news_category('trump', 'Trump', self.handle_trump_news)

    def handle_trump_news(self, articles, avg_sentiment):
        """Handle Trump-related news and adjust risk if needed."""
        if avg_sentiment < -0.5:
            self.adjust_risk(0.5, duration_minutes=30)
            self.news_queue.put(f"Negative Trump news detected (Sentiment: {avg_sentiment:.2f})")
        elif avg_sentiment > 0.5:
            self.news_queue.put(f"Positive Trump news detected (Sentiment: {avg_sentiment:.2f})")

    def check_nasdaq_news(self):
        """Check news related to broader market indices (not NASDAQ futures-specific)."""
        self._check_news_category('nasdaq', 'market indices earnings', self.handle_nasdaq_news)

    def handle_nasdaq_news(self, articles, avg_sentiment):
        """Handle market index-related news."""
        if avg_sentiment < -0.5:
            self.adjust_risk(0.6, duration_minutes=20)
            self.news_queue.put(f"Negative market index news detected (Sentiment: {avg_sentiment:.2f})")

    def check_currency_news(self):
        """Check news related to currency exchange rates."""
        self._check_news_category('currency', 'currency exchange forex', self.handle_currency_news)

    def handle_currency_news(self, articles, avg_sentiment):
        """Handle currency-related news, critical for EURUSD and GBPJPY."""
        if avg_sentiment < -0.4:
            self.adjust_risk(0.7, duration_minutes=15)
            self.news_queue.put(f"Negative currency news detected (Sentiment: {avg_sentiment:.2f})")

    def check_macroeconomic_news(self):
        """Check macroeconomic news affecting forex markets."""
        self._check_news_category('macroeconomic', 'macroeconomic GDP', self.handle_macroeconomic_news)

    def handle_macroeconomic_news(self, articles, avg_sentiment):
        """Handle macroeconomic news."""
        if avg_sentiment < -0.5:
            self.adjust_risk(0.5, duration_minutes=30)
            self.news_queue.put(f"Negative macroeconomic news detected (Sentiment: {avg_sentiment:.2f})")

    def check_geopolitical_news(self):
        """Check geopolitical news impacting forex volatility."""
        self._check_news_category('geopolitical', 'geopolitical conflict', self.handle_geopolitical_news)

    def handle_geopolitical_news(self, articles, avg_sentiment):
        """Handle geopolitical news."""
        if avg_sentiment < -0.6:
            self.adjust_risk(0.4, duration_minutes=60)
            self.news_queue.put(f"Negative geopolitical news detected (Sentiment: {avg_sentiment:.2f})")

    def check_global_trends(self):
        """Check global market trends affecting forex."""
        self._check_news_category('global_trends', 'global market trends', self.handle_global_trends)

    def handle_global_trends(self, articles, avg_sentiment):
        """Handle global trends news."""
        if avg_sentiment < -0.5:
            self.adjust_risk(0.6, duration_minutes=20)
            self.news_queue.put(f"Negative global trends detected (Sentiment: {avg_sentiment:.2f})")

    def check_investor_sentiment(self):
        """Check investor sentiment news."""
        self._check_news_category('investor_sentiment', 'investor sentiment forex', self.handle_investor_sentiment)

    def handle_investor_sentiment(self, articles, avg_sentiment):
        """Handle investor sentiment news."""
        if avg_sentiment < -0.5:
            self.adjust_risk(0.7, duration_minutes=15)
            self.news_queue.put(f"Negative investor sentiment detected (Sentiment: {avg_sentiment:.2f})")

    def check_market_liquidity(self):
        """Check market liquidity news affecting forex trading."""
        self._check_news_category('market_liquidity', 'market liquidity forex', self.handle_market_liquidity)

    def handle_market_liquidity(self, articles, avg_sentiment):
        """Handle market liquidity news."""
        if avg_sentiment < -0.5:
            self.adjust_risk(0.6, duration_minutes=20)
            self.news_queue.put(f"Negative market liquidity news detected (Sentiment: {avg_sentiment:.2f})")

    def check_fiscal_policy(self):
        """Check fiscal policy news impacting forex markets."""
        self._check_news_category('fiscal_policy', 'fiscal policy tax', self.handle_fiscal_policy)

    def handle_fiscal_policy(self, articles, avg_sentiment):
        """Handle fiscal policy news."""
        if avg_sentiment < -0.5:
            self.adjust_risk(0.5, duration_minutes=30)
            self.news_queue.put(f"Negative fiscal policy news detected (Sentiment: {avg_sentiment:.2f})")

    def _get_tweets(self, query):
        """
        Fetch recent tweets based on a query.

        Args:
            query (str): Search query for tweets.

        Returns:
            list: List of tweet data, or empty list if failed.
        """
        try:
            headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}
            params = {'query': query, 'max_results': 10}
            response = requests.get(self.twitter_endpoint, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('data', [])
        except requests.RequestException as e:
            self.logger.error(f"Error fetching tweets: {e}")
            return []

    def handle_tweet(self, tweet):
        """Analyze and handle a single tweet."""
        sentiment = self.sid.polarity_scores(tweet['text'])['compound']
        text_lower = tweet['text'].lower()
        if "announcement" in text_lower or "update" in text_lower:
            if sentiment < -0.3:
                self.adjust_risk(0.7, duration_minutes=15)
                self.news_queue.put(f"Negative tweet detected: {tweet['text']} (Sentiment: {sentiment:.2f})")

    def check_twitter(self):
        """Check tweets from configured queries."""
        queries = self.config.get('market_news', {}).get('twitter_queries', ['forex', 'EURUSD', 'GBPJPY'])
        for query in queries:
            tweets = self._get_tweets(query)
            for tweet in tweets:
                self.handle_tweet(tweet)

    def _get_truthsocial_posts(self):
        """Placeholder for fetching Truth Social posts (not implemented)."""
        # Implement when API is available; currently returns empty list
        return []

    def handle_truthsocial_post(self, post):
        """Analyze and handle a Truth Social post."""
        sentiment = self.sid.polarity_scores(post['text'])['compound']
        if sentiment < -0.3:
            self.adjust_risk(0.7, duration_minutes=15)
            self.news_queue.put(f"Negative Truth Social post detected (Sentiment: {sentiment:.2f})")

    def check_truthsocial(self):
        """Check Truth Social posts (placeholder)."""
        posts = self._get_truthsocial_posts()
        for post in posts:
            self.handle_truthsocial_post(post)

    def check_economic_calendar(self):
        """Placeholder for checking economic calendar events."""
        # Implement with an actual economic calendar API (e.g., Forex Factory API when available)
        events = []  # Example: [{'name': 'GDP Release', 'impact': 'high', 'time': datetime.utcnow()}]
        for event in events:
            if event.get('impact') == 'high' and (event['time'] - datetime.utcnow()) < timedelta(hours=1):
                self.adjust_risk(0.5, duration_minutes=60)
                self.news_queue.put(f"High-impact event imminent: {event.get('name')}")

    def adjust_risk(self, factor, duration_minutes):
        """
        Adjust trading risk based on sentiment or events.

        Args:
            factor (float): Risk adjustment factor (e.g., 0.5 reduces risk by 50%).
            duration_minutes (int): Duration of the adjustment in minutes.
        """
        if self.risk_adjusted_until and datetime.utcnow() < self.risk_adjusted_until:
            return  # Skip if already adjusted
        self.risk_factor = factor
        self.risk_adjusted_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        self.logger.info(f"Risk adjusted to {factor} for {duration_minutes} minutes")
        self.news_queue.put(f"Risk adjusted to {factor} for {duration_minutes} minutes")

    def run_all_checks(self):
        """Run continuous checks on all data sources."""
        while True:
            try:
                self.check_trump_news()
                self.check_nasdaq_news()  # Broad market context, not futures-specific
                self.check_currency_news()
                self.check_macroeconomic_news()
                self.check_geopolitical_news()
                self.check_global_trends()
                self.check_investor_sentiment()
                self.check_market_liquidity()
                self.check_fiscal_policy()
                self.check_economic_calendar()
                self.check_twitter()
                self.check_truthsocial()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in run_all_checks: {e}")
                time.sleep(60)  # Wait before retrying


# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "market_news": {
            "news_api_key": "your_news_api_key",
            "twitter_bearer_token": "your_twitter_bearer_token",
            "twitter_queries": ["forex", "EURUSD", "GBPJPY"]
        }
    }
    news_queue = queue.Queue()
    news_api = MarketNewsAPI(config, news_queue)
    news_api.run_all_checks()