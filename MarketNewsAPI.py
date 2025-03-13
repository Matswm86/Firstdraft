import os
import logging
from datetime import datetime, timedelta
import pytz  # Added for UTC consistency
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

        self.news_api_key = config.get('market_news', {}).get('news_api_key')
        self.twitter_bearer_token = config.get('market_news', {}).get('twitter_bearer_token')
        if not self.news_api_key or not self.twitter_bearer_token:
            raise ValueError("Missing API keys for NewsAPI or Twitter.")

        self.news_endpoint = "https://newsapi.org/v2/everything"
        self.twitter_endpoint = "https://api.twitter.com/2/tweets/search/recent"
        self.truthsocial_endpoint = "https://truthsocial.com/api/v1/posts"  # Placeholder

        self.last_checked = {
            'trump': datetime.now(pytz.UTC) - timedelta(days=1),
            'nasdaq': datetime.now(pytz.UTC) - timedelta(days=1),
            'currency': datetime.now(pytz.UTC) - timedelta(days=1),
            'macroeconomic': datetime.now(pytz.UTC) - timedelta(days=1),
            'geopolitical': datetime.now(pytz.UTC) - timedelta(days=1),
            'global_trends': datetime.now(pytz.UTC) - timedelta(days=1),
            'investor_sentiment': datetime.now(pytz.UTC) - timedelta(days=1),
            'market_liquidity': datetime.now(pytz.UTC) - timedelta(days=1),
            'fiscal_policy': datetime.now(pytz.UTC) - timedelta(days=1)
        }

        self.risk_adjusted_until = None
        self.risk_factor = 1.0

        self.sid = SentimentIntensityAnalyzer()

    def _fetch_news(self, query, from_date):
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
        self.last_checked[category] = datetime.now(pytz.UTC)  # Updated to pytz UTC
        handler(articles, avg_sentiment)

    def check_trump_news(self):
        self._check_news_category('trump', 'Trump', self.handle_trump_news)

    def handle_trump_news(self, articles, avg_sentiment):
        if avg_sentiment < -0.5:
            self.adjust_risk(0.5, duration_minutes=30)
            self.news_queue.put(f"Negative Trump news detected (Sentiment: {avg_sentiment:.2f})")
        elif avg_sentiment > 0.5:
            self.news_queue.put(f"Positive Trump news detected (Sentiment: {avg_sentiment:.2f})")

    def check_nasdaq_news(self):
        self._check_news_category('nasdaq', 'market indices earnings', self.handle_nasdaq_news)

    def handle_nasdaq_news(self, articles, avg_sentiment):
        if avg_sentiment < -0.5:
            self.adjust_risk(0.6, duration_minutes=20)
            self.news_queue.put(f"Negative market index news detected (Sentiment: {avg_sentiment:.2f})")

    def check_currency_news(self):
        self._check_news_category('currency', 'currency exchange forex', self.handle_currency_news)

    def handle_currency_news(self, articles, avg_sentiment):
        if avg_sentiment < -0.4:
            self.adjust_risk(0.7, duration_minutes=15)
            self.news_queue.put(f"Negative currency news detected (Sentiment: {avg_sentiment:.2f})")

    def check_macroeconomic_news(self):
        self._check_news_category('macroeconomic', 'macroeconomic GDP', self.handle_macroeconomic_news)

    def handle_macroeconomic_news(self, articles, avg_sentiment):
        if avg_sentiment < -0.5:
            self.adjust_risk(0.5, duration_minutes=30)
            self.news_queue.put(f"Negative macroeconomic news detected (Sentiment: {avg_sentiment:.2f})")

    def check_geopolitical_news(self):
        self._check_news_category('geopolitical', 'geopolitical conflict', self.handle_geopolitical_news)

    def handle_geopolitical_news(self, articles, avg_sentiment):
        if avg_sentiment < -0.6:
            self.adjust_risk(0.4, duration_minutes=60)
            self.news_queue.put(f"Negative geopolitical news detected (Sentiment: {avg_sentiment:.2f})")

    def check_global_trends(self):
        self._check_news_category('global_trends', 'global market trends', self.handle_global_trends)

    def handle_global_trends(self, articles, avg_sentiment):
        if avg_sentiment < -0.5:
            self.adjust_risk(0.6, duration_minutes=20)
            self.news_queue.put(f"Negative global trends detected (Sentiment: {avg_sentiment:.2f})")

    def check_investor_sentiment(self):
        self._check_news_category('investor_sentiment', 'investor sentiment forex', self.handle_investor_sentiment)

    def handle_investor_sentiment(self, articles, avg_sentiment):
        if avg_sentiment < -0.5:
            self.adjust_risk(0.7, duration_minutes=15)
            self.news_queue.put(f"Negative investor sentiment detected (Sentiment: {avg_sentiment:.2f})")

    def check_market_liquidity(self):
        self._check_news_category('market_liquidity', 'market liquidity forex', self.handle_market_liquidity)

    def handle_market_liquidity(self, articles, avg_sentiment):
        if avg_sentiment < -0.5:
            self.adjust_risk(0.6, duration_minutes=20)
            self.news_queue.put(f"Negative market liquidity news detected (Sentiment: {avg_sentiment:.2f})")

    def check_fiscal_policy(self):
        self._check_news_category('fiscal_policy', 'fiscal policy tax', self.handle_fiscal_policy)

    def handle_fiscal_policy(self, articles, avg_sentiment):
        if avg_sentiment < -0.5:
            self.adjust_risk(0.5, duration_minutes=30)
            self.news_queue.put(f"Negative fiscal policy news detected (Sentiment: {avg_sentiment:.2f})")

    def _get_tweets(self, query):
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
        sentiment = self.sid.polarity_scores(tweet['text'])['compound']
        text_lower = tweet['text'].lower()
        if "announcement" in text_lower or "update" in text_lower:
            if sentiment < -0.3:
                self.adjust_risk(0.7, duration_minutes=15)
                self.news_queue.put(f"Negative tweet detected: {tweet['text']} (Sentiment: {sentiment:.2f})")

    def check_twitter(self):
        queries = self.config.get('market_news', {}).get('twitter_queries', ['forex', 'EURUSD', 'GBPJPY'])
        for query in queries:
            tweets = self._get_tweets(query)
            for tweet in tweets:
                self.handle_tweet(tweet)

    def _get_truthsocial_posts(self):
        return []  # Placeholder

    def handle_truthsocial_post(self, post):
        sentiment = self.sid.polarity_scores(post['text'])['compound']
        if sentiment < -0.3:
            self.adjust_risk(0.7, duration_minutes=15)
            self.news_queue.put(f"Negative Truth Social post detected (Sentiment: {sentiment:.2f})")

    def check_truthsocial(self):
        posts = self._get_truthsocial_posts()
        for post in posts:
            self.handle_truthsocial_post(post)

    def check_economic_calendar(self):
        events = []  # Placeholder
        for event in events:
            if event.get('impact') == 'high' and (event['time'] - datetime.now(pytz.UTC)) < timedelta(hours=1):  # Updated to pytz UTC
                self.adjust_risk(0.5, duration_minutes=60)
                self.news_queue.put(f"High-impact event imminent: {event.get('name')}")

    def adjust_risk(self, factor, duration_minutes):
        if self.risk_adjusted_until and datetime.now(pytz.UTC) < self.risk_adjusted_until:  # Updated to pytz UTC
            return
        self.risk_factor = factor
        self.risk_adjusted_until = datetime.now(pytz.UTC) + timedelta(minutes=duration_minutes)  # Updated to pytz UTC
        self.logger.info(f"Risk adjusted to {factor} for {duration_minutes} minutes")
        self.news_queue.put(f"Risk adjusted to {factor} for {duration_minutes} minutes")

    def run_all_checks(self):
        while True:
            try:
                self.check_trump_news()
                self.check_nasdaq_news()
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
                time.sleep(300)
            except Exception as e:
                self.logger.error(f"Error in run_all_checks: {e}")
                time.sleep(60)


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