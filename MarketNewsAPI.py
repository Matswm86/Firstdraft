import os
import logging
from datetime import datetime, timedelta
import pytz
import requests
import time
import queue
import threading
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import json
from pathlib import Path
import re
import nltk


class MarketNewsAPI:
    """
    Market News API for retrieving and analyzing financial news.
    Handles news collection, sentiment analysis, and risk adjustments.
    """

    def __init__(self, config, news_queue=None):
        """
        Initialize MarketNewsAPI with configuration settings.

        Args:
            config (dict): Configuration dictionary with news API settings
            news_queue (queue.Queue, optional): Queue for sending news updates to other modules
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create news queue if not provided
        self.news_queue = queue.Queue() if news_queue is None else news_queue

        # Initialize API credentials
        self._initialize_api_keys()

        # Set up news categories and their last check times
        self.categories = {
            'forex': {'query': 'forex currency "exchange rate"', 'interval': 15},
            'macro': {'query': 'macroeconomic GDP inflation', 'interval': 30},
            'rates': {'query': 'interest rates central bank', 'interval': 30},
            'geopolitical': {'query': 'geopolitical conflict trade war', 'interval': 60},
            'market_sentiment': {'query': 'market sentiment investor confidence', 'interval': 15},
            'technical_analysis': {'query': 'technical analysis forex', 'interval': 30},
            'eurusd': {'query': 'EURUSD euro dollar', 'interval': 10},
            'gbpjpy': {'query': 'GBPJPY pound yen', 'interval': 10}
        }

        # Add last check time to all categories with UTC
        for category in self.categories:
            self.categories[category]['last_checked'] = datetime.now(pytz.UTC) - timedelta(
                minutes=self.categories[category]['interval'])

        # Ensure NLTK data is available
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            self.logger.info("Downloading NLTK VADER lexicon...")
            nltk.download('vader_lexicon', quiet=True)

        # Initialize sentiment analyzer
        self._initialize_sentiment_analyzer()

        # Set up HTTP session with retry
        self.session = self._create_http_session()

        # Risk adjustment settings
        self.risk_adjusted_until = None
        self.risk_factor = 1.0
        self.min_risk_factor = float(config.get('market_news', {}).get('min_risk_factor', 0.3))

        # News cache with tuned expiry
        self.news_cache = {}
        self.cache_duration = timedelta(hours=12)

        # Create a stop event for threads
        self.stop_event = threading.Event()
        self.check_thread = None

        # News API settings
        self.news_endpoint = "https://newsapi.org/v2/everything"
        self.news_api_max_results = int(config.get('market_news', {}).get('max_results', 10))

        # Twitter API settings
        self.twitter_endpoint = "https://api.twitter.com/2/tweets/search/recent"
        self.twitter_api_max_results = int(config.get('market_news', {}).get('twitter_max_results', 10))

        # Economic calendar
        self.economic_events = self._load_economic_calendar()

        # News sentiment thresholds
        self.sentiment_thresholds = config.get('market_news', {}).get('sentiment_thresholds', {
            'very_negative': -0.6,
            'negative': -0.3,
            'neutral': 0.1,
            'positive': 0.4,
            'very_positive': 0.7
        })

        self.logger.info("Market News API initialized")

    def _initialize_api_keys(self):
        """Initialize API keys from config or environment variables"""
        try:
            # News API key
            self.news_api_key = self.config.get('market_news', {}).get('news_api_key')
            if not self.news_api_key or self.news_api_key == "your_news_api_key":
                self.news_api_key = os.environ.get('NEWS_API_KEY')

            # Twitter API key
            self.twitter_bearer_token = self.config.get('market_news', {}).get('twitter_bearer_token')
            if not self.twitter_bearer_token or self.twitter_bearer_token == "your_twitter_bearer_token":
                self.twitter_bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')

            # Log availability of APIs
            self.news_api_enabled = bool(self.news_api_key)
            self.twitter_api_enabled = bool(self.twitter_bearer_token)

            if not self.news_api_enabled and not self.twitter_api_enabled:
                self.logger.warning("No news sources enabled. MarketNewsAPI will have limited functionality.")

            self.logger.info(f"News API enabled: {self.news_api_enabled}, Twitter API enabled: {self.twitter_api_enabled}")
        except Exception as e:
            self.logger.error(f"Error initializing API keys: {str(e)}")

    def _initialize_sentiment_analyzer(self):
        """Initialize the sentiment analyzer with improved error handling"""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.sid = SentimentIntensityAnalyzer()
            self.sentiment_analyzer_available = True
            self.logger.info("VADER sentiment analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize VADER sentiment analyzer: {str(e)}. Using fallback analyzer.")
            self.sentiment_analyzer_available = False
            self.sid = self._simple_sentiment_analyzer

    def _simple_sentiment_analyzer(self, text):
        """
        Simple fallback sentiment analyzer when NLTK is not available.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Sentiment scores
        """
        positive_words = ['increase', 'gain', 'positive', 'rise', 'up', 'higher', 'bull', 'bullish', 'growth', 'strong']
        negative_words = ['decrease', 'loss', 'negative', 'fall', 'down', 'lower', 'bear', 'bearish', 'recession', 'weak']
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        total_count = positive_count + negative_count
        score = float((positive_count - negative_count) / total_count) if total_count > 0 else 0.0
        return {'compound': score}

    def _create_http_session(self):
        """Create an HTTP session with retry logic"""
        try:
            session = requests.Session()
            retries = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"]
            )
            adapter = HTTPAdapter(max_retries=retries)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            return session
        except Exception as e:
            self.logger.error(f"Error creating HTTP session: {str(e)}")
            return requests.Session()  # Fallback to basic session

    def _load_economic_calendar(self):
        """Load economic calendar data from file"""
        calendar_file = self.config.get('market_news', {}).get('calendar_file', 'data/economic_calendar.json')
        try:
            calendar_path = Path(calendar_file)
            if calendar_path.exists():
                with calendar_path.open('r', encoding='utf-8') as f:
                    events = json.load(f)
                # Ensure events have UTC datetime objects
                for event in events:
                    if 'time' in event:
                        event['time'] = datetime.fromisoformat(event['time'].replace('Z', '+00:00')).replace(tzinfo=pytz.UTC)
                self.logger.info(f"Loaded {len(events)} economic events from {calendar_file}")
                return events
            else:
                self.logger.warning(f"Economic calendar file not found at {calendar_file}")
                return []
        except Exception as e:
            self.logger.error(f"Error loading economic calendar: {str(e)}")
            return []

    def _fetch_news(self, query, from_date, max_results=None):
        """
        Fetch news articles from News API with cache.

        Args:
            query (str): Search query
            from_date (datetime): Start date for search
            max_results (int, optional): Maximum results to return

        Returns:
            list: List of news articles
        """
        if not self.news_api_enabled:
            self.logger.debug("News API not enabled; skipping fetch")
            return []

        cache_key = f"news_{query}_{from_date.strftime('%Y-%m-%d')}"
        if (cache_key in self.news_cache and
                (datetime.now(pytz.UTC) - self.news_cache[cache_key]['timestamp']) < self.cache_duration):
            self.logger.debug(f"Using cached news data for query: {query}")
            return self.news_cache[cache_key]['data']

        max_results = max_results or self.news_api_max_results

        try:
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': max_results,
                'apiKey': self.news_api_key
            }
            response = self.session.get(self.news_endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])
            self.news_cache[cache_key] = {
                'data': articles,
                'timestamp': datetime.now(pytz.UTC)
            }
            self.logger.info(f"Fetched {len(articles)} news articles for query: '{query}'")
            return articles
        except requests.RequestException as e:
            self.logger.error(f"HTTP error fetching news for '{query}': {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching news for '{query}': {str(e)}")
            return []

    def _analyze_sentiment(self, text):
        """
        Analyze sentiment of a text with optimized preprocessing.

        Args:
            text (str): Text to analyze

        Returns:
            float: Sentiment score (-1 to 1)
        """
        if not text or not isinstance(text, str):
            return 0.0

        try:
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            text = re.sub(r'[^\w\s]', '', text).strip()  # Remove special chars
            if not text:
                return 0.0

            sentiment = self.sid.polarity_scores(text) if self.sentiment_analyzer_available else self.sid(text)
            return float(sentiment['compound'])
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for text '{text[:50]}...': {str(e)}")
            return 0.0

    def _check_news_category(self, category):
        """
        Check for news in a specific category.

        Args:
            category (str): News category to check

        Returns:
            dict: News analysis results
        """
        if category not in self.categories:
            self.logger.warning(f"Unknown news category: {category}")
            return {"status": "error", "message": "Unknown category"}

        category_data = self.categories[category]
        query = category_data['query']
        last_checked = category_data['last_checked']
        interval = category_data['interval']
        now = datetime.now(pytz.UTC)

        if (now - last_checked).total_seconds() < interval * 60:
            self.logger.debug(f"Skipping check for {category}; last checked at {last_checked}")
            return {"status": "skipped", "message": f"Too soon to check again; last checked {last_checked}"}

        self.categories[category]['last_checked'] = now
        from_date = now - timedelta(hours=24)
        articles = self._fetch_news(query, from_date)

        if not articles:
            return {"status": "no_data", "message": f"No articles found for {category}"}

        sentiments = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}".strip()
            if text:
                sentiment = self._analyze_sentiment(text)
                if abs(sentiment) > 0.1:  # Significant sentiments only
                    sentiments.append(sentiment)

        avg_sentiment = float(sum(sentiments) / len(sentiments)) if sentiments else 0.0

        result = {
            "status": "success",
            "category": category,
            "avg_sentiment": avg_sentiment,
            "sentiment_category": "neutral",
            "article_count": len(articles),
            "analyzed_count": len(sentiments),
            "timestamp": now.isoformat()
        }

        thresholds = self.sentiment_thresholds
        if avg_sentiment <= thresholds['very_negative']:
            result["sentiment_category"] = "very_negative"
        elif avg_sentiment <= thresholds['negative']:
            result["sentiment_category"] = "negative"
        elif avg_sentiment >= thresholds['very_positive']:
            result["sentiment_category"] = "very_positive"
        elif avg_sentiment >= thresholds['positive']:
            result["sentiment_category"] = "positive"

        self._process_news_sentiment(category, result)
        return result

    def _process_news_sentiment(self, category, result):
        """
        Process news sentiment analysis results and adjust risk.

        Args:
            category (str): News category
            result (dict): Sentiment analysis results
        """
        sentiment_category = result['sentiment_category']
        avg_sentiment = result['avg_sentiment']

        try:
            if sentiment_category in ['very_negative', 'negative']:
                adjustment_strength = abs(avg_sentiment) * 0.7
                risk_factor = max(self.min_risk_factor, 1.0 - adjustment_strength)
                duration_minutes = {
                    'forex': 15, 'eurusd': 15, 'gbpjpy': 15, 'market_sentiment': 15,
                    'geopolitical': 60, 'macro': 30, 'rates': 30, 'technical_analysis': 30
                }.get(category, 30)
                self.adjust_risk(risk_factor, duration_minutes)
                message = (
                    f"Negative {category} news detected (Sentiment: {avg_sentiment:.2f}). "
                    f"Risk adjusted to {risk_factor:.2f} for {duration_minutes} minutes."
                )
                self.news_queue.put({
                    "type": "risk_adjustment",
                    "category": category,
                    "sentiment": float(avg_sentiment),
                    "risk_factor": float(risk_factor),
                    "duration_minutes": duration_minutes,
                    "message": message,
                    "timestamp": datetime.now(pytz.UTC).isoformat()
                })
                self.logger.info(message)

            elif sentiment_category in ['very_positive', 'positive']:
                message = f"Positive {category} news detected (Sentiment: {avg_sentiment:.2f})."
                self.news_queue.put({
                    "type": "news_alert",
                    "category": category,
                    "sentiment": float(avg_sentiment),
                    "message": message,
                    "timestamp": datetime.now(pytz.UTC).isoformat()
                })
                self.logger.info(message)
        except Exception as e:
            self.logger.error(f"Error processing news sentiment: {str(e)}")

    def manually_check_category(self, category):
        """
        Manually trigger a check for a specific news category.

        Args:
            category (str): Category to check

        Returns:
            dict: Results of the check
        """
        if category not in self.categories:
            self.logger.warning(f"Unknown category: {category}")
            return {"status": "error", "message": "Unknown category"}
        self.categories[category]['last_checked'] = datetime.now(pytz.UTC) - timedelta(hours=1)
        return self._check_news_category(category)

    def manually_adjust_risk(self, factor, duration_minutes):
        """
        Manually adjust risk level.

        Args:
            factor (float): Risk factor (0-1)
            duration_minutes (int): Duration in minutes

        Returns:
            dict: Result of adjustment
        """
        try:
            factor = float(factor)
            duration_minutes = int(duration_minutes)
            if not 0 < factor <= 1:
                return {"status": "error", "message": "Factor must be between 0 and 1"}
            if duration_minutes <= 0:
                return {"status": "error", "message": "Duration must be positive"}
            self.adjust_risk(factor, duration_minutes)
            return {"status": "success", "message": f"Risk adjusted to {factor:.2f} for {duration_minutes} minutes"}
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid risk adjustment parameters: {str(e)}")
            return {"status": "error", "message": f"Invalid parameters: {str(e)}"}

    def test_sentiment_analyzer(self, text):
        """
        Test sentiment analyzer with sample text.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Sentiment analysis results
        """
        try:
            sentiment = self._analyze_sentiment(text)
            sentiment_category = "neutral"
            thresholds = self.sentiment_thresholds

            if sentiment <= thresholds['very_negative']:
                sentiment_category = "very_negative"
            elif sentiment <= thresholds['negative']:
                sentiment_category = "negative"
            elif sentiment >= thresholds['very_positive']:
                sentiment_category = "very_positive"
            elif sentiment >= thresholds['positive']:
                sentiment_category = "positive"

            return {
                "text": text,
                "sentiment_score": float(sentiment),
                "sentiment_category": sentiment_category,
                "analyzer_type": "VADER" if self.sentiment_analyzer_available else "Fallback"
            }
        except Exception as e:
            self.logger.error(f"Error testing sentiment analyzer: {str(e)}")
            return {"status": "error", "message": str(e)}

    def get_recent_news(self, category=None, limit=10):
        """
        Get recent news items.

        Args:
            category (str, optional): Category to filter by
            limit (int): Maximum number of news items to return

        Returns:
            list: Recent news items
        """
        try:
            if category and category not in self.categories:
                self.logger.warning(f"Invalid category: {category}")
                return []
            query = self.categories[category]['query'] if category else "forex"
            from_date = datetime.now(pytz.UTC) - timedelta(hours=24)
            news = self._fetch_news(query, from_date, max_results=limit)
            processed_news = []
            for item in news:
                title = item.get('title', 'No Title')
                description = item.get('description', '')
                text = f"{title} {description}".strip()
                sentiment = self._analyze_sentiment(text)
                processed_news.append({
                    "title": title,
                    "description": description,
                    "url": item.get('url', ''),
                    "published_at": item.get('publishedAt', ''),
                    "source": item.get('source', {}).get('name', 'Unknown'),
                    "sentiment": float(sentiment)
                })
            return processed_news
        except Exception as e:
            self.logger.error(f"Error getting recent news: {str(e)}")
            return []

    def _get_tweets(self, query):
        """
        Get tweets from Twitter API with cache.

        Args:
            query (str): Search query

        Returns:
            list: List of tweets
        """
        if not self.twitter_api_enabled:
            self.logger.debug("Twitter API not enabled; skipping fetch")
            return []

        cache_key = f"twitter_{query}"
        if (cache_key in self.news_cache and
                (datetime.now(pytz.UTC) - self.news_cache[cache_key]['timestamp']) < timedelta(minutes=15)):
            self.logger.debug(f"Using cached Twitter data for query: {query}")
            return self.news_cache[cache_key]['data']

        try:
            headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}
            params = {
                'query': query,
                'max_results': self.twitter_api_max_results,
                'tweet.fields': 'created_at,author_id',
                'expansions': 'author_id',
                'user.fields': 'username'
            }
            response = self.session.get(self.twitter_endpoint, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            tweets = data.get('data', [])
            if 'includes' in data and 'users' in data['includes']:
                users = {user['id']: user for user in data['includes']['users']}
                for tweet in tweets:
                    author_id = tweet.get('author_id')
                    if author_id and author_id in users:
                        tweet['user'] = users[author_id]
            self.news_cache[cache_key] = {
                'data': tweets,
                'timestamp': datetime.now(pytz.UTC)
            }
            self.logger.info(f"Fetched {len(tweets)} tweets for query: '{query}'")
            return tweets
        except requests.RequestException as e:
            self.logger.error(f"HTTP error fetching tweets for '{query}': {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching tweets for '{query}': {str(e)}")
            return []

    def check_twitter(self):
        """
        Check Twitter for relevant tweets across configured queries.
        """
        queries = self.config.get('market_news', {}).get('twitter_queries', ['forex', 'EURUSD', 'GBPJPY'])
        for query in queries:
            tweets = self._get_tweets(query)
            for tweet in tweets:
                self.handle_tweet(tweet)

    def handle_tweet(self, tweet):
        """
        Process a single tweet and adjust risk if necessary.

        Args:
            tweet (dict): Tweet data
        """
        try:
            text = tweet.get('text', '')
            sentiment = self._analyze_sentiment(text)
            text_lower = text.lower()
            if "announcement" in text_lower or "update" in text_lower:
                if sentiment < -0.3:
                    self.adjust_risk(0.7, duration_minutes=15)
                    message = f"Negative tweet detected: {text[:50]}... (Sentiment: {sentiment:.2f})"
                    self.news_queue.put({
                        "type": "risk_adjustment",
                        "category": "twitter",
                        "sentiment": float(sentiment),
                        "risk_factor": 0.7,
                        "duration_minutes": 15,
                        "message": message,
                        "timestamp": datetime.now(pytz.UTC).isoformat()
                    })
                    self.logger.info(message)
        except Exception as e:
            self.logger.error(f"Error handling tweet: {str(e)}")

    def _get_truthsocial_posts(self):
        """
        Placeholder for fetching Truth Social posts (not implemented).

        Returns:
            list: Empty list as placeholder
        """
        self.logger.warning("Truth Social integration not implemented")
        return []

    def handle_truthsocial_post(self, post):
        """
        Process a Truth Social post and adjust risk if necessary.

        Args:
            post (dict): Post data
        """
        try:
            text = post.get('text', '')
            sentiment = self._analyze_sentiment(text)
            if sentiment < -0.3:
                self.adjust_risk(0.7, duration_minutes=15)
                message = f"Negative Truth Social post detected (Sentiment: {sentiment:.2f})"
                self.news_queue.put({
                    "type": "risk_adjustment",
                    "category": "truthsocial",
                    "sentiment": float(sentiment),
                    "risk_factor": 0.7,
                    "duration_minutes": 15,
                    "message": message,
                    "timestamp": datetime.now(pytz.UTC).isoformat()
                })
                self.logger.info(message)
        except Exception as e:
            self.logger.error(f"Error handling Truth Social post: {str(e)}")

    def check_truthsocial(self):
        """
        Check Truth Social for relevant posts.
        """
        posts = self._get_truthsocial_posts()
        for post in posts:
            self.handle_truthsocial_post(post)

    def check_economic_calendar(self):
        """
        Check economic calendar for upcoming high-impact events.
        """
        try:
            now = datetime.now(pytz.UTC)
            for event in self.economic_events:
                event_time = event.get('time')
                if not event_time:
                    continue
                time_until = (event_time - now).total_seconds() / 3600  # Hours until event
                if event.get('impact') == 'high' and 0 < time_until <= 1:
                    self.adjust_risk(0.5, duration_minutes=60)
                    message = f"High-impact event imminent: {event.get('name', 'Unknown')} at {event_time.isoformat()}"
                    self.news_queue.put({
                        "type": "risk_adjustment",
                        "category": "economic_calendar",
                        "sentiment": 0.0,
                        "risk_factor": 0.5,
                        "duration_minutes": 60,
                        "message": message,
                        "timestamp": now.isoformat()
                    })
                    self.logger.info(message)
        except Exception as e:
            self.logger.error(f"Error checking economic calendar: {str(e)}")

    def adjust_risk(self, factor, duration_minutes):
        """
        Adjust risk factor for a specified duration.

        Args:
            factor (float): Risk factor (0-1)
            duration_minutes (int): Duration in minutes
        """
        try:
            now = datetime.now(pytz.UTC)
            if self.risk_adjusted_until and now < self.risk_adjusted_until:
                self.logger.debug(f"Risk adjustment skipped; active until {self.risk_adjusted_until}")
                return
            self.risk_factor = max(self.min_risk_factor, min(1.0, float(factor)))
            self.risk_adjusted_until = now + timedelta(minutes=duration_minutes)
            self.logger.info(f"Risk factor adjusted to {self.risk_factor:.2f} until {self.risk_adjusted_until}")
        except Exception as e:
            self.logger.error(f"Error adjusting risk: {str(e)}")

    def run_all_checks(self):
        """
        Run continuous checks across all news categories and sources.
        """
        if self.check_thread and self.check_thread.is_alive():
            self.logger.info("Background checks already running")
            return

        def check_loop():
            while not self.stop_event.is_set():
                try:
                    for category in self.categories:
                        self._check_news_category(category)
                    self.check_twitter()
                    self.check_truthsocial()
                    self.check_economic_calendar()
                    time.sleep(300)  # 5-minute interval
                except Exception as e:
                    self.logger.error(f"Error in news check loop: {str(e)}")
                    time.sleep(60)

        self.check_thread = threading.Thread(target=check_loop, daemon=True)
        self.check_thread.start()
        self.logger.info("Started background news checks")

    # Category-specific check methods
    def check_trump_news(self):
        self._check_news_category('forex')  # Placeholder mapping

    def check_nasdaq_news(self):
        self._check_news_category('market_sentiment')  # Placeholder mapping

    def check_currency_news(self):
        self._check_news_category('forex')

    def check_macroeconomic_news(self):
        self._check_news_category('macro')

    def check_geopolitical_news(self):
        self._check_news_category('geopolitical')

    def check_global_trends(self):
        self._check_news_category('market_sentiment')  # Placeholder mapping

    def check_investor_sentiment(self):
        self._check_news_category('market_sentiment')

    def check_market_liquidity(self):
        self._check_news_category('forex')  # Placeholder mapping

    def check_fiscal_policy(self):
        self._check_news_category('macro')  # Placeholder mapping

    def get_status(self):
        """
        Get current status of MarketNewsAPI.

        Returns:
            dict: Status information
        """
        try:
            now = datetime.now(pytz.UTC)
            return {
                "news_api_enabled": self.news_api_enabled,
                "twitter_api_enabled": self.twitter_api_enabled,
                "sentiment_analyzer_available": self.sentiment_analyzer_available,
                "risk_factor": float(self.risk_factor),
                "risk_adjusted_until": self.risk_adjusted_until.isoformat() if self.risk_adjusted_until else None,
                "background_checks_running": bool(self.check_thread and self.check_thread.is_alive()),
                "categories": {
                    category: {
                        "last_checked": data['last_checked'].isoformat(),
                        "interval_minutes": int(data['interval'])
                    } for category, data in self.categories.items()
                },
                "upcoming_economic_events": len([
                    event for event in self.economic_events
                    if event.get('time') and event['time'] > now
                ]),
                "timestamp": now.isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error generating status: {str(e)}")
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "market_news": {
            "news_api_key": "your_news_api_key",
            "twitter_bearer_token": "your_twitter_bearer_token",
            "twitter_queries": ["forex", "EURUSD", "GBPJPY", "Dollar", "Trump", "Musk", "Euro"],
            "min_risk_factor": 0.3,
            "max_results": 10,
            "twitter_max_results": 10,
            "calendar_file": "data/economic_calendar.json",
            "sentiment_thresholds": {
                "very_negative": -0.6,
                "negative": -0.3,
                "neutral": 0.1,
                "positive": 0.4,
                "very_positive": 0.7
            }
        }
    }
    news_queue = queue.Queue()
    news_api = MarketNewsAPI(config, news_queue)
    news_api.run_all_checks()