import logging
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ta  # Technical analysis library for feature engineering


class MachineLearning:
    def __init__(self, config):
        """
        Initialize the MachineLearning module for The 5%ers MT5 trading with EURUSD and GBPJPY.

        Args:
            config (dict): Configuration dictionary with model settings.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}  # Dictionary to hold multiple models
        self.trained = False  # Flag to indicate if models are trained

        # Load models during initialization
        self.load_models()

    def load_models(self):
        """
        Load and initialize machine learning models based on configuration.
        """
        model_configs = self.config.get('machine_learning', {}).get('models', [])
        if not model_configs:
            self.logger.warning("No models specified in config; ML functionality disabled")
            return

        for model_config in model_configs:
            model_type = model_config.get('type')
            if model_type == 'random_forest':
                self.models['random_forest'] = RandomForestClassifier(
                    n_estimators=model_config.get('n_estimators', 100),
                    max_depth=model_config.get('max_depth', 10),
                    random_state=42  # For reproducibility
                )
                self.logger.info(f"Initialized {model_type} model with n_estimators={model_config.get('n_estimators', 100)}, "
                                 f"max_depth={model_config.get('max_depth', 10)}")
            else:
                self.logger.warning(f"Model type '{model_type}' not yet supported")

    def train_models(self, X_train, y_train):
        """
        Train the machine learning models using provided training data.

        Args:
            X_train (pd.DataFrame): Feature data for training.
            y_train (pd.Series): Target labels for training.
        """
        if not self.models:
            self.logger.error("No models loaded to train")
            return

        try:
            for name, model in self.models.items():
                self.logger.info(f"Training {name} model...")
                model.fit(X_train, y_train)
                self.logger.info(f"{name} model trained successfully")
            self.trained = True
        except Exception as e:
            self.logger.error(f"Failed to train models: {str(e)}")
            self.trained = False

    def predict(self, X):
        """
        Generate predictions from each model and combine them (e.g., via voting).

        Args:
            X (pd.DataFrame): Feature data for prediction.

        Returns:
            np.array: Combined predictions, or None if models aren't trained or prediction fails.
        """
        if not self.trained:
            self.logger.error("Models are not trained yet")
            return None

        if X.empty:
            self.logger.error("No feature data provided for prediction")
            return None

        try:
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(X)

            # Simple majority voting for binary classification
            if len(self.models) > 1:
                combined_predictions = np.array([max(set(pred), key=pred.count)
                                               for pred in zip(*predictions.values())])
            else:
                combined_predictions = list(predictions.values())[0]
            self.logger.debug(f"Generated predictions for {len(X)} data points")
            return combined_predictions
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return None

    def enhance_signals(self, signals, data):
        """
        Enhance trading signals using machine learning models for EURUSD and GBPJPY.

        Args:
            signals (list): List of trading signals from SignalGenerator.
            data (pd.DataFrame): Market data used for feature generation.

        Returns:
            list: Enhanced signals (filtered or adjusted by ML predictions).
        """
        self.logger.info("Enhancing signals with machine learning...")
        if not self.trained:
            self.logger.warning("Models not trained; returning original signals")
            return signals

        if not signals or data.empty:
            self.logger.warning("No signals or data provided; returning original signals")
            return signals

        # Generate features from market data
        features = self.generate_features(data)

        # Align features with signals (assuming signals correspond to latest data points)
        if len(features) < len(signals):
            self.logger.error(f"Feature data length ({len(features)}) less than signals length ({len(signals)})")
            return signals
        elif len(features) > len(signals):
            features = features.tail(len(signals))  # Trim to match signals

        # Predict using the ensemble
        predictions = self.predict(features)
        if predictions is None:
            self.logger.error("Prediction failed; returning original signals")
            return signals

        # Enhance signals: filter based on positive predictions (1 for trade confirmation)
        enhanced_signals = []
        for signal, pred in zip(signals, predictions):
            if pred == 1:  # Assuming 1 indicates a valid trade
                enhanced_signals.append(signal)
            else:
                self.logger.debug(f"Signal filtered out by ML prediction: {signal}")

        self.logger.info(f"Enhanced signals: kept {len(enhanced_signals)} out of {len(signals)}")
        return enhanced_signals

    def generate_features(self, data):
        """
        Generate features from market data for machine learning models.

        Args:
            data (pd.DataFrame): Market data with columns like 'Close'.

        Returns:
            pd.DataFrame: Feature set for model prediction, or empty DataFrame on failure.
        """
        if data.empty:
            self.logger.error("No market data provided for feature generation")
            return pd.DataFrame()

        features = data.copy()
        try:
            # Example features for EURUSD and GBPJPY
            features['ma_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            features['rsi'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
            features['volatility'] = data['Close'].rolling(window=20, min_periods=1).std()
            features['macd'] = ta.trend.MACD(data['Close']).macd()
            features['bollinger_upper'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
            features['bollinger_lower'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()

            # Drop NaN values from indicators
            features = features.dropna()
            self.logger.debug(f"Generated features with shape: {features.shape}")
            return features
        except Exception as e:
            self.logger.error(f"Feature generation failed: {str(e)}")
            return pd.DataFrame()

    def train_test_split_data(self, data, target, test_size=0.2):
        """
        Split data into training and testing sets for model evaluation.

        Args:
            data (pd.DataFrame): Feature data.
            target (pd.Series): Target labels.
            test_size (float): Proportion of data for testing.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) or None if splitting fails.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                                test_size=test_size,
                                                                random_state=42)
            self.logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Train-test split failed: {str(e)}")
            return None


# Example usage (commented out for reference)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        "machine_learning": {
            "models": [
                {"type": "random_forest", "n_estimators": 100, "max_depth": 10}
            ]
        }
    }
    ml = MachineLearning(config)
    # Example data and signals
    data = pd.DataFrame({'Close': np.random.rand(100)})
    signals = [{"action": "buy", "entry_price": p, "symbol": "EURUSD"} for p in data['Close']]
    target = pd.Series(np.random.randint(0, 2, size=100))  # Dummy target: 1 for trade, 0 for no trade
    features = ml.generate_features(data)
    split = ml.train_test_split_data(features, target)
    if split:
        X_train, X_test, y_train, y_test = split
        ml.train_models(X_train, y_train)
        enhanced = ml.enhance_signals(signals, features)
        print(f"Enhanced signals: {len(enhanced)}")