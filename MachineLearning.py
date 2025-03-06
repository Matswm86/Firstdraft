import logging
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ta  # Technical analysis library for feature engineering (ensure it's installed)

class MachineLearning:
    def __init__(self, config):
        """Initialize the MachineLearning module with a configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = {}  # Dictionary to hold multiple models
        self.trained = False  # Flag to indicate if models are trained

    def load_models(self):
        """Load and initialize machine learning models based on configuration."""
        model_configs = self.config.get('models', [])
        for model_config in model_configs:
            model_type = model_config.get('type')
            if model_type == 'random_forest':
                self.models['random_forest'] = RandomForestClassifier(
                    n_estimators=model_config.get('n_estimators', 100),
                    max_depth=model_config.get('max_depth', 10)
                )
                self.logger.info(f"Initialized {model_type} model")
            # Add other model types here in the future (e.g., 'gradient_boosting', 'neural_network')
            else:
                self.logger.warning(f"Model type {model_type} not yet supported")

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
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")
            model.fit(X_train, y_train)
            self.logger.info(f"{name} model trained successfully")
        self.trained = True

    def predict(self, X):
        """
        Generate predictions from each model and combine them (e.g., via voting).

        Args:
            X (pd.DataFrame): Feature data for prediction.

        Returns:
            np.array: Combined predictions, or None if models aren't trained.
        """
        if not self.trained:
            self.logger.error("Models are not trained yet")
            return None

        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        # Simple majority voting for binary classification
        # Adjust this for regression or multi-class scenarios later
        combined_predictions = np.array([max(set(pred), key=pred.count)
                                       for pred in zip(*predictions.values())])
        return combined_predictions

    def enhance_signals(self, signals, data):
        """
        Enhance trading signals using machine learning models.

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

        # Generate features from market data
        features = self.generate_features(data)

        # Ensure feature DataFrame aligns with signals length
        if len(features) != len(signals):
            self.logger.error("Feature data length does not match signals length")
            return signals

        # Predict using the ensemble
        predictions = self.predict(features)

        if predictions is None:
            self.logger.error("Prediction failed; returning original signals")
            return signals

        # Filter signals: keep only those where prediction is positive (e.g., 1)
        # This assumes binary classification; adjust as needed
        enhanced_signals = [signal for signal, pred in zip(signals, predictions) if pred == 1]
        self.logger.info(f"Enhanced signals: kept {len(enhanced_signals)} out of {len(signals)}")
        return enhanced_signals

    def generate_features(self, data):
        """
        Generate features from market data for machine learning models.

        Args:
            data (pd.DataFrame): Market data with columns like 'Close'.

        Returns:
            pd.DataFrame: Feature set for model prediction.
        """
        features = data.copy()
        try:
            # Example features: moving average and RSI
            features['ma_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            features['rsi'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
            # Drop NaN values that might arise from indicators
            features = features.dropna()
            return features
        except Exception as e:
            self.logger.error(f"Feature generation failed: {str(e)}")
            return data  # Return original data as fallback

    def train_test_split_data(self, data, target, test_size=0.2):
        """
        Split data into training and testing sets for model evaluation.

        Args:
            data (pd.DataFrame): Feature data.
            target (pd.Series): Target labels.
            test_size (float): Proportion of data for testing.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(data, target, test_size=test_size, random_state=42)

# Example usage (commented out for reference)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     config = {
#         "models": [
#             {"type": "random_forest", "n_estimators": 100, "max_depth": 10}
#         ]
#     }
#     ml = MachineLearning(config)
#     ml.load_models()
#     # Example data and signals
#     data = pd.DataFrame({'Close': np.random.rand(100)})
#     signals = [1] * 100  # Dummy signals
#     # ml.train_models(X_train, y_train)  # Requires prepared X_train, y_train
#     # enhanced = ml.enhance_signals(signals, data)