import logging
import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import ta  # Technical analysis library
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb  # XGBoost for gradient boosting
import lightgbm as lgb  # LightGBM for gradient boosting


class MachineLearning:
    def __init__(self, config):
        """
        Initialize the MachineLearning module with ensemble capabilities.

        Args:
            config (dict): Configuration dictionary with model settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract ML config
        self.ml_config = config.get('machine_learning', {})

        # Model storage
        self.models = {}  # Individual models
        self.ensemble = None  # Combined ensemble model
        self.scaler = StandardScaler()  # Feature scaling
        self.trained = False
        self.features_columns = []  # Store feature column names

        # Model persistence
        self.model_dir = self.ml_config.get('model_dir', 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        # Performance tracking
        self.performance = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }

        # Initialize models
        self.load_models()

        self.logger.info(f"MachineLearning initialized: Trained={self.trained}, Models={len(self.models)}")

    def load_models(self):
        """
        Initialize machine learning models or load pre-trained models.
        """
        # Check if we should load saved models
        if self.ml_config.get('load_saved_models', False):
            try:
                saved_model_path = os.path.join(self.model_dir, 'ensemble.joblib')
                if os.path.exists(saved_model_path):
                    self.ensemble = joblib.load(saved_model_path)
                    self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
                    self.features_columns = joblib.load(os.path.join(self.model_dir, 'features.joblib'))
                    loaded_performance = joblib.load(os.path.join(self.model_dir, 'performance.joblib'))
                    # Ensure type safety for performance metrics
                    for key in self.performance:
                        if key in loaded_performance:
                            self.performance[key] = float(loaded_performance[key])
                    self.trained = True
                    self.logger.info("Loaded pre-trained ensemble model")
                    return
                else:
                    self.logger.warning("No saved models found, initializing new models")
            except Exception as e:
                self.logger.error(f"Error loading saved models: {str(e)}")

        # Initialize models from configuration
        model_configs = self.ml_config.get('models', [])
        if not model_configs:
            self.logger.warning("No models specified in config; ML functionality limited")
            return

        for model_config in model_configs:
            model_type = model_config.get('type', '').lower()
            model_name = model_config.get('name', model_type)

            try:
                if model_type == 'random_forest':
                    self.models[model_name] = RandomForestClassifier(
                        n_estimators=int(model_config.get('n_estimators', 100)),
                        max_depth=int(model_config.get('max_depth', 10)),
                        min_samples_split=int(model_config.get('min_samples_split', 2)),
                        min_samples_leaf=int(model_config.get('min_samples_leaf', 1)),
                        class_weight=model_config.get('class_weight', 'balanced'),
                        random_state=42
                    )
                elif model_type == 'gradient_boosting':
                    self.models[model_name] = GradientBoostingClassifier(
                        n_estimators=int(model_config.get('n_estimators', 100)),
                        learning_rate=float(model_config.get('learning_rate', 0.1)),
                        max_depth=int(model_config.get('max_depth', 3)),
                        random_state=42
                    )
                elif model_type == 'xgboost':
                    self.models[model_name] = xgb.XGBClassifier(
                        n_estimators=int(model_config.get('n_estimators', 100)),
                        learning_rate=float(model_config.get('learning_rate', 0.1)),
                        max_depth=int(model_config.get('max_depth', 3)),
                        random_state=42
                    )
                elif model_type == 'lightgbm':
                    self.models[model_name] = lgb.LGBMClassifier(
                        n_estimators=int(model_config.get('n_estimators', 100)),
                        learning_rate=float(model_config.get('learning_rate', 0.1)),
                        max_depth=int(model_config.get('max_depth', -1)),
                        random_state=42
                    )
                else:
                    self.logger.warning(f"Model type '{model_type}' not supported")
                    continue

                self.logger.debug(f"Initialized {model_name} model")
            except Exception as e:
                self.logger.error(f"Error initializing {model_name} model: {str(e)}")

        # Set up ensemble if multiple models are defined
        if len(self.models) > 1:
            self.ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in self.models.items()],
                voting=self.ml_config.get('voting_method', 'soft')
            )
            self.logger.info(
                f"Initialized ensemble with {len(self.models)} models using {self.ml_config.get('voting_method', 'soft')} voting"
            )
        elif len(self.models) == 1:
            model_name, model = next(iter(self.models.items()))
            self.ensemble = model
            self.logger.info(f"Using single model: {model_name}")
        else:
            self.logger.warning("No models could be initialized")

    def train_models(self, data, target_generator=None, optimize=False):
        """
        Train the machine learning ensemble with proper time-series validation.

        Args:
            data (pd.DataFrame): Historical market data
            target_generator (callable, optional): Function to generate target labels
            optimize (bool): Whether to optimize hyperparameters

        Returns:
            bool: True if training was successful
        """
        if self.ensemble is None:
            self.logger.error("No models initialized for training")
            return False

        try:
            # Generate features
            features = self.generate_features(data)
            if features.empty:
                self.logger.error("Failed to generate features for training")
                return False

            # Store feature columns for future reference
            self.features_columns = features.columns.tolist()

            # Generate target labels if not provided
            if target_generator and callable(target_generator):
                target = target_generator(data)
            else:
                # Default target: 1 if close price increases, 0 otherwise
                target = (data['close'].shift(-1) > data['close']).astype(int)

            # Align features and target, drop NaN values
            features = features.loc[target.index]
            features = features.dropna()
            target = target.loc[features.index]

            if len(features) == 0:
                self.logger.error("No valid data after preprocessing")
                return False

            # Scale features
            scaled_features = self.scaler.fit_transform(features)

            # Train with time-series validation if optimizing
            if optimize:
                self._optimize_models(scaled_features, target)
            else:
                self.ensemble.fit(scaled_features, target)

            # Evaluate on training data
            predictions = self.ensemble.predict(scaled_features)
            self._update_performance_metrics(target, predictions, data.loc[features.index])

            # Save the trained models
            self._save_models()

            self.trained = True
            self.logger.info(
                f"Model training completed successfully. Accuracy: {self.performance['accuracy']:.2f}, Win Rate: {self.performance['win_rate']:.2f}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            return False

    def _optimize_models(self, X, y):
        """
        Optimize model hyperparameters using time-series cross-validation.

        Args:
            X (np.array): Feature data
            y (np.array): Target labels
        """
        self.logger.info("Optimizing model hyperparameters...")
        try:
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # If we have individual models, optimize each one
            if len(self.models) > 1:
                for name, model in self.models.items():
                    if hasattr(model, 'get_params'):
                        param_grid = self._get_param_grid(name, model)
                        if not param_grid:
                            self.logger.warning(f"No parameter grid defined for {name}, skipping optimization")
                            continue

                        grid_search = GridSearchCV(
                            model, param_grid, cv=tscv, scoring='f1',
                            n_jobs=-1, verbose=1
                        )
                        grid_search.fit(X, y)

                        self.models[name] = grid_search.best_estimator_
                        self.logger.info(f"Optimized {name} model: {grid_search.best_params_}")

                # Recreate ensemble with optimized models
                self.ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in self.models.items()],
                    voting=self.ml_config.get('voting_method', 'soft')
                )

            # Train the ensemble on the full dataset
            self.ensemble.fit(X, y)
        except Exception as e:
            self.logger.error(f"Error optimizing models: {str(e)}")
            raise

    def _get_param_grid(self, model_name, model):
        """
        Get appropriate parameter grid for hyperparameter optimization.

        Args:
            model_name (str): Name of the model
            model: Model instance

        Returns:
            dict: Parameter grid for GridSearchCV
        """
        try:
            if 'random_forest' in model_name.lower():
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif 'gradient' in model_name.lower():
                return {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            elif 'xgboost' in model_name.lower():
                return {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            elif 'lightgbm' in model_name.lower():
                return {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7, -1],
                    'num_leaves': [31, 50, 100]
                }
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error generating parameter grid for {model_name}: {str(e)}")
            return {}

    def _update_performance_metrics(self, true_labels, predictions, price_data):
        """
        Update performance metrics for the trained model.

        Args:
            true_labels (np.array): True target labels
            predictions (np.array): Model predictions
            price_data (pd.DataFrame): Price data for financial metrics
        """
        try:
            # Classification metrics
            self.performance['accuracy'] = float(accuracy_score(true_labels, predictions))
            self.performance['precision'] = float(precision_score(true_labels, predictions, zero_division=0))
            self.performance['recall'] = float(recall_score(true_labels, predictions, zero_division=0))
            self.performance['f1'] = float(f1_score(true_labels, predictions, zero_division=0))

            # Financial metrics if price data is available
            if 'close' in price_data.columns:
                price_changes = price_data['close'].pct_change().shift(-1)
                predicted_returns = np.where(predictions == 1, price_changes, -price_changes)

                # Win rate
                valid_returns = [r for r in predicted_returns if not np.isnan(r)]
                winning_trades = sum(1 for r in valid_returns if r > 0)
                total_trades = len(valid_returns)
                self.performance['win_rate'] = float(winning_trades / total_trades) if total_trades > 0 else 0.0

                # Profit factor
                gross_profit = float(sum(r for r in valid_returns if r > 0))
                gross_loss = float(abs(sum(r for r in valid_returns if r < 0)))
                self.performance['profit_factor'] = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')

                # Sharpe ratio (simplified, assuming risk-free rate = 0)
                if len(valid_returns) > 1:
                    mean_return = float(np.nanmean(predicted_returns))
                    std_return = float(np.nanstd(predicted_returns))
                    self.performance['sharpe_ratio'] = float(mean_return / std_return) if std_return > 0 else 0.0
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")

    def _save_models(self):
        """Save trained models and related data."""
        try:
            # Save ensemble model
            joblib.dump(self.ensemble, os.path.join(self.model_dir, 'ensemble.joblib'))

            # Save scaler
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))

            # Save feature columns
            joblib.dump(self.features_columns, os.path.join(self.model_dir, 'features.joblib'))

            # Save performance metrics
            joblib.dump(self.performance, os.path.join(self.model_dir, 'performance.joblib'))

            # Save timestamp
            with open(os.path.join(self.model_dir, 'timestamp.txt'), 'w', encoding='utf-8') as f:
                f.write(datetime.now(pytz.UTC).isoformat())

            self.logger.info(f"Models saved to {self.model_dir}")
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")

    def predict(self, data, include_probabilities=False):
        """
        Generate predictions for market data.

        Args:
            data (pd.DataFrame): Market data
            include_probabilities (bool): Whether to include prediction probabilities

        Returns:
            dict: Predictions and probabilities if requested
        """
        if not self.trained or self.ensemble is None:
            self.logger.error("Models not trained yet")
            return None

        try:
            # Generate features
            features = self.generate_features(data)
            if features.empty:
                self.logger.error("Failed to generate features for prediction")
                return None

            # Ensure we have the same features as during training
            if self.features_columns:
                common_cols = list(set(features.columns) & set(self.features_columns))
                if len(common_cols) < len(self.features_columns):
                    missing = set(self.features_columns) - set(features.columns)
                    self.logger.warning(f"Missing feature columns: {missing}")
                    for col in missing:
                        features[col] = 0.0  # Fill missing with zeros

                # Reorder columns to match training order
                features = features[self.features_columns]

            # Scale features
            scaled_features = self.scaler.transform(features)

            # Generate predictions
            predictions = self.ensemble.predict(scaled_features)

            result = {'predictions': predictions.tolist()}  # Convert to list for JSON serialization

            # Include probabilities if requested and available
            if include_probabilities and hasattr(self.ensemble, 'predict_proba'):
                probabilities = self.ensemble.predict_proba(scaled_features)
                result['probabilities'] = probabilities.tolist()  # Convert to list for JSON serialization

            return result
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return None

    def enhance_signals(self, signals, data, confidence_threshold=0.6):
        """
        Enhance trading signals using ML predictions.

        Args:
            signals (list): Trading signals
            data (pd.DataFrame): Market data
            confidence_threshold (float): Minimum prediction confidence

        Returns:
            list: Enhanced signals
        """
        self.logger.info(f"Enhancing {len(signals)} signals with machine learning")
        if not self.trained or not signals:
            return signals

        try:
            # Get predictions with probabilities
            prediction_result = self.predict(data, include_probabilities=True)
            if prediction_result is None or 'predictions' not in prediction_result:
                self.logger.warning("No valid predictions returned, returning original signals")
                return signals

            predictions = prediction_result['predictions']
            probabilities = prediction_result.get('probabilities', None)

            # Match predictions to signals
            enhanced_signals = []
            prediction_idx = 0

            for signal in signals:
                if prediction_idx >= len(predictions):
                    self.logger.debug("Ran out of predictions for signals")
                    break

                pred = predictions[prediction_idx]
                conf = max(probabilities[prediction_idx]) if probabilities else 1.0  # Default to 1 if no probs
                prediction_idx += 1

                expected_pred = 1 if signal['action'].lower() == 'buy' else 0
                if pred == expected_pred and conf >= confidence_threshold:
                    enhanced_signal = signal.copy()
                    enhanced_signal['ml_confidence'] = float(conf)
                    enhanced_signal['ml_enhanced'] = True
                    enhanced_signals.append(enhanced_signal)
                else:
                    self.logger.debug(f"Signal filtered: action={signal['action']}, pred={pred}, confidence={conf:.2f}")

            self.logger.info(f"Enhanced signals: kept {len(enhanced_signals)} out of {len(signals)}")
            return enhanced_signals
        except Exception as e:
            self.logger.error(f"Error enhancing signals: {str(e)}")
            return signals

    def generate_features(self, data):
        """
        Generate a comprehensive feature set from market data.

        Args:
            data (pd.DataFrame): Market data

        Returns:
            pd.DataFrame: Features for prediction
        """
        if data.empty:
            self.logger.warning("Empty data provided for feature generation")
            return pd.DataFrame()

        try:
            df = data.copy()
            df.columns = [col.lower() for col in df.columns]  # Ensure lowercase columns

            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()

            features = pd.DataFrame(index=df.index)

            # Price features
            for period in [5, 10, 20, 50, 100, 200]:
                if len(df) > period:
                    features[f'ma_{period}'] = df['close'].rolling(window=period).mean()
                    features[f'close_to_ma_{period}'] = df['close'] / features[f'ma_{period}']
                    features[f'volatility_{period}'] = df['close'].rolling(window=period).std()
                    features[f'price_change_{period}'] = df['close'].pct_change(periods=period)

            # Momentum indicators
            features['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            features['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
            features['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()

            # MACD
            macd = ta.trend.MACD(df['close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            features['bb_upper'] = bollinger.bollinger_hband()
            features['bb_lower'] = bollinger.bollinger_lband()
            features['bb_mid'] = bollinger.bollinger_mavg()
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_mid']
            features['bb_pct'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

            # Volume indicators
            features['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            features['volume_change'] = df['volume'].pct_change()
            features['vol_ratio'] = df['volume'] / features['volume_ma_10']

            # ATR
            features['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()

            # ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            features['adx'] = adx.adx()
            features['plus_di'] = adx.adx_pos()
            features['minus_di'] = adx.adx_neg()

            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
            features['ichimoku_a'] = ichimoku.ichimoku_a()
            features['ichimoku_b'] = ichimoku.ichimoku_b()

            # Time-based features
            if 'timestamp' in df.columns or df.index.dtype == 'datetime64[ns]':
                datetime_idx = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else df.index
                datetime_idx = datetime_idx.map(lambda x: x.replace(tzinfo=pytz.UTC) if x.tzinfo is None else x)
                features['day_of_week'] = datetime_idx.dayofweek
                features['hour_of_day'] = datetime_idx.hour
                for day in range(5):  # Trading days
                    features[f'day_{day}'] = (features['day_of_week'] == day).astype(int)

            # Drop columns with all NaN and fill remaining NaN
            features = features.dropna(axis=1, how='all').fillna(method='ffill').fillna(0)

            self.logger.info(f"Generated {features.shape[1]} features for {features.shape[0]} data points")
            return features
        except Exception as e:
            self.logger.error(f"Feature generation failed: {str(e)}")
            return pd.DataFrame()

    def online_update(self, new_data, true_labels=None):
        """
        Update model with new data (online learning).

        Args:
            new_data (pd.DataFrame): New market data
            true_labels (pd.Series, optional): True labels for the new data

        Returns:
            bool: True if update was successful
        """
        if not self.trained:
            self.logger.error("Model must be trained before online updates")
            return False

        try:
            # Generate features for new data
            features = self.generate_features(new_data)
            if features.empty:
                self.logger.warning("No features generated for online update")
                return False

            # Ensure features match training features
            if self.features_columns:
                for col in set(self.features_columns) - set(features.columns):
                    features[col] = 0.0
                features = features[self.features_columns]

            # Scale features
            scaled_features = self.scaler.transform(features)

            # Generate default labels if not provided
            if true_labels is None:
                true_labels = (new_data['close'].shift(-1) > new_data['close']).astype(int)
                true_labels = true_labels.iloc[:len(features)]

            # Perform partial fit if available
            if hasattr(self.ensemble, 'partial_fit'):
                self.ensemble.partial_fit(scaled_features, true_labels)
                self._save_models()  # Save updated model
                self.logger.info(f"Performed online update with {len(features)} samples")
                return True
            else:
                self.logger.warning("Model doesn't support online learning, skipping update")
                return False
        except Exception as e:
            self.logger.error(f"Online update failed: {str(e)}")
            return False

    def get_feature_importance(self):
        """
        Get feature importance from the models.

        Returns:
            dict: Feature importance scores
        """
        if not self.trained:
            self.logger.error("Models not trained yet")
            return {}

        try:
            feature_importance = {}
            if hasattr(self.ensemble, 'feature_importances_'):
                importances = self.ensemble.feature_importances_
                for i, feature in enumerate(self.features_columns):
                    feature_importance[feature] = float(importances[i])
            elif hasattr(self.ensemble, 'estimators_'):
                for name, model in self.ensemble.estimators_:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        model_importance = {feature: float(importances[i]) for i, feature in enumerate(self.features_columns)}
                        feature_importance[name] = model_importance
            return feature_importance
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return {}

    def get_performance_metrics(self):
        """
        Get model performance metrics.

        Returns:
            dict: Performance metrics
        """
        return {k: float(v) for k, v in self.performance.items()}  # Ensure float for serialization