import pandas as pd
import numpy as np
import logging
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from scipy.signal import argrelextrema
from hurst import compute_Hc
import ta.trend as trend
import traceback


class MarketStructure:
    """
    Market Structure analysis module for technical price action analysis.
    Focuses on price patterns, market regimes, and technical relationships.
    """

    def __init__(self, config: Dict, parent):
        """
        Initialize the MarketStructure module.

        Args:
            config (Dict): Configuration dictionary
            parent: Parent SignalGenerator instance
        """
        self.parent = parent
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize parameters
        self.symbols = parent.symbols
        self.timeframes = parent.timeframes

        # Setup configuration parameters
        self.swing_point_lookback = config.get('swing_point_lookback', 20)
        self.pattern_sensitivity = config.get('pattern_sensitivity', 0.7)
        self.hurst_update_interval = config.get('hurst_update_interval', 300)  # seconds
        self.structure_weights = config.get('structure_weights', {
            'trend_strength': 3,
            'hurst_exponent': 2,
            'price_patterns': 3,
            'support_resistance': 2,
            'multi_timeframe': 4,
            'market_regime': 2
        })

        # Hurst exponent cache
        self.hurst_cache = {symbol: {tf: None for tf in parent.timeframes}
                            for symbol in parent.symbols}
        self.hurst_last_update = {symbol: {tf: None for tf in parent.timeframes}
                                  for symbol in parent.symbols}

        # Market regime tracking
        self.market_regimes = {symbol: {tf: 'undefined' for tf in ['1h', '4h', 'daily']}
                               for symbol in self.symbols}
        self.regime_last_update = {symbol: {tf: None for tf in ['1h', '4h', 'daily']}
                                   for symbol in self.symbols}

        self.logger.info("MarketStructure module initialized")

    def analyze(self, symbol: str, timeframe: str, df: pd.DataFrame,
                current_time: datetime) -> Dict[str, Any]:
        """
        Perform comprehensive market structure analysis on price data.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data
            current_time (datetime): Current market time

        Returns:
            Dict[str, Any]: Complete market structure analysis
        """
        try:
            if df is None or df.empty or len(df) < 14:
                return {'valid': False, 'reason': 'Insufficient data'}

            # Update market regime if needed
            self._update_market_regime(symbol, timeframe, df, current_time)

            # Detect current market regime
            regime = self.detect_market_regime(symbol, timeframe, df)

            # Analyze price direction and trend
            trend_analysis = self.analyze_trend(symbol, timeframe, df)

            # Detect price patterns
            patterns = self.detect_price_patterns(symbol, timeframe, df)

            # Calculate Hurst exponent
            hurst = self.calculate_hurst_exponent(symbol, timeframe, df)

            # Analyze multi-timeframe structure
            mtf_alignment = self.analyze_multi_timeframe_structure(symbol, timeframe)

            # Detect support and resistance levels
            price = df['close'].iloc[-1]
            sr_levels = self.detect_support_resistance(symbol, price, df)

            # Check for break of structure or change of character
            bos = self.check_structure_break(symbol, df, direction='both')

            # Get optimal entry/exit levels
            optimal_levels = self.calculate_optimal_levels(
                symbol, trend_analysis['direction'], price, df
            )

            # Calculate structure score
            structure_score = self.calculate_structure_score(
                symbol, timeframe, df, regime, trend_analysis,
                patterns, hurst, mtf_alignment, bos
            )

            # Create comprehensive analysis result
            analysis = {
                'valid': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'regime': regime,
                'direction': trend_analysis['direction'],
                'strength': trend_analysis['strength'],
                'patterns': patterns,
                'hurst': hurst,
                'mtf_alignment': mtf_alignment,
                'support_resistance': sr_levels,
                'structure_break': bos,
                'optimal_levels': optimal_levels,
                'structure_score': structure_score
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error in market structure analysis for {symbol} on {timeframe}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return {'valid': False, 'reason': f'Error: {str(e)}'}

    def detect_market_regime(self, symbol: str, timeframe: str, df: pd.DataFrame) -> str:
        """
        Detect current market regime using volatility, trend strength, and momentum metrics.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data

        Returns:
            str: 'trending', 'ranging', 'volatile', or 'undefined'
        """
        try:
            # Check for cached regime
            if timeframe in self.market_regimes.get(symbol, {}):
                cached_regime = self.market_regimes[symbol][timeframe]
                last_update = self.regime_last_update[symbol][timeframe]

                # Use cached value if recent enough
                if last_update:
                    update_interval = self.parent.timeframe_intervals.get(timeframe, timedelta(hours=1))
                    if datetime.now(pytz.UTC) - last_update < update_interval:
                        return cached_regime

            if df.empty or len(df) < 50:
                return 'undefined'

            # Check for ADX (trend strength)
            if 'adx' not in df.columns:
                # Calculate ADX if not present
                adx_indicator = trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
                adx_value = adx_indicator.adx().iloc[-1]
            else:
                adx_value = df['adx'].iloc[-1]

            # Calculate volatility metrics
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
            if len(df) >= 70:
                avg_volatility = df['close'].pct_change().rolling(20).std().rolling(50).mean().iloc[-1]
            else:
                avg_volatility = volatility

            # Calculate Hurst exponent if not already calculated
            hurst = self.calculate_hurst_exponent(symbol, timeframe, df)

            # Determine regime
            if adx_value > 25 and hurst > 0.6:
                regime = 'trending'
            elif volatility > 1.5 * avg_volatility:
                regime = 'volatile'
            elif adx_value < 20 and 0.4 < hurst < 0.6:
                regime = 'ranging'
            else:
                regime = 'undefined'

            # Update cache
            if timeframe in self.market_regimes.get(symbol, {}):
                self.market_regimes[symbol][timeframe] = regime
                self.regime_last_update[symbol][timeframe] = datetime.now(pytz.UTC)

            return regime

        except Exception as e:
            self.logger.error(f"Error detecting market regime for {symbol} on {timeframe}: {str(e)}")
            return 'undefined'

    def _update_market_regime(self, symbol: str, timeframe: str, df: pd.DataFrame, current_time: datetime) -> None:
        """
        Update market regime cache if needed.

        Args:
            symbol (str): Symbol to update
            timeframe (str): Timeframe to update
            df (pd.DataFrame): DataFrame with OHLC data
            current_time (datetime): Current market time
        """
        try:
            if timeframe not in self.market_regimes.get(symbol, {}):
                return

            last_update = self.regime_last_update[symbol][timeframe]
            if last_update is None:
                # No previous update, detect now
                regime = self.detect_market_regime(symbol, timeframe, df)
                self.market_regimes[symbol][timeframe] = regime
                self.regime_last_update[symbol][timeframe] = current_time
            else:
                # Check if update interval has passed
                update_interval = self.parent.timeframe_intervals.get(timeframe, timedelta(hours=1))
                if (current_time - last_update) > update_interval:
                    regime = self.detect_market_regime(symbol, timeframe, df)
                    self.market_regimes[symbol][timeframe] = regime
                    self.regime_last_update[symbol][timeframe] = current_time
        except Exception as e:
            self.logger.error(f"Error updating market regime: {str(e)}")

    def analyze_trend(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price trend direction and strength.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data

        Returns:
            Dict[str, Any]: Trend analysis results
        """
        try:
            if df.empty or 'close' not in df.columns or 'adaptive_ma' not in df.columns:
                return {'direction': 'sideways', 'strength': 0}

            last_close = df['close'].iloc[-1]
            last_ma = df['adaptive_ma'].iloc[-1]

            # Calculate trend direction
            if last_close > last_ma * 1.005:
                direction = "uptrend"
            elif last_close < last_ma * 0.995:
                direction = "downtrend"
            else:
                direction = "sideways"

            # Calculate trend strength
            trend_strength = 0

            # 1. Price distance from MA
            ma_distance = abs(last_close - last_ma) / last_ma
            trend_strength += min(1.0, ma_distance * 100)

            # 2. MA slope
            if 'sma_slope' in df.columns:
                slope = df['sma_slope'].iloc[-1]
                slope_strength = min(1.0, abs(slope) * 10000)
                trend_strength += slope_strength

                # Ensure trend direction matches slope direction
                if (direction == "uptrend" and slope < 0) or (direction == "downtrend" and slope > 0):
                    trend_strength *= 0.5  # Penalize conflicting signals

            # 3. ADX trend strength if available
            if 'adx' in df.columns:
                adx = df['adx'].iloc[-1]
                trend_strength += min(1.0, adx / 50)

            # 4. Consistent closes in trend direction
            price_direction = np.sign(df['close'].diff(1).iloc[-5:])
            directional_consistency = np.abs(price_direction.sum()) / len(price_direction)
            trend_strength += directional_consistency

            # Normalize final strength score to 0-10 range
            normalized_strength = min(10, trend_strength * 2.5)

            return {
                'direction': direction,
                'strength': normalized_strength
            }

        except Exception as e:
            self.logger.error(f"Error analyzing trend for {symbol} on {timeframe}: {str(e)}")
            return {'direction': 'sideways', 'strength': 0}

    def detect_price_patterns(self, symbol: str, timeframe: str, df: pd.DataFrame) -> List[str]:
        """
        Detect common price action patterns.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data

        Returns:
            List[str]: Detected patterns
        """
        try:
            if df.empty or len(df) < 5:
                return []

            patterns = []

            # Calculate key values for pattern detection
            high_tail = df['high'].iloc[-1] - max(df['open'].iloc[-1], df['close'].iloc[-1])
            low_tail = min(df['open'].iloc[-1], df['close'].iloc[-1]) - df['low'].iloc[-1]
            body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
            body_direction = np.sign(df['close'].iloc[-1] - df['open'].iloc[-1])  # 1 for bullish, -1 for bearish
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001

            # 1. Pin Bar / Rejection Candle
            if high_tail > 2 * body and high_tail > atr * 0.5:
                patterns.append('pin_bar_high')
            if low_tail > 2 * body and low_tail > atr * 0.5:
                patterns.append('pin_bar_low')

            # 2. Inside Bar
            if (df['high'].iloc[-1] < df['high'].iloc[-2] and df['low'].iloc[-1] > df['low'].iloc[-2]):
                patterns.append('inside_bar')

            # 3. Outside Bar
            if (df['high'].iloc[-1] > df['high'].iloc[-2] and df['low'].iloc[-1] < df['low'].iloc[-2]):
                patterns.append('outside_bar')

            # 4. Engulfing Pattern
            prev_body = abs(df['close'].iloc[-2] - df['open'].iloc[-2])
            prev_body_direction = np.sign(df['close'].iloc[-2] - df['open'].iloc[-2])

            if (body > prev_body and  # Current body larger than previous
                    body_direction == 1 and prev_body_direction == -1 and  # Bull engulfing bear
                    df['close'].iloc[-1] > df['open'].iloc[-2] and  # Close above previous open
                    df['open'].iloc[-1] < df['close'].iloc[-2]):  # Open below previous close
                patterns.append('bullish_engulfing')

            elif (body > prev_body and  # Current body larger than previous
                  body_direction == -1 and prev_body_direction == 1 and  # Bear engulfing bull
                  df['close'].iloc[-1] < df['open'].iloc[-2] and  # Close below previous open
                  df['open'].iloc[-1] > df['close'].iloc[-2]):  # Open above previous close
                patterns.append('bearish_engulfing')

            # 5. Doji (tiny body with tails)
            if body < atr * 0.1 and (high_tail + low_tail) > atr * 0.5:
                patterns.append('doji')

            # 6. Three Bar Pattern (three consecutive in same direction)
            if len(df) >= 3:
                last_3_direction = [np.sign(df['close'].iloc[-i] - df['open'].iloc[-i]) for i in range(1, 4)]
                if all(d == 1 for d in last_3_direction):
                    patterns.append('three_bull_candles')
                elif all(d == -1 for d in last_3_direction):
                    patterns.append('three_bear_candles')

            # 7. Two-Bar Reversal
            if len(df) >= 2:
                if (df['close'].iloc[-2] < df['open'].iloc[-2] and  # Previous bearish
                        df['close'].iloc[-1] > df['open'].iloc[-1] and  # Current bullish
                        df['close'].iloc[-1] > df['high'].iloc[-2]):  # Close above previous high
                    patterns.append('bullish_reversal')

                elif (df['close'].iloc[-2] > df['open'].iloc[-2] and  # Previous bullish
                      df['close'].iloc[-1] < df['open'].iloc[-1] and  # Current bearish
                      df['close'].iloc[-1] < df['low'].iloc[-2]):  # Close below previous low
                    patterns.append('bearish_reversal')

            return patterns

        except Exception as e:
            self.logger.error(f"Error detecting price patterns for {symbol} on {timeframe}: {str(e)}")
            return []

    def calculate_hurst_exponent(self, symbol: str, timeframe: str, df: pd.DataFrame) -> float:
        """
        Calculate Hurst exponent with caching for efficiency.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data

        Returns:
            float: Hurst exponent value (0-1)
        """
        try:
            # Check for cached value
            current_time = datetime.now(pytz.UTC)
            if timeframe in self.hurst_last_update.get(symbol, {}):
                last_update = self.hurst_last_update[symbol][timeframe]
                if last_update and (current_time - last_update).total_seconds() < self.hurst_update_interval:
                    cached_value = self.hurst_cache[symbol][timeframe]
                    if cached_value is not None:
                        return cached_value

            # Calculate if no valid cache
            if df.empty or len(df) < 50:
                return 0.5  # Default to random walk

            # Use maximum of 100 points for efficiency
            series = df['close'].values[-100:] if len(df) >= 100 else df['close'].values

            H, _, _ = compute_Hc(series, kind='price')

            # Update cache
            if timeframe in self.hurst_cache.get(symbol, {}):
                self.hurst_cache[symbol][timeframe] = H
                self.hurst_last_update[symbol][timeframe] = current_time

            return H
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent for {symbol} on {timeframe}: {str(e)}")
            return 0.5  # Default to random walk on error

    def analyze_multi_timeframe_structure(self, symbol: str, trading_tf: str) -> Dict[str, Any]:
        """
        Analyze price structure alignment across multiple timeframes.

        Args:
            symbol (str): Symbol to analyze
            trading_tf (str): Base timeframe for analysis

        Returns:
            Dict[str, Any]: Multi-timeframe alignment results
        """
        try:
            # Define timeframe relationships based on trading timeframe
            tf_relationships = {
                '1min': ['1min', '5min', '15min'],
                '5min': ['5min', '15min', '30min'],
                '15min': ['15min', '30min', '1h'],
                '30min': ['30min', '1h', '4h'],
                '1h': ['1h', '4h', 'daily'],
                '4h': ['4h', 'daily'],
                'daily': ['daily']
            }

            # Get relevant timeframes with weights (higher weight for higher timeframes)
            alignment_tfs = tf_relationships.get(trading_tf, [trading_tf])
            weights = {tf: 1.0 / (i + 1) for i, tf in enumerate(alignment_tfs)}
            total_weight = sum(weights.values())

            # Check trend direction on each timeframe
            alignment_score = 0
            trend_details = {}

            for tf in alignment_tfs:
                df = self.parent.indicator_histories.get(symbol, {}).get(tf)
                if df is None or df.empty:
                    continue

                # Get trend direction
                trend_analysis = self.analyze_trend(symbol, tf, df)
                direction = trend_analysis['direction']

                trend_details[tf] = {
                    'direction': direction,
                    'strength': trend_analysis['strength']
                }

                # Add to alignment score based on direction
                if direction == 'uptrend':
                    alignment_score += weights[tf]
                elif direction == 'downtrend':
                    alignment_score -= weights[tf]

            # Normalize score to [-1, 1] range
            if total_weight > 0:
                normalized_score = alignment_score / total_weight
            else:
                normalized_score = 0

            # Determine overall alignment
            result = {
                'score': normalized_score,
                'trending_up': normalized_score > 0.7,
                'trending_down': normalized_score < -0.7,
                'strong_alignment': abs(normalized_score) > 0.8,
                'details': trend_details
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing multi-timeframe structure for {symbol}: {str(e)}")
            return {
                'score': 0,
                'trending_up': False,
                'trending_down': False,
                'strong_alignment': False,
                'details': {}
            }

    def detect_swing_points(self, symbol: str, df: pd.DataFrame, direction: str = 'both') -> Dict[str, List[int]]:
        """
        Detect swing high and swing low points.

        Args:
            symbol (str): Symbol to analyze
            df (pd.DataFrame): DataFrame with OHLC data
            direction (str): 'high', 'low', or 'both'

        Returns:
            Dict[str, List[int]]: Indices of swing points
        """
        try:
            if df.empty or len(df) < 20:
                return {'highs': [], 'lows': []}

            # Determine window size based on volatility
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001
            price = df['close'].iloc[-1]
            window = max(2, min(5, int(atr / price * 10000)))

            results = {}

            if direction in ['high', 'both']:
                # Find swing highs
                highs = list(argrelextrema(df['high'].values, np.greater, order=window)[0])

                # Filter by volume if available
                if 'volume' in df.columns:
                    avg_volume = df['volume'].mean()
                    highs = [i for i in highs if i < len(df) and df['volume'].iloc[i] > avg_volume * 0.8]

                results['highs'] = highs

            if direction in ['low', 'both']:
                # Find swing lows
                lows = list(argrelextrema(df['low'].values, np.less, order=window)[0])

                # Filter by volume if available
                if 'volume' in df.columns:
                    avg_volume = df['volume'].mean()
                    lows = [i for i in lows if i < len(df) and df['volume'].iloc[i] > avg_volume * 0.8]

                results['lows'] = lows

            return results

        except Exception as e:
            self.logger.error(f"Error detecting swing points for {symbol}: {str(e)}")
            return {'highs': [], 'lows': []}

    def detect_support_resistance(self, symbol: str, price: float, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Detect key support and resistance levels.

        Args:
            symbol (str): Symbol to analyze
            price (float): Current price
            df (pd.DataFrame): DataFrame with OHLC data

        Returns:
            Dict[str, List[float]]: Support and resistance levels
        """
        try:
            if df.empty or len(df) < 50:
                return {'support': [], 'resistance': []}

            # Get swing points
            swing_points = self.detect_swing_points(symbol, df)

            support_levels = []
            resistance_levels = []

            # 1. Add levels from swing points
            for idx in swing_points.get('lows', []):
                if idx < len(df):
                    level = df['low'].iloc[idx]
                    if level < price:  # Only add as support if below current price
                        support_levels.append(level)

            for idx in swing_points.get('highs', []):
                if idx < len(df):
                    level = df['high'].iloc[idx]
                    if level > price:  # Only add as resistance if above current price
                        resistance_levels.append(level)

            # 2. Add key moving averages as potential S/R
            for ma_name in ['adaptive_ma', 'vwap']:
                if ma_name in df.columns:
                    ma_level = df[ma_name].iloc[-1]
                    if ma_level < price:
                        support_levels.append(ma_level)
                    elif ma_level > price:
                        resistance_levels.append(ma_level)

            # 3. Add VWAP bands if available
            if 'vwap_upper_1' in df.columns and df['vwap_upper_1'].iloc[-1] > price:
                resistance_levels.append(df['vwap_upper_1'].iloc[-1])

            if 'vwap_upper_2' in df.columns and df['vwap_upper_2'].iloc[-1] > price:
                resistance_levels.append(df['vwap_upper_2'].iloc[-1])

            if 'vwap_lower_1' in df.columns and df['vwap_lower_1'].iloc[-1] < price:
                support_levels.append(df['vwap_lower_1'].iloc[-1])

            if 'vwap_lower_2' in df.columns and df['vwap_lower_2'].iloc[-1] < price:
                support_levels.append(df['vwap_lower_2'].iloc[-1])

            # 4. Add round numbers as potential S/R
            # Find appropriate decimal place based on price range
            if price < 1:
                round_increment = 0.001
            elif price < 10:
                round_increment = 0.01
            elif price < 100:
                round_increment = 0.1
            elif price < 1000:
                round_increment = 1
            else:
                round_increment = 10

            # Add nearest round numbers
            lower_round = np.floor(price / round_increment) * round_increment
            upper_round = np.ceil(price / round_increment) * round_increment

            if lower_round < price:
                support_levels.append(lower_round)
            if upper_round > price:
                resistance_levels.append(upper_round)

            # Sort and filter out duplicates
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)))

            return {
                'support': support_levels,
                'resistance': resistance_levels
            }

        except Exception as e:
            self.logger.error(f"Error detecting support/resistance for {symbol}: {str(e)}")
            return {'support': [], 'resistance': []}

    def check_structure_break(self, symbol: str, df: pd.DataFrame, direction: str = 'both') -> Dict[str, bool]:
        """
        Check for breaks of market structure (BOS) and changes of character (CHoCH).

        Args:
            symbol (str): Symbol to analyze
            df (pd.DataFrame): DataFrame with OHLC data
            direction (str): 'up', 'down', or 'both'

        Returns:
            Dict[str, bool]: Structure break flags
        """
        try:
            if df.empty or len(df) < 20:
                return {'bos_up': False, 'bos_down': False, 'choch': False}

            result = {
                'bos_up': False,
                'bos_down': False,
                'choch': False
            }

            # Get ATR for volatility context
            current_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]) else 0
            avg_atr = df['avg_atr'].iloc[-1] if 'avg_atr' in df.columns and not pd.isna(df['avg_atr'].iloc[-1]) else 0

            # Adjust lookback based on volatility
            lookback = 20 if current_atr == 0 or avg_atr == 0 else (5 if current_atr > 1.5 * avg_atr else 20)

            # Check Break of Structure (upward)
            if direction in ['up', 'both']:
                recent_highs = df['high'].iloc[-lookback:-1].max()
                if df['high'].iloc[-1] > recent_highs:
                    result['bos_up'] = True

            # Check Break of Structure (downward)
            if direction in ['down', 'both']:
                recent_lows = df['low'].iloc[-lookback:-1].min()
                if df['low'].iloc[-1] < recent_lows:
                    result['bos_down'] = True

            # Check Change of Character (shift from bullish to bearish or vice versa)
            if len(df) >= lookback * 2:
                # Compare recent trend with previous trend
                recent_close_change = df['close'].iloc[-1] - df['close'].iloc[-lookback]
                previous_close_change = df['close'].iloc[-lookback] - df['close'].iloc[-lookback * 2]

                if (recent_close_change * previous_close_change < 0 and
                        abs(recent_close_change) > current_atr * 0.5):
                    result['choch'] = True

            return result

        except Exception as e:
            self.logger.error(f"Error checking structure break for {symbol}: {str(e)}")
            return {'bos_up': False, 'bos_down': False, 'choch': False}

    def calculate_optimal_levels(self, symbol: str, direction: str, entry_price: float,
                                 df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate optimal stop loss and take profit levels based on market structure.

        Args:
            symbol (str): Symbol to analyze
            direction (str): Trade direction ('uptrend', 'downtrend', or 'sideways')
            entry_price (float): Entry price for the trade
            df (pd.DataFrame): DataFrame with OHLC data

        Returns:
            Dict[str, float]: Optimal levels including stop loss and take profit
        """
        try:
            if df.empty or len(df) < 20:
                return {'stop_loss': None, 'take_profit': None}

            # Get ATR for volatility-based calculation
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001

            # Get swing points
            swing_points = self.detect_swing_points(symbol, df)

            # Get support/resistance levels
            sr_levels = self.detect_support_resistance(symbol, entry_price, df)

            # Determine action
            action = 'buy' if direction == 'uptrend' else 'sell' if direction == 'downtrend' else None

            if not action:
                # Default ATR-based levels if no clear direction
                stop_loss = entry_price - (atr * 2) if action == 'buy' else entry_price + (atr * 2)
                take_profit = entry_price + (atr * 3) if action == 'buy' else entry_price - (atr * 3)
                return {'stop_loss': stop_loss, 'take_profit': take_profit}

            # Calculate stop loss based on swing points and market structure
            if action == 'buy':
                # Find appropriate swing low for stop loss
                recent_lows = []
                for idx in swing_points.get('lows', []):
                    if idx < len(df) - 1 and df['low'].iloc[idx] < entry_price:
                        recent_lows.append(df['low'].iloc[idx])

                if recent_lows:
                    # Use highest swing low below entry as base for stop
                    nearest_low = max(recent_lows)
                    # Add buffer for stop loss
                    stop_loss = nearest_low - atr * 0.5
                else:
                    # Fallback to ATR-based stop
                    stop_loss = entry_price - atr * 2

                # Check support levels as alternatives
                support_levels = sr_levels.get('support', [])
                if support_levels:
                    # Find nearest support below entry
                    valid_supports = [s for s in support_levels if s < entry_price]
                    if valid_supports:
                        nearest_support = max(valid_supports)
                        # Choose the better (higher) of the two stop methods
                        stop_loss = max(stop_loss, nearest_support - atr * 0.3)

                # Calculate take profit based on resistance levels or price projection
                resistance_levels = sr_levels.get('resistance', [])
                if resistance_levels:
                    # Find nearest resistance above entry
                    valid_resistances = [r for r in resistance_levels if r > entry_price]
                    if valid_resistances:
                        nearest_resistance = min(valid_resistances)
                        take_profit = nearest_resistance - atr * 0.5
                    else:
                        # Project take profit based on risk multiple
                        risk = entry_price - stop_loss
                        take_profit = entry_price + (risk * 1.5)  # 1.5:1 risk-reward minimum
                else:
                    # Fallback to ATR-based take profit
                    take_profit = entry_price + atr * 3

            else:  # 'sell'
                # Find appropriate swing high for stop loss
                recent_highs = []
                for idx in swing_points.get('highs', []):
                    if idx < len(df) - 1 and df['high'].iloc[idx] > entry_price:
                        recent_highs.append(df['high'].iloc[idx])

                if recent_highs:
                    # Use lowest swing high above entry as base for stop
                    nearest_high = min(recent_highs)
                    # Add buffer for stop loss
                    stop_loss = nearest_high + atr * 0.5
                else:
                    # Fallback to ATR-based stop
                    stop_loss = entry_price + atr * 2

                # Check resistance levels as alternatives
                resistance_levels = sr_levels.get('resistance', [])
                if resistance_levels:
                    # Find nearest resistance above entry
                    valid_resistances = [r for r in resistance_levels if r > entry_price]
                    if valid_resistances:
                        nearest_resistance = min(valid_resistances)
                        # Choose the better (lower) of the two stop methods
                        stop_loss = min(stop_loss, nearest_resistance + atr * 0.3)

                # Calculate take profit based on support levels or price projection
                support_levels = sr_levels.get('support', [])
                if support_levels:
                    # Find nearest support below entry
                    valid_supports = [s for s in support_levels if s < entry_price]
                    if valid_supports:
                        nearest_support = max(valid_supports)
                        take_profit = nearest_support + atr * 0.5
                    else:
                        # Project take profit based on risk multiple
                        risk = stop_loss - entry_price
                        take_profit = entry_price - (risk * 1.5)  # 1.5:1 risk-reward minimum
                else:
                    # Fallback to ATR-based take profit
                    take_profit = entry_price - atr * 3

            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0

            # Ensure minimum risk-reward ratio of 1.5:1
            if risk_reward_ratio < 1.5:
                if action == 'buy':
                    take_profit = entry_price + (risk * 1.5)
                else:
                    take_profit = entry_price - (risk * 1.5)
                risk_reward_ratio = 1.5

            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio
            }

        except Exception as e:
            self.logger.error(f"Error calculating optimal levels for {symbol}: {str(e)}")
            return {'stop_loss': None, 'take_profit': None}

    def calculate_structure_score(self, symbol: str, timeframe: str, df: pd.DataFrame,
                                  regime: str, trend_analysis: Dict, patterns: List[str],
                                  hurst: float, mtf_alignment: Dict, structure_break: Dict) -> float:
        """
        Calculate comprehensive structure score based on all structural elements.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data
            regime (str): Market regime
            trend_analysis (Dict): Trend analysis results
            patterns (List[str]): Detected price patterns
            hurst (float): Hurst exponent
            mtf_alignment (Dict): Multi-timeframe alignment results
            structure_break (Dict): Structure break analysis

        Returns:
            float: Structure score (0-10 scale)
        """
        try:
            score = 0
            max_score = 0

            # 1. Trend Strength (0-3 points)
            trend_weight = self.structure_weights.get('trend_strength', 3)
            max_score += trend_weight

            trend_strength = trend_analysis.get('strength', 0)
            normalized_trend_strength = min(1.0, trend_strength / 10)
            score += normalized_trend_strength * trend_weight

            # 2. Hurst Exponent (0-2 points)
            hurst_weight = self.structure_weights.get('hurst_exponent', 2)
            max_score += hurst_weight

            trend_direction = trend_analysis.get('direction', 'sideways')
            if trend_direction == 'uptrend' or trend_direction == 'downtrend':
                # Reward trending (Hurst > 0.5) for trend trades
                hurst_score = max(0, (hurst - 0.5) * 2) if hurst > 0.5 else 0
            else:
                # Reward mean-reversion (Hurst < 0.5) for sideways markets
                hurst_score = max(0, (0.5 - hurst) * 2) if hurst < 0.5 else 0

            score += hurst_score * hurst_weight

            # 3. Price Patterns (0-3 points)
            pattern_weight = self.structure_weights.get('price_patterns', 3)
            max_score += pattern_weight

            pattern_score = 0
            bullish_patterns = ['pin_bar_low', 'bullish_engulfing', 'bullish_reversal', 'three_bull_candles']
            bearish_patterns = ['pin_bar_high', 'bearish_engulfing', 'bearish_reversal', 'three_bear_candles']

            # Reward confirmation patterns
            if trend_direction == 'uptrend':
                confirming_patterns = [p for p in patterns if p in bullish_patterns]
                contradicting_patterns = [p for p in patterns if p in bearish_patterns]

                pattern_score += 0.5 * min(2, len(confirming_patterns))
                pattern_score -= 0.25 * min(2, len(contradicting_patterns))

            elif trend_direction == 'downtrend':
                confirming_patterns = [p for p in patterns if p in bearish_patterns]
                contradicting_patterns = [p for p in patterns if p in bullish_patterns]

                pattern_score += 0.5 * min(2, len(confirming_patterns))
                pattern_score -= 0.25 * min(2, len(contradicting_patterns))

            # Common patterns for specific regimes
            if regime == 'ranging' and 'inside_bar' in patterns:
                pattern_score += 0.5
            if regime == 'volatile' and 'outside_bar' in patterns:
                pattern_score += 0.5
            if 'doji' in patterns:
                pattern_score += 0.25

            # Cap and add pattern score
            pattern_score = max(0, min(1.0, pattern_score))
            score += pattern_score * pattern_weight

            # 4. Multi-Timeframe Alignment (0-4 points)
            mtf_weight = self.structure_weights.get('multi_timeframe', 4)
            max_score += mtf_weight

            mtf_score = abs(mtf_alignment.get('score', 0))

            # Bonus for strong alignment
            if mtf_alignment.get('strong_alignment', False):
                mtf_score = min(1.0, mtf_score * 1.2)

            # Check direction alignment
            mtf_trending_up = mtf_alignment.get('trending_up', False)
            mtf_trending_down = mtf_alignment.get('trending_down', False)

            if ((trend_direction == 'uptrend' and mtf_trending_up) or
                    (trend_direction == 'downtrend' and mtf_trending_down)):
                mtf_score = min(1.0, mtf_score * 1.2)  # Bonus for alignment
            elif ((trend_direction == 'uptrend' and mtf_trending_down) or
                  (trend_direction == 'downtrend' and mtf_trending_up)):
                mtf_score *= 0.5  # Penalty for misalignment

            score += mtf_score * mtf_weight

            # 5. Market Regime (0-2 points)
            regime_weight = self.structure_weights.get('market_regime', 2)
            max_score += regime_weight

            regime_score = 0
            if regime == 'trending' and trend_direction != 'sideways':
                regime_score = 1.0  # Full score for trend trading in trending regime
            elif regime == 'ranging' and trend_direction == 'sideways':
                regime_score = 1.0  # Full score for range trading in ranging regime
            elif regime == 'volatile':
                # In volatile regimes, reward strong signals
                if trend_strength > 7:
                    regime_score = 0.8
                elif trend_strength > 5:
                    regime_score = 0.5
                else:
                    regime_score = 0.2
            else:
                regime_score = 0.5  # Partial score for less optimal regime-strategy matches

            score += regime_score * regime_weight

            # 6. Break of Structure / Change of Character (0-2 points bonus)
            bos_up = structure_break.get('bos_up', False)
            bos_down = structure_break.get('bos_down', False)
            choch = structure_break.get('choch', False)

            bos_score = 0
            if trend_direction == 'uptrend' and bos_up:
                bos_score = 2  # Bonus for BOS confirming uptrend
            elif trend_direction == 'downtrend' and bos_down:
                bos_score = 2  # Bonus for BOS confirming downtrend
            elif choch:
                bos_score = 1  # Smaller bonus for change of character

            score += bos_score
            max_score += 2  # Account for potential bonus

            # Normalize final score to 0-10 scale
            normalized_score = (score / max_score) * 10
            return normalized_score

        except Exception as e:
            self.logger.error(f"Error calculating structure score for {symbol} on {timeframe}: {str(e)}")
            return 0.0