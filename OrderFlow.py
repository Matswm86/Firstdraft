import pandas as pd
import numpy as np
import logging
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import traceback


class OrderFlow:
    """
    Order Flow analysis module for examining transaction dynamics,
    buying/selling pressure, and liquidity characteristics.
    """

    def __init__(self, config: Dict, parent):
        """
        Initialize the OrderFlow module.

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

        # Configuration parameters
        self.delta_threshold = config.get('delta_threshold', 500)
        self.imbalance_threshold = config.get('imbalance_threshold', 0.2)
        self.absorption_ratio = config.get('absorption_ratio', 1.5)

        # Order flow weights for scoring
        self.flow_weights = config.get('flow_weights', {
            'delta': 3,
            'bid_ask_imbalance': 2,
            'absorption': 2,
            'effort_vs_result': 2,
            'liquidity': 2,
            'transaction_intensity': 2
        })

        # Caches for order flow metrics
        self.volume_profile_cache = {symbol: {} for symbol in self.symbols}
        self.institutional_activity_markers = {symbol: {'timestamp': None, 'level': None} for symbol in self.symbols}

        # Footprint data for volume by price
        self.footprint_data = {symbol: {tf: {} for tf in self.timeframes} for symbol in self.symbols}

        self.logger.info("OrderFlow module initialized")

    def analyze(self, symbol: str, timeframe: str, df: pd.DataFrame,
                real_time_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive order flow analysis.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data
            real_time_data (Dict): Real-time market data

        Returns:
            Dict[str, Any]: Complete order flow analysis
        """
        try:
            if df is None or df.empty or len(df) < 5:
                return {'valid': False, 'reason': 'Insufficient data'}

            if real_time_data is None:
                return {'valid': False, 'reason': 'No real-time data available'}

            # Analyze delta (buying/selling pressure)
            delta_analysis = self.analyze_delta(symbol, timeframe, df, real_time_data)

            # Analyze bid-ask dynamics
            bid_ask = self.analyze_bid_ask_dynamics(symbol, real_time_data)

            # Detect liquidity zones
            liquidity = self.detect_liquidity_zones(symbol, timeframe, df)

            # Check absorption
            absorption = self.check_absorption(symbol, timeframe, df, real_time_data)

            # Analyze effort vs. result
            effort_result = self.analyze_effort_vs_result(symbol, timeframe, df)

            # Detect institutional activity
            institutional = self.detect_institutional_activity(symbol, df, real_time_data)

            # Calculate transaction intensity
            intensity = self.calculate_transaction_intensity(symbol, real_time_data)

            # Determine overall order flow direction
            direction = self._determine_flow_direction(
                delta_analysis, bid_ask, absorption, institutional
            )

            # Calculate flow score
            flow_score = self.calculate_flow_score(
                symbol, timeframe, delta_analysis, bid_ask, liquidity,
                absorption, effort_result, institutional, intensity
            )

            # Create comprehensive analysis result
            analysis = {
                'valid': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'delta': delta_analysis,
                'bid_ask': bid_ask,
                'liquidity': liquidity,
                'absorption': absorption,
                'effort_result': effort_result,
                'institutional': institutional,
                'intensity': intensity,
                'flow_score': flow_score
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error in order flow analysis for {symbol} on {timeframe}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return {'valid': False, 'reason': f'Error: {str(e)}'}

    def analyze_delta(self, symbol: str, timeframe: str, df: pd.DataFrame,
                      real_time_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze volume delta (buying vs. selling pressure).

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data
            real_time_data (Dict): Real-time market data

        Returns:
            Dict[str, Any]: Delta analysis results
        """
        try:
            if df.empty or len(df) < 5:
                return {'valid': False}

            # Extract delta history from real-time data
            delta_history = list(real_time_data.get('delta_history', []))
            if len(delta_history) < 20:
                return {'valid': False}

            # Calculate recent and medium-term delta
            recent_delta = sum(delta_history[-20:])
            medium_delta = sum(delta_history[-50:]) if len(delta_history) >= 50 else recent_delta
            long_delta = sum(delta_history) if delta_history else 0

            # Calculate normalized strength (0-1)
            max_observed_delta = max(abs(recent_delta), abs(medium_delta), 100)
            normalized_strength = min(1.0, abs(recent_delta) / max_observed_delta)

            # Determine delta direction
            if recent_delta > 0:
                delta_direction = 'up'
            elif recent_delta < 0:
                delta_direction = 'down'
            else:
                delta_direction = 'neutral'

            # Calculate delta consistency
            recent_signs = [1 if d > 0 else -1 if d < 0 else 0 for d in delta_history[-20:]]
            consistency = abs(sum(recent_signs)) / len(recent_signs) if recent_signs else 0

            # Detect delta divergence with price
            divergence = False
            price_change = None

            if len(df) >= 5:
                price_change = df['close'].iloc[-1] - df['close'].iloc[-5]
                price_change_sign = 1 if price_change > 0 else -1 if price_change < 0 else 0
                delta_sign = 1 if recent_delta > 0 else -1 if recent_delta < 0 else 0

                # Divergence occurs when price and delta move in opposite directions
                if price_change_sign != 0 and delta_sign != 0 and price_change_sign != delta_sign:
                    divergence = True

            # Calculate delta momentum (acceleration/deceleration)
            delta_momentum = 'increasing'
            if len(delta_history) >= 40:
                very_recent_delta = sum(delta_history[-10:])
                previous_delta = sum(delta_history[-20:-10])

                if abs(very_recent_delta) < abs(previous_delta):
                    delta_momentum = 'decreasing'
                elif abs(very_recent_delta) == abs(previous_delta):
                    delta_momentum = 'stable'

            return {
                'valid': True,
                'recent_delta': recent_delta,
                'medium_delta': medium_delta,
                'long_delta': long_delta,
                'direction': delta_direction,
                'strength': normalized_strength,
                'consistency': consistency,
                'divergence': divergence,
                'price_change': price_change,
                'momentum': delta_momentum
            }

        except Exception as e:
            self.logger.error(f"Error analyzing delta for {symbol} on {timeframe}: {str(e)}")
            return {'valid': False}

    def analyze_bid_ask_dynamics(self, symbol: str, real_time_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze bid-ask spread, imbalance, and dynamics.

        Args:
            symbol (str): Symbol to analyze
            real_time_data (Dict): Real-time market data

        Returns:
            Dict[str, Any]: Bid-ask analysis results
        """
        try:
            imbalance = real_time_data.get('bid_ask_imbalance', 0)
            price = real_time_data.get('last_price')

            if price is None:
                return {'valid': False}

            # Determine imbalance direction and strength
            direction = 'neutral'
            if imbalance > self.imbalance_threshold:
                direction = 'bid_dominant'  # More buying pressure
            elif imbalance < -self.imbalance_threshold:
                direction = 'ask_dominant'  # More selling pressure

            # Convert to normalized strength (0-1)
            strength = min(1.0, abs(imbalance) / 0.5)  # 0.5 is theoretical max imbalance

            return {
                'valid': True,
                'imbalance': imbalance,
                'direction': direction,
                'strength': strength,
                'significant': abs(imbalance) > self.imbalance_threshold
            }

        except Exception as e:
            self.logger.error(f"Error analyzing bid-ask dynamics for {symbol}: {str(e)}")
            return {'valid': False}

    def detect_liquidity_zones(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect zones of significant liquidity and price interest.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data

        Returns:
            Dict[str, Any]: Liquidity zone analysis
        """
        try:
            if df.empty or len(df) < 20:
                return {'valid': False}

            current_price = df['close'].iloc[-1]

            # Identify high volume bars
            if 'volume' not in df.columns:
                return {'valid': False}

            avg_volume = df['volume'].rolling(window=20).mean()
            high_volume_threshold = avg_volume.mean() * 1.5
            high_volume_bars = df[df['volume'] > high_volume_threshold]

            if high_volume_bars.empty:
                return {
                    'valid': True,
                    'zones': [],
                    'nearest_zone': None,
                    'in_liquidity_zone': False
                }

            # Identify price ranges with high volume
            price_ranges = []
            for _, bar in high_volume_bars.iterrows():
                price_range = {
                    'low': bar['low'],
                    'high': bar['high'],
                    'volume': bar['volume'],
                    'volume_ratio': bar['volume'] / high_volume_threshold
                }
                price_ranges.append(price_range)

            # Merge overlapping price ranges
            merged_ranges = self._merge_overlapping_ranges(price_ranges)

            # Identify current liquidity zones (within 1% of current price)
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else (current_price * 0.01)
            active_zones = []
            nearest_zone = None
            min_distance = float('inf')

            for zone in merged_ranges:
                # Calculate distance to zone
                if zone['high'] >= current_price >= zone['low']:
                    distance = 0  # Price is within zone
                elif current_price < zone['low']:
                    distance = zone['low'] - current_price
                else:  # current_price > zone['high']
                    distance = current_price - zone['high']

                # Convert distance to ATR units
                distance_atr = distance / atr if atr > 0 else 0

                zone_info = {
                    'low': zone['low'],
                    'high': zone['high'],
                    'volume_ratio': zone['volume_ratio'],
                    'distance': distance,
                    'distance_atr': distance_atr,
                    'active': distance_atr < 2  # Active if within 2 ATR
                }

                if zone_info['active']:
                    active_zones.append(zone_info)

                if distance < min_distance:
                    min_distance = distance
                    nearest_zone = zone_info

            in_liquidity_zone = any(zone['high'] >= current_price >= zone['low'] for zone in active_zones)

            return {
                'valid': True,
                'zones': active_zones,
                'nearest_zone': nearest_zone,
                'in_liquidity_zone': in_liquidity_zone
            }

        except Exception as e:
            self.logger.error(f"Error detecting liquidity zones for {symbol} on {timeframe}: {str(e)}")
            return {'valid': False}

    def _merge_overlapping_ranges(self, price_ranges: List[Dict]) -> List[Dict]:
        """
        Merge overlapping price ranges to form coherent liquidity zones.

        Args:
            price_ranges (List[Dict]): Individual price ranges to merge

        Returns:
            List[Dict]: Merged price ranges
        """
        if not price_ranges:
            return []

        # Sort by low price
        sorted_ranges = sorted(price_ranges, key=lambda x: x['low'])

        merged = []
        current = sorted_ranges[0]

        for next_range in sorted_ranges[1:]:
            if current['high'] >= next_range['low']:
                # Overlapping ranges - merge
                current['high'] = max(current['high'], next_range['high'])
                current['volume'] = current['volume'] + next_range['volume']
                current['volume_ratio'] = max(current['volume_ratio'], next_range['volume_ratio'])
            else:
                # Non-overlapping - add current to merged list and move to next
                merged.append(current)
                current = next_range

        # Add the last range
        merged.append(current)

        return merged

    def check_absorption(self, symbol: str, timeframe: str, df: pd.DataFrame,
                         real_time_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for absorption of buying/selling pressure at key levels.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data
            real_time_data (Dict): Real-time market data

        Returns:
            Dict[str, Any]: Absorption analysis
        """
        try:
            if df.empty or len(df) < 5:
                return {'valid': False}

            # Extract recent bars
            recent_bars = df.iloc[-5:]
            current_price = df['close'].iloc[-1]

            # Get volume information
            if 'volume' not in recent_bars.columns:
                return {'valid': False}

            price_range = recent_bars['high'].max() - recent_bars['low'].min()
            if price_range <= 0:
                return {'valid': False}

            # Calculate volume density
            total_volume = recent_bars['volume'].sum()
            volume_density = total_volume / price_range if price_range > 0 else 0

            # Get delta information
            delta_history = list(real_time_data.get('delta_history', []))
            if len(delta_history) < 20:
                return {'valid': False}

            recent_delta = sum(delta_history[-20:])

            # Check for divergences
            price_change = recent_bars['close'].iloc[-1] - recent_bars['close'].iloc[0]
            price_direction = np.sign(price_change)
            delta_direction = np.sign(recent_delta)

            divergence = price_direction != 0 and delta_direction != 0 and price_direction != delta_direction

            # Check for absorption
            absorption_detected = False
            absorption_direction = 'neutral'

            # Condition 1: High volume with minimal price movement
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else (current_price * 0.01)
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]

            minimal_movement = abs(price_change) < atr * 0.5
            high_volume = total_volume > avg_volume * self.absorption_ratio

            if minimal_movement and high_volume:
                absorption_detected = True

                # Determine if bulls or bears are being absorbed
                if recent_delta > 0:
                    # Bullish pressure is being absorbed (resistance)
                    absorption_direction = 'bull_absorption'
                elif recent_delta < 0:
                    # Bearish pressure is being absorbed (support)
                    absorption_direction = 'bear_absorption'

            return {
                'valid': True,
                'absorption_detected': absorption_detected,
                'direction': absorption_direction,
                'volume_density': volume_density,
                'divergence': divergence,
                'price_change': price_change,
                'delta': recent_delta
            }

        except Exception as e:
            self.logger.error(f"Error checking absorption for {symbol} on {timeframe}: {str(e)}")
            return {'valid': False}

    def analyze_effort_vs_result(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the relationship between volume effort and price result.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe to analyze
            df (pd.DataFrame): DataFrame with OHLC data

        Returns:
            Dict[str, Any]: Effort vs. result analysis
        """
        try:
            if df.empty or len(df) < 20 or 'volume' not in df.columns:
                return {'valid': False}

            # Calculate volume moving average
            volume_ma = df['volume'].rolling(window=20).mean()

            # Look at the last few bars
            recent_bars = df.iloc[-5:].copy()
            recent_bars['volume_ratio'] = recent_bars['volume'] / volume_ma.iloc[-5:]

            # Calculate price changes
            if len(recent_bars) >= 2:
                recent_bars['price_change'] = recent_bars['close'].diff()
                recent_bars['price_change_abs'] = recent_bars['price_change'].abs()

                # Calculate price change per unit volume
                recent_bars['efficiency'] = recent_bars['price_change_abs'] / recent_bars['volume']
                recent_bars['efficiency'].replace([np.inf, -np.inf], np.nan, inplace=True)
                recent_bars['efficiency'].fillna(0, inplace=True)

                efficiency = recent_bars['efficiency'].iloc[-1]
                avg_efficiency = recent_bars['efficiency'].mean()

                # High volume bar with low efficiency indicates poor effort/result ratio
                current_volume_ratio = recent_bars['volume_ratio'].iloc[-1]
                high_volume = current_volume_ratio > 1.5
                low_efficiency = efficiency < avg_efficiency * 0.5

                # Effort-result anomaly detection
                anomaly = high_volume and low_efficiency

                # Directional efficiency
                current_price_change = recent_bars['price_change'].iloc[-1]
                directional_efficiency = np.sign(current_price_change)

                return {
                    'valid': True,
                    'efficiency': efficiency,
                    'avg_efficiency': avg_efficiency,
                    'volume_ratio': current_volume_ratio,
                    'anomaly_detected': anomaly,
                    'directional_efficiency': directional_efficiency,
                    'high_volume': high_volume,
                    'low_efficiency': low_efficiency
                }
            else:
                return {'valid': False}

        except Exception as e:
            self.logger.error(f"Error analyzing effort vs result for {symbol} on {timeframe}: {str(e)}")
            return {'valid': False}

    def detect_institutional_activity(self, symbol: str, df: pd.DataFrame,
                                      real_time_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potential institutional buying/selling activity.

        Args:
            symbol (str): Symbol to analyze
            df (pd.DataFrame): DataFrame with OHLC data
            real_time_data (Dict): Real-time market data

        Returns:
            Dict[str, Any]: Institutional activity analysis
        """
        try:
            if df.empty or len(df) < 20:
                return {'valid': False}

            # Get current price
            current_price = df['close'].iloc[-1]

            # Characteristics of institutional activity:
            # 1. Large delta moves
            delta_history = list(real_time_data.get('delta_history', []))
            if len(delta_history) < 50:
                return {'valid': False}

            recent_delta = sum(delta_history[-20:])
            medium_delta = sum(delta_history[-50:])

            # Calculate volume metrics if available
            volume_characteristics = {}
            if 'volume' in df.columns:
                avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
                last_volume = df['volume'].iloc[-1]
                volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1

                # Large volume is a sign of institutional activity
                large_volume = volume_ratio > 2.0
                volume_characteristics = {
                    'volume_ratio': volume_ratio,
                    'large_volume': large_volume
                }

            # Calculate delta threshold based on historical data
            delta_std = np.std(delta_history) if delta_history else 100
            delta_threshold = max(self.delta_threshold, 2 * delta_std)

            # Detect large delta moves
            large_delta = abs(recent_delta) > delta_threshold

            # Determine direction
            direction = 'neutral'
            if recent_delta > delta_threshold:
                direction = 'buying'
            elif recent_delta < -delta_threshold:
                direction = 'selling'

            # Check for consistent pressure
            consistent_pressure = np.sign(recent_delta) == np.sign(medium_delta) and abs(medium_delta) > delta_threshold

            # Strength score (0-1)
            delta_ratio = abs(recent_delta) / delta_threshold if delta_threshold > 0 else 0
            strength = min(1.0, delta_ratio)

            # Update institutional activity marker if significant activity detected
            significant_activity = large_delta and (
                        volume_characteristics.get('large_volume', False) or consistent_pressure)

            if significant_activity and symbol in self.institutional_activity_markers:
                self.institutional_activity_markers[symbol] = {
                    'timestamp': datetime.now(pytz.UTC),
                    'level': current_price,
                    'direction': direction
                }

            return {
                'valid': True,
                'activity_detected': large_delta,
                'significant_activity': significant_activity,
                'direction': direction,
                'strength': strength,
                'consistent_pressure': consistent_pressure,
                'volume_characteristics': volume_characteristics,
                'recent_delta': recent_delta,
                'delta_threshold': delta_threshold
            }

        except Exception as e:
            self.logger.error(f"Error detecting institutional activity for {symbol}: {str(e)}")
            return {'valid': False}

    def calculate_transaction_intensity(self, symbol: str, real_time_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure transaction intensity and momentum.

        Args:
            symbol (str): Symbol to analyze
            real_time_data (Dict): Real-time market data

        Returns:
            Dict[str, Any]: Transaction intensity analysis
        """
        try:
            # Transaction intensity is measured by the rate of transactions and volume
            last_volume = real_time_data.get('last_volume', 0)
            delta_history = list(real_time_data.get('delta_history', []))

            if len(delta_history) < 20:
                return {'valid': False}

            # Calculate absolute delta (total transacted volume regardless of direction)
            abs_delta = [abs(d) for d in delta_history[-20:]]
            total_volume = sum(abs_delta)
            avg_transaction_size = total_volume / len(abs_delta) if abs_delta else 0

            # Calculate acceleration/deceleration
            if len(abs_delta) >= 20:
                recent_volume = sum(abs_delta[-10:])
                previous_volume = sum(abs_delta[-20:-10])

                acceleration = (recent_volume - previous_volume) / previous_volume if previous_volume > 0 else 0
                momentum = 'accelerating' if acceleration > 0.1 else 'decelerating' if acceleration < -0.1 else 'stable'
            else:
                acceleration = 0
                momentum = 'stable'

            # Calculate intensity ratio based on recent activity
            intensity = total_volume / 100  # Base value, normalize based on symbol

            # Normalize to 0-1 range
            normalized_intensity = min(1.0, intensity / 1000)

            return {
                'valid': True,
                'intensity': normalized_intensity,
                'total_volume': total_volume,
                'avg_transaction_size': avg_transaction_size,
                'acceleration': acceleration,
                'momentum': momentum,
                'high_intensity': normalized_intensity > 0.7
            }

        except Exception as e:
            self.logger.error(f"Error calculating transaction intensity for {symbol}: {str(e)}")
            return {'valid': False}

    def estimate_slippage(self, symbol: str, order_size: float) -> float:
        """
        Estimate slippage based on market depth and order flow metrics.

        Args:
            symbol (str): Symbol to estimate for
            order_size (float): Order size in lots

        Returns:
            float: Estimated slippage in price points
        """
        try:
            # Get real-time market data
            rtd = self.parent.real_time_data.get(symbol)
            if not rtd:
                return 0.001  # Default minimal slippage

            # Get delta history for liquidity estimation
            delta_history = list(rtd.get('delta_history', []))
            if len(delta_history) < 20:
                return 0.001

            # Get recent transaction volume
            recent_volume = sum(abs(d) for d in delta_history[-20:])

            # Impact increases with order size relative to market volume
            impact_factor = min(2.0, order_size / (recent_volume * 0.01)) if recent_volume > 0 else 0.1

            # Get bid-ask spread as baseline for slippage
            bid = rtd.get('bid', 0)
            ask = rtd.get('ask', 0)

            if bid > 0 and ask > 0:
                spread = ask - bid
            else:
                # Default to 0.1% of price if spread not available
                price = rtd.get('last_price', 1)
                spread = price * 0.001

            # Slippage is typically a fraction of spread plus impact
            base_slippage = spread * 0.5  # Half the spread
            impact_slippage = spread * impact_factor  # Additional impact based on order size

            total_slippage = base_slippage + impact_slippage

            # Get DataFrame for ATR context
            df = None
            for tf in ['1h', '15min', '5min']:
                df = self.parent.indicator_histories.get(symbol, {}).get(tf)
                if df is not None and not df.empty:
                    break

            # Use ATR to ensure slippage is reasonable
            if df is not None and 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                # Cap slippage at 20% of ATR
                max_slippage = atr * 0.2
                total_slippage = min(total_slippage, max_slippage)

            return total_slippage

        except Exception as e:
            self.logger.error(f"Error estimating slippage for {symbol}: {str(e)}")
            return 0.001  # Default minimal slippage on error

    def _determine_flow_direction(self, delta_analysis: Dict[str, Any],
                                  bid_ask: Dict[str, Any],
                                  absorption: Dict[str, Any],
                                  institutional: Dict[str, Any]) -> str:
        """
        Determine overall order flow direction based on multiple factors.

        Args:
            delta_analysis (Dict): Delta analysis results
            bid_ask (Dict): Bid-ask analysis results
            absorption (Dict): Absorption analysis results
            institutional (Dict): Institutional activity analysis

        Returns:
            str: Order flow direction ('up', 'down', or 'neutral')
        """
        try:
            # Default to neutral
            direction = 'neutral'

            # Check if analyses are valid
            if not delta_analysis.get('valid', False) or not bid_ask.get('valid', False):
                return direction

            # Get individual directions with weights
            delta_direction = delta_analysis.get('direction', 'neutral')
            delta_strength = delta_analysis.get('strength', 0)
            delta_weight = 3

            bid_ask_direction = 'up' if bid_ask.get('direction') == 'bid_dominant' else 'down' if bid_ask.get(
                'direction') == 'ask_dominant' else 'neutral'
            bid_ask_strength = bid_ask.get('strength', 0)
            bid_ask_weight = 2

            # Consider institutional activity with high weight
            inst_direction = institutional.get('direction', 'neutral')
            inst_strength = institutional.get('strength', 0)
            inst_weight = 4 if institutional.get('significant_activity', False) else 0

            # Handle absorption (it can override other signals)
            absorption_detected = absorption.get('absorption_detected', False)
            absorption_direction = absorption.get('direction', 'neutral')

            if absorption_detected:
                if absorption_direction == 'bull_absorption':
                    # Bullish pressure being absorbed indicates potential bearish move
                    absorption_direction_translated = 'down'
                    absorption_weight = 3
                elif absorption_direction == 'bear_absorption':
                    # Bearish pressure being absorbed indicates potential bullish move
                    absorption_direction_translated = 'up'
                    absorption_weight = 3
                else:
                    absorption_direction_translated = 'neutral'
                    absorption_weight = 0
            else:
                absorption_direction_translated = 'neutral'
                absorption_weight = 0

            # Calculate weighted direction score
            up_score = 0
            down_score = 0

            if delta_direction == 'up':
                up_score += delta_weight * delta_strength
            elif delta_direction == 'down':
                down_score += delta_weight * delta_strength

            if bid_ask_direction == 'up':
                up_score += bid_ask_weight * bid_ask_strength
            elif bid_ask_direction == 'down':
                down_score += bid_ask_weight * bid_ask_strength

            if inst_direction == 'buying':
                up_score += inst_weight * inst_strength
            elif inst_direction == 'selling':
                down_score += inst_weight * inst_strength

            if absorption_direction_translated == 'up':
                up_score += absorption_weight
            elif absorption_direction_translated == 'down':
                down_score += absorption_weight

            # Determine final direction based on scores
            if up_score > down_score + 0.5:  # Small threshold to avoid frequent changes
                direction = 'up'
            elif down_score > up_score + 0.5:
                direction = 'down'

            return direction

        except Exception as e:
            self.logger.error(f"Error determining flow direction: {str(e)}")
            return 'neutral'

    def calculate_flow_score(self, symbol: str, timeframe: str,
                             delta_analysis: Dict[str, Any],
                             bid_ask: Dict[str, Any],
                             liquidity: Dict[str, Any],
                             absorption: Dict[str, Any],
                             effort_result: Dict[str, Any],
                             institutional: Dict[str, Any],
                             intensity: Dict[str, Any]) -> float:
        """
        Calculate comprehensive order flow score.

        Args:
            symbol (str): Symbol to analyze
            timeframe (str): Timeframe being analyzed
            delta_analysis (Dict): Delta analysis results
            bid_ask (Dict): Bid-ask analysis results
            liquidity (Dict): Liquidity zone analysis
            absorption (Dict): Absorption analysis results
            effort_result (Dict): Effort vs. result analysis
            institutional (Dict): Institutional activity analysis
            intensity (Dict): Transaction intensity analysis

        Returns:
            float: Order flow score (0-10 scale)
        """
        try:
            score = 0
            max_score = 0

            # 1. Delta Analysis (0-3 points)
            delta_weight = self.flow_weights.get('delta', 3)
            max_score += delta_weight

            if delta_analysis.get('valid', False):
                delta_strength = delta_analysis.get('strength', 0)
                delta_consistency = delta_analysis.get('consistency', 0)

                # Reward strong and consistent delta
                delta_score = delta_strength * 0.7 + delta_consistency * 0.3
                score += delta_score * delta_weight

            # 2. Bid-Ask Imbalance (0-2 points)
            bid_ask_weight = self.flow_weights.get('bid_ask_imbalance', 2)
            max_score += bid_ask_weight

            if bid_ask.get('valid', False):
                bid_ask_strength = bid_ask.get('strength', 0)

                # Only reward significant imbalances
                if bid_ask.get('significant', False):
                    score += bid_ask_strength * bid_ask_weight

            # 3. Absorption Analysis (0-2 points)
            absorption_weight = self.flow_weights.get('absorption', 2)
            max_score += absorption_weight

            if absorption.get('valid', False) and absorption.get('absorption_detected', False):
                # Absorption is an important signal when detected
                score += absorption_weight

            # 4. Effort vs. Result Analysis (0-2 points)
            effort_weight = self.flow_weights.get('effort_vs_result', 2)
            max_score += effort_weight

            if effort_result.get('valid', False):
                # Reward normal effort-result relationship (not anomalous)
                anomaly = effort_result.get('anomaly_detected', False)
                efficiency = effort_result.get('efficiency', 0) / max(0.0001,
                                                                      effort_result.get('avg_efficiency', 0.0001))

                # Cap efficiency ratio at 2
                efficiency_ratio = min(2.0, efficiency) / 2.0  # Normalize to 0-1

                if not anomaly:
                    score += efficiency_ratio * effort_weight

            # 5. Liquidity Zones (0-2 points)
            liquidity_weight = self.flow_weights.get('liquidity', 2)
            max_score += liquidity_weight

            if liquidity.get('valid', False):
                # Check if price is in a liquidity zone
                in_liquidity_zone = liquidity.get('in_liquidity_zone', False)
                nearest_zone = liquidity.get('nearest_zone')

                if in_liquidity_zone:
                    score += liquidity_weight
                elif nearest_zone and nearest_zone.get('distance_atr', float('inf')) < 1:
                    # Partial score if near a liquidity zone
                    score += liquidity_weight * 0.5

            # 6. Institutional Activity (0-3 points, potential bonus)
            inst_weight = self.flow_weights.get('institutional_activity', 3)
            max_score += inst_weight

            if institutional.get('valid', False):
                significant_activity = institutional.get('significant_activity', False)
                inst_strength = institutional.get('strength', 0)

                if significant_activity:
                    score += inst_strength * inst_weight

                    # Extra bonus for confirmed institutional activity
                    consistent_pressure = institutional.get('consistent_pressure', False)
                    if consistent_pressure:
                        score += inst_weight * 0.5  # Additional bonus
                        max_score += inst_weight * 0.5  # Account for this in max

            # 7. Transaction Intensity (0-2 points)
            intensity_weight = self.flow_weights.get('transaction_intensity', 2)
            max_score += intensity_weight

            if intensity.get('valid', False):
                normalized_intensity = intensity.get('intensity', 0)
                high_intensity = intensity.get('high_intensity', False)

                if high_intensity:
                    score += intensity_weight * normalized_intensity

            # Normalize final score to 0-10 scale
            normalized_score = (score / max_score) * 10 if max_score > 0 else 0
            return normalized_score

        except Exception as e:
            self.logger.error(f"Error calculating flow score for {symbol} on {timeframe}: {str(e)}")
            return 0.0