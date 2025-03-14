#!/usr/bin/env python3
"""
Main entry point for the Trading Bot application.
Handles configuration loading, logging setup, and bot initialization.
"""

import json
import os
import sys
import logging
import logging.handlers
import argparse
import signal
from datetime import datetime
import pytz
import traceback
import time


def load_config(config_path):
    """
    Load the configuration file and substitute environment variables.

    Args:
        config_path (str): Path to the config file

    Returns:
        dict: Processed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file contains invalid JSON
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Recursively substitute environment variables
        config = substitute_env_vars(config)
        return config
    except FileNotFoundError:
        logging.critical(f"Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logging.critical(f"Invalid JSON in config file: {str(e)}")
        raise


def substitute_env_vars(item):
    """
    Recursively substitute environment variables in a configuration item.

    Args:
        item: Configuration item (dict, list, or scalar)

    Returns:
        The item with environment variables substituted
    """
    if isinstance(item, dict):
        return {k: substitute_env_vars(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [substitute_env_vars(i) for i in item]
    elif isinstance(item, str) and item.startswith("ENV:"):
        env_var = item[4:]
        env_value = os.environ.get(env_var)
        if env_value is None:
            logging.warning(f"Environment variable {env_var} not found")
            return f"ENV_NOT_FOUND_{env_var}"
        return env_value
    return item


def setup_logging(config):
    """
    Set up logging based on configuration.

    Args:
        config (dict): Logging configuration
    """
    try:
        # Extract logging configuration
        log_level_name = config.get('log_level', 'INFO').upper()
        log_file = config.get('log_file', 'logs/trading_bot.log')
        max_log_size = config.get('max_log_size', 10 * 1024 * 1024)  # 10 MB
        backup_count = config.get('backup_count', 5)
        log_format = config.get('log_format',
                                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Get log level
        log_level = getattr(logging, log_level_name, logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Clear any existing handlers
        root_logger.handlers = []

        # Create formatter
        formatter = logging.Formatter(log_format)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Create file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_log_size, backupCount=backup_count, encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Logging configured with level {log_level_name}, file: {log_file}")
    except Exception as e:
        logging.critical(f"Error setting up logging: {str(e)}")
        raise


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    try:
        parser = argparse.ArgumentParser(description="Trading Bot Application")
        parser.add_argument(
            "-c", "--config",
            default="config.json",
            help="Path to configuration file (default: config.json)"
        )
        parser.add_argument(
            "-m", "--mode",
            choices=["live", "backtest"],
            help="Override trading mode from config"
        )
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )
        return parser.parse_args()
    except Exception as e:
        logging.error(f"Error parsing arguments: {str(e)}")
        sys.exit(1)


def handle_signals(bot):
    """
    Set up signal handlers for graceful shutdown.

    Args:
        bot: The trading bot instance
    """
    def signal_handler(sig, frame):
        logging.info(f"Received signal {sig}, shutting down...")
        if bot:
            bot.stop()
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logging.debug("Signal handlers registered")
    except Exception as e:
        logging.error(f"Error setting up signal handlers: {str(e)}")


def validate_config(config):
    """
    Validate essential configuration parameters.

    Args:
        config (dict): Configuration dictionary

    Returns:
        bool: True if configuration is valid

    Raises:
        ValueError: If essential configuration is missing
    """
    try:
        required_sections = ['central_trading_bot', 'symbols']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        mode = config['central_trading_bot'].get('mode')
        if mode not in ['live', 'backtest']:
            raise ValueError(f"Invalid trading mode: {mode}. Must be 'live' or 'backtest'")

        if not config['symbols']:
            raise ValueError("No trading symbols specified in configuration")

        if mode == 'live':
            if 'mt5_settings' not in config:
                raise ValueError("Missing MT5 settings for live trading")
            required_mt5 = ['server', 'login', 'password']
            for setting in required_mt5:
                if setting not in config['mt5_settings'] or not config['mt5_settings'][setting]:
                    raise ValueError(f"Missing or empty required MT5 setting: {setting}")

        logging.debug("Configuration validated successfully")
        return True
    except ValueError as e:
        logging.critical(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logging.critical(f"Unexpected error validating config: {str(e)}")
        raise


def main():
    """
    Main entry point for the application.
    """
    # Init basic logging for startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse command line arguments
    args = parse_arguments()

    try:
        # Load configuration
        logging.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Override mode if specified
        if args.mode:
            logging.info(f"Overriding trading mode to {args.mode}")
            config['central_trading_bot']['mode'] = args.mode

        # Set verbose logging if requested
        if args.verbose:
            config.setdefault('trade_logger', {})['log_level'] = 'DEBUG'

        # Validate configuration
        validate_config(config)

        # Set up proper logging
        trade_logger_config = config.get('trade_logger', {})
        setup_logging(trade_logger_config)

        # Import and create trading bot instance
        from CentralTradingBot import CentralTradingBot
        bot = CentralTradingBot(config)

        # Set up signal handlers
        handle_signals(bot)

        # Load previous state if available
        bot.load_state()

        # Start the bot based on mode
        mode = config['central_trading_bot']['mode']
        if mode == 'live':
            logging.info("Starting bot in live trading mode")
            bot.start()
        else:
            logging.info("Starting bot in backtest mode")
            bot.start()

        # Keep main thread alive for live mode
        while bot.state == 'running':
            time.sleep(1)

        # Generate summary report
        try:
            report = bot.generate_report()
            if report:
                logging.info("Generated performance report")
                timestamp = datetime.now(pytz.UTC).strftime('%Y%m%d_%H%M%S')
                report_dir = 'reports'
                os.makedirs(report_dir, exist_ok=True)
                report_file = os.path.join(report_dir, f"report_{timestamp}.json")
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2)
                logging.info(f"Report saved to {report_file}")
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")

        logging.info("Trading bot has completed execution")
        return 0  # Success

    except FileNotFoundError:
        logging.critical(f"Configuration file not found: {args.config}")
        print(f"ERROR: Configuration file not found: {args.config}")
        print("Please specify a valid configuration file with --config")
        return 1

    except json.JSONDecodeError as e:
        logging.critical(f"Invalid JSON in configuration file: {str(e)}")
        print(f"ERROR: Invalid JSON in configuration file: {str(e)}")
        return 1

    except ValueError as e:
        logging.critical(f"Configuration error: {str(e)}")
        print(f"ERROR: Configuration error: {str(e)}")
        return 1

    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
        print("\nShutting down gracefully...")
        if 'bot' in locals():
            bot.stop()
        return 0

    except Exception as e:
        logging.critical(f"Unhandled exception: {str(e)}")
        logging.debug(traceback.format_exc())
        print(f"ERROR: An unexpected error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)