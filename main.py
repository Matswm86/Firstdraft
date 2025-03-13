import json
import os
import logging
from CentralTradingBot import CentralTradingBot


def load_config(config_path='config.json'):
    """
    Load the configuration file and substitute any environment variable placeholders.

    Args:
        config_path (str): Path to the config file (default: 'config.json').

    Returns:
        dict: Processed configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: Config file '{config_path}' not found!")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error: Invalid JSON in '{config_path}'!")
        exit(1)

    # Recursively substitute environment variable placeholders (e.g., "ENV:VAR_NAME")
    def substitute_env(item):
        if isinstance(item, dict):
            return {k: substitute_env(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [substitute_env(x) for x in item]
        elif isinstance(item, str) and item.startswith("ENV:"):
            env_var = item[4:]
            return os.environ.get(env_var, f"Missing_{env_var}")
        else:
            return item

    return substitute_env(config)


def setup_logging(log_config):
    """
    Set up logging based on the configuration.

    Args:
        log_config (dict): Logging configuration from config.json.
    """
    log_level = log_config.get("log_level", "INFO").upper()
    log_file = log_config.get("trade_log_file", "logs/trades.csv")  # Using trade log file for consistency
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename='trading_bot.log',  # Separate log file for bot events
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging has been set up.")


def main():
    """
    Main entry point: Load config, initialize bot, and start trading based on mode.
    """
    # Load configuration and set up logging
    config = load_config('config.json')
    setup_logging(config.get('trade_logger', {}))
    logging.info("Starting trading bot for The 5%ers on MT5...")

    bot = None  # Initialize bot as None to avoid UnboundLocalError
    try:
        # Initialize the central trading bot with configuration
        bot = CentralTradingBot(config)

        # Determine the mode and start the appropriate process
        mode = config.get('central_trading_bot', {}).get('mode', 'live').lower()
        if mode == 'backtest':
            logging.info("Running in backtest mode with existing NASDAQ futures data...")
            bot.start_backtest()
        elif mode == 'live':
            logging.info("Running in live mode with EURUSD and GBPJPY on MT5...")
            bot.start_live_trading()
        else:
            logging.error(f"Invalid trading mode '{mode}' specified in configuration.")
            exit(1)
    except Exception as e:
        logging.error(f"Failed to start bot: {str(e)}")
        if bot is not None:  # Only shutdown if bot was successfully created
            bot.shutdown()  # Ensure MT5 connection is closed on error
        exit(1)


if __name__ == '__main__':
    main()