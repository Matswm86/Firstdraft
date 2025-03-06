import json
import os
import logging
import pandas as pd
from central_trading_bot import CentralTradingBot


def load_config(config_path='config.json'):
    """
    Load the configuration file and substitute any environment variable placeholders.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error("Error: config.json not found!")
        exit(1)
    except json.JSONDecodeError:
        logging.error("Error: Invalid JSON in config.json!")
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
    """
    log_level = log_config.get("log_level", "INFO").upper()
    log_file = log_config.get("log_file", "trading_bot.log")
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging has been set up.")
def main():
    # Load configuration from config.json and substitute ENV variables
    config = load_config('config.json')
    setup_logging(config.get('trade_logger', {}))

      # Initialize the central trading bot with configuration
    bot = CentralTradingBot(config)

    # Determine the mode (backtest or live) and start the corresponding process
    mode = config.get('central_trading_bot', {}).get('mode', 'live').lower()
    if mode == 'backtest':
        bot.start_backtest()
    elif mode == 'live':
        bot.start_live_trading()
    else:
        logging.error("Invalid trading mode specified in configuration.")


if __name__ == '__main__':
    main()
