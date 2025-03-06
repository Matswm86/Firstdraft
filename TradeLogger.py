import logging


class TradeLogger:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_trades(self, signals):
        self.logger.info(f"Logging {len(signals)} trades.")
