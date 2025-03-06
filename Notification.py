import logging


class Notification:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def send_notification(self, message, details):
        self.logger.info(f"Notification: {message} - Details: {details}")
