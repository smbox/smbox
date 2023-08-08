import time
from datetime import datetime

class Logger:
    _instance = None
    LEVELS = {
        'ERROR': 1,
        'INFO': 2,
        'DEBUG': 3
    }

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, verbosity_level='INFO'):
        # Check if the instance is already initialized
        if hasattr(self, '_initialized'):
            return

        if verbosity_level not in self.LEVELS:
            raise ValueError(f"Invalid verbosity level: {verbosity_level}. Allowed values are: {', '.join(self.LEVELS.keys())}")

        self.verbosity_level = verbosity_level
        self._initialized = True

    @staticmethod
    def _get_current_time():
        current_time = time.time()
        formatted_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
        return formatted_time

    def log(self, message, level='INFO'):
        current_time = self._get_current_time()
        if self.LEVELS[level] <= self.LEVELS[self.verbosity_level]:
            print(f"{current_time}: {message}")
