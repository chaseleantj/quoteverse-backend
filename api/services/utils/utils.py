import os
import time

from api.settings import settings


def time_it(func):
    """
    Decorator to time a function and log the time taken if the LOG_TIME_IT environment variable is set to true.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if settings.LOG_TIME_IT:
            print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds to run.")
        return result
    return wrapper