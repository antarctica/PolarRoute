import logging
import time
import tracemalloc

from datetime import timedelta
from functools import wraps

"""
Utilities that might be of use
"""


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def memory_trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start(20)
        res = func(*args, **kwargs)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('traceback')

        stat = top_stats[0]
        logging.info("{} memory blocks: {.1f} KiB".
                     format(stat.count, stat.size / 1024))
        logging.info("\n".join(stat.traceback.format()))
        return res
    return wrapper


def timed_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info("Timed call to {} took {:02f} seconds".
                     format(func.__name__, end - start))
        return res
    return wrapper


# CLI utilities
def setup_logging(func,
                  log_format="[%(asctime)-17s :%(levelname)-8s] - %(message)s"):
    """Wraps a CLI endpoint and sets up logging for it

    This is probably not the smoothest implementation, but it's an educational
    one for people who aren't aware of decorators and how they're implemented.
    In addition, it supports a nice pattern for CLI endpoints

    TODO: start handling level configuration from logging yaml config

    :param func:
    :param log_format:
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        parsed_args = func(*args, **kwargs)
        level = logging.INFO

        if hasattr(parsed_args, "verbose") and parsed_args.verbose:
            level = logging.DEBUG

        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt="%d-%m-%y %T",
        )

        logging.getLogger("cdsapi").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        return parsed_args
    return wrapper


