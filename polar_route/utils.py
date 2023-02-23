import logging
import time
import tracemalloc

from datetime import datetime, timedelta
from functools import wraps
from calendar import monthrange
"""
Utilities that might be of use
"""
def rectangle_overlap(a_coords, b_coords):
    # Rectangles must have parallel lines
    # Coords are tuples of format
    # ((x0, y0), (x2, y2))
    # where 0/2 denote the min/max extent 
    # of the x/y coords of the rect
    min_a_coords = a_coords[0]
    max_a_coords = a_coords[1]
    min_b_coords = b_coords[0]
    max_b_coords = b_coords[1]
    
    dx = min(max_a_coords[0], max_b_coords[0]) - \
         max(min_a_coords[0], min_b_coords[0])
         
    dy = min(max_a_coords[1], max_b_coords[1]) - \
         max(min_a_coords[1], min_b_coords[1])
         
    return dx*dy
    
    
def frac_of_month(year, month, start_date=None, end_date=None):
    
    # Determine the number of days in the month specified
    days_in_month = monthrange(year, month)[1]
    # If not specified, default to beginning/end of month
    if start_date is None:
        start_date = str_to_datetime(f'{year}-{month}-01')
    if end_date is None:
        end_date = str_to_datetime(f'{year}-{month}-{days_in_month}')
        
    # Ensure that input to fn was valid
    assert(start_date.month == month), 'Start date not in same month!'
    assert(end_date.month == month), 'End date not in same month!'
    # Determine overlap from dates (inclusive)
    days_overlap = (end_date - start_date).days + 1
    # Return fraction
    return days_overlap / days_in_month
    
    
def str_to_datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')

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


if __name__ == '__main__':
    aa = str_to_datetime('2020-03-01')
    ab = str_to_datetime('2020-03-15')
    
    ba = str_to_datetime('2020-02-01')
    bb = str_to_datetime('2020-04-01')
    
    ax = date_range(aa, ab)
    bx = date_range(ba, bb)
    
    print(frac_of_month(2020, 2, start_date='2020-02-29'))
    # print(time_overlap(ax, bx))
    