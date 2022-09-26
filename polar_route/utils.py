import logging

from functools import wraps

"""

"""

def setup_logging(func,
                  log_format="[%(asctime)-17s :%(levelname)-8s] - %(message)s"):
    """Wraps a CLI endpoint and sets up logging for it

    This is probably not the smoothest implementation, but it's an educational
    one for people who aren't aware of decorators and how they're implemented.
    In addition, it supports a nice pattern for CLI endpoints

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

        # TODO: better way of handling these on a case by case basis
        logging.getLogger("cdsapi").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        return parsed_args
    return wrapper
