import logging

_default_handler = logging.StreamHandler()
_default_handler.setLevel(logging.WARNING)
_default_formatter = logging.Formatter('{name} {levelname:8s} {message}',
                                       style='{')
_default_handler.setFormatter(_default_formatter)


def get_logger(module_name):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.WARNING)
    logger.addHandler(_default_handler)
    logger.propagate = True
    return logger
