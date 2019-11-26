import logging
import logging.config
from logging.handlers import RotatingFileHandler

LOGGER_NAME = 'dnfal'
LOGGER_FORMAT = '[%(asctime)s] %(levelname)s %(module)s:%(funcName)s:%(lineno)d "%(message)s"'

LEVEL_DEBUG = 'debug'
LEVEL_INFO = 'info'
LEVEL_WARNING = 'warning'
LEVEL_ERROR = 'error'
LEVEL_CRITICAL = 'critical'



logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': LOGGER_FORMAT
        }
    },
    'loggers': {
        LOGGER_NAME: {
            'level': 'INFO',
            'propagate': False,
            'handlers': [],
        },
    },
})

logger = logging.getLogger(LOGGER_NAME)


def config_logger(
    level: str,
    to_console: bool = True,
    file_path: str = ''
):
    logger.setLevel(level.upper())
    formatter = logging.Formatter(LOGGER_FORMAT)

    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file_path:
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=100 * 1024,
            backupCount=10
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not to_console and not file_path:
        null_handler = logging.NullHandler()
        logger.addHandler(null_handler)
