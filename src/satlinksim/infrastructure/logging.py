import logging
import sys
import structlog

def configure_logging(level=logging.INFO):
    """
    Configures structlog to output structured JSON logs in production 
    and pretty-formatted logs in development.
    """
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if sys.stderr.isatty():
        # Pretty printing for interactive terminals
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # JSON output for production/logs
        processors.append(structlog.processors.dict_tracebacks)
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge standard logging to structlog if needed
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

def get_logger(name: str):
    return structlog.get_logger(name)
