import logging

# Simple logger configuration used across the project.
# Modules can import this logger via "from ..logging.logger import logger"

logger = logging.getLogger("simclever_intercom")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
