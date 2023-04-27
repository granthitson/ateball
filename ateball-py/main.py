import logging

import sys
import os
import traceback

import datetime

#files
import utils
from ateball import AteBall

logger = logging.getLogger("ateball")

def configure_logging():
    log_level = os.environ.get("LOG_LEVEL")

    time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

    formatter = utils.Formatter()

    fHandler = logging.FileHandler(f"logs/{time}.log", mode="w")
    fHandler.setFormatter(formatter)
    fHandler.setLevel(log_level)

    sHandler = logging.StreamHandler(sys.stdout)
    sHandler.setFormatter(formatter)
    sHandler.setLevel(log_level)

    logger.addHandler(fHandler)
    logger.addHandler(sHandler)
    logger.setLevel(log_level)
    logger.propagate = False

def main():
    configure_logging()

    try:
        ateball = AteBall()
        ateball.start()
        ateball.quit_event.wait()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
    finally:  
        if not ateball.quit_event.is_set():
            ateball.quit()

        logger.info("exited")
        logger.handlers.clear()
        logging.shutdown()

        sys.exit(0)

if __name__ == "__main__":
    main()