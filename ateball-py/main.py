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
    time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

    formatter = utils.Formatter()

    fHandler = logging.FileHandler(f"logs/{time}.log", mode="w")
    fHandler.setFormatter(formatter)
    fHandler.setLevel(logging.DEBUG)

    sHandler = logging.StreamHandler(sys.stdout)
    sHandler.setFormatter(formatter)
    sHandler.setLevel(logging.DEBUG)

    logger.addHandler(fHandler)
    logger.addHandler(sHandler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

def main():
    configure_logging()

    try:
        ateball = AteBall(8888)
        ateball.start()
        ateball.quit_event.wait()
    except KeyboardInterrupt: # causes error with webdriver - max retry error
        logger.debug("User interrupted execution.")
        webdriver.driver = None
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