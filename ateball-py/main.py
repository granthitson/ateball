import logging

import sys
import os
import traceback

import asyncio

#files
import utils
from ateball import AteBall

logger = logging.getLogger("ateball")

def configure_logging():
    formatter = utils.Formatter()

    fHandler = logging.FileHandler("debug.log", mode="w")
    fHandler.setFormatter(formatter)
    fHandler.setLevel(logging.DEBUG)

    sHandler = logging.StreamHandler(sys.stdout)
    sHandler.setFormatter(formatter)
    sHandler.setLevel(logging.DEBUG)

    logger.addHandler(fHandler)
    logger.addHandler(sHandler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("chromedriver_autoinstaller").setLevel(logging.INFO)

async def main():
    configure_logging()

    try:
        ateball = AteBall(8888)
        await ateball.start()
        await ateball.quit_event.wait()
    except KeyboardInterrupt: # causes error with webdriver - max retry error
        logger.debug("User interrupted execution.")
        webdriver.driver = None
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
    finally:  
        await ateball.shutdown()

        logger.info("exited")
        logger.handlers.clear()
        logging.shutdown()

        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())