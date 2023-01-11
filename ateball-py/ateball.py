import logging
import traceback
from contextlib import suppress

import math
import time
import pyautogui

import threading
import queue as q

from enum import Enum
from enum import IntEnum

#files
import games
import utils
import constants

class AteBall():
    def __init__(self, port):
        self.ipc = utils.IPC()

        self.click_offset = [0, 0]

        self.processing_play_request = threading.Event()
        self.active_game = None

        self.quit_event = threading.Event()

        self.exception = None

        self.logger = logging.getLogger("ateball")

    def process_message(self):
        self.logger.info("waiting for command...")
        while not self.quit_event.is_set():
            try:
                response = None # change to object/dict to send response

                msg = self.ipc.incoming.get()

                m_type = msg["type"]
                # self.logger.debug(f"incoming msg: {msg}")
                
                if m_type == "play":
                    if not self.processing_play_request.is_set() and self.active_game is None:
                        threading.Thread(target=self.play, args=(msg,), daemon=True).start()
                    else:
                        response = { "type" : "BUSY"}
                elif m_type == "cancel":
                    self.cancel()
                elif m_type == "quit":
                    self.quit()
                else:
                    pass
            except q.Empty() as e:
                pass
            except KeyError as e:
                response["status"] = "failed"
                response["msg"] = str(e)
            else: 
                if self.exception is not None:
                    self.logger.error(f"error processing message : {self.exception}")
                    if response is not None:
                        response["status"] = "failed"
                        response["msg"] = str(self.exception)
                else:
                    if (isinstance(response, dict)):
                        response["id"] = msg["id"]

                if response:
                    self.ipc.outgoing.put(response)


    ###initialization

    def start(self):
        try:
            self.logger.info("Starting Ateball...")
            self.init_start_time = time.time()

            threading.Thread(target=self.ipc.listen, daemon=True).start()
            threading.Thread(target=self.ipc.send, daemon=True).start()

            if self.ipc.listen_event.wait(5):
                #start receiving msgs and initialize
                threading.Thread(target=self.process_message, daemon=True).start()
        except Exception as e:
            self.logger.error(f"{type(e)} - {e}")
            self.quit()
        else:
            self.ipc.send_message({"type" : "INIT"})

    ###initialization

    ###menu

    def play(self, data):
        try:
            self.processing_play_request.set()
        
            location = data['location'] if "location" in data else ""
        
            Game = getattr(games, data['gamemode'].upper())
            self.active_game = Game(self.ipc, location, daemon=True)
            self.active_game.start()
        except Exception as e:
            self.logger.debug(e)
            if isinstance(e, KeyError):
                self.logger.error(f"error playing game: invalid gamemode location - {data['location']}")
            elif isinstance(e, AttributeError):
                self.logger.error(f"error playing game: invalid gamemode - {data['gamemode']}")
            else:
                self.logger.error(f"error playing game: {traceback.format_exc()}")
        finally:
            self.processing_play_request.clear()
            self.logger.debug(self.processing_play_request.is_set())

    def cancel(self):
        try:
            if self.active_game:
                self.active_game.cancel()
            else:
                raise Exception("no game to cancel")
        except Exception as e:
            self.logger.error(f"ran into error cancelling game: {e}")
        else:
            self.active_game = None

    ###menu

    def quit(self):
        self.ipc.quit()
        self.quit_event.set()