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
import game
import utils
import constants

class AteBall():
    def __init__(self, port):
        self.ipc = utils.IPC()
        # self.task_scheduler = utils.TaskScheduler()

        #statuses

        self.click_offset = [0, 0]

        self.processing_play_request = threading.Event()

        self.quit_event = threading.Event()

        self.exception = None

        self.logger = logging.getLogger("ateball")

    def send_message(self, msg):
        self.ipc.outgoing.put(msg)

    def process_message(self):
        self.logger.info("waiting for command...")
        while not self.quit_event.is_set():
            try:
                response = None # change to object/dict to send response

                msg = self.ipc.incoming.get()

                action = msg["action"]
                if "data" in msg:
                    mdata = msg["data"]

                self.logger.debug(f"incoming msg: {msg}")
                
                if action == "login":
                    self.create_task(self.login)
                    # self.task_scheduler.add_task(self.login, 1)
                elif action == "logout":
                    pass
                elif action == "game_region":
                    self.create_task(self.findGameRegion)
                    # self.task_scheduler.add_task(self.findGameRegion, 1)
                elif action == "accept_prompt":
                    self.accept_prompt()
                elif action == "deny_prompt":
                    self.deny_prompt()
                elif action == "play":
                    if not self.processing_play_request.is_set():
                        self.create_task(self.play, mdata)
                    else:
                        response = self.busy_msg()
                elif action == "cancel":
                    self.logger.debug("cancelling active task - 1")
                    # self.task_scheduler.cancel_task()
                elif action == "quit":
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
                    self.logger.error(f"error completing action: {self.exception}")
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
            self.send_message({"type" : "INIT"})

    ###initialization

    ###menu

    def play(self, data):
        try:
            self.processing_play_request.set()

            self.current_gamemode = Gamemode[data['gamemode'].upper()]
            self.current_gamemode_info = constants.gamemodes[self.current_gamemode.name]
        
            if "location" in data and data['location'] in constants.locations:
                self.current_gamemode_info["location"] = {
                    "name" : data['location'],
                    "data" : constants.locations[data['location']]
                }
            else:
                self.current_gamemode_info["location"] = None

            if "bet" in data and int(data['bet']) in constants.bets:
                bet = int(data['bet'])
                self.current_gamemode_info["bet"] = {
                    "value" : bet,
                    "data" : constants.bets[bet]
                }
            else:
                self.current_gamemode_info["bet"] = None

            self.current_gamemode_info["username"] = data['username'] if "username" in data else None

            if "parent" in self.current_gamemode_info:
                parent = self.current_gamemode_info["parent"]
                self.logger.info(f"Playing {parent} - {self.current_gamemode.name}...")
            else:
                self.logger.info(f"Playing {self.current_gamemode.name}...")

            gamemode_pos = self.create_task(self.searchForGamemode)
            if gamemode_pos:
                self.logger.info(f"Navigated to {self.current_gamemode.name} gamemode..")

                self.prompt_user(self.current_gamemode, 30)
                self.prompt.clear()
                self.prompt.wait()
                
                if not self.accept_prompt_event.is_set():
                    self.logger.debug("play game prompt denied/timeout")
                    self.game_event.notify(self.game_cancelled_event)
                else:
                    self.click(gamemode_pos)
                
                    self.game = game.Game(self.current_gamemode, self.game_region)
                    self.create_task(self.game.start)
                    if self.game.game_start.wait(timeout=10):
                        self.game_event.notify(self.game_start_event)
                    else:
                        self.game_event.notify(self.game_cancelled_event)
                        raise Exception("could not determine if game started")
        except Exception as e:
            if isinstance(e, KeyError):
                if "location" in data:
                    self.logger.error(f"error playing game: invalid gamemode location - {data['location']}")
                elif "bet" in data:
                    self.logger.error(f"error playing game: invalid gamemode bet - {data['bet']}")
                else:
                    self.logger.error(f"error playing game: invalid gamemode - {data['gamemode']}")
            else:
                self.logger.error(f"error playing game: {e}")
        finally:
            if not self.game_start_event.is_set() or self.game_cancelled_event.is_set():
                self.create_task(self.searchForMenu)

            self.processing_play_request.clear()

    ###menu

    def quit(self):
        self.ipc.quit()
        self.quit_event.set()