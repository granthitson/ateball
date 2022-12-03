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
from utils import ResponseType as rtype
from utils import ResponseAction as raction
from utils import ResponseStatus as rstatus
import constants

class LoginStatus(Enum):
    LOGGED_OUT = 0
    LOGGED_IN = 1

class Gamemode(IntEnum):
    PENDING = 0
    ONE_ON_ONE = 1
    TOURNAMENT = 2
    NO_GUIDELINE = 3
    NINE_BALL = 4
    SPIN_WIN = 5
    LUCKY_SHOT = 6
    CHALLENGE = 7
    PASS_N_PLAY = 8
    QUICK_FIRE = 9
    GUEST = 10

class GameStatus(Enum):
    PENDING = 0
    INITIALIZED = 1

class AteBall():
    def __init__(self, port):
        self.ipc = utils.IPC()
        # self.task_scheduler = utils.TaskScheduler()

        #statuses

        self.loginStatus = LoginStatus.LOGGED_OUT

        self.gameMode = Gamemode.PENDING
        self.gameStatus = GameStatus.PENDING

        self.game_container = None
        self.game_region = [0, 0, constants.game_width, constants.game_height]
        self.click_offset = [0, 0]

        self.login_event = threading.Event()

        self.game_region_event = threading.Event()

        self.accept_prompt_event = threading.Event()
        self.deny_prompt_event = threading.Event()
        self.prompt = utils.OrEvent(self.accept_prompt_event, self.deny_prompt_event)

        self.menu_acquired_event = threading.Event()
        self.menu_exception_event = threading.Event()
        self.menu_search_event = utils.OrEvent(self.menu_acquired_event, self.menu_exception_event)

        self.processing_play_request = threading.Event()

        self.location_carousel_event = threading.Event()
        self.bet_selection_event = threading.Event()

        self.current_gamemode = None
        self.current_gamemode_info = None

        self.game_start_event = threading.Event()
        self.game_cancelled_event = threading.Event()
        self.game_event = utils.OrEvent(self.game_start_event, self.game_cancelled_event)
        self.game = None

        self.unique_dismiss_points = set()
        self.dismiss_points = q.PriorityQueue() 
        self.dismiss_lock = threading.Lock()
        self.dismiss_event = threading.Event()

        self.quit_event = threading.Event()

        self.exception = None

        self.tasks = set()

        self.logger = logging.getLogger("ateball")

    def send_message(self, msg):
        self.ipc.outgoing.put(msg)

    def status_msg(self, resptype=rtype.INFO):
        return {
            "type" : resptype.name,
            "action" : raction.STATUS.name,
            "status" : {
                "login" : self.loginStatus.name,
                "mode" : self.gameMode.name,
                "game" : self.gameStatus.name,
            }
        }
    
    def busy_msg(self):
        return {
            "type" : rtype.INFO.name,
            "action" : raction.BUSY.name,
        }

    def prompt_user(self, mode, timeout):
        self.logger.debug(f"prompting user to play: {mode.name}")

        msg = {
            "type" : rtype.PROMPT.name,
            "action" : raction.PLAY.name,
            "mode" : mode.name,
            "timeout" : timeout
        }

        self.send_message(msg)

    def accept_prompt(self):
        self.logger.debug("user accepted prompt")
        self.prompt.notify(self.accept_prompt_event)

    def deny_prompt(self):
        self.logger.debug("user denied prompt")
        self.prompt.notify(self.deny_prompt_event)

    def process_message(self):
        self.logger.info("waiting for command...")
        while not self.quit_event.is_set():
            try:
                response = None # change to object/dict to send response

                msg = self.ipc.incoming.get()
                self.logger.debug(msg)
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

                self.ipc.outgoing.put(response)


    ###initialization

    def start(self):
        try:
            self.logger.info("Starting Ateball...")
            self.init_start_time = time.time()

            #wait for websocket server to start or throw
            threading.Thread(target=self.ipc.listen, daemon=True).start()

            if self.ipc.listen_event.wait(5):
                #start receiving msgs and initialize
                threading.Thread(target=self.process_message, daemon=True).start()

                self.send_message(self.status_msg(rtype.INIT))
        except Exception as e:
            self.logger.error(f"{type(e)} - {e}")
            self.quit()
        else:
            self.send_message(self.status_msg(rtype.INIT))

    # def findGameRegion(self):
    #     try:
    #         self.game_region_event.clear()

    #         self.logger.info("Getting game region...")

    #         if not self.login_event.wait(5):
    #             raise Exception("login event not set")

    #         # self.game_container = await self.webdriver.presenceOfElement(".play-area", 10000)

    #         calibration = utils.ImageHelper.imageSearch(constants.img_calibration, point_type="corner", time_limit=20)
    #         if calibration:
    #             browser_pos = (0, 0)

    #             self.game_region[0], self.game_region[1] = calibration[0], calibration[1]
    #             self.click_offset[0], self.click_offset[1] = browser_pos[0]-calibration[0], browser_pos[1]-calibration[1]
    #             if self.game:
    #                 self.game.regions.recalculate()
    #         else:
    #             raise Exception(f"could not find calibration img")
    #     except Exception as e:
    #         self.exception = Exception(f"error getting game region: {e}")
    #     else:
    #         self.logger.debug(f"game window acquired: {self.game_region}")
    #         self.logger.debug(f"click offset: {self.click_offset} - {browser_pos} - {calibration[0], calibration[1]}")
    #         self.game_region_event.set()

    # def verifyLoginState(self):
    #     self.logger.info("Verifying login state...")

    #     await self.webdriver.presenceOfElement(".play-area", 10000)
    #     avatar = await self.webdriver.getElementBy(".avatar-container")
    #     if avatar is None:
    #         login_state = LoginStatus.LOGGED_OUT
    #         self.logger.debug("No account logged in.")
    #     else:
    #         login_state = LoginStatus.LOGGED_IN
    #         self.logger.debug("User logged in.")

    #     if login_state != self.loginStatus:
    #         raise Exception("Login state not authorized")
    #     else:
    #         self.login_event.set()

    # async def updateLoginState(self):
    #     self.logger.info("Verifying login state...")

    #     await self.webdriver.presenceOfElement(".play-area", 10000)
    #     avatar = await self.webdriver.getElementBy(".avatar-container")
    #     if avatar is None:
    #         self.loginStatus = LoginStatus.LOGGED_OUT
    #         self.logger.debug("No account logged in.")
    #         self.login_event.set()
    #     else:
    #         self.loginStatus = LoginStatus.LOGGED_IN
    #         self.logger.debug("User logged in.")
    #         self.login_event.set()

    # async def waitForLogoScene(self):
    #     start_time = time.time()

    #     try:
    #         iframe_container = await self.webdriver.presenceOfElement("#iframe-game", 3000)
    #         iframe = await iframe_container.contentFrame()
    #         await self.webdriver.waitForVisibility("#loadingBox", True, iframe)
    #         await self.webdriver.waitForVisibility("#loadingBox", False, iframe)
    #     except Exception as e:
    #         self.menu_search_event.notify(self.menu_exception_event)
    #         raise Exception(f"error waiting for logo scene - {e}")
    #     else:
    #         time_taken = time.time() - start_time
    #         self.logger.debug(f"logoscene ended - {time_taken:.2f}s")

    # async def searchForMenu(self):
    #     try:
    #         self.logger.info("Searching for game menu...")
    #         start_time = time.time()

    #         init = self.init_event.is_set()

    #         self.menu_search_event.clear()

    #         if not await asyncio.wait_for(self.game_region_event.wait(), timeout=5):
    #             raise Exception("game region undefined")
    #         else:
    #             end_conditions = [self.menu_search_event.is_set, self.menu_search_event.has_timed_out, self.game_start_event.is_set, self.quit_event.is_set]

    #             if not init:
    #                 await self.waitForLogoScene()

    #             if not self.processing_play_request.is_set() or self.menuStatus == MenuStatus.GUEST:
    #                 self.create_task(self.navigateToMainMenu, end_conditions)
    #             else:
    #                 self.create_task(self.navigateBackToMainMenu, end_conditions)

    #             await self.menu_search_event.wait(60) # search for menu with 60sec timeout
    #             if not (self.menu_acquired_event.is_set() or self.game_start_event.is_set()):
    #                 if self.menu_search_event.has_timed_out():
    #                     raise Exception("timed out")
    #                 else:
    #                     raise Exception("error searching for menu targets")
    #     except Exception as e:
    #         self.exception = Exception(f"could not find game menu: {type(e)} - {e} - {traceback.format_exc()}")
    #     else:
    #         time_taken = time.time() - start_time
    #         self.logger.info(f"Game menu found - {time_taken:.2f}s...")
    #         await self.send_message(self.status_msg())
    #     finally:
    #         self.prompt.clear()
    
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