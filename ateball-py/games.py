import logging

import time

import threading
import queue as q

import os
import cv2
import numpy as np

import win32gui, win32ui, win32con

import json
from pathlib import Path

from abc import ABC

#files
from hole import Hole
import utils
import constants

class Game(threading.Thread, ABC):
    location: str
    img_game_start: str
    
    def __init__(self, ipc, location, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ipc = ipc

        self.hwnd = win32gui.FindWindow(None, os.getenv("APP_NAME"))

        self.name = self.__class__.__name__
        self.location = constants.locations[location] if location != "" else location

        self.game_start = threading.Event()
        self.game_cancelled = threading.Event()

        self.game_end = threading.Event()
        self.game_exception = threading.Event()
        self.game_over_event = utils.OrEvent(self.game_end, self.game_exception)

        self.game_path = None
        self.game_num = 0

        self.game_constants = utils.JSONHelper.loadJSON(Path("ateball-py", "game_constants.json"))

        self.regions = utils.RegionData(self.game_constants["regions"])
        self.hole_locations = [ Hole(hole["name"], hole["image"], (hole["x"], hole["y"]), self.regions.table_offset) for hole in self.game_constants["hole_locations"] ]
        self.walls = []
        
        self.suit = None
        self.turn_num = 0
        self.round_start_image = None

        self.current_turn = threading.Event()
        self.turn_start = threading.Event()
        self.turn_exception = threading.Event()
        self.turn_start_event = utils.OrEvent(self.turn_start, self.turn_exception, self.game_over_event)

        self.current_round = None

        self.logger = logging.getLogger("ateball.games")

    def run(self):
        try:
            self.logger.info(f"Playing {self.name}...")

            self.wait_for_game_start()
            if self.game_start.is_set():
                self.game_path = str(Path("ateball-py", "games", f"{self.game_num}-{self.name}-{self.location}"))
                os.makedirs(self.game_path, exist_ok=True)

                self.ipc.send_message({"type" : "GAME-START"})

                self.logger.info(f"\nGame #{self.game_num}\n")
                
                threading.Thread(target=self.wait_for_turn_start, daemon=True).start()

                while not self.game_over_event.is_set():
                    self.turn_start_event.wait(self.game_constants["round_time"] * 2)
                    if self.turn_start.is_set():
                        self.turn_start.clear()

                        self.ipc.send_message({"type" : "ROUND-START"})

                        round_path = Path(self.game_path, f"round-{self.turn_num}")
                        os.makedirs(round_path, exist_ok=True)

                        cv2.imwrite(str(Path(round_path, "round_start.png")), self.round_start_image)

                        self.logger.info(f"\nTurn #{self.turn_num}\n")
                        
                        round_data = self.get_round_data(round_path)
                        self.current_round = Round(round_data, self.game_constants)
                        result = self.current_round.start()
                    else:
                        if not self.turn_exception.is_set():
                            self.logger.debug("timed out waiting for turn")
                            break
        except Exception as e:
            self.logger.debug(e)
        finally:
            self.game_over_event.notify(self.game_end)
            if self.game_cancelled.is_set():
                self.ipc.send_message({"type" : "GAME-CANCELLED"})
            elif self.game_end.is_set():
                self.ipc.send_message({"type" : "GAME-END"})

    def cancel(self):
        self.logger.debug("cancelling game")
        self.game_cancelled.set()
        self.game_end.wait()

    def window_capture(self):
        w, h = (self.regions.game[2], self.regions.game[3])

        if not self.hwnd:
            raise Exception("window not found")

        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        offset = (self.regions.window_offset[0] + left, self.regions.window_offset[1] + top)

        # can't capture hardware accelerated window
        desktop = win32gui.GetDesktopWindow()
        wDC = win32gui.GetWindowDC(desktop)

        # wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj=win32ui.CreateDCFromHandle(wDC)
        cDC=dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0,0), (w, h), dcObj, offset, win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        # img = np.fromstring(signedIntsArray, dtype='uint8')
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (h, w, 4)

        img = img[...,:3]
        img = np.ascontiguousarray(img)

        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        return img

    def wait_for_game_start(self):
        self.logger.info("Waiting for game to start...")
        while not self.game_start.is_set() and not self.game_cancelled.is_set() and not self.game_over_event.is_set():
            try:
                image = self.window_capture()
                if image.any():
                    pos = utils.ImageHelper.imageSearchLock(self.img_game_start, image, region=self.regions.turn_start)
                    if pos:
                        self.get_game_num()
                        self.game_start.set()
                time.sleep(.25)
            except q.Empty:
                time.sleep(.25)
            except Exception as e:
                self.logger.error(f"error monitoring game start: {e}")

    def get_game_num(self):
        json_path = Path("ateball-py", "game.json")
        json_path.touch(exist_ok=True)

        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                data["num"] += 1 
                self.game_num = data["num"]
                with open(json_path, "w") as f:
                    f.write(json.dumps(data))
            except json.decoder.JSONDecodeError as e:
                data = {"num" : 1}
                self.game_num = data["num"]
                with open(json_path, "w") as f:
                    f.write(json.dumps(data))

    def wait_for_turn_start(self):
        self.logger.info("Waiting for turn to start...")
        turn_mask = cv2.imread(utils.ImageHelper.imagePath(constants.img_turn_mask), 0)

        while not self.game_over_event.is_set():
            try:
                image = self.window_capture()
                if image.any():
                    #mask turn cycle timer
                    turn_status = self.get_turn_mask(image, turn_mask)
                    height, width = turn_status.shape

                    #sort left to right based on x coord
                    contours = utils.CV2Helper.getContours(turn_status, lambda c: utils.CV2Helper.contourCenter(c)[0])

                    # check who has turn, based on location of contour
                    center = utils.CV2Helper.contourCenter(contours[0])
                    if center[0] < width/2 and (not self.current_turn.is_set() or (self.current_round and self.current_round.round_over_event.is_set())):
                        self.turn_num += 1
                        self.round_start_image = image
                        self.current_turn.set()
                        self.turn_start_event.notify(self.turn_start)
                    elif center[0] > width/2 and self.current_turn.is_set():
                        self.current_turn.clear()
                time.sleep(.25)
            except q.Empty:
                time.sleep(.25)
            except ZeroDivisionError as e:
                time.sleep(.25)
            except Exception as e:
                self.logger.debug(f"error monitoring turn start: {e}")

        self.logger.debug("done wiating")

    def get_turn_mask(self, image, turn_mask):
        image = image[self.regions.turn_mask[1]:self.regions.turn_mask[1] + self.regions.turn_mask[3], self.regions.turn_mask[0]:self.regions.turn_mask[0] + self.regions.turn_mask[2]]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        upper_gray = np.array([255, 255, 255])
        lower_gray = np.array([0, 0, 141])

        masked = cv2.bitwise_and(image_rgb, image_rgb, mask=turn_mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

        img = cv2.inRange(hsv, lower_gray, upper_gray)

        return img

    def get_round_data(self, image_path):
        return {
            "save_image_path" : image_path,
            "round_image" : self.round_start_image,
            "suit" : self.suit,
            "all_balls" : {},
            "unpocketed_balls" : [],
            "pocketed_balls" : {},
            "targets" : {},
            "nontargets" : {},
        }

class ONE_ON_ONE(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "game/bet_marker.png"

class TOURNAMENT(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "game/bet_marker.png"

class NO_GUIDELINE(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "game/bet_marker.png"

class NINE_BALL(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "game/bet_marker.png"

class LUCKY_SHOT(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "game/lucky_shot.png"

class CHALLENGE(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "game/bet_marker.png"

class PASS_N_PLAY(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "game/practice_marker.png"

class QUICK_FIRE(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "game/practice_marker.png"

class GUEST(Game):
    def __init__(self, location, *args, **kwargs):
        super().__init__(location, *args, **kwargs)

        self.img_game_start = "game\\bet_marker.png"