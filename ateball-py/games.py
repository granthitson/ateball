import logging
import traceback

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
from round import Round
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

        self.image_stack = []

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

        self.player_turn = threading.Event()
        self.opponent_turn = threading.Event()
        self.turn_start = threading.Event()
        self.turn_start_event = utils.OrEvent(self.turn_start, self.game_over_event)

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
                    self.turn_start_event.wait()
                    if self.player_turn.is_set() and self.turn_start.is_set():
                        self.turn_start.clear()

                        self.ipc.send_message({"type" : "ROUND-START"})

                        round_path = Path(self.game_path, f"round-{self.turn_num}")
                        os.makedirs(round_path, exist_ok=True)

                        cv2.imwrite(str(Path(round_path, "round_start.png")), self.round_start_image)

                        self.logger.info(f"Turn #{self.turn_num}")
                        
                        round_data = self.get_round_data(round_path)
                        game_const = { "regions" : self.regions, "hole_locations" : self.hole_locations, "balls" : self.balls}

                        self.current_round = Round(round_data, game_const)
                        result = self.current_round.start()
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
                    pos = utils.ImageHelper.imageSearch(self.img_game_start, image, region=self.regions.turn_start)
                    if pos:
                        self.get_game_num()
                        self.game_start.set()
                time.sleep(.25)
            except Exception as e:
                self.logger.error(f"error monitoring game start: {e}")
                time.sleep(.25)

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

        # load turn timer masks
        turn_timer_mask = cv2.imread(utils.ImageHelper.imagePath(constants.img_turn_timers_mask), 0)
        turn_timer_mask_single =cv2.imread(utils.ImageHelper.imagePath(constants.img_turn_timers_mask_single), 0)

        old_player_status = None
        while not self.game_over_event.is_set():
            try:
                image = self.window_capture()
                if image.any():
                    # keep track of last 20 image frames
                    if len(self.image_stack) >= 20:
                        self.image_stack.pop(0)
                    self.image_stack.append(image)

                    # get mean color of each turn timer - colors represent different state at beginning of round
                    turn_status = self.get_turn_status(turn_timer_mask, turn_timer_mask_single)

                    # get status of player/opponent turn timer - color, open/closed, timeout
                    player_status, opponent_status = turn_status["player"]["status"], turn_status["opponent"]["status"]
                    player_timer_start, opponent_timer_start = turn_status["player"]["hierarchy"], turn_status["opponent"]["hierarchy"]
                    timed_out = ((player_status == "timeout" and player_timer_start) or (opponent_status == "timeout" and opponent_timer_start))

                    if self.player_turn.is_set():
                        if opponent_status != "pending":
                            if opponent_status == "successive":
                                # on successive pockets, player turn starts when player timer flashes white and contour is 'closed'
                                # *as long as player status in the previous frame is not 'started'
                                if (player_status == "started" and player_timer_start) and old_player_status != "started":
                                    self.end_existing_round()

                                    self.turn_num += 1
                                    self.set_round_image(timed_out)
                                    self.player_turn.set()
                                    self.turn_start_event.notify(self.turn_start)
                            else:
                                # player turn ends when opponent timer flashes white and contour is 'closed'
                                if opponent_status == "started" and opponent_timer_start:
                                    self.end_existing_round()

                                    self.set_round_image(timed_out)
                                    self.opponent_turn.set()
                    elif self.opponent_turn.is_set():
                        # player turn starts when player timer flashes white and contour is 'closed'
                        if player_status == "started" and player_timer_start:
                            self.end_existing_round()

                            self.turn_num += 1
                            self.set_round_image(timed_out)
                            self.player_turn.set()
                            self.turn_start_event.notify(self.turn_start)
                    else:
                        # player and opponent turn will both not be set at start of game

                        if player_status != "pending":
                            self.end_existing_round()

                            self.turn_num += 1
                            self.set_round_image(timed_out)
                            self.player_turn.set()
                            self.turn_start_event.notify(self.turn_start)
                        else:
                            self.player_turn.clear()
                            self.opponent_turn.set()
            except Exception as e:
                self.logger.error(traceback.format_exc())
            finally:
                # keep track of last frame's player status
                old_player_status = player_status

    def get_turn_status(self, mask, mask_single):
        # status indicated by the color of turn timer and whether timer is 'open' or 'closed' (at the start of a turn)

        # slice specified image to size - match size of mask
        turn_timer = utils.CV2Helper.slice_image(self.image_stack[-1], self.regions.turn_timer)

        # mask out background - show only turn timers
        turn_timers_masked = cv2.bitwise_and(turn_timer, turn_timer, mask=mask)

        # slice image to seperate timers
        p_timer = utils.CV2Helper.slice_image(turn_timers_masked, self.regions.player_turn_timer)
        o_timer = utils.CV2Helper.slice_image(turn_timers_masked, self.regions.opponent_turn_timer)

        # match mean color to closest color
        closest_color_player = utils.CV2Helper.get_closest_color(p_timer, mask_single, self.game_constants["turn_status"])
        closest_color_opponent = utils.CV2Helper.get_closest_color(o_timer, mask_single, self.game_constants["turn_status"])

        # get hierarchy of turn timer contours (closed vs open)
        p_timer_hsv = cv2.cvtColor(p_timer.copy(), cv2.COLOR_BGR2HSV)
        o_timer_hsv = cv2.cvtColor(o_timer.copy(), cv2.COLOR_BGR2HSV)

        p_hierarchy_mask = utils.CV2Helper.create_mask(p_timer_hsv, np.array([0, 0, 100]), np.array([180, 255, 255]), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        o_hierarchy_mask = utils.CV2Helper.create_mask(o_timer_hsv, np.array([0, 0, 100]), np.array([180, 255, 255]), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        p_contours, p_hierarchy = cv2.findContours(p_hierarchy_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        o_contours, o_hierarchy = cv2.findContours(o_hierarchy_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        return {
            "player" : {
                "status" : closest_color_player,
                "hierarchy" : len(p_hierarchy[0]) > 1 if p_hierarchy is not None else False
            },
            "opponent" : {
                "status" : closest_color_opponent,
                "hierarchy" : len(o_hierarchy[0]) > 1 if o_hierarchy is not None else False
            },
        }

    def end_existing_round(self):
        self.opponent_turn.clear()
        self.player_turn.clear()

        self.ipc.send_message({"type" : "ROUND-END"})

        if self.current_round is not None and not self.current_round.round_over_event.is_set():
            self.current_round.round_over_event.notify(self.current_round.round_cancel)
            self.current_round = None
            self.logger.info(f"Turn #{self.turn_num} Complete")

        self.export_round_data()

    def set_round_image(self, turn_timed_out):
        if self.round_start_image is not None:
            # in case of timeout, use last recorded round_start_image
            self.round_start_image = self.round_start_image if turn_timed_out else self.image_stack[0]
        else:
            self.round_start_image = self.image_stack[0]

    def export_round_data(self):
        pass

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