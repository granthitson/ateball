import logging

import time

import threading
import queue as q

import os
import cv2
import numpy as np

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

        self.name = self.__class__.__name__
        self.location = constants.locations[location] if location != "" else location

        self.game_start = threading.Event()
        self.game_cancelled = threading.Event()

        self.game_end = threading.Event()
        self.game_exception = threading.Event()
        self.game_over_event = utils.OrEvent(self.game_end, self.game_exception)

        self.game_num = 0

        self.regions = utils.RegionData()

        self.hole_locations = [ Hole(hole[0], hole[1], hole[2], self.regions.table_offset) for hole in constants.hole_locations ]
        self.suit = None

        self.turn_num = 0
        self.current_turn = False

        self.turn_start_event = threading.Event()
        self.turn_exception_event = threading.Event()
        self.turn_start = utils.OrEvent(self.turn_start_event, self.turn_exception_event, self.game_over_event)

        self.current_round = None

        self.logger = logging.getLogger("ateball.games")

    def run(self): #user waits for turn
        try:
            self.logger.info(f"Playing {self.name}...")

            self.wait_for_game_start()
            if self.game_start.is_set():
                self.ipc.send_message({"type" : "GAME-START"})

                self.logger.info(f"Game #{self.game_num}\n")
                
                threading.Thread(target=self.wait_for_turn_start, daemon=True).start()

                while not self.game_over_event.is_set():
                    self.turn_start.wait(constants.round_time * 2)
                    if self.turn_start_event.is_set():
                        self.turn_start.clear()

                        self.ipc.send_message({"type" : "ROUND-START"})

                        save_img_path = f"games\\game{self.game_num}-{self.name}-{self.location}\\round{self.turn_num}\\"
                        os.makedirs(save_img_path, exist_ok=True)

                        self.logger.info(f"-- Turn #{self.turn_num} --")
                        
                        self.current_round = round.Round(self.suit, self.regions, self.hole_locations, save_img_path)
                        self.current_round.start()
                    else:
                        if not self.turn_exception_event.is_set():
                            self.logger.debug("timed out waiting for turn")
                            break
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

    def wait_for_game_start(self):
        self.logger.info("Waiting for game to start...")
        while not self.game_cancelled.is_set() and not self.game_over_event.is_set():
            pos = utils.ImageHelper.imageSearchLock(self.img_game_start)
            if pos:
                self.game_start.set()
                self.set_game_num()
            time.sleep(.2)

    def set_game_num(self):
        with open("gamecounter.txt", "r") as g:
            data = g.readlines()
            for line in data:
                if line is None:
                    g.write("0")
                self.game_num = str(int(line) + 1)
                with open("gamecounter.txt", "w") as g2:
                    g2.write(self.game_num)

    def get_turn_mask(self):
        turn_region = (332, 10, 570, 82)

        upper_gray = np.array([255, 255, 255])
        lower_gray = np.array([0, 0, 141])

        screen = np.array(ImageGrab.grab(bbox=turn_region))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        turn_mask = cv2.imread(utils.ImageHelper.imagePath(constants.img_turn_mask), 0)

        masked = cv2.bitwise_and(screen, screen , mask=turn_mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, lower_gray, upper_gray)

    def wait_for_turn_start(self):
        try:
            while not self.game_over_event.is_set():
                #mask turn cycle timer
                turn_status = self.get_turn_mask()
                height, width = turn_status.shape

                #sort left to right based on x coord
                contours = utils.CV2Helper.getContours(turn_status, lambda c: utils.CV2Helper.contourCenter(c)[0])

                # check who has turn, based on location of contour
                center = utils.CV2Helper.contourCenter(contours[0])
                if center[0] < width/2 and (not self.current_turn or self.current_round.complete_event.is_set()):
                    self.turn_num += 1
                    self.current_turn = True
                    self.turn_start.notify(self.turn_start_event)
                elif center[0] > width/2 and self.current_turn:
                    self.current_turn = False
        except Exception as e:
            self.logger.error(f"something went wrong monitoring turns: {e}")
            if type(e) is ZeroDivisionError:
                self.logger.error("are you ingame? is the turn timer visible?")
            self.turn_start.notify(self.turn_exception_event)

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