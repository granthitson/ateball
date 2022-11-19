import logging

import os
import time
import pyautogui
import cv2
import numpy as np
import imutils

from PIL import ImageGrab

import threading

import round
from hole import Hole
import utils
import constants

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)

class Game:
    def __init__(self, gametype, game_region):
        self.game_start = threading.Event()
        self.game_end = threading.Event()
        self.game_cancelled = threading.Event()
        self.game_exception = threading.Event()

        self.game_num = 0
        self.game_type = gametype

        self.regions = utils.RegionData(game_region)

        self.hole_locations = [ Hole(hole[0], hole[1], hole[2], self.regions.table_offset) for hole in constants.hole_locations ]

        self.suit = None

        self.turn_num = 0
        self.current_turn = False

        self.turn_start_event = threading.Event()
        self.turn_exception_event = threading.Event()
        self.turn_start = utils.OrEvent(self.turn_start_event, self.turn_exception_event)

        self.current_round = None

        self.logger = logging.getLogger("ateball.game")
        self.logger.setLevel(logging.DEBUG)

    def getGameNum(self):
        with open("gamecounter.txt", "r") as g:
            data = g.readlines()
            for line in data:
                if line is None:
                    g.write("0")
                self.game_num = str(int(line) + 1)
                with open("gamecounter.txt", "w") as g2:
                    g2.write(self.game_num)

    def start(self): #user waits for turn
        pos = utils.ImageHelper.imageSearch(constants.img_game_marker, self.regions.game, time_limit=10)
        if pos:
            self.game_start.set()

            self.getGameNum()
            self.logger.info(f"Game #{self.game_num}\n")
            
            threading.Thread(target=self.turnCycle, daemon=True).start()

            while True:
                self.turn_start.wait(constants.round_time * 2)
                if self.turn_start_event.is_set():
                    self.turn_start.clear()

                    save_img_path = f"games\\game{self.game_num}-{self.game_type.name}\\round{self.turn_num}\\"
                    os.makedirs(save_img_path, exist_ok=True)

                    self.logger.info(f"-- Turn #{self.turn_num} --")
                    
                    self.current_round = round.Round(self.suit, self.regions, self.hole_locations, save_img_path)
                    self.current_round.start()
                else:
                    if not self.turn_exception_event.is_set():
                        self.logger.debug("timed out waiting for turn")

                    break

    def getTurnStatus(self):
        turn_region = (self.regions.game[0]+332 , self.regions.game[1]+10, self.regions.game[0]+570, self.regions.game[1] + 82)

        upper_gray = np.array([255, 255, 255])
        lower_gray = np.array([0, 0, 141])

        screen = np.array(ImageGrab.grab(bbox=turn_region))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        turn_mask = cv2.imread(utils.ImageHelper.imagePath(constants.img_turn_mask), 0)

        masked = cv2.bitwise_and(screen, screen , mask=turn_mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, lower_gray, upper_gray)

    def turnCycle(self):
        try:
            while True:
                #mask turn cycle timer
                turn_status = self.getTurnStatus()
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

def main():
    game_region = [409, 135, constants.game_width, constants.game_height] #without extension
    # game_region = [357, 135, constants.game_width, constants.game_height] #with extension
    g = Game(utils.GameType.PASSNPLAY, game_region)
    g.start()

if __name__ == '__main__':
    main()