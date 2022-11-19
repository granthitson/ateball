import logging

import cv2
import numpy as np
import pyautogui
import time

import constants

logger = logging.getLogger("ateball.ball")

class Ball:
    def __init__(self, suit, name, color, target):
        self.center = (0, 0)
        self.offsetCenter = (0, 0)
        self.rotatedCenter = (0, 0)
        self.area = 0

        self.target = target
        self.suit = suit
        self.color = color
        self.name = name
        self.number = int(name[0]) if name != ("cueball" or "eightball") else None
        self.mask = None
        self.RGB = (255, 255, 255)
        self.pocketed = False

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __copy__(self):
        return Ball(self.suit, self.name, self.color, self.target)

    def setCenter(self, center, region):
        self.center = (center[0], center[1])
        self.offsetCenter = (center[0] + constants.offset[0], center[1] + constants.offset[1])

        self.setPocketed(region)

    def setPocketed(self, region):
        if self.center[0] > region[2]:
            self.pocketed = True

    def maskSetup(self, hsv):
        if self.color == "white":
            upper_whitecue = np.array([27, 36, 255])
            lower_whitecue = np.array([18, 0, 120])
            maskwhiteself = cv2.inRange(hsv, lower_whitecue, upper_whitecue)
            maskwhiteself = cv2.morphologyEx(maskwhiteself, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            self.mask = maskwhiteself
            self.RGB = 255, 255, 255
        elif self.color == "black":
            upper_black = np.array([0, 0, 110])
            lower_black = np.array([0, 0, 30])
            maskblackself = cv2.inRange(hsv, lower_black, upper_black)
            maskblackself = cv2.morphologyEx(maskblackself, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            self.mask = maskblackself
            self.RGB = 17, 17, 17
        elif self.color == "yellow":
            upper_yellow = np.array([31, 255, 255])
            lower_yellow = np.array([19, 50, 160])
            maskyellowself = cv2.inRange(hsv, lower_yellow, upper_yellow)
            maskyellowself = cv2.morphologyEx(maskyellowself, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            self.mask = maskyellowself
            self.RGB = 30, 240, 247
        elif self.color == "blue":
            upper_blue = np.array([120, 220, 255])
            lower_blue = np.array([105, 50, 160])
            maskblueself = cv2.inRange(hsv, lower_blue, upper_blue)
            maskblueself = cv2.morphologyEx(maskblueself, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            self.mask = maskblueself
            self.RGB = 196, 112, 58
        elif self.color == "lightred":
            upper_lightred = np.array([180, 255, 250])
            lower_lightred = np.array([170, 0, 150])
            masklightredself = cv2.inRange(hsv, lower_lightred, upper_lightred)
            masklightredself = cv2.morphologyEx(masklightredself, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            self.mask = masklightredself
            self.RGB = 21, 6, 242
        elif self.color == "purple":
            upper_purple = np.array([139, 255, 255])
            lower_purple = np.array([130, 11, 70])
            maskpurpleself = cv2.inRange(hsv, lower_purple, upper_purple)
            maskpurpleself = cv2.morphologyEx(maskpurpleself, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            self.mask = maskpurpleself
            self.RGB = 130, 0, 120
        elif self.color == "orange":
            upper_orange = np.array([12, 255, 255])
            lower_orange = np.array([10, 134, 0])
            maskorangeself = cv2.inRange(hsv, lower_orange, upper_orange)
            maskorangeself = cv2.morphologyEx(maskorangeself, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            self.mask = maskorangeself
            self.RGB = 9, 101, 241
        elif self.color == "green":
            upper_green = np.array([70, 255, 255])
            lower_green = np.array([50, 0, 140])
            maskgreenself = cv2.inRange(hsv, lower_green, upper_green)
            maskgreenself = cv2.morphologyEx(maskgreenself, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            self.mask = maskgreenself
            self.RGB = 76, 149, 62
        else:
            upper_darkred = np.array([7, 255, 255])
            lower_darkred = np.array([3, 100, 100])
            maskdarkredself = cv2.inRange(hsv, lower_darkred, upper_darkred)
            maskdarkredself = cv2.morphologyEx(maskdarkredself, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            self.mask = maskdarkredself
            self.RGB = 6, 24, 102

    def isPocketed(self, min, max):
        if (self.center[0] + min) < min or (self.center[0] + min) > max:
            self.pocketed = True
        else:
            self.pocketed = False

    def moveTo(self):
        pyautogui.moveTo(self.offsetCenter)
        time.sleep(.3)

    def dragTo(self, target, duration=.5):
        pyautogui.dragTo(target, duration=duration)
        time.sleep(.3)

class Cue(Ball):
    def __init__(self):
        super().__init__(None, "cueball", "white", False)

        self.RGB = (255, 255, 255)
        self.number = None

class Eight(Ball):
    def __init__(self, target):
        super().__init__(None, "eightball", "black", target)

        self.RGB = (0, 0, 0)
        self.number = 8

