import logging

import math

import cv2
import numpy as np

from constants import constants

class Point:
    def __init__(self, center):
        self.center = (center[0], center[1])

        self.logger = logging.getLogger("ateball.balls.point")

    def distance(self, point):
        return round(math.hypot(self.center[0] - point[0], self.center[1] - point[1]), 2)

    def average(self, point):
        return (int((self.center[0] + point[0])/2),int((self.center[1] + point[1])/2))

class Ball(Point):
    def __init__(self, center, suit=None, name=None, color=None, target=False):
        super().__init__(center)

        self.suit = suit
        self.name = name
        self.color = color
        self.number = int(name[0]) if name is not None else None
        self.target = target

        self.rgb = (0, 255, 0)
        self.mask = None
        self.pocketed = False

        self.offset_center = (0, 0)
        self.rotated_center = (0, 0)
        self.area = 0

        self.logger = logging.getLogger("ateball.ball")

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __copy__(self):
        return Ball(self.center, self.suit, self.name, self.color, self.target)

    def update(self, center):
        self.center = (center[0], center[1])
        self.offset_center = (center[0] + constants.regions.table[0], center[1] + constants.regions.table[1])

    def draw(self, image):
        cv2.circle(image, self.center, 10, self.rgb, 1)

class Cue(Ball):
    def __init__(self, center):
        super().__init__(center, None, "cueball", "white", False)

        self.rgb = (255, 255, 255)
        self.number = None

class Eight(Ball):
    def __init__(self, center):
        super().__init__(center, None, "eightball", "black", False)

        self.rgb = (0, 0, 0)
        self.number = 8

