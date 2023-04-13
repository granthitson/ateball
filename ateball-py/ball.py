import logging

import cv2
import numpy as np

from utils import Point
from constants import constants

class Ball(Point):
    def __init__(self, center, suit=None, name=None, color=None, target=False):
        super().__init__(center)

        self.suit = suit
        self.name = name
        self.color = color
        self.number = int(name[0]) if name is not None else None
        self.target = target

        self.bgr = (0, 255, 0)

        self.ball_mask = None
        self.white_mask = None
        self.stick_mask = None
        self.glove_mask = None

        self.white_total = None
        self.color_total = None
        self.ratio = None

        self.pocketed = False

        self.offset_center = (0, 0)
        self.rotated_center = (0, 0)
        self.area = 0

        self.logger = logging.getLogger("ateball.ball")

    def __str__(self):
        return f"Ball({self.name})"

    def __repr__(self):
        return f"Ball({self.name})"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, Ball):
            dist = self.distance(other)
            if dist:
                self.logger.debug(f"{self} - {other} : {dist}")
            return (dist < 5)
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __copy__(self):
        return Ball(self.center, self.suit, self.name, self.color, self.target)

    def update(self, center):
        self.center = (center[0], center[1])
        self.offset_center = (center[0] + constants.regions.table[0], center[1] + constants.regions.table[1])

    def get_features(self):
        # not happy with this process - works most of the time, but want more reliable

        # compensate for stick blocking
        stick_total = np.sum(np.count_nonzero(self.stick_mask, axis=-1))
        if stick_total > 9:
            solid_white_total =  80 - (stick_total * .33)
            solid_color_total =  230 - (stick_total * .66)
        else:
            solid_white_total =  80 - stick_total
            solid_color_total =  230

        # generate avg ratio comparing total white pixels and colored pixels against their max totals of solids
        self.white_total = np.sum(np.count_nonzero(self.white_mask, axis=-1))
        self.color_total = np.sum(np.count_nonzero(self.ball_mask, axis=-1)) / 3
        self.ratio = ((self.white_total / solid_white_total) + (self.color_total / solid_color_total)) / 2

    def set_identity(self, data, color_info):
        self.suit = data.suit if "suit" in data.__dict__ else None
        self.name = data.name
        self.number = data.number if "number" in data.__dict__ else None
        self.color = data.color if "color" in data.__dict__ else None
        self.bgr = color_info.bgr if "color" in data.__dict__ else (0, 255, 0)

    def draw(self, image):
        if self.suit == "solid":
            cv2.circle(image, self.center, 8, (self.bgr[0], self.bgr[1], self.bgr[2]), 2)
        elif self.suit == "stripe":
            cv2.line(image, (self.center[0] - 8, self.center[1]), (self.center[0] + 8, self.center[1]), (255, 255, 255), 2)
            cv2.circle(image, self.center, 8, (self.bgr[0], self.bgr[1], self.bgr[2]), 2)
        else:
            cv2.circle(image, self.center, 8, (self.bgr[0], self.bgr[1], self.bgr[2]), 2)

class Cue(Ball):
    def __init__(self, center):
        super().__init__(center, None, "cueball", "white", False)

        self.bgr = (255, 255, 255)
        self.number = None

class Eight(Ball):
    def __init__(self, center):
        super().__init__(center, None, "eightball", "black", False)

        self.bgr = (0, 0, 0)
        self.number = 8

