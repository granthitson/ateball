import logging

import cv2
import numpy as np

from utils import Point
from constants import constants

class Ball(Point):
    def __init__(self, center, suit=None, name=None, number=None, color=None, target=False, pocketed=False):
        super().__init__(center)

        self.name = name
        self.number = number
        self.color = color

        self.suit = suit
        self.is_target = target
        self.pocketed = pocketed

        self.bgr = (0, 255, 0)
        self.mask_info = BallMaskInfo()

        self.target_vector = None
        self.deflect_vector = None
        self.neighbors = {}

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
            return (dist < 5)
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

    def copy(self):
        return Ball(self.center, self.suit, self.name, self.color, self.is_target, self.pocketed)

    def serialize(self):
        return {
            "info" : {
                "name" : self.name,
                "number" : self.number,
                "color" : self.color,
                "suit" : self.suit,
                "target" : self.is_target,
                "pocketed" : self.pocketed
            },
            "center" : {
                "x" : (self.center[0] - constants.ball.radius) if self.center[0] is not None else 0,
                "y" : (self.center[1] - constants.ball.radius) if self.center[1] is not None else 0
            },
            "vectors" : {
                "target" : (self.target_vector - constants.ball.radius) if self.target_vector is not None else None,
                "deflect" : (self.deflect_vector - constants.ball.radius) if self.deflect_vector is not None else None
            }
        }

    def update(self, center):
        self.center = (center[0], center[1])
        self.offset_center = (center[0] + constants.regions.table[0], center[1] + constants.regions.table[1])

    def set_identity(self, data, color_info, is_target):
        self.suit = data.suit if "suit" in data.__dict__ else None
        self.name = data.name
        self.number = data.number
        self.color = data.color if "color" in data.__dict__ else None
        self.bgr = color_info.bgr if "color" in data.__dict__ else (0, 255, 0)
        self.is_target = is_target

    def get_closest_neighbors(self, balls, exclude={}):
        # return { n:balls[n] for n, d in self.neighbors.items() if (n not in exclude) and d <= (constants.ball.radius * 5) }
        return { n:balls[n] for n, d in self.neighbors.items() if (n not in exclude) and d <= (constants.ball.radius * 4) }

    def draw(self, image):
        if self.suit == "solid":
            cv2.circle(image, self.center, 8, (self.bgr[0], self.bgr[1], self.bgr[2]), 2)
        elif self.suit == "stripe":
            cv2.line(image, (self.center[0] - 8, self.center[1]), (self.center[0] + 8, self.center[1]), (255, 255, 255), 2)
            cv2.circle(image, self.center, 8, (self.bgr[0], self.bgr[1], self.bgr[2]), 2)
        else:
            cv2.circle(image, self.center, 8, (self.bgr[0], self.bgr[1], self.bgr[2]), 2)

class BallMaskInfo:
    def __init__(self):
        self.color_mask = None
        self.white_mask = None
        self.stick_mask = None
        self.glove_mask = None

        self.stick_total = None
        self.glove_total = None
        self.white_total = None
        self.color_total = None
        self.ratio = None

        self.logger = logging.getLogger("ateball.ball_mask_info")

    def update_masks(self, color_mask, white_mask, glove_mask, stick_mask):
        self.color_mask = color_mask
        self.white_mask = white_mask
        self.glove_mask = glove_mask
        self.stick_mask = stick_mask

    def update_mask_totals(self):
        self.stick_total = np.count_nonzero(self.stick_mask)
        self.glove_total = np.count_nonzero(self.glove_mask)
        self.white_total = np.count_nonzero(self.white_mask)
        self.color_total = np.count_nonzero(self.color_mask) / 3

        # compensate for stick blocking
        obsuring_area_ratio = (self.stick_total / constants.ball.area)
        if obsuring_area_ratio >= .05:
            solid_ratio = constants.ball.suit.solid.white_threshold / constants.ball.suit.solid.color_threshold
            solid_white_total =  constants.ball.suit.solid.white_threshold - (self.stick_total * solid_ratio)
            solid_color_total =  constants.ball.suit.solid.color_threshold - (self.stick_total * (1 - solid_ratio))
        else:
            solid_white_total =  constants.ball.suit.solid.white_threshold - self.stick_total
            solid_color_total =  constants.ball.suit.solid.color_threshold

        # generate avg ratio comparing total white pixels and colored pixels against their max totals of solids
        self.ratio = ((self.white_total / solid_white_total) + (self.color_total / solid_color_total)) / 2

class BallCluster(object):
    def __init__(self, balls):
        self.balls = balls

        self.min_bound = [0, 0]
        self.max_bound = [0, 0]

        self.logger = logging.getLogger("ateball.ball_cluster")

        for n, b in self.balls.items():
            if self.min_bound[0] == 0 or (b.center[0] - constants.ball.radius) < self.min_bound[0]:
                self.min_bound[0] = int(b.center[0] - constants.ball.radius)

            if self.min_bound[1] == 0 or (b.center[1] - constants.ball.radius) < self.min_bound[1]:
                self.min_bound[1] = int(b.center[1] - constants.ball.radius)

            if self.max_bound[0] == 0 or (b.center[0] + constants.ball.radius) > self.max_bound[0]:
                self.max_bound[0] = int(b.center[0] + constants.ball.radius)

            if self.max_bound[1] == 0 or (b.center[1] + constants.ball.radius) > self.max_bound[1]:
                self.max_bound[1] = int(b.center[1] + constants.ball.radius)

    def serialize(self):
        return {
            "min" : self.min_bound,
            "max" : self.max_bound
        }

    def draw(self, image):
        cv2.rectangle(image, self.min_bound, self.max_bound, (255, 255, 255), 1)
        cv2.line(image, self.min_bound, self.max_bound, (255, 255, 255), 1)