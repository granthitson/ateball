import logging
import copy

import threading

import cv2
import math
import numpy as np

from ball import Ball
from utils import Point, CV2Helper
from constants import constants

class Table(object):
    def __init__(self, gamemode):
        self.table = constants.regions.table
        self.table_end = Point((self.table[2], self.table[3]/2))
        self.table_center = Point((self.table[2]/2, self.table[3]/2))

        self.table_background_mask = np.array(constants.gamemodes.__dict__[gamemode].table_mask_lower), np.array(constants.gamemodes.__dict__[gamemode].table_mask_upper)
        self.table_background_black_mask = np.array(constants.table.black_mask.lower), np.array(constants.table.black_mask.upper)

        self.balls = []
        self.updated = threading.Event()

        self.walls = [Wall(name, data) for i, (name, data) in enumerate(constants.table.walls.__dict__.items())]
        self.holes = [Hole(name, data, self) for i, (name, data) in enumerate(constants.table.holes.__dict__.items())]

        self.images = {
            "table" : None,
            "combined_mask" : None,
            "mask" : None,
            "none" : None
        }

        self.logger = logging.getLogger("ateball.table")

    def copy(self):
        return copy.copy(self)

    def prepare_table(self, img):
        table = CV2Helper.slice_image(img, constants.regions.table)

        height, width, channels = table.shape
        hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)

        # mask out table color for visibility
        table_invert_mask = cv2.inRange(hsv, *self.table_background_mask)
        table_invert_mask = cv2.bitwise_not(table_invert_mask)

        # mask out holes
        hole_mask = cv2.inRange(hsv, *self.table_background_black_mask)

        # combine masks
        table_mask = cv2.bitwise_and(table_invert_mask, hole_mask)
        table_masked_out = cv2.bitwise_and(table, table, mask=table_mask)

        self.images["table"] = table
        self.images["combined_mask"] = table_masked_out
        self.images["mask"] = table_mask
        self.images["none"] = np.zeros((height, width, channels), np.uint8)

    def get_ball_locations(self, available_identities):
        self.updated.clear()

        image = self.images["combined_mask"].copy()
        table_masked_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # approximate location with hough circles - TODO improve approximation
        points = cv2.HoughCircles(table_masked_gray, cv2.HOUGH_GRADIENT, 1, 17, param1=20, param2=9, minRadius=9, maxRadius=11)
        points = np.uint16(np.around(points))[0, :]

        # create list of Balls from points
        self.balls = [Ball((p[0], p[1])) for p in points]
   
        self.updated.set()

    def draw(self, config):
        # draw on correct image type
        i_type = config["image_type"] if "image_type" in config else "table"
        image = self.images[i_type].copy()

        if "show_walls" in config and config["show_walls"]:
            for w in self.walls:
                w.draw(image)

        if "show_holes" in config and config["show_holes"]:
            for h in self.holes:
                h.draw(image)
                h.draw_points(image)

        for b in self.balls:
            b.draw(image)

        return self.images["table"], image

    def capture(self):
        return (self.images["combined_mask"], self.balls)

class Wall:
    def __init__(self, name, data):
        self.name = name

        self.start = tuple(data.start)
        self.end = tuple(data.end)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def draw(self, image):
        cv2.line(image, self.start, self.end, (0, 0, 255), 1)

class Hole:
    def __init__(self, name, data, table):
        self.name = name
        self.image = data.image
        
        self.center = Point(tuple(data.center))
        self.rotated_center = Point((0, 0))

        self.corner = Point(tuple(data.corner))
        self.outer_left = Point(tuple(data.outer_left))
        self.inner_left = Point(tuple(data.inner_left))
        self.inner_right = Point(tuple(data.inner_right))
        self.outer_right = Point(tuple(data.outer_right))

        self.hole_gap_rise, self.hole_gap_run, self.hole_gap_slope = self.outer_left.get_rise_run_slope(self.outer_right)

        self.angle_from_center = table.table_center.get_angle(table.table_end, self.center)
        self.min_x = min(self.outer_left, self.outer_right, key = lambda p: p.center[0])
        self.max_x = max(self.outer_left, self.outer_right, key = lambda p: p.center[0])

        self.radius = int(self.min_x.distance(self.center))

        self.opening_angle = self.min_x.get_angle(self.corner, self.max_x) if self.min_x.center[0] > table.table_center.center[0] else self.max_x.get_angle(self.corner, self.min_x)
        self.opening_angle_start = 180 if self.angle_from_center >= 180 else 0
        self.opening_angle_end = 360 if self.angle_from_center >= 180 else 180

        self.points = self.create_points()

        self.logger = logging.getLogger("ateball.hole")

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def create_points(self):
        min_x = min(self.inner_left.center[0], self.inner_right.center[0])
        max_x = max(self.inner_left.center[0], self.inner_right.center[0])

        min_y = min(self.inner_left.center[1], self.inner_right.center[1])
        max_y = max(self.inner_left.center[1], self.inner_right.center[1])

        if self.hole_gap_slope > 0:
            y = min_y
            increment = 1
        elif self.hole_gap_slope < 0:
            y = max_y
            increment = -1
        else:
            y = min_y
            increment = 0

        points = []
        for x in range(min_x, max_x+1, 1):
            points.append(Point((x, y)))

            y += increment

        points.sort(key=lambda p: (p.center[0], p.center[1]))

        return points

    def draw(self, image):
        cv2.ellipse(image, self.center.center, (self.radius, self.radius), self.opening_angle, self.opening_angle_start, self.opening_angle_end, (52, 222, 235))

    def draw_points(self, image):
        self.center.draw(image)
        self.outer_left.draw(image)
        self.inner_left.draw(image)
        self.inner_right.draw(image)
        self.outer_right.draw(image)
        self.corner.draw(image)
