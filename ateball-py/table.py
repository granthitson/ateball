import logging

import cv2
import math

from utils import Point
from constants import constants

class Table:
    def __init__(self):
        self.table = constants.regions.table
        self.table_end = Point((self.table[2], self.table[3]/2))
        self.table_center = Point((self.table[2]/2, self.table[3]/2))

        self.balls = []

        self.walls = [Wall(name, data) for i, (name, data) in enumerate(constants.table.walls.__dict__.items())]
        self.holes = [Hole(name, data, self) for i, (name, data) in enumerate(constants.table.holes.__dict__.items())]

        self.logger = logging.getLogger("ateball.table")

    def get_ball_positions(self):
        return [b.center for b in self.balls]

    def draw(self, config, image):
        if "show_walls" in config and config["show_walls"]:
            for w in self.walls:
                w.draw(image)

        if "show_holes" in config and config["show_holes"]:
            for h in self.holes:
                h.draw(image)
                h.draw_points(image)

        for b in self.balls:
            b.draw(image)

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
