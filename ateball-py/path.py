import logging

import math
import cv2
import numpy as np

from utils import Line, clamp
from constants import constants

logger = logging.getLogger("ateball.path")

class BallPath:
    def __init__(self, cueball, target_ball, target_hole, hole_target_point, ball_inside_pocket):
        self.cueball = cueball
        self.cue_to_ball_trajectory = None
        self.obscuring_cue_to_ball_trajectory = []

        self.target_ball = target_ball
        self.target_ball_point = None
        self.ball_to_hole_trajectory = None
        self.obscuring_ball_to_hole_trajectory = []
        
        self.target_hole = target_hole
        self.target_hole_bound_l = None
        self.target_hole_bound_r = None
        self.hole_target_point = hole_target_point

        self.ball_inside_pocket = ball_inside_pocket
        self.direct_to_hole = False
        self.rebound = False

        self.difficulty = 0 #out of 10
        self.benefit = 0 #out of 10

        self.logger = logging.getLogger("ateball.ballpath")

    def __str__(self):
        return f"shot on {self.target_ball.name} to {self.target_hole.name} - benefit: {self.benefit} - difficulty: {self.difficulty}"

    def __repr__(self):
        return str(self)

    def get_uuid(self):
        return hash(f"{self.cueball}{self.target_ball}{self.target_ball_point}{self.target_hole}{self.hole_target_point}")

    def update_difficulty(self, add=1):
        self.difficulty += add

    def update_benefit(self, add=1):
        self.benefit += add

    def to_json(self):
        return {
            "cueball" : {
                "ball" : self.cueball,
                "path" : self.cue_to_ball_trajectory,
                "obscuring" : { b.name : b for b in self.obscuring_cue_to_ball_trajectory }
            },
            "target_ball" : {
                "ball" : self.target_ball,
                "point" : self.target_ball_point,
                "path" : self.cue_to_ball_trajectory,
                "obscuring" : { b.name : b for b in self.obscuring_ball_to_hole_trajectory }
            },
            "target_hole" : {
                "hole" : self.target_hole,
                "point" : self.hole_target_point
            },
            "info" : {
                "ball_inside_pocket" : self.ball_inside_pocket,
                "direct_to_hole" : self.direct_to_hole,
                "rebound" : self.rebound,
                "difficulty" : self.difficulty,
                "benefit" : self.benefit,
            }
        }

    def is_ball_trajectory_clear(self, targets):
        # get angle of entry and ratio compared to (half) max_entry_angle
        entry_angle = self.hole_target_point.get_angle(self.target_ball, self.target_hole.midpoint_table_intersection)
        entry_angle = math.fabs(360 - entry_angle if entry_angle > 180 else entry_angle)
        entry_angle_ratio = 1 - (entry_angle / self.target_hole.max_angle_of_entry)

        # ball must exist within elgible angle of entry, other requires further computation to determine bounce shot
        within_angle_of_entry = entry_angle <= self.target_hole.max_angle_of_entry
        if not within_angle_of_entry:
            # if entry_angle exceeds bounds, it requires further investigation (bounce shot)
            return False

        dy, dx, dydx = self.target_ball.get_slope_to(self.hole_target_point)
        bp1, bp2 = self.target_ball.find_points_on_either_side(constants.ball.radius * 2, dy, dx)

        if dydx < 0:
            self.target_ball_point = max(bp1, bp2, key=lambda p: p.center[0] if dx else p.center[1]) if dy < 0 else min(bp1, bp2, key=lambda p: p.center[0] if dx else p.center[1])
        elif dydx > 0:
            self.target_ball_point = min(bp1, bp2, key=lambda p: p.center[0] if dx else p.center[1]) if dy < 0 else max(bp1, bp2, key=lambda p: p.center[0] if dx else p.center[1])
        else:
            self.target_ball_point = min(bp1, bp2, key=lambda p: p.center[0] if dx else p.center[1]) if dy < 0 else max(bp1, bp2, key=lambda p: p.center[0] if dx else p.center[1])

        self.ball_to_hole_trajectory = Line(self.hole_target_point, self.target_ball_point)
        non_obscured_distances = []
        for n, b in targets.items():
            if n == self.target_ball.name:
                continue

            distance = self.ball_to_hole_trajectory.dist(b)
            if distance <= (constants.ball.radius * 2):
                self.target_hole.inside_pocket(b)
                self.obscuring_ball_to_hole_trajectory.append(b)
            else:
                non_obscured_distances.append(distance)

        if non_obscured_distances:
            distance_scores = list(map(lambda d: clamp(1 - (d - ((constants.ball.radius * 2) + 1) / (constants.ball.radius * 2)), 0, 1), non_obscured_distances))
            self.difficulty += sum(distance_scores) / len(non_obscured_distances)

        if self.obscuring_ball_to_hole_trajectory:
            # check for alternate, can obscuring be hit?
            return False

        self.direct_to_hole = entry_angle <= self.target_hole.max_angle_of_entry
        self.rebound = not self.direct_to_hole and not self.ball_inside_pocket

        # factor distance from ball to hole into difficulty
        self.difficulty += clamp(self.target_ball.distance(self.target_hole.center) / 766, 0, 1)
        
        return True

    def is_cueball_trajectory_clear(self, image, targets):
        left, above = self.cueball.center[0] < self.target_ball.center[0], self.cueball.center[1] < self.target_ball.center[1]
        right, below = not left, not above

        cue_to_ball_angle = self.target_ball_point.get_angle(self.cueball, self.target_hole.hole_gap_center)
        if constants.ball.entry_angle_bound.min > cue_to_ball_angle or cue_to_ball_angle > constants.ball.entry_angle_bound.max:
            # if entry_angle exceeds bounds, it requires further investigation (bounce shot)
            return False

        self.cue_to_ball_trajectory = Line(self.cueball, self.target_ball_point)
        non_obscured_distances = []
        for n, b in targets.items():
            if n == self.target_ball.name:
                continue

            distance = self.cue_to_ball_trajectory.dist(b)
            if distance <= (constants.ball.radius * 2):
                self.obscuring_cue_to_ball_trajectory.append(b)
            else:
                non_obscured_distances.append(distance) 

        if non_obscured_distances:
            distance_scores = list(map(lambda d: clamp(1 - (d - ((constants.ball.radius * 2) + 1) / (constants.ball.radius * 2)), 0, 1), non_obscured_distances))
            self.difficulty += sum(distance_scores) / len(non_obscured_distances)

        if self.obscuring_cue_to_ball_trajectory:
            # self.logger.debug(f"{self.target_ball.name} is obscured to {self.target_hole.name} - {cue_to_ball_angle}")
            return False

        # factor distance from cue to ball into difficulty
        self.difficulty += clamp(self.cueball.distance(self.target_ball) / 766, 0, 1)

        return True

    def draw(self, image):
        if self.target_ball_point is not None:
            self.target_ball_point.draw(image, radius=(2,2), bgr=(0, 0, 0))

        # draw ball to hole trajectory
        if self.ball_to_hole_trajectory is not None:
            self.ball_to_hole_trajectory.draw(image, bgr=(0, 255, 0))

        # draw cue to ball trajectory
        if self.cue_to_ball_trajectory is not None:
            self.cue_to_ball_trajectory.draw(image, bgr=(0, 255, 0))

