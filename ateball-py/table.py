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
    def __init__(self, gamemode_info):
        self.gamemode_info = gamemode_info

        self.table = constants.regions.table
        self.table_end = Point((self.table[2], self.table[3]/2))
        self.table_center = Point((self.table[2]/2, self.table[3]/2))

        self.table_background_mask = np.array(self.gamemode_info.table_mask.mask_lower), np.array(self.gamemode_info.table_mask.mask_upper)
        self.table_background_black_mask = np.array(constants.table.masks.table.black_mask.lower), np.array(constants.table.masks.table.black_mask.upper)
        
        self.table_pocketed_background_color_mask = np.array(constants.table.masks.pocketed.color_mask.lower), np.array(constants.table.masks.pocketed.color_mask.upper)
        self.table_pocketed_background_white_mask = np.array(constants.table.masks.pocketed.white_mask.lower), np.array(constants.table.masks.pocketed.white_mask.upper)

        self.stick_mask = np.array(constants.table.masks.table.stick_mask.lower), np.array(constants.table.masks.table.stick_mask.upper)
        self.glove_mask = np.array(constants.table.masks.table.glove_mask.lower), np.array(constants.table.masks.table.glove_mask.upper)
        self.white_mask = np.array(constants.table.masks.table.white_mask.lower), np.array(constants.table.masks.table.white_mask.upper)

        self._ball_colors = constants.table.balls.__dict__[self.gamemode_info.balls].colors.__dict__
        self.ball_color_look_up = {c : d.match_bgr for c, d in self._ball_colors.items()}

        # dict lookup of currently available
        self.all_available_table_targets = {c : copy.copy(d) for c, d in constants.table.balls.__dict__[self.gamemode_info.balls].identities.__dict__.items()}
        self.available_table_targets = {c : copy.copy(d) for c, d in constants.table.balls.__dict__[self.gamemode_info.balls].identities.__dict__.items()}
        self.hittable_table_targets = {c : d for c, d in self.available_table_targets.items() if c not in ["white", "target"]}

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

    def identify_targets(self, image):
        self.updated.clear()

        prepared_table, prepared_pocketed = self.__prepare_table(image)
        self.balls = self.__identify_targets(prepared_table, prepared_pocketed)
   
        self.updated.set()

    def __prepare_table(self, image):
        # table --
        table = CV2Helper.slice_image(image, constants.regions.table)
        height, width, channels = table.shape

        hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)

        # mask out table color for visibility
        table_invert_mask = cv2.inRange(hsv, *self.table_background_mask)
        table_invert_mask = cv2.bitwise_not(table_invert_mask)

        # mask out holes
        hole_mask = cv2.inRange(hsv, *self.table_background_black_mask)

        # combine masks
        table_mask = cv2.bitwise_and(table_invert_mask, hole_mask)
        table_masked = cv2.bitwise_and(table, table, mask=table_mask)

        # pocketed --
        targets_pocketed = CV2Helper.slice_image(image, constants.regions.targets_pocketed)
        targets_pocketed_hsv = cv2.cvtColor(targets_pocketed, cv2.COLOR_BGR2HSV)

        targets_pocketed_white_mask = cv2.inRange(targets_pocketed_hsv, *self.table_pocketed_background_white_mask)
        targets_pocketed_color_mask = cv2.inRange(targets_pocketed_hsv, *self.table_pocketed_background_color_mask)
        targets_pocketed_mask = cv2.bitwise_or(targets_pocketed_color_mask, targets_pocketed_white_mask)

        targets_pocketed_masked = cv2.bitwise_and(targets_pocketed, targets_pocketed, mask=targets_pocketed_mask)

        self.images["table"] = table
        self.images["combined_mask"] = table_masked
        self.images["mask"] = table_mask
        self.images["none"] = np.zeros((height, width, channels), np.uint8)

        return table_masked, targets_pocketed_masked

    def __identify_targets(self, table, pocketed):
        ball_table_positions = self.__find_targets(table)
        ball_pocketed_positions = self.__find_targets(pocketed, pocketed=True)

        ball_positions = ball_pocketed_positions + ball_table_positions

        to_identify = {}
        identified = []

        # identify balls - based on characteristics
        for img, b in ball_positions:
            roi, region = CV2Helper.roi(img, b.center, 10)

            self.__adjust_center(b, roi, region)

            self.__mask_out_unimportant(b, roi, region)
            self.__identify_ball(b, to_identify, identified)

        # identify any leftover balls - likely one of the pairs is obstructed by view
        for c, balls in to_identify.items():
            b = balls[0]
            color_info = self._ball_colors[c]

            if math.fabs(1 - b.mask_info.ratio) < math.fabs(.5 - b.mask_info.ratio):
                b.set_identity(self.available_table_targets[c].stripe, color_info)
            else:
                b.set_identity(self.available_table_targets[c].solid, color_info)

            identified.append(b)

            if b.pocketed:
                del self.available_table_targets[c]

        return identified

    def __find_targets(self, image, **kwargs):
        image = image.copy()
        table_masked_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # approximate location with hough circles - TODO improve approximation
        points = cv2.HoughCircles(table_masked_gray, cv2.HOUGH_GRADIENT, 1, 17, param1=20, param2=9, minRadius=9, maxRadius=11)
        points = np.uint16(np.around(points))[0, :] if points is not None and points.any() else []

        # return list tuples (source_img, Ball) from points
        return [(image, Ball((p[0], p[1]), **kwargs)) for p in points]

    # compounding error when hough circle is off
    def __adjust_center(self, b, image, region):
        try:
            height, width, channels = image.shape

            # create circular mask around approximated center
            circle_mask = np.zeros((height,width), np.uint8)
            cv2.circle(circle_mask, (b.center[0]-region[0], b.center[1]-region[1]), 10, [255, 255, 255], -1)
            correction_mask = cv2.bitwise_and(image, image, mask=circle_mask)

            # crop image from edge to first nonzero row/col
            correction_cropped = np.nonzero(correction_mask) 
            top = correction_cropped[0].min()
            bottom = correction_cropped[0].max()
            left = correction_cropped[1].min()
            right = correction_cropped[1].max()
            
            # calculate corrected center based on minimums/maximums
            w_correction = ((width-1) - (right - left)) // 2
            y_correction = ((height-1) - (bottom - top)) // 2
            y_correction = y_correction if top > 0 else -y_correction
            w_correction = w_correction if left > 0 else -w_correction
            corrected = b.center[0] + w_correction, b.center[1] + y_correction

            b.update(corrected)
        except Exception as e:
            pass

    def __mask_out_unimportant(self, b, image, region):
        # image_blur = cv2.GaussianBlur(image, (3, 3), 0)
        height, width, channels = image.shape

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

        # create roi smaller than ball to avoid color bias of balls in close proximity
        blank_image = np.zeros((height,width), np.uint8)
        cv2.circle(blank_image, (b.center[0]-region[0], b.center[1]-region[1]), (constants.ball.radius-1), [255, 255, 255], -1)

        # mask out white to avoid color bias and get more accurate mean
        white_mask = cv2.inRange(hsv, *self.white_mask)

        # create inversion mask to exlude white pixels - dilate to ensure good white masking
        mask_white_invert = cv2.dilate(white_mask, np.ones((3, 3), np.uint8))
        mask_white_invert = cv2.bitwise_not(mask_white_invert)
        mask_white_invert = cv2.bitwise_and(mask_white_invert, blank_image)

        # TODO improve stick masking - should be color/design invariant
        # mask out default dan pool stick and create inversion mask to exclude stick pixels
        stick_mask = cv2.inRange(hsv, *self.stick_mask)
        stick_mask_invert = cv2.dilate(stick_mask, np.ones((2, 2), np.uint8))
        stick_mask_invert = cv2.bitwise_not(stick_mask)

        # limit stick mask inside area of ball (exlude any mask inside roi, but outside ball)
        stick_mask = cv2.bitwise_and(blank_image, stick_mask)

        # in the case the player can move the cue ball, mask out glove
        glove_mask = cv2.inRange(hsv, *self.glove_mask)
        glove_mask_invert = cv2.dilate(glove_mask, np.ones((2, 2), np.uint8))
        glove_mask_invert = cv2.bitwise_not(glove_mask_invert)

        # combine masks and mask out undesired areas
        mask = cv2.bitwise_and(mask_white_invert, glove_mask_invert)
        mask = cv2.bitwise_and(mask, stick_mask_invert)

        color_mask = cv2.bitwise_and(image, image, mask=mask)

        b.mask_info.update_masks(color_mask, white_mask, glove_mask, stick_mask)

    def __identify_ball(self, b, to_identify, identified):
        b.mask_info.update_mask_totals()

        available_targets = self.all_available_table_targets if b.pocketed else self.available_table_targets

        # eliminate false positives near pool stick
        if b.mask_info.stick_total > (b.mask_info.glove_total + b.mask_info.white_total + b.mask_info.color_total):
            return

        # identify cue ball if glove is present
        if (b.mask_info.glove_total / constants.ball.area) > .4:
            b.set_identity(available_targets["white"], self._ball_colors["white"])
            identified.append(b)
            return

        # identify target - color_total should be* less than 5% of total area possible 
        if (b.mask_info.color_total / constants.ball.area) < .05:
            # filter out false positives (appear around cue ball glove) - false positives will not have sufficient white pixels
            if (b.mask_info.white_total / constants.ball.area) > .2:
                b.set_identity(available_targets["target"], {})
                identified.append(b)
            return

        # get mean of colored pixels - excluding black pixels
        color = b.mask_info.color_mask[~np.all(b.mask_info.color_mask == 0, axis=-1)].mean(axis=0)
        color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]

        # get list of colors - ordered by smallest deltae
        color_proximities = CV2Helper.color_deltas(color, self.ball_color_look_up)
        for dist, c in color_proximities:
            # color should be in list of currently available targets
            if c in available_targets:
                color_info = self._ball_colors[c]

                # verify color identifiction - hue should be within its mask range
                if color_hsv[0] >= color_info.mask_lower[0] and color_hsv[0] <= color_info.mask_upper[0]:
                    
                    # set identity after identifying all balls of same color (except cue/eightball/target)
                    if c in ["white", "black", "target"]:
                        b.set_identity(available_targets[c], color_info)
                        identified.append(b)
                        break
                    else:
                        total_identities = len(available_targets[c].__dict__)

                        # skip identification if there should be another ball of the same color
                        if c not in to_identify and total_identities > 1:
                            to_identify[c] = [b]
                        else:
                            # identify with only available identity
                            if total_identities == 1:
                                identity = available_targets[c].__dict__[list(available_targets[c].__dict__)[0]]
                                b.set_identity(identity, color_info)
                                identified.append(b)
                                if c in to_identify:
                                    del to_identify[c]
                                if b.pocketed:
                                    del available_targets[c]
                            elif total_identities == 2:
                                # approx suit by comparing ratios of white to colored pixels
                                b1 = to_identify[c][0]

                                if (b.mask_info.ratio) > (b1.mask_info.ratio):
                                    b.set_identity(available_targets[c].stripe, color_info)
                                    b1.set_identity(available_targets[c].solid, color_info)
                                else:
                                    b.set_identity(available_targets[c].solid, color_info)
                                    b1.set_identity(available_targets[c].stripe, color_info)

                                identified.append(b)
                                identified.append(b1)
                                del to_identify[c]
                                if b.pocketed:
                                    del available_targets[c]
                    break

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
            if not b.pocketed:
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
