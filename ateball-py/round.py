import logging
import traceback

import math
import os
import random
import time

import threading

import cv2
import imutils
import numpy as np

from pathlib import Path

from ball import Ball
from ball import Cue
from ball import Eight
import path
import utils
from constants import constants

logger = logging.getLogger("ateball.round")

class Round:

    def __init__(self, data):   
        # game constants

        self.regions = constants.regions
        self.hole_locations = constants.table.holes

        self.solid_balls = constants.balls.solid
        self.stripe_balls = constants.balls.stripe
        self.colors = constants.balls.colors

        self.available_targets = [*self.solid_balls, constants.balls.eight, *self.stripe_balls]
        # game constants
        
        # round constants
        self.img_path = data["save_image_path"]
        self.images = { 
            "game" : data["round_image"],
            "table" : None,
            "pocketed" : None,
            "targets_bot" : None,
            "targets_opponent" : None,
        }
        #round constants

        self.table_hsv = None

        self.get_targets_event = threading.Event()


        self.all_balls = data["all_balls"]

        self.unpocketed_balls = data["unpocketed_balls"]
        self.pocketed_balls = data["pocketed_balls"]

        self.suit = data["suit"]
        self.targets = data["targets"]
        self.nontargets = data["nontargets"]

        self.chosen_ball = None

        # self.cueball = Cue()

        self.viable_paths = []
        self.unviable_paths = []

        self.clear_table = (0,0)
        self.can_pick_up = False

        self.round_cancel = threading.Event()
        self.round_complete = threading.Event()
        self.round_exception = threading.Event()
        self.round_over_event = utils.OrEvent(self.round_complete, self.round_cancel, self.round_exception)

        self.logger = logging.getLogger("ateball.round")

    def start(self):
        try:
            start_time = time.time()

            self.logger.info("ROUND START")

            self.images["table"] = self.images["game"].copy()[self.regions.table[1]:self.regions.table[1]+self.regions.table[3], self.regions.table[0]:self.regions.table[0]+self.regions.table[2]]
            self.images["pocketed"] = self.images["game"].copy()[self.regions.pocketed[1]:self.regions.pocketed[1]+self.regions.pocketed[3], self.regions.pocketed[0]:self.regions.pocketed[0]+self.regions.pocketed[2]]
            self.images["targets_bot"] = self.images["game"].copy()[self.regions.targets_bot[1]:self.regions.targets_bot[1]+self.regions.targets_bot[3], self.regions.targets_bot[0]:self.regions.targets_bot[0]+self.regions.targets_bot[2]]
            self.images["targets_opponent"] = self.images["game"].copy()[self.regions.targets_opponent[1]:self.regions.targets_opponent[1]+self.regions.targets_opponent[3], self.regions.targets_opponent[0]:self.regions.targets_opponent[0]+self.regions.targets_opponent[2]]

            cv2.imwrite(str(Path(self.img_path, "table.png")), self.images["table"])
            cv2.imwrite(str(Path(self.img_path, "pocketed.png")), self.images["pocketed"])
            cv2.imwrite(str(Path(self.img_path, "targets_bot.png")), self.images["targets_bot"])
            cv2.imwrite(str(Path(self.img_path, "targets_opponent.png")), self.images["targets_opponent"])

            threading.Thread(target=self.get_targets_on_table).start()

            self.round_complete.set()
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.round_exception.set()
        finally:
            self.logger.debug("ROUND COMPLETE")

### Gather info on balls to hit ###

    def create_targets(self):
        # count number of targets at start of round
        self.logger.debug("Counting targets...\n")

        image = self.images["targets_bot"].copy()
        mask = cv2.imread(utils.ImageHelper.imagePath(constants.img_target_mask), 0)
        
        # mask out background of player targets
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        upper_gray = np.array([255, 255, 255])
        lower_gray = np.array([0, 0, 32])

        image_hsv = cv2.inRange(image_hsv, lower_gray, upper_gray)
        image_hsv1 = cv2.morphologyEx(image_hsv, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        mask1 = cv2.bitwise_and(mask, image_hsv1)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # combine mask with targets image
        masked = utils.CV2Helper.mask_image(image, mask1)
        
        # count targets by # of contours
        contours = utils.CV2Helper.getContours(mask1)
        num_of_targets = len(contours)

        if len(self.targets) != num_of_targets:
            for i in range(num_of_targets):
                self.targets.append(Ball(self.suit))
        else:
            if not self.targets:
                for i in range(14):
                    self.targets.append(Ball(self.suit))

        self.logger.debug(f"total targets: {num_of_targets}")

    def get_targets_on_table(self):
        self.logger.debug("Finding targets...")

        table = self.images["table"].copy()
        self.table_hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)

        table_masked = self.mask_out_table(table, self.table_hsv)
        table_masked_gray = cv2.cvtColor(table_masked, cv2.COLOR_BGR2GRAY)

        # identify pool balls using hough circles
        circles = cv2.HoughCircles(table_masked_gray, cv2.HOUGH_GRADIENT, 1, 17, param1=20, param2=9, minRadius=9, maxRadius=11)
        circles = np.uint16(np.around(circles))
        
        hough_points = []
        for i in circles[0, :]:
            cv2.circle(table, (i[0], i[1]), i[2], (0, 255, 0), 1)
            hough_points.append((i[0], i[1]))

        self.logger.debug(f"targets on table: {len(hough_points)}")

        cv2.imwrite(str(Path(self.img_path, "hough.png")), table)

    def mask_out_table(self, img, hsv):
        # mask for table color - table color masked out for visibility
        blue_lower = np.array([90, 80, 0])
        blue_higher = np.array([106, 255, 255])
        table_mask = cv2.inRange(hsv, blue_lower, blue_higher)
        table_mask = cv2.bitwise_not(table_invert_mask)

        black_lower = np.array([0, 0, 35])
        black_higher = np.array([180, 255, 255])
        hole_mask = cv2.inRange(hsv, black_lower, black_higher)

        table_hole_mask = cv2.bitwise_and(table_mask, hole_mask)
        table_masked_out = cv2.bitwise_and(img, img, mask=table_hole_mask)

        return table_masked_out

    def get_targets(self):
        start = time.time()

        image = self.images["targets_bot"].copy()
        mask = cv2.imread(utils.ImageHelper.imagePath(constants.img_target_mask), 0)

        image = cv2.bitwise_and(image, image, mask=mask)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for color, color_data in self.colors.items():
            ball_mask = utils.CV2Helper.create_mask(image_hsv, color_data["mask_lower"], color_data["mask_upper"], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours = utils.CV2Helper.getContours(ball_mask)

        self.get_targets_event.set()
        print(time.time() - start)


    def determine_suit(self, region):
        for name, data in solid_balls.items():
            pos = utils.ImageHelper.imageSearch(name, region, confidence=.90)
            if pos is not None:
                self.suit = "solid"
                break
        
        for b in constants.stripes.items():
            pos = utils.ImageHelper.imageSearch(k, region, confidence=.90)
            if pos is not None:
                self.suit = "stripe"
                break

    def determine_targets(self, leftReg, rightReg):
        self.add_target_balls(self.suit, constants.solids, leftReg)
        if self.suit == "solid":
            self.add_target_balls("stripe", constants.stripesDark, rightReg, False)
        elif self.suit == "stripe":
            self.add_target_balls("solid", constants.solidsDark, rightReg, False)

    def add_target_balls(self, suit, diction, region=None, target=True):
        for k, v in diction.items():
            if region == None:
                name = k.replace(".png", "")
                b = ball.Ball(suit, name, v, True)
                self.all_balls[name] = b
            else:
                pos = utils.ImageHelper.imageSearch(k, region, confidence=.90)
                if pos is not None:
                    if target is True:
                        name = k.replace(".png", "")
                        b = ball.Ball(suit, name, v, True)
                        self.all_balls[name] = b
                    else:
                        name = k.replace("Dark.png", "")
                        b = ball.Ball(suit, name, v, False)
                        self.all_balls[name] = b
                else:
                    name = k.replace(".png", "")
                    self.all_balls[name] = ball.Ball(suit, name, v, False, True)

### Gather info on balls to hit ###


### Outline Balls - Determine center/suit ###

    def outline_balls(self):
        """
        Outlines pool balls using Hough Circles and Contours. Hough Circles, while sometimes inaccurate, find the center
        of each ball on the screen more accurately than contours. This includes extraneous/uneeded circles found.

        Then, for each ball in ballList, contours are found using each balls' specific mask. Depending on the color/type
        of ball (cueball/eightball/playballs), the list of contours found is limited to the two largest. Certain circumstances
        allow for more than two, such as a split contour.

        Typically, for each of balls, assuming there is still two contours of a specific color (which means that
        there are both a solid and striped colored ball on the table), each contour is investigated. As long as the contour
        has not already been used, and is within a certain area/radius, information is collected in regard to the pool ball;
        the information includes area of the ball's colored mask, area of white contours in a region of interest around
        the center, and a number of total contours within said region of interest (a count of more than two white contours
        usually means a striped ball is present). The contour in question is then matched to the closest point identified
        by the Hough Circles.

        Depending on how many contours are being investigated, the information collected on them are assigned to variables,
        ball_1 if there is one contour, ball_2 if there is two. Then, in order to differentiate between two contours of
        the same color, a confidence level is generated based on the information collected from each ball. The confidence level
        is only generated on ball_1; if the confidence level is < 0, the ball is striped, > 0 is solid.

        The correct center point is then assigned to ball.

        :return: None - only updates paramters in each ball
        """
        self.logger.debug("Outlining pool balls...\n")

        table = self.images["table"].copy()
        table_hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)

        # image of area where sunk balls go
        pocketed = self.images["pocketed"].copy()
        pocketed_hsv = cv2.cvtColor(pocketed, cv2.COLOR_BGR2HSV)

        # white mask of all balls (except cue)
        upper_white = np.array([0, 255, 255])
        lower_white = np.array([0, 0, 125])
        white_mask_table = cv2.inRange(table_hsv, lower_white, upper_white)
        white_mask_pocketed = cv2.inRange(pocketed_hsv, lower_white, upper_white)

        table_masked = self.mask_out_table(table, table_hsv)
        table_masked_hsv = cv2.cvtColor(table_masked, cv2.COLOR_BGR2HSV)
        table_masked_gray = cv2.cvtColor(table_masked, cv2.COLOR_BGR2GRAY)

        # Hough Circles for identifying ball locations more accurate than contours
        circles = cv2.HoughCircles(table_masked_gray, cv2.HOUGH_GRADIENT, 1, 17, param1=20, param2=9, minRadius=9, maxRadius=11)
        circles = np.uint16(np.around(circles))
        
        hough_points = []
        for i in circles[0, :]:
            cv2.circle(table, (i[0], i[1]), i[2], (0, 255, 0), 1)
            hough_points.append((i[0], i[1]))

        # cv2.imshow("table", table)
        # cv2.imshow("table_masked", table_masked)
        # cv2.imshow("table_masked_gray", table_masked_gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return

        cv2.imwrite(str(Path(self.img_path, "hough.png")), table)

        ignore = set()

        unidentified = self.all_balls.copy()
        for name, b in unidentified.items():
            b.create_mask(table_masked) # prepare each ball's mask based on its color

            contours = self.contourSetup(b.mask, True)  # list of contours found

            if name == "eightball":
                contours = contours[len(contours) - 1:len(contours)] # limits contours to single largest contour
            else:
                contoursTemp = contours[len(contours) - 2:len(contours)] # limits contorus to two largest contours

                for k, c in enumerate(contoursTemp):
                    carea = cv2.contourArea(c)
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    if radius > 25: # in some cases, a message pops up on screen - replaces it
                        contoursTemp[k] = contours[len(contours) - 3]
                    else:
                        if carea > 0 and carea < 80: # if a contour that should be together is split, it will result in two smaller contours
                            b.mask = cv2.morphologyEx(b.mask, cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8))
                            contoursTemp = self.contourSetup(b.mask, True)

                contours = contoursTemp

            ball_1, ball_2 = None, None

            large = False
            count = 0
            # find center based on color
            for c in contours:
                carea = cv2.contourArea(c)
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (x,y)

                if center not in ignore and carea > 25:
                    count += 1
                    if radius < 13:
                        roiMask = self.roi(white_mask_table, utils.PointHelper.roundTuple(center), 10)
                        roiContours = self.contourSetup(roiMask, False)
                        whiteArea, whiteCount = self.whiteMaskedArea(roiContours, True, 0)

                        closestIndex, closest, dist = self.matchHoughCircleToMask(hough_points, center, 10)
                        if dist == math.inf:
                            continue
                        else:
                            if ball_1 is None:
                                ball_1 = (hough_points[closestIndex][0], hough_points[closestIndex][1], carea, whiteArea, whiteCount, center)
                            else:
                                ball_2 = (hough_points[closestIndex][0], hough_points[closestIndex][1], carea, whiteArea, whiteCount, center)
                    else:
                        large = True

                        closestIndex, closestIndex1 = self.matchHoughCircleToMask(hough_points, center, 15, True)
                        
                        closestIndex, closest, dist = closestIndex
                        closestIndex1, closest1, dist1 = closestIndex1

                        if dist == math.inf and dist1 == math.inf:
                            continue
                        else:
                            points = [closest, closest1]
                            for p in points:
                                roiMask = self.roi(b.mask, utils.PointHelper.roundTuple(p), 10)
                                contours = self.contourSetup(roiMask, False)

                                for c in contours:
                                    carea1 = cv2.contourArea(c)
                                    (x1, y1), radius1 = cv2.minEnclosingCircle(c)
                                    center1 = (x1,y1)

                                    if carea > 30:
                                        roiMask = self.roi(white_mask_table, utils.PointHelper.roundTuple(p), 10)
                                        contours = self.contourSetup(roiMask, False)
                                        whiteArea, whiteCount = self.whiteMaskedArea(contours, True, 0)

                                        if ball_1 is None:
                                            ball_1 = (p[0], p[1], carea1, whiteArea, whiteCount, center1)
                                        else:
                                            ball_2 = (p[0], p[1], carea1, whiteArea, whiteCount, center1)

                                    ignore.add(center)

            if b.name == "eightball":
                if ball_1 is None:
                    ball_1 = self.backUpCenter(contours, white_mask_table)
                    b.center = (ball_1[0], ball_1[1])
                else:
                    b.center = (ball_1[0], ball_1[1])

                b.offsetCenter = (b.center[0] + self.regions.table_offset[0], b.center[1] + self.regions.table_offset[1])
                self.drawStripe(b.center, b.bgr)
            else:
                if ball_2 is None and count == 1: #looking for pocketed balls
                    b1 = unidentified[f"{b.number+8}ball"]

                    b1.create_mask(pocketed_hsv)
                    contours = self.contourSetup(b1.mask, True)
                    contours = contours[len(contours) - 2:len(contours)]

                    for c in contours:
                        carea = cv2.contourArea(c)
                        (x, y), radius = cv2.minEnclosingCircle(c)
                        center = utils.PointHelper.roundTuple((x,y))

                        if carea > 15:
                            roiMask1 = self.roi(white_mask_pocketed, center, 10)
                            contours1 = self.contourSetup(roiMask1, False)
                            whiteArea, whiteCount = self.whiteMaskedArea(contours1, True, 0)

                            if ball_1 is None:
                                ball_1 = (center[0] + 725, center[1] + 40, carea, whiteArea, whiteCount, (center[0] + 725, center[1] + 40))
                            else:
                                ball_2 = (center[0] + 725, center[1] + 40, carea, whiteArea, whiteCount, (center[0] + 725, center[1] + 40))

                if ball_2 is None:
                    if ball_1 is not None:
                        if b.suit == "solid":
                            b.set_center(ball_1[5], self.regions.table)

                            self.drawSolid(b.center, b.bgr)
                        elif b.suit == "stripe":
                            b.set_center(ball_1[5], self.regions.table)
                            
                            self.drawStripe(b.center, b.bgr)

                        ignore.add(ball_1[5])
                    else:
                        continue
                else:
                    confidence = self.suitConfidence(ball_1, ball_2)

                    if confidence > 0:
                        if b.suit == "solid":
                            b.set_center(ball_1[5], self.regions.table)
                            
                            b1 = unidentified[f"{b.number+8}ball"]
                            b1.set_center(ball_2[5], self.regions.table)

                            self.drawSolid(b.center, b.bgr)
                            self.drawStripe(b1.center, b.bgr)
                        elif b.suit == "stripe":
                            b.set_center(ball_2[5], self.regions.table)

                            b1 = unidentified[f"{b.number+8}ball"]
                            b1.set_center(ball_1[5], self.regions.table)

                            self.drawStripe(b.center, b.bgr)
                            self.drawSolid(b1.center, b.bgr)
                    elif confidence < 0:
                        if b.suit == "stripe":
                            b.set_center(ball_1[5], self.regions.table)
                            
                            b1 = unidentified[f"{b.number+8}ball"]
                            b1.set_center(ball_2[5], self.regions.table)

                            self.drawStripe(b.center, b.bgr)
                            self.drawSolid(b1.center, b.bgr)
                        elif b.suit == "solid":
                            b.set_center(ball_2[5], self.regions.table)
                            
                            b1 = unidentified[f"{b.number+8}ball"]
                            b1.set_center(ball_1[5], self.regions.table)

                            self.drawSolid(b.center, b.bgr)
                            self.drawStripe(b1.center, b.bgr)
                    else:
                        # random chance at this point
                        #print("confidence is 0")
                        if b.suit == "solid":
                            b.set_center(ball_1[5], self.regions.table)
                            
                            b1 = unidentified[f"{b.number+8}ball"]
                            b1.set_center(ball_2[5], self.regions.table)

                            self.drawSolid(b.center, b.bgr)
                            self.drawStripe(b1.center, b.bgr)
                        elif b.suit == "stripe":
                            b.set_center(ball_1[5], self.regions.table)
                            
                            b1 = unidentified[f"{b.number+8}ball"]
                            b1.set_center(ball_2[5], self.regions.table)

                            self.drawStripe(b.center, b.bgr)
                            self.drawSolid(b1.center, b.bgr)

                    ignore.add(ball_1[5])
                    ignore.add(ball_2[5])

        self.savePic()

    def sortBalls(self):
        self.targets = {n:b for n,b in self.all_balls.items() if b.target is True and b.pocketed is False}
        self.nontargets = {n:b for n,b in self.all_balls.items() if b.target is False or b.pocketed is True}

        self.unpocketed_balls = {n:b for n,b in self.all_balls.items() if b.pocketed is False}
        self.pocketed_balls = {n:b for n,b in self.all_balls.items() if b.pocketed is True}

    def contourSetup(self, mask, sort=True):
        """
        Allows for creation of list of contours without repeating constantly. Can return unsorted list or sorted list.

        :param mask: numpy array of mask
        :param sort: boolean - True to return sorted list - default is sorted list
        :return: sorted/unsorted list
        """
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        if sort is True:
            contours = sorted(contours, key=cv2.contourArea)

        return contours

    def roi(self, img, center, radius):
        """
        Creates a mask around region of interest.

        :param img: image to combine mask with
        :param center: tuple of point to center region around
        :param radius: how far around the point the region should go
        :return: mask with only region of interest
        """
        mask = np.zeros_like(img)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        masked = cv2.bitwise_and(img, mask)
        return masked

    def whiteMaskedArea(self, contours, count, mini):
        """
        Adds up total area of white mask areas and adds up the count of how many contours there are above a minimum.

        :param contours:
        :param count:
        :param min:
        :return:
        """
        whiteArea = 0
        whiteCount = 0
        for c in contours:
            carea = cv2.contourArea(c)
            if carea > mini:
                whiteArea += carea
                whiteCount += 1

        if count is True:
            return whiteArea, whiteCount
        else:
            return whiteArea

    def matchHoughCircleToMask(self, houghPoints, center, maxi, large=False):
        if not large:
            closest = (0, (0,0), math.inf)
            for k, p in enumerate(houghPoints):
                dist = math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)

                if dist < maxi and dist < closest[2]:
                    closest = (k, p, dist)

            return closest
        else:
            closest = (0, (0,0), math.inf)
            closest1 = (0, (0,0), math.inf)
            for k, p in enumerate(houghPoints):
                dist = math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)

                if dist < maxi:
                    if dist < closest[2]:
                        closest1 = closest
                        closest = (k, p, dist)
                    elif dist < closest1[2]:
                        closest1 = (k, p, dist)

            return closest, closest1
  
    def backUpCenter(self, contours, maskwhite):
        ball_1 = (0, 0, 0, 0, 0, 0)
        for c in contours:
            carea = cv2.contourArea(c)
            (x, y), radius = cv2.minEnclosingCircle(c)
            if carea > 30 and carea > ball_1[2]:
                center = utils.PointHelper.roundTuple((x,y))

                roiMask1 = self.roi(maskwhite, center, 10)
                contours1 = cv2.findContours(roiMask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours1 = contours1[0] if imutils.is_cv2() else contours1[1]

                whiteArea = 0
                whiteCount = 0
                for c1 in contours1:
                    c1area = cv2.contourArea(c1)
                    whiteCount += 1
                    if c1area > 0:
                        whiteArea += c1area

                ball_1 = (center[0], center[1], carea, whiteArea, whiteCount, center)

        return ball_1

    def suitConfidence(self, ball_1, ball_2):
        """
        Finds the confidence level of ball_1. Compares area of each color contour, area of each white color contour, and
        the number of white contours found within a region of interest.

        Ranges from -1 to 1. -1 being striped, 0 being uncertain, and 1 being solid.

        Confidence is added and subtracted based on typical differences in each suit. Solid balls will typically have a
        colored area of greater than 200, around 240+. The area of the white area will be around 50ish. The count will be
        either 1 or 0, though in some cases it can be two.

        The number of white contours can be a big difference maker in balls that have very similar colored areas and white
        areas. Thus, if confidence is low (between -.1 and .1), confidence given is more generous than if the bot is already
        more certain which ball is solid/striped.

        The areas of colored area and white area can result in misidentification. Sometimes a striped ball will have a
        larger colored area and smaller white area. However, if this striped ball has two white areas, and the solid ball
        only has 1/0,  the confidence that is given is multiplied by the ratio of the colored area to the white area in
        order to combat misidentification. In the cases where the "bot" is completely accurate in its assessment, multiplying
        by the color ratio, will only emphasize its decision even more, although it's capped at -1 and 1.

        :param ball_1: ball_1 from ballOutline
        :param ball_2: ball_2 from ballOutline
        :return: float - -1 to 1 - confidence level of ball_1
        """
        confidence = 0.0 #-1 to 1: -1 being striped - 1 being solid

        area1 = ball_1[2]
        whiteArea1 = ball_1[3]
        whiteCount1 = ball_1[4]
        if ball_1[3] == 0:
            colorRatio1 = 9999
        else:
            colorRatio1 = ball_1[2] / ball_1[3]

        area2 = ball_2[2]
        whiteArea2 = ball_2[3]
        whiteCount2 = ball_2[4]
        if ball_2[3] == 0:
            colorRatio2 = 9999
        else:
            colorRatio2 = ball_2[2] / ball_2[3]

        aDifference = area1-area2
        if aDifference >= 50:
            confidence += 0.25
        else:
            if aDifference > 0:
                confidence += (aDifference/50) * 0.25
            else:
                if aDifference <= -50:
                    confidence = -0.25
                else:
                    confidence += math.fabs(aDifference/50) * -0.25

        whiteDifference = whiteArea1 - whiteArea2
        if whiteDifference >= 50:
            confidence += -0.25
        else:
            if whiteDifference > 0:
                confidence += (whiteDifference/50) * -0.25
            else:
                if whiteDifference <= -50:
                    confidence += 0.25
                else:
                    confidence += math.fabs(whiteDifference/50) * 0.25

        if whiteArea2 == 0:
            whiteRatio = 999
        else:
            whiteRatio = whiteArea1 / whiteArea2

        if whiteCount1 > whiteCount2:
            if 0.10 > confidence > -0.10:
                confidence += -0.5
            else:
                if confidence == -0.50:
                    confidence += (math.fabs(whiteDifference/ 100)* colorRatio2 * whiteRatio) * -0.25
                else:
                    confidence += (math.fabs(whiteDifference / 100)* colorRatio2 * whiteRatio) * -0.5
        elif whiteCount1 < whiteCount2:
            if 0.10 > confidence > -0.10:
                confidence += 0.5
            else:
                if math.fabs(confidence) > 0.45:
                    confidence += (math.fabs(whiteDifference / 100)* colorRatio1 * whiteRatio) * 0.25
                else:
                    confidence += (math.fabs(whiteDifference / 100) * colorRatio1 * whiteRatio) * 0.5

        confidence = utils.PointHelper.clamp(confidence, -1, 1)

        return confidence

### Outline Balls - Determine center/suit ###


### Obtain shot on ball ###

    def checkForBreak(self):
        """
        Determines if bot needs to break balls. If the bot doesn't currently have a suit assigned, it checks if the centers
        of all pool balls other than the cueball are within a certain area.
        :return: True/False depending on if bot needs to break.
        """

        breakRack = False
        if self.suit == None:
            print("Checking for break...")

            count = 0
            for n, b in self.unpocketed_balls.items():
                if 691 > b.center[0] > 489 and 242 > b.center[1] > 120:
                    count += 1
                    continue

            if count > 10:
                breakRack = True

        if breakRack is True:
            print("Breaking.")

            #get head ball
            closest = [None, None]
            for n, b in self.unpocketed_balls.items():
                distance = utils.PointHelper.measureDistance(self.cueball.center, b.center)
                if closest[1] is None or distance < closest[1]:
                    closest = [b, distance]

            randNum = 6#random.randint(0, 10)
            if randNum < 5:
                closest[0].moveTo()
                closest[0].dragTo(self.cueball.offsetCenter)
            else:
                print("Randomized Break.")
                offsetX, offsetY = random.randrange(-75, -25), random.randrange(-50, 50)
                self.cueball.moveTo()
                self.cueball.dragTo((self.cueball.offsetCenter[0] + offsetX, self.cueball.offsetCenter[1] + offsetY), duration=.5)
                self.cueball.center = (self.cueball.offsetCenter[0] - self.regions.table_offset[0], self.cueball.offsetCenter[1] - self.regions.table_offset[1])

                closest[0].moveTo()
                self.cueball.dragTo(self.cueball.offsetCenter, duration=.5)

                return True

        else:
            return False

    def queryDirectShot(self):
        if self.suit is None:
            solid, stripe = 0, 0
            for n,b in self.pocketed_balls.items():
                if b.suit == "solid":
                    solid += 1
                else:
                    stripe += 1

            if solid > stripe:
                self.targets = {n:b for n,b in self.all_balls.items() if b.target is True and b.pocketed is False and b.suit == "solid"}
                self.nontargets = {n:b for n,b in self.all_balls.items() if b.target is False or b.pocketed is True and b.suit == "stripe"}
            elif solid < stripe:
                self.targets = {n:b for n,b in self.all_balls.items() if b.target is True and b.pocketed is False and b.suit == "stripe"}
                self.nontargets = {n:b for n,b in self.all_balls.items() if b.target is False or b.pocketed is True and b.suit == "solid"}

        print(f"List of unpocketed balls: {self.unpocketed_balls}")
        print(f"List of pocketed balls: {self.pocketed_balls}\n")

        print(f"List of target balls: {self.targets}")
        print(f"List of nontarget balls: {self.nontargets}\n")

        startTime = time.time()
        for n, b in self.targets.items():
            if b.center == (0, 0):
                print(f"Position of {b.name} unknown.")
                if b.name == "eightball":
                    print("Position of eightball unknown.")
                    if len(targets) == 1:
                        break

                continue
            else:
                self.logger.debug(f"Looking for shot on - {b.name}.\n")

                self.createEligibleShots(b)

        self.viable_paths.sort(key=lambda x: (x.difficulty, len(x.blockingBallToHole), len(x.blockingCueToBall)))
        self.viable_paths.sort(key=lambda x: (x.benefit), reverse=True)

        endTime = time.time() - startTime
        print(f"List of eligible holes: {len(self.viable_paths)}")
        print(f"{endTime} seconds.")

        start = time.time()
        # self.openCVDrawHoles(b)
        # self.openCVDraw(b)
        # self.savePic()
        # self.chosen_ball = b

        # cv2.imshow('round', self.round_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # return True
        print(f"Time to check paths {time.time() - start} seconds")

    def queryReboundedShot(self, ball):
        pass

### Obtain shot on ball ###


### Check paths ###

    def createEligibleShots(self, ball):
        start = time.time()
        os.makedirs(self.img_path + "\paths", exist_ok=True)
        os.makedirs(self.img_path + "\paths\\ball-hole", exist_ok=True)
        os.makedirs(self.img_path + "\paths\\ball-hole\\rotated", exist_ok=True)
        os.makedirs(self.img_path + "\paths\\cue-ball", exist_ok=True)
        os.makedirs(self.img_path + "\paths\\cue-ball\\rotated", exist_ok=True)

        viable, unviable = [], []
        for hole in self.hole_locations:
            for points in hole.points:
                for point in points:
                    p = path.Path(self.cueball, ball, hole, point, self.img_path)

                    if p.isHittable(self.regions.hittable, self.regions.back_up):
                        viable = p.isViableTarget()
                        if not viable:
                            p.updateDifficulty(5)
                            unviable.append(p)
                            continue

                        clearPathB = p.checkPathofBall(self.unpocketed_balls)
                        if not clearPathB:
                            continue

                        clearPathC = p.checkPathofCue(self.unpocketed_balls)
                        if not clearPathC:
                            unviable.append(p)
                            continue

                        viable.append(p) 
        
        print(f"create shots {time.time() - start} seconds")
    
    def findBestPath(self, path):
        minBlocking = []
       
        for index, point in enumerate(path.hittablePointsToHole):
            clean = cv2.imread(self.img_path + "pooltable.png", 1)

            angle = utils.PointHelper.findAngle(point, path.ball.center, (path.ball.center[0], point[1]))
            
            rise, run, slope = utils.PointHelper.findRiseRunSlope(point, path.ball.center)
            if slope != 0:
                if slope < 0:
                    angle = -angle
            else:
                if rise == 0:
                    angle = -90

            path.hole.rotatedCenter = utils.PointHelper.rotateAround(path.ball.center, point, angle)

            tempPoints = []
            for b in self.unpocketed_balls:
                if b.name == path.ball.name:
                    continue
                else:
                    b.rotatedCenter = utils.PointHelper.rotateAround(path.ball.center, b.center, angle)
                    tempPoints.append(b)

            #bounding box
            minX = min(path.hole.rotatedCenter[0] - 10, path.hole.rotatedCenter[0] + 10, path.ball.center[0] - 10, path.ball.center[0] + 10)
            maxX = max(path.hole.rotatedCenter[0] - 10, path.hole.rotatedCenter[0] + 10, path.ball.center[0] - 10, path.ball.center[0] + 10)
            minY = min(path.hole.rotatedCenter[1], path.hole.rotatedCenter[1], path.ball.center[1], path.ball.center[1])
            maxY = max(path.hole.rotatedCenter[1], path.hole.rotatedCenter[1], path.ball.center[1], path.ball.center[1])

            t1, t2 = utils.PointHelper.findPointsOnEitherSideOf(path.ball.center, 10, -run, rise)
            t3, t4 = utils.PointHelper.findPointsOnEitherSideOf(point, 10, -run, rise)

            #checking for balls blocking
            blocking = []
            for b in tempPoints:
                if b.name != path.ball.name:
                    cv2.circle(clean, utils.PointHelper.tupleToInt(b.center), 9,(255, 0, 0), 2)

                    distance1 = utils.PointHelper.measureDistance(b.rotatedCenter, (minX, utils.PointHelper.clamp(b.rotatedCenter[1], minY, maxY)))
                    distance2 = utils.PointHelper.measureDistance(b.rotatedCenter, (maxX, utils.PointHelper.clamp(b.rotatedCenter[1], minY, maxY)))
                    distance3 = utils.PointHelper.measureDistance(b.rotatedCenter, (utils.PointHelper.clamp(b.rotatedCenter[0], minX, maxX), minY) )
                    distance4 = utils.PointHelper.measureDistance(b.rotatedCenter, (utils.PointHelper.clamp(b.rotatedCenter[0], minX, maxX), maxY) )

                    if distance1 <= 10 or distance2 <= 10 or distance3 <= 10 or distance4 <= 10:
                        cv2.circle(clean, utils.PointHelper.tupleToInt(b.center), 9,(255, 0, 255), 2)

                        blocking.append(b)
            

            #checking for walls blockling
            rotatedLeftMarkH = utils.PointHelper.rotateAround(path.ball.center, path.hole.leftMarkH, angle)
            rotatedRightMarkH = utils.PointHelper.rotateAround(path.ball.center, path.hole.rightMarkH, angle)

            distance1 = utils.PointHelper.measureDistance(rotatedLeftMarkH, (path.leftMarkB1[0], rotatedLeftMarkH[1]) )
            distance2 = utils.PointHelper.measureDistance(rotatedRightMarkH, (path.rightMarkB1[0], rotatedRightMarkH[1]) )


            if "t" in path.hole.name:
                if "r" in path.hole.name:
                    angleLessThanL = utils.PointHelper.findAngle((point[0]-100, point[1]), point, path.hole.leftMarkH)
                    angleLessThanR = utils.PointHelper.findAngle((point[0], point[1]+100), point, path.hole.rightMarkH)

                    testAngle = utils.PointHelper.findAngle((path.hole.leftMarkH[0]-100, path.hole.leftMarkH[1]), path.hole.leftMarkH, path.leftMarkB1)
                    testAngle2 = utils.PointHelper.findAngle((path.hole.rightMarkH[0], path.hole.rightMarkH[1]+100), path.hole.rightMarkH, path.rightMarkB1)

                    intersectPointLeft = utils.PointHelper.line_intersection((t1,t3), (path.hole.leftMarkH,(path.hole.leftMarkH[0]-100, path.hole.leftMarkH[1])))
                    intersectPointRight = utils.PointHelper.line_intersection((t2,t4), (path.hole.rightMarkH,(path.hole.rightMarkH[0], path.hole.rightMarkH[1]+100)))
                    
                    distanceIntersectLeftMarkH = utils.PointHelper.measureDistance(intersectPointLeft, path.hole.leftMarkH)
                    distanceIntersectRightMarkH = utils.PointHelper.measureDistance(intersectPointRight, path.hole.rightMarkH)
                    
                    if testAngle < angleLessThanL and distance1 < 10:
                        cv2.line(clean, path.hole.leftMarkH, (path.hole.leftMarkH[0]-50,path.hole.leftMarkH[1]),(255, 0, 255), 1)
                        blocking.append(Wall(path.hole.leftMarkH, (path.hole.leftMarkH[0]-50,path.hole.leftMarkH[1])))
                    else:
                        if (intersectPointLeft[0] < path.hole.leftMarkH[0] and intersectPointLeft[0] >= path.ball.center[0]) and (distanceIntersectLeftMarkH >= 6 or testAngle < 45):
                            cv2.line(clean, path.hole.leftMarkH, (path.hole.leftMarkH[0]-50,path.hole.leftMarkH[1]),(255, 0, 255), 1)
                            blocking.append(Wall(path.hole.leftMarkH, (path.hole.leftMarkH[0]-50,path.hole.leftMarkH[1])))
                
                    if testAngle2 < angleLessThanR and distance2 < 10:
                        cv2.line(clean, path.hole.rightMarkH, (path.hole.rightMarkH[0],path.hole.rightMarkH[1]+50),(255, 0, 255), 1)
                        blocking.append(Wall(path.hole.rightMarkH, (path.hole.rightMarkH[0],path.hole.rightMarkH[1]+50)))
                    else:
                        if (intersectPointRight[1] > path.hole.rightMarkH[1] and intersectPointRight[1] <= path.ball.center[1]) and (distanceIntersectRightMarkH >= 6 or testAngle2 < 45):
                            cv2.line(clean, path.hole.rightMarkH, (path.hole.rightMarkH[0],path.hole.rightMarkH[1]+50),(255, 0, 255), 1)
                            blocking.append(Wall(path.hole.rightMarkH, (path.hole.rightMarkH[0],path.hole.rightMarkH[1]+50)))

            path.hole.targetPoint = path.hittablePointsToHole[index]

            path.setupParamsFromBallToHole()
            path.setupParamsFromCueToBall()

            # slopeDiff = math.fabs(path.slopeToHole - slope)
            minBlocking.append([index, blocking, max(distance1, distance2) - min(distance1, distance2)])
                                
            cv2.line(clean, utils.PointHelper.tupleToInt(t1), utils.PointHelper.tupleToInt(t2),(255, 255, 255), 1)
            cv2.line(clean, utils.PointHelper.tupleToInt(t2), utils.PointHelper.tupleToInt(t4),(255, 255, 255), 1)
            cv2.line(clean, utils.PointHelper.tupleToInt(t4), utils.PointHelper.tupleToInt(t3),(255, 255, 255), 1)
            cv2.line(clean, utils.PointHelper.tupleToInt(t3), utils.PointHelper.tupleToInt(t1),(255, 255, 255), 1)

            self.openCVDrawHoles(clean, path.hole)
            result = self.rotate_bound(clean, angle)

            cv2.imshow('test', clean) 
            cv2.imshow('rotated', result) 
            cv2.waitKey(0)
            cv2.destroyAllWindows() 

        minBlocking.sort(key=lambda x: (len(x[1]), x[2]))

        if len(minBlocking) > 0:
            minBlocking = [v for k, v in enumerate(minBlocking) if not any(isinstance(y, Wall) for y in minBlocking[k][1])]
            try:
                leastBlocking = next(x for x in minBlocking) 
            except StopIteration:
                return False   


            targetIndex = leastBlocking[0] #optimal target hole point
            blockingBalls = leastBlocking[1] #balls blocking shot to hole

            path.hole.targetPoint = path.hittablePointsToHole[targetIndex]
            path.blockingBallToHole = blockingBalls
            path.hittablePointsToHole = [path.hittablePointsToHole[x[0]] for x in minBlocking] #remove unworthy target points

            path.updateDifficulty(.33*len(blockingBalls))

            path.setupParamsFromBallToHole()
            path.setupParamsFromCueToBall()

            angle = utils.PointHelper.findAngle(path.hole.targetPoint, path.ball.center, (path.ball.center[0], path.hole.targetPoint[1]))
            if path.slopeToHole < 0:
                angle = -angle
            else:
                if path.slopeToHoleRise == 0:
                    angle = -90

            clean = cv2.imread(self.img_path + "pooltable.png", 1)

            self.openCVDrawHoles(clean, path.hole)

            for b in self.unpocketed_balls:
                if b.name != path.ball.name:
                    cv2.circle(clean, utils.PointHelper.tupleToInt(b.center), 9,(255, 0, 0), 2)

                    if b in blockingBalls:
                        cv2.circle(clean, utils.PointHelper.tupleToInt(b.center), 9,(255, 0, 255), 2)

            for w in path.blockingBallToHole:
                if isinstance(w, Wall):
                    cv2.line(clean, w.startingPoint, w.endingPoint,(255, 0, 255), 1)

            t1, t2 = utils.PointHelper.findPointsOnEitherSideOf(path.ball.center, 10, -path.slopeToHoleRun, path.slopeToHoleRise)
            t3, t4 = utils.PointHelper.findPointsOnEitherSideOf(path.hole.targetPoint, 10, -path.slopeToHoleRun, path.slopeToHoleRise)

            cv2.line(clean, utils.PointHelper.tupleToInt(t1), utils.PointHelper.tupleToInt(t2),(255, 255, 255), 1)
            cv2.line(clean, utils.PointHelper.tupleToInt(t2), utils.PointHelper.tupleToInt(t4),(255, 255, 255), 1)
            cv2.line(clean, utils.PointHelper.tupleToInt(t4), utils.PointHelper.tupleToInt(t3),(255, 255, 255), 1)
            cv2.line(clean, utils.PointHelper.tupleToInt(t3), utils.PointHelper.tupleToInt(t1),(255, 255, 255), 1)

            result = self.rotate_bound(clean, angle)
            cv2.imwrite(f"{self.img_path}\paths\\ball-hole\\rotated\\{path.ball.name}-{path.hole.name}-{targetIndex}_rotated.png", result)

            cv2.imwrite(f"{self.img_path}\paths\\ball-hole\\{path.ball.name}-{path.hole.name}-{targetIndex}_path.png", clean)

            cv2.imshow('test', clean) 
            cv2.imshow('rotated', result) 
            cv2.waitKey(0)
            cv2.destroyAllWindows() 

            return True
        else:
            return False

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def findResultingPositions(self):
        pass

### Check paths ###


### Drawing ###

    def drawStripe(self, center, colorRGB):
        cv2.circle(self.round_image, utils.PointHelper.tupleToInt(center), 9, colorRGB, 2)
        cv2.circle(self.round_image, utils.PointHelper.tupleToInt(center), 10, (0, 0, 0), 1)
        cv2.line(self.round_image, utils.PointHelper.tupleToInt((center[0], center[1] + 10)), utils.PointHelper.tupleToInt((center[0], center[1] - 10)), (0, 0, 0))
        cv2.line(self.round_image, utils.PointHelper.tupleToInt((center[0] - 10, center[1])), utils.PointHelper.tupleToInt((center[0] + 10, center[1])), (0, 0, 0))

    def drawSolid(self, center, colorRGB):
        cv2.circle(self.round_image, utils.PointHelper.tupleToInt((center[0], center[1])), 10, colorRGB, -1)
        cv2.circle(self.round_image, utils.PointHelper.tupleToInt((center[0], center[1])), 10, (0, 0, 0), 1)
        cv2.line(self.round_image, utils.PointHelper.tupleToInt((center[0], center[1] + 10)), utils.PointHelper.tupleToInt((center[0], center[1] - 10)), (255, 255, 255))
        cv2.line(self.round_image, utils.PointHelper.tupleToInt((center[0] - 10, center[1])), utils.PointHelper.tupleToInt((center[0] + 10, center[1])), (255, 255, 255))

    def savePic(self):
        cv2.imwrite(self.img_path + "pooltableOutlined.png", self.round_image)

    def openCVDrawHoles(self, img, hole):
        cv2.circle(img, utils.PointHelper.tupleToInt(hole.innerBoundLeft), 1,(255, 255, 255), -1)
        cv2.circle(img, utils.PointHelper.tupleToInt(hole.innerBoundRight), 1,(255, 255, 255), -1)

        cv2.circle(img, utils.PointHelper.tupleToInt(hole.leftMarkH), 1,(0, 255, 255), -1)
        cv2.circle(img, utils.PointHelper.tupleToInt(hole.rightMarkH), 1,(0, 255, 255), -1)

        cv2.circle(img, utils.PointHelper.tupleToInt(hole.corner), 1,(255, 255, 255), -1)
        cv2.circle(img, utils.PointHelper.tupleToInt(hole.center), 1,(255, 255, 255), -1)

    def openCVDraw(self, ball):
        cv2.circle(self.round_image, utils.PointHelper.tupleToInt(ball.hitPoint), 9, (0, 0, 0), 1)
        cv2.line(self.round_image, utils.PointHelper.tupleToInt(ball.currentHole),utils.PointHelper.tupleToInt(ball.hitPoint),(0, 0, 0), 1)
        cv2.line(self.round_image, utils.PointHelper.tupleToInt(ball.hitPoint), utils.PointHelper.tupleToInt(self.cueball.center),(0, 0, 0), 1)
        cv2.line(self.round_image, utils.PointHelper.tupleToInt(ball.leftMarkH), utils.PointHelper.tupleToInt(ball.leftMarkB1), (235, 186, 25), 1)
        cv2.line(self.round_image, utils.PointHelper.tupleToInt(ball.rightMarkH), utils.PointHelper.tupleToInt(ball.rightMarkB1),(199, 18, 27), 1)
        cv2.line(self.round_image, utils.PointHelper.tupleToInt(ball.leftMarkB2), utils.PointHelper.tupleToInt(ball.leftMarkC), (3, 202, 252), 1)
        cv2.line(self.round_image, utils.PointHelper.tupleToInt(ball.rightMarkB2), utils.PointHelper.tupleToInt(ball.rightMarkC),(52, 52, 235), 1)
        cv2.line(self.round_image, utils.PointHelper.tupleToInt(ball.leftMarkB1), utils.PointHelper.tupleToInt(ball.rightMarkB1),(255, 0, 0), 1)
        cv2.line(self.round_image, utils.PointHelper.tupleToInt(ball.leftMarkB2), utils.PointHelper.tupleToInt(ball.rightMarkB2),(255, 255, 255), 1)

### Drawing ###


### Hit Ball ###

    def hitBall(self, ball):
        self.logger.debug("Hitting ball.")

        self.powerAmount(ball)

        print(ball.slopeToHPRun, ball.slopeToHPRise)
        if ball.slopeToHPRun > 0:
            print(1)
            if ball.slopeToHPRise > 0:
                print("1a")
                ball.hitPointStart = (ball.hitPointStart[0] + 2, ball.hitPointStart[1] + 2)
            else:
                print("1b")
                ball.hitPointStart = (ball.hitPointStart[0] - 2, ball.hitPointStart[1] - 2)
        else:
            print(2)
            if ball.slopeToHPRise > 0:
                print("2a")
                ball.hitPointStart = (ball.hitPointStart[0] - 2, ball.hitPointStart[1] + 2)
            else:
                print("2b")
                ball.hitPointStart = (ball.hitPointStart[0] + 2, ball.hitPointStart[1] + 3)

        print(ball.hitPoint, ball.hitPointStart, ball.hitPointEnd)
        pyautogui.moveTo((ball.hitPointStart[0] + self.regions.table_offset[0], ball.hitPointStart[1] + self.regions.table_offset[1]))
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pyautogui.moveTo((ball.hitPointStart[0] + self.regions.table_offset[0], ball.hitPointStart[1] + self.regions.table_offset[1]))
            pyautogui.mouseDown(button="left")
            pyautogui.moveTo((ball.hitPointEnd[0] + self.regions.table_offset[0]), (ball.hitPointEnd[1] + self.regions.table_offset[1]))
            pyautogui.dragTo((ball.hitPointEnd[0] + self.regions.table_offset[0], ball.hitPointEnd[1] + self.regions.table_offset[1]),
                             button="left", duration=1)
        
        pyautogui.screenshot(self.img_path + "confirmation.png",
                             region=self.regions.table)

        cv2.circle(self.round_image, utils.PointHelper.tupleToInt(ball.hitPoint), 3, (0, 0, 255), -1)
        cv2.circle(self.round_image, utils.PointHelper.tupleToInt(ball.hitPointStart), 3, (0, 0, 255), -1)
        cv2.circle(self.round_image, utils.PointHelper.tupleToInt(ball.hitPointEnd), 2, (0, 100, 255),-1)
        self.savePic()

    def powerAmount(self, ball):
        maxDistance = 1384
        pullDistance = 264
        velocity = 2768

        ratioCue = ball.distancetoCue / 664.0
        ratioBall = ball.distanceToHole / 664.0
        ratio = ratioCue + ratioBall
        if ratio < 1.0:
            distance = (ratio * 336) * 1.5
        else:
            distance = 336

        velocity = ratio * velocity

        print(ratioCue, ratioBall, ratio, velocity)


        if ball.slopeToHP > 0:
            if ball.slopeToHPRise > 0:
                #print(111)
                x = (ball.hitPointStart[0] + (distance * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))
                y = (ball.hitPointStart[1] + ((ball.slopeToHP * distance) * math.sqrt( 1 / (1 + ball.slopeToHP ** 2))))
            else:
                #print(222)
                x = (ball.hitPointStart[0] - (distance * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))
                y = (ball.hitPointStart[1] - ((ball.slopeToHP * distance) * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))

            ball.hitPointEnd = (x, y)
        elif ball.slopeToHP < 0:
            if ball.slopeToHPRise > 0:
                #print(333)
                x = (ball.hitPointStart[0] - (distance * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))
                y = (ball.hitPointStart[1] - ((ball.slopeToHP * distance) * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))
            else:
                #print(444)
                x = (ball.hitPointStart[0] + (distance * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))
                y = (ball.hitPointStart[1] + ((ball.slopeToHP * distance) * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))

            ball.hitPointEnd = (x, y)
        else:
            if ball.slopeToHPRise == 0:
                #print(555)
                x = (ball.hitPointStart[0] + distance)
                y = ball.hitPointStart[1]
            else:
                #print(666)
                x = ball.hitPointStart[0]
                y = (ball.hitPointStart[1] - distance)

            ball.hitPointEnd = (x, y)
        #ball.stats()

### Hit Ball ###


### Pre-round info collection ###

    def canPickUpCueBall(self):
        pos = utils.timedImageSearch(constants.img_glove, 2, self.regions.table, .95)
        if pos is not None:
            pos = (pos[0]-8, pos[1]-1)
            pyautogui.mouseDown(pos[0], pos[1], duration=.5)
            time.sleep(.4)
            pyautogui.screenshot((self.img_path + "pooltable.png"), region=self.regions.table)
            pyautogui.mouseUp()

            return True

        return False

    def getCleanRoundImage(self): #not a proiority, but attempts to get image in which all balls are most visible
        self.can_pick_up = self.canPickUpCueBall()

        ballPoints = self.quickBallCount()
        cueball = self.findCueBall()

        index = 0
        for k, p in enumerate(ballPoints):
            if math.fabs(p[0] - cueball[0]) < 6 and math.fabs(p[1] - cueball[1]) < 6:
                self.cueball.center = (p[0], p[1])
                self.cueball.offsetCenter = (p[0] + self.regions.table_offset[0], p[1] + self.regions.table_offset[1])
                index = k

        ballPoints.pop(index)
        
        if not self.can_pick_up:
            self.moveMouseOut(ballPoints)
        else:
            return True

    def findCueBall(self):
        if self.can_pick_up:
            tableMaskedOut = cv2.imread(self.img_path + "pooltable.png", 1)
        else:
            tableMaskedOut = cv2.imread(self.img_path + "quickBallCount.png", 1)

        tableMaskedOut = cv2.imread(self.img_path + "quickBallCount.png", 1)
        hsvTableMaskedOut = cv2.cvtColor(tableMaskedOut, cv2.COLOR_BGR2HSV)

        upper_whitecue = np.array([27, 36, 255])
        lower_whitecue = np.array([18, 0, 120])
        maskwhiteball = cv2.inRange(hsvTableMaskedOut, lower_whitecue, upper_whitecue)
        maskwhiteball = cv2.morphologyEx(maskwhiteball, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours = self.contourSetup(maskwhiteball, True)
        (x, y), radius = cv2.minEnclosingCircle(contours[-1] )
        return (int(x),int(y))

    def quickBallCount(self):
        roundImageCopy = cv2.imread(self.img_path + "pooltable.png", 1).copy()
        hsvRndImg = cv2.cvtColor(roundImageCopy, cv2.COLOR_BGR2HSV)

        invertMask = self.maskOutTable(roundImageCopy, hsvRndImg)
        cv2.imwrite(self.img_path + "quickBallCount.png", invertMask)

        tableMaskedOutBW = cv2.imread(self.img_path + "quickBallCount.png", 0)

        circles = cv2.HoughCircles(tableMaskedOutBW, cv2.HOUGH_GRADIENT, 1, 17,
                                    param1=20, param2=9, minRadius=9, maxRadius=11)
        circles = np.uint16(np.around(circles))
        houghPoints = []
        for i in circles[0, :]:
            houghPoints.append((i[0], i[1]))

        count = len(houghPoints)
        self.logger.debug(f"QuickBallCount discovered {count} balls.\n")

        return houghPoints

    def maskOutTable(self, img, hsv):
        self.round_image = cv2.imread(self.img_path + "pooltable.png", 1)

        # mask for table color - table color masked out for visibility
        blueTableLow = np.array([90, 80, 0])
        blueTableHigh = np.array([106, 255, 255])
        blackTableLow = np.array([0, 0, 35])
        blackTableHigh = np.array([180, 255, 255])
        tableInvertMask = cv2.inRange(hsv, blueTableLow, blueTableHigh)
        tableInvertMask = cv2.bitwise_not(tableInvertMask)
        holeInvertMask = cv2.inRange(hsv, blackTableLow, blackTableHigh)
        tableInvertMask = cv2.bitwise_and(tableInvertMask, holeInvertMask)
        tableInvertMask = cv2.bitwise_and(img, img, mask=tableInvertMask)

        return tableInvertMask

    def moveMouseOut(self, points):
        pass

### Pre-round info collection ###
