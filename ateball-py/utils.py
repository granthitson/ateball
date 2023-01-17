import logging

import os
import sys
import time

from enum import Enum
from enum import IntEnum

import threading
import queue as q

import json

import math
import pyautogui
import cv2
import numpy as np

import constants

logger = logging.getLogger("ateball.utils")

class SetupError(Exception):
    pass
class IPCSetupError(SetupError):
    pass
class TurnCycleError(Exception):
    pass

class Formatter(logging.Formatter):
    def __init__(self):
        super().__init__()

        self.datefmt = '%H:%M:%S'

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            self._style._fmt = "%(asctime)s - %(name)s - %(levelname)s - %(lineno)s: %(message)s"
        return super().format(record)

class OrEvent:
    def __init__(self, event1, event2, *args):
        self.condition = threading.Condition()
        self.events = [event1, event2, *args]
        self.timeout = threading.Event()

    def wait(self, timeout):
        logger.debug("or event start")
        with self.condition:
            while not any([e.is_set() for e in self.events]) and not self.timeout.is_set():
                logger.debug(f"or event waiting")
                timed_out = not self.condition.wait(timeout)
                if timed_out:
                    self.timeout.set()
                    logger.debug("or event timeout")

        result = " - ".join([str(e.is_set()) for e in self.events])
        logger.debug(f"or event end: {result} - {self.timeout.is_set()}")

    def notify(self, event):
        event.set()
        with self.condition:
            self.condition.notify()

    def clear(self):
        for e in self.events:
            e.clear()
        self.timeout.clear()

    def is_set(self):
        is_set = False
        for e in self.events:
            if e.is_set():
                is_set = True

        return is_set

    def has_timed_out(self):
        return self.timeout.is_set()

class IPC:
    def __init__(self):
        self.incoming = q.Queue()
        self.outgoing = q.Queue()
        
        self.listen_event = threading.Event()
        self.listen_exception_event = threading.Event() 
        self.exception_event = threading.Event() 
        self.stop_event = threading.Event()

        self.exception = None

        self.logger = logging.getLogger("ateball.utils.ipc")

    def listen(self):
        try:
            self.logger.info("ipc listening...")
            self.listen_event.set()
            while not self.stop_event.is_set():
                data = sys.stdin.readline()
                if data:
                    msg = json.loads(data)
                    self.incoming.put(msg)
        except Exception as e:
            self.listen_exception_event.set()
            self.logger.exception(f"error handling ipc messages: {e}")

    def send(self):
        try:
            while not self.stop_event.is_set():
                try:
                    msg = self.outgoing.get()
                    sys.stdout.write(f"{json.dumps(msg)}\n")
                    sys.stdout.flush()
                except q.Empty:
                    pass
        except Exception as e:
            self.start_exception_event.set()
            self.logger.exception(f"error handling ipc messages: {e}")

    def send_message(self, msg):
        self.outgoing.put(msg)

    def quit(self):
        self.stop_event.set()

class RegionData:
    def __init__(self, data):
        self.window_offset = (data["window_offset"]["x"], data["window_offset"]["y"])

        self.game = (data["game"]["x"], data["game"]["y"], data["game"]["width"], data["game"]["height"])

        self.turn_start = (data["turn_start"]["x"], data["turn_start"]["y"], data["turn_start"]["width"], data["turn_start"]["height"])
        self.turn_mask = (data["turn_mask"]["x"], data["turn_mask"]["y"], data["turn_mask"]["width"], data["turn_mask"]["height"])

        self.table = (data["table"]["x"], data["table"]["y"], data["table"]["width"], data["table"]["height"])
        #offset from top-left-corner of screen to table corner
        # offsetx, offsety, width, height
        self.table_offset = (data["table"]["x"], data["table"]["y"])
        
        # offsetx, offsety, width, height
        self.pocketed = (data["pocketed"]["x"], data["pocketed"]["y"], data["pocketed"]["width"], data["pocketed"]["height"])

        self.targets_bot = (data["targets_bot"]["x"], data["targets_bot"]["y"], data["targets_bot"]["width"], data["targets_bot"]["height"])
        self.targets_opponent = (data["targets_opponent"]["x"], data["targets_opponent"]["y"], data["targets_opponent"]["width"], data["targets_opponent"]["height"])
        
        # defined as within half ball_diameter distance to wall/edge of table
        self.hittable = (data["hittable"]["x"], data["hittable"]["y"], data["hittable"]["width"], data["hittable"]["height"])
        # defined as within fourth ball_diameter distance to wall/edge of table - if ball is near hole
        self.back_up = (data["back_up"]["x"], data["back_up"]["y"], data["back_up"]["width"], data["back_up"]["height"])

class Wall:
    def __init__(self, startingPoint, endingPoint):
        self.startingPoint = startingPoint
        self.endingPoint = endingPoint

    def __str__(self):
        return "Wall"

    def __repr__(self):
        return str(self)

class JSONHelper:
    logger = logging.getLogger("ateball.utils.JSONHelper")

    @staticmethod
    def loadJSON(path):
        with open(path, "r") as f:
            data = json.load(f)
            return data

class ImageHelper:
    logger = logging.getLogger("ateball.utils.ImageHelper")

    @staticmethod
    def imagePath(filename):
        return os.path.join(os.getcwd(), "ateball-py", "images", filename)

    @staticmethod
    def get(pos, point_type="center", ):
        if point_type == "corner":
            return (pos[0], pos[1])
        else:
            return (pos[0] + (pos[2]/2), pos[1] + (pos[3]/2))

    @staticmethod
    def locateImage(needle, haystack, region=None, threshold=.90):
        try:
            if region is not None:
                haystack = haystack[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]

            needle = cv2.imread(ImageHelper.imagePath(needle))
            w, h = needle.shape[:-1]

            w1, h1 = haystack.shape[:-1]

            res = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)

            loc = np.where( res >= threshold)
            results = list(map(lambda p: (p[0], p[1], w, h), zip(*loc[::-1])))

            if len(results) > 0:
                result = results[0]
            else:
                result = None
        except Exception as e:
            logger.error(f"error locating image - {needle} in {haystack}")
            result = None

        return result

    @staticmethod
    def imageSearch(needle, haystack, region=None, point_type="center", threshold=.95):
        pos = ImageHelper.locateImage(needle, haystack, region, threshold)
        if pos is not None:
            pos = ImageHelper.get(pos, "center")
            
        ImageHelper.logger.debug(f"image search complete: {needle} - {pos}")

        return pos

    @staticmethod
    def imageSearchLock(needle, haystack, region=None, threshold=.95, lock_time=1):
        # image is present if it exists before and after x seconds
        pos = ImageHelper.locateImage(needle, haystack, region, threshold)
        if pos is not None:
            threading.Event().wait(lock_time)
            pos = ImageHelper.locateImage(needle, haystack, region, threshold)
            if pos is not None:
                ImageHelper.logger.debug(f"image lock acquired: {needle}")
                return ImageHelper.get(pos, "center")

        return None

class CV2Helper:
    logger = logging.getLogger("ateball.utils.CV2Helper")

    @staticmethod
    def getContours(mask, key=None):
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        return sorted(contours, key=key) if key is not None else contours

    @staticmethod
    def contourCenter(c):
        try:
            M = cv2.moments(c)
            return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        except ZeroDivisionError as e:
            raise TurnCycleError("zerodivisionerror getting center of contour")

class PointHelper:
    logger = logging.getLogger("ateball.utils.PointHelper")

    @staticmethod
    def getBrightness(p):
        return sum([pyautogui.pixel(p[0], p[1])[0], pyautogui.pixel(p[0], p[1])[1], pyautogui.pixel(p[0], p[1])[2]])/3
     
    @staticmethod
    def averagePoint(p1, p2):
        return ((p1[0] + p2[0])/2,(p1[1] + p2[1])/2)

    @staticmethod
    def isTupleInRange(t1, t2, xRange, yRange):
        if (abs(t1[0]-t2[0]) < xRange and abs(t1[1]-t2[1]) < yRange):
            return True
        else:
            return False

    @staticmethod
    def roundTuple(tuple):
        return (round(tuple[0]),round(tuple[1]))

    @staticmethod
    def tupleToInt(tuple):
        return (int(tuple[0]),int(tuple[1]))

    @staticmethod
    def tupleToFloat(tuple):
        return (float(tuple[0]),float(tuple[1]))

    @staticmethod
    def clamp(n, mini, maxi):
        return max(min(maxi, n), mini)

    @staticmethod
    def measureDistance(firstInput, secondInput):
        dist = math.sqrt((int(secondInput[0]) - int(firstInput[0])) ** 2 + (int(secondInput[1]) - int(firstInput[1])) ** 2)
        return dist

    @staticmethod
    def findAngle(p1, p2, p3):
        a = tupleToFloat(p1)
        b = tupleToFloat(p2)
        c = tupleToFloat(p3)

        try:
            angle = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        except ZeroDivisionError:
            return 90.0

        if (angle < 0):
            if math.fabs(angle) > 180:
                angle = 360 - math.fabs(angle)
            else:
                angle = math.fabs(angle)

        return angle

    @staticmethod
    def findRiseRunSlope(p1, p2):
        rise = float(p1[1]) - float(p2[1])
        run = float(p1[0]) - float(p2[0])
        if run != 0:
            slope = rise / run
        else:
            slope = 0

        return rise, run, slope

    @staticmethod
    def findPointsOnEitherSideOf(p, distance, rise, run, invert=False):
        try:
            slope = rise/run
        except ZeroDivisionError:
            slope = 0

        if invert is False:
            if slope != 0:
                left = findPointsAlongSlope(p, distance, slope, True)
                right = findPointsAlongSlope(p, distance, slope)
            else:
                if math.fabs(rise) > 0:
                    left = (p[0], p[1]-distance)
                    right = (p[0], p[1]+distance)
                else:
                    left = (p[0]-distance, p[1])
                    right = (p[0]+distance, p[1])
        else:
            if slope != 0:
                left = findPointsAlongSlope(p, distance, slope)
                right = findPointsAlongSlope(p, distance, slope, True)
            else:
                if math.fabs(rise) > 0:
                    left = (p[0], p[1]+distance)
                    right = (p[0], p[1]-distance)
                else:
                    left = (p[0]+distance, p[1])
                    right = (p[0]-distance, p[1])

        return left, right

    @staticmethod
    def findPointsAlongSlope(p, distance, slope, subtract=False):
        if subtract:
            x = (p[0] - (distance * math.sqrt(1 / (1 + slope ** 2))))
            y = (p[1] - ((slope * distance) * math.sqrt(1 / (1 + slope ** 2))))
        else:
            x = (p[0] + (distance * math.sqrt(1 / (1 + slope ** 2))))
            y = (p[1] + ((slope * distance) * math.sqrt(1 / (1 + slope ** 2))))

        return x, y

    @staticmethod
    def rotateAround(anchor, point, angle):
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        pX = int(point[0]) - int(anchor[0])
        pY = int(point[1]) - int(anchor[1])

        newX = (pX * c - pY * s) + anchor[0]
        newY = (pX * s + pY * c) + anchor[1]

        center = tupleToInt((newX, newY))

        return center

    @staticmethod
    def line_intersection(line1, line2, xmin=None, xmax=None, ymin=None, ymax=None):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        if xmin is not None:
            if x < xmin:
                return None

        if xmax is not None:
            if x > xmax:
                return None

        if ymin is not None:
            if y < ymin:
                return None

        if ymax is not None:
            if y > ymax:
                return None

        return x, y
