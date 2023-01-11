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
    '''
    `game_region` is a list to maintain mutability and changes if the window is moved.
    Changes to game_region  will not propagate to other regions regardless structure (list/tuple),
    so performance is gained by using tuples which are immutable.
    '''
    def __init__(self):
        self.table = (106, 176, 690, 360)
        #offset from top-left-corner of screen to table corner
        # offsetx, offsety, width, height
        self.table_offset = (106, 176)
        
        # offsetx, offsety, width, height
        self.pocketed = (self.table[2] + 725, self.table[1] + 0, 50, self.table[3])

        self.targets_bot = (self.table[0] + 7, self.table[1] - 119, 210, 30)
        self.targets_opponent = (self.targets_bot[0] + 465, self.targets_bot[1], self.targets_bot[2], self.targets_bot[3])
        
        # defined as within half ball_diameter distance to wall/edge of table
        self.hittable = (
            self.table[0]+constants.ball_diameter*2, self.table[1]+constants.ball_diameter*2, 
            self.table[2]-constants.ball_diameter*4, self.table[3]-constants.ball_diameter*4
        )
        # defined as within fourth ball_diameter distance to wall/edge of table - if ball is near hole
        self.back_up = (
            self.table[0]+constants.ball_diameter/2, self.table[1]+constants.ball_diameter/2, 
            self.table[2]-constants.ball_diameter, self.table[3]-constants.ball_diameter
        )  

class Wall:
    def __init__(self, startingPoint, endingPoint):
        self.startingPoint = startingPoint
        self.endingPoint = endingPoint

    def __str__(self):
        return "Wall"

    def __repr__(self):
        return str(self)

class ImageHelper:
    logger = logging.getLogger("ateball.utils.ImageHelper")

    @staticmethod
    def imagePath(filename):
        return os.path.join(os.getcwd(), "ateball-py", "images", filename)

    @staticmethod
    def locateImage(image, region=None, confidence=.99):
        if region is None:
            pos = pyautogui.locateOnScreen(ImageHelper.imagePath(image), confidence=confidence)
        else:
            pos = pyautogui.locateOnScreen(ImageHelper.imagePath(image), region=region, confidence=confidence)
        
        return pos

    @staticmethod
    def locateAllImages(image, region=None, confidence=.99):
        if region is None:
            pos = pyautogui.locateAllOnScreen(ImageHelper.imagePath(image), confidence=confidence)
        else:
            pos = pyautogui.locateAllOnScreen(ImageHelper.imagePath(image), region=region, confidence=confidence)
        
        return pos

    @staticmethod
    def imageSearch(image, region=None, point_type="center", confidence=.95, time_limit=0):
        time_taken = 0 
        startTime = time.time()

        pos = ImageHelper.locateImage(image, region, confidence)
        while (pos is None and time_taken < time_limit):
            pos = ImageHelper.locateImage(image, region, confidence)      
            time_taken = (time.time() - startTime)

        if pos is not None:
            if point_type == "corner":
                pos = (int(pos.left), int(pos.top))
            else:
                pos = (int(pos.left + (pos.width/2)), int(pos.top + (pos.height/2)))
            
        ImageHelper.logger.debug(f"image search complete: {image} - {pos} - {time_taken:.2f}s")

        return pos

    @staticmethod
    def imageSearchAll(image, region=None, point_type="center", confidence=.95, time_limit=0):
        time_taken = 0 
        startTime = time.time()

        points = ImageHelper.locateAllImages(image, region, confidence)
        while (points is None and time_taken < time_limit):
            points = ImageHelper.locateAllImages(image, region, confidence)

        if point_type == "center":
            points = [(int(p.left + (p.width/2)), int(p.top + (p.height/2))) for p in points]
        elif point_type == "corner":
            points = [(int(p.left), int(p.top)) for p in points]
        else:
            points = [p for p in points]

        time_taken += (time.time() - startTime)

        ImageHelper.logger.debug(f"image search complete: {image} - {points} - {time_taken:.2f}s")

        return points

    @staticmethod
    def imageSearchLock(image, region=None, confidence=.95, lock_time=1):
        # image is present if it exists before and after x seconds
        pos = ImageHelper.locateImage(image, region, confidence)
        if pos is not None:
            threading.Event().wait(lock_time)
            pos = ImageHelper.locateImage(image, region, confidence)
            if pos is not None:
                ImageHelper.logger.debug(f"image lock acquired: {image}")
                return (int(pos.left + (pos.width/2)), int(pos.top + (pos.height/2))) #return center

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
