import logging

import os
import sys
import time

import threading
import queue as q

import json

import math
import pyautogui
import cv2
import numpy as np
import skimage

import win32gui, win32ui, win32con
from pathlib import Path

logger = logging.getLogger("ateball.utils")

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

    def wait(self, timeout=None):
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

class WindowCapturer(threading.Thread):
    def __init__(self, region, offset, fps, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hwnd = win32gui.FindWindow(None, os.getenv("APP_NAME"))
        if not self.hwnd:
            raise Exception("window not found")

        self.region = (region[2], region[3])
        self.offset = offset

        self.fps = fps
        self.last_tick = 0
        self.on_tick = threading.Event()

        self.image = None

        self.stop_event = threading.Event()

        self.logger = logging.getLogger("ateball.utils.WindowCapturer")

    def tick(self, fps):
        # synchronize loop with fps
        self.on_tick.clear()

        interval = 1 / fps 
        current_time = time.time()
        delta = current_time - self.last_tick

        if delta < interval:
            time.sleep(interval - delta)

        self.last_tick = time.time()

    def run(self):
        self.stop_event.clear()

        while not self.stop_event.is_set():
            self.tick(self.fps)

            w, h = self.region

            left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
            offset = (self.offset[0] + left, self.offset[1] + top)

            # can't capture hardware accelerated window
            desktop = win32gui.GetDesktopWindow()
            wDC = win32gui.GetWindowDC(desktop)

            # wDC = win32gui.GetWindowDC(self.hwnd)
            dcObj=win32ui.CreateDCFromHandle(wDC)
            cDC=dcObj.CreateCompatibleDC()
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
            cDC.SelectObject(dataBitMap)
            cDC.BitBlt((0,0), (w, h), dcObj, offset, win32con.SRCCOPY)

            signedIntsArray = dataBitMap.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype='uint8').reshape((h, w, 4))
            img.shape = (h, w, 4)

            img = img[...,:3]
            self.image = np.ascontiguousarray(img)

            # Free Resources
            dcObj.DeleteDC()
            cDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, wDC)
            win32gui.DeleteObject(dataBitMap.GetHandle())

            # allow threads to retrieve latest
            self.on_tick.set()

    def get(self):
        # get latest image added to stack
        self.on_tick.wait()
        return self.image if self.image.any() else np.array([])

    def stop(self):
        self.stop_event.set()

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
    def imread(path, *args):
        image = cv2.imread(path, *args)
        if image is None:
            raise Exception(f"image does not exist at path: {path}")

        return image

    @staticmethod
    def get_closest_color(img, mask, lookup):
        # get mean bgr color value for both timers
        mean = cv2.mean(img, mask=mask)[:3]

        # match mean color to nearest color match
        deltas = CV2Helper.color_deltas(mean, lookup)

        return deltas[0][1]

    @staticmethod
    def color_deltas(color, color_lookup):
        # match color to closest color listed in lookup

        # create list of differences for each lookup value
        differences = [
            [CV2Helper.color_difference(color, bgr), name]
            for name, bgr in color_lookup.items()
        ]

        # sort differences (ascending) - and return color with least difference
        differences.sort()
        return differences
    
    @staticmethod
    def color_difference(color1, color2):
        # measure deltaE using LAB colorspace

        color1_lab = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_BGR2LAB)[0][0]
        color2_lab = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_BGR2LAB)[0][0]

        # return deltaE of lab colors
        return skimage.color.deltaE_cie76(color1, color2)

    def resize(image, factor):
        width = int(image.shape[1] * factor)
        height = int(image.shape[0] * factor)
        dim = (width, height)

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def slice_image(image, region):
        # cut image to size (within confines of image)
        height, width, channels = image.shape
        x, y, width, height = max(0, region[0]), max(0, region[1]), min(region[2], width), min(region[3], height)
        return image[y:y + height, x:x + width]
    
    @staticmethod
    def roi(image, center, radius):
        region = (center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        return CV2Helper.slice_image(image, region), region

    @staticmethod
    def create_mask(hsv, lower, upper, op=None, kernal=None):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if op is not None:
            mask = cv2.morphologyEx(mask, op, kernal)

        return mask

class Point:
    def __init__(self, center):
        self.center = (center[0], center[1])

        self.logger = logging.getLogger("ateball.utils.Point")

    def __str__(self):
        return f"{self.center}"

    def __repr__(self):
        return str(self)

    def add(self, p):
        return (self.center[0] + p.center[0], self.center[1] + p.center[0])

    def distance(self, p):
        return round(math.hypot(self.center[0] - p.center[0], self.center[1] - p.center[1]), 2)

    def average(self, p):
        return (int((self.center[0] + p.center[0])/2),int((self.center[1] + p.center[1])/2))

    def get_rise_run_slope(self, p):
        rise = float(self.center[1]) - float(p.center[1])
        run = float(self.center[0]) - float(p.center[0])
        if run != 0:
            slope = rise / run
        else:
            slope = 0

        return rise, run, slope

    def get_angle(self, p1, p2):
        # get angle where point is vertex

        a = p1.center
        b = self.center
        c = p2.center

        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang + 360 if ang < 0 else ang

    def to_float(self):
        return (float(self.center[0]), float(self.center[1]))

    def to_int(self):
        return (int(self.center[0]), int(self.center[1]))

    def round(self, precision=2):
        return (round(self.center[0]), round(self.center[1]))

    def findPointsOnEitherSideOf(self, distance, rise, run, invert=False):
        try:
            slope = rise/run
        except ZeroDivisionError:
            slope = 0

        if invert is False:
            if slope != 0:
                left = findPointsAlongSlope(self.center, distance, slope, True)
                right = findPointsAlongSlope(self.center, distance, slope)
            else:
                if math.fabs(rise) > 0:
                    left = (self.center[0], self.center[1]-distance)
                    right = (self.center[0], self.center[1]+distance)
                else:
                    left = (self.center[0]-distance, self.center[1])
                    right = (self.center[0]+distance, self.center[1])
        else:
            if slope != 0:
                left = findPointsAlongSlope(self.center, distance, slope)
                right = findPointsAlongSlope(self.center, distance, slope, True)
            else:
                if math.fabs(rise) > 0:
                    left = (self.center[0], self.center[1]+distance)
                    right = (self.center[0], self.center[1]-distance)
                else:
                    left = (self.center[0]+distance, self.center[1])
                    right = (self.center[0]-distance, self.center[1])

        return left, right

    def findPointsAlongSlope(self, distance, slope, subtract=False):
        if subtract:
            x = (self.center[0] - (distance * math.sqrt(1 / (1 + slope ** 2))))
            y = (self.center[1] - ((slope * distance) * math.sqrt(1 / (1 + slope ** 2))))
        else:
            x = (self.center[0] + (distance * math.sqrt(1 / (1 + slope ** 2))))
            y = (self.center[1] + ((slope * distance) * math.sqrt(1 / (1 + slope ** 2))))

        return x, y

    def rotateAround(self, anchor, angle):
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        pX = int(self.center[0]) - int(self.center[0])
        pY = int(self.center[1]) - int(self.center[1])

        newX = (pX * c - pY * s) + anchor[0]
        newY = (pX * s + pY * c) + anchor[1]

        center = tupleToInt((newX, newY))

        return center

    def line_intersection(self, line1, line2, xmin=None, xmax=None, ymin=None, ymax=None):
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

    def draw(self, image, radius=1, rgb=(0, 255, 0), dtype=1):
        cv2.circle(image, self.center, radius, rgb, dtype)

def clamp(n, low, high):
    return max(min(high, n), low)