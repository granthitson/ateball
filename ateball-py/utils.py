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

import constants

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
        self.on_tick = threading.Condition()

        self.image_stack = []
        self.image_stack_limit = 20
        self.video_writer = None

        self.record_event = threading.Event()
        self.stop_event = threading.Event()

        self.logger = logging.getLogger("ateball.utils.WindowCapturer")

    def tick(self, fps):
        # synchronize loop with fps

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
            img = np.ascontiguousarray(img)

            # Free Resources
            dcObj.DeleteDC()
            cDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, wDC)
            win32gui.DeleteObject(dataBitMap.GetHandle())

            # keep track of last x image frames - set by image_stack_limit
            if len(self.image_stack) >= self.image_stack_limit:
                self.image_stack.pop(0)
            self.image_stack.append(img)

            # allow threads to retrieve latest
            with self.on_tick:
                self.on_tick.notify()

            if self.record_event.is_set():
                self.video_writer.write(img)
    
    def record(self, path, filename, format=cv2.VideoWriter_fourcc(*'XVID')):
        # write images to avi file
        self.video_writer = cv2.VideoWriter(str(Path(path, f"{filename}.avi")), format, self.fps, self.region)
        self.record_event.set()

    def get_first(self):
        # get latest image added to stack
        with self.on_tick:
            self.on_tick.wait()
            return self.image_stack[-1] if self.image_stack else np.array([])
    
    def get_last(self):
        # get oldest image added to stack
        with self.on_tick:
            self.on_tick.wait()
            return self.image_stack[0] if self.image_stack else np.array([])

    def stop(self):
        self.stop_event.set()
        if self.record_event.is_set():
            self.video_writer.release()
            self.record_event.clear()

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

    def get_closest_color(img, mask, lookup):
        # get mean bgr color value for both timers
        mean = cv2.mean(img, mask=mask)[:3]

        # match mean color to nearest color match
        closest_color = CV2Helper.get_color_name(mean, lookup)

        return closest_color

    def get_color_name(color, color_lookup):
        # match color to closest color listed in lookup

        # create list of differences for each lookup value
        differences = [
            [CV2Helper.color_difference(color, bgr), name]
            for name, bgr in color_lookup.items()
        ]

        # sort differences (ascending) - and return color with least difference
        differences.sort()
        return differences[0][1]
    
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
        # cut image to size
        return image[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]

    @staticmethod
    def create_mask(hsv, lower, upper, op=None, kernal=None):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if op is not None:
            mask = cv2.morphologyEx(mask, op, kernal)

        return mask

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
