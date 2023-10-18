import logging

import os
import sys
import time
import copy

import threading
import queue as q

import json

import math
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
        while not any([e.is_set() for e in self.events]) and not self.timeout.is_set():
            with self.condition:
                timed_out = not self.condition.wait(timeout)
                if timed_out:
                    self.timeout.set()

        return any([e.is_set() for e in self.events])

    def notify(self, event):
        if event in self.events:
            event.set()
            with self.condition:
                self.condition.notify_all()
        else:
            raise ValueError("event does not belong to OrEvent")

    def clear(self):
        for e in self.events:
            e.clear()
        self.timeout.clear()

    def is_set(self):
        return any([e.is_set() for e in self.events])

    def has_timed_out(self):
        return self.timeout.is_set()

class IPC:
    def __init__(self):
        self.incoming = q.Queue()
        self.outgoing = q.Queue()
        
        self.listen_event = threading.Event()
        self.listen_exception_event = threading.Event() 
        self.send_exception_event = threading.Event() 
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
                    sys.stdout.write(f"{json.dumps(msg, cls=IPCJSONEncoder)}\n")
                    sys.stdout.flush()
                except q.Empty:
                    pass
        except Exception as e: 
            self.send_exception_event.set()
            self.logger.exception(f"error handling ipc messages: {e}")
            self.logger.exception(msg)

    def send_message(self, msg):
        self.outgoing.put(msg)

    def quit(self):
        self.stop_event.set()

class IPCJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        try:
            # handle custom objects
            return obj.serialize()
        except Exception as e:
            logger.exception(f"could not encode object of type {type(obj)} - {obj}")
            return super(IPCJSONEncoder, self).default(obj)

class WindowCapturer(threading.Thread):
    def __init__(self, region, offset, fps, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hwnd = win32gui.FindWindow(None, os.getenv("APP_NAME"))
        if not self.hwnd:
            raise Exception("window not found")

        self.region = (region[2], region[3])
        self.offset = offset

        self.image = np.array([])

        self.exception = threading.Event()
        self.exit = threading.Event()
        self.stop_event = OrEvent(self.exception, self.exit)

        self.fps = fps
        self.last_tick = 0
        self.on_tick = threading.Event()
        self.on_tick_event = OrEvent(self.on_tick, self.exception, self.exit)

        self.logger = logging.getLogger("ateball.utils.WindowCapturer")

    def tick(self, fps):
        # synchronize loop with fps
        interval = 1 / fps 
        current_time = time.time()
        delta = current_time - self.last_tick

        if delta < interval:
            time.sleep(interval - delta)

        self.last_tick = time.time()
        self.on_tick.clear()

    def run(self):
        self.stop_event.clear()
        self.on_tick.clear()

        w, h = self.region

        self.hwnd = win32gui.FindWindow(None, os.getenv("APP_NAME"))
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        offset = (self.offset[0] + left, self.offset[1] + top)

        # can't capture hardware accelerated window
        desktop = win32gui.GetDesktopWindow()
        wDC = win32gui.GetWindowDC(desktop)

        # wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj=win32ui.CreateDCFromHandle(wDC)
        cDC=dcObj.CreateCompatibleDC()

        while not self.stop_event.is_set():
            try:
                # allow threads to retrieve latest
                self.tick(self.fps)

                dataBitMap = win32ui.CreateBitmap()
                dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
                cDC.SelectObject(dataBitMap)
                cDC.BitBlt((0,0), (w, h), dcObj, offset, win32con.SRCCOPY)

                signedIntsArray = dataBitMap.GetBitmapBits(True)
                img = np.frombuffer(signedIntsArray, dtype='uint8').reshape((h, w, 4))
                img.shape = (h, w, 4)

                img = img[...,:3]
                self.image = np.ascontiguousarray(img)
                self.on_tick_event.notify(self.on_tick)

                win32gui.DeleteObject(dataBitMap.GetHandle())
            except (win32ui.error, Exception) as e:
                self.logger.error(f"unable to capture window: {e}")
                self.on_tick_event.notify(self.exception)
                self.stop_event.notify(self.exception)
        
        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)

    def get(self):
        # get latest image added to stack
        self.on_tick_event.wait()
        return self.image if self.image.any() else np.array([])

    def stop(self):
        self.on_tick_event.notify(self.exit)
        self.stop_event.notify(self.exit)

class CV2Helper:
    logger = logging.getLogger("ateball.utils.CV2Helper")

    @staticmethod
    def image_path(filename):
        return os.path.join(os.getcwd(), "ateball-py", "images", filename)

    @staticmethod
    def imread(path, *args):
        image = cv2.imread(CV2Helper.image_path(path), *args)
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
        return skimage.color.deltaE_cie76(color1_lab, color2_lab)

    @staticmethod
    def resize(image, factor):
        width = int(image.shape[1] * factor)
        height = int(image.shape[0] * factor)
        dim = (width, height)

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def slice_image(image, region):
        # cut image to size (within confines of image)
        height, width = image.shape[0], image.shape[1]
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

class Point(object):
    def __init__(self, center):
        self.center = (center[0], center[1])

        self.logger = logging.getLogger("ateball.utils.Point")

    def __str__(self):
        return f"{self.center}"

    def __repr__(self):
        return f"Point({self.center})"

    def __hash__(self):
        return hash(self.__repr__())

    def __lt__(self, other):
        return self.center < other.center

    def __le__(self, other):
        return self.center <= other.center

    def __eq__(self, other):
        if isinstance(other, Point):
            return (self.center) == (other.center)
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __gt__(self, other):
        return self.center > other.center

    def __ge__(self, other):
        return self.center >= other.center

    def __add__(self, p):
        if isinstance(p, tuple):
            return Point((self.center[0] + p[0], self.center[1] + p[1]))
        elif isinstance(p, Point):
            return Point((self.center[0] + p.center[0], self.center[1] + p.center[1]))
        elif isinstance(p, int) or isinstance(p, float):
            return Point((self.center[0] + p, self.center[1] + p))

    def __sub__(self, p):
        if isinstance(p, tuple):
            return Point((self.center[0] - p[0], self.center[1] - p[1]))
        elif isinstance(p, Point):
            return Point((self.center[0] - p.center[0], self.center[1] - p.center[1]))
        elif isinstance(p, int) or isinstance(p, float):
            return Point((self.center[0] - p, self.center[1] - p))

    def __rmul__(self, p):
        if isinstance(p, tuple):
            return Point((self.center[0] * p[0], self.center[1] * p[1]))
        elif isinstance(p, Point):
            return Point((self.center[0] * p.center[0], self.center[1] * p.center[1]))
        elif isinstance(p, int) or isinstance(p, float):
            return Point((self.center[0] * p, self.center[1] * p))

    def __truediv__(self, p):
        if isinstance(p, tuple):
            return Point((self.center[0] / p[0], self.center[1] / p[1]))
        elif isinstance(p, Point):
            return Point((self.center[0] / p.center[0], self.center[1] / p.center[1]))
        elif isinstance(p, int) or isinstance(p, float):
            return Point((self.center[0] / p, self.center[1] / p))

    def serialize(self):
        return {
            "x" : self.center[0] if self.center[0] is not None else 0,
            "y" : self.center[1] if self.center[1] is not None else 0
        }

    def distance(self, p):
        return round(math.hypot(self.center[0] - p.center[0], self.center[1] - p.center[1]), 2)

    def average(self, p):
        return (int((self.center[0] + p.center[0])/2),int((self.center[1] + p.center[1])/2))

    def get_slope_to(self, p):
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

    def is_on_left(self, a, b):
        return (b.center[0] - a.center[0])*(self.center[1] - a.center[1]) - (b.center[1] - a.center[1])*(self.center[0] - a.center[0]) > 0

    def to_float(self):
        return (float(self.center[0]), float(self.center[1]))

    def to_int(self):
        return (int(self.center[0]), int(self.center[1]))

    def round(self, precision=2):
        return (round(self.center[0]), round(self.center[1]))

    def find_points_on_either_side(self, distance, rise, run):
        left = self.find_points_along_slope(-distance, rise, run)
        right = self.find_points_along_slope(distance, rise, run)

        return left, right

    def find_points_along_slope(self, distance, rise, run):
        try:
            slope = rise / run

            dy = ((slope * distance) * math.sqrt(1 / (1 + slope ** 2)))
            dx = (distance * math.sqrt(1 / (1 + slope ** 2)))
        except ZeroDivisionError:
            dy, dx = (0, distance) if rise == 0 else (distance, 0)

        return Point((self.center[0] + dx, self.center[1] + dy))

    def draw(self, image, radius=(1, 1), angle=0, angle_start=0, end_angle=360, bgr=(0, 255, 0)):
        cv2.ellipse(image, self.to_int(), radius, angle, angle_start, end_angle, bgr)

class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __str__(self):
        return f"{self.p1} to {self.p2}"

    def __repr__(self):
        return f"Line({self.p1} to {self.p2})"

    def serialize(self):
        return {
            "start" : self.p1.center,
            "end" : self.p2.center
        }

    def to_vector(self):
        radius = self.p1.distance(self.p2)
        angle = math.atan2((self.p2.center[1] - self.p1.center[1]), (self.p2.center[0] - self.p1.center[0])) * (180 / math.pi)
        return Vector(self.p1, radius, angle)

    def intersects_line(self, line):
        p = self.p1
        r = (self.p2-self.p1)

        q = line.p1
        s = (line.p2-line.p1)

        t = np.cross((q - p).center,s.center)/(np.cross(r.center,s.center))

        # This is the intersection point
        return p + t*r

    def dist(self, p3): # x3,y3 is the point
        x1, y1 = self.p1.center
        x2, y2 = self.p2.center
        x3, y3 = p3.center

        px = x2-x1
        py = y2-y1

        norm = px*px + py*py

        u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        # Note: If the actual distance does not matter,
        # if you only want to compare what this function
        # returns to other results of this function, you
        # can just return the squared distance instead
        # (i.e. remove the sqrt) to gain a little performance

        dist = (dx*dx + dy*dy)**.5

        return dist

    def draw(self, image, bgr):
        cv2.line(image, self.p1.to_int(), self.p2.to_int(), bgr, 1)

class Vector(object):
    def __init__(self, origin, radius, theta):
        self.origin = Point(origin.center)
        self.radius = radius
        self.theta = theta

        self.logger = logging.getLogger("ateball.utils.Vector")

    def __str__(self):
        return f"Vector({self.radius}, {self.theta})"

    def __repr__(self):
        return f"Vector({self.origin} - {self.theta}deg - {self.radius})"

    def serialize(self):
        return {
            "origin" : self.origin,
            "radius" : self.radius,
            "angle" : self.theta
        }

    def __add__(self, n):
        return Vector(copy.copy(self.origin) + n, self.radius, self.theta)

    def __sub__(self, n):
        return Vector(copy.copy(self.origin) - n, self.radius, self.theta)

    def draw(self, image, bgr=(0, 255, 0), thickness=2):
        radians = self.theta *  (math.pi / 180)
        vector = self.origin + ((self.radius * math.cos(radians)), (self.radius * math.sin(radians)))

        cv2.line(image, self.origin.to_int(), vector.to_int(), bgr, thickness)

def clamp(n, low, high):
    return max(min(high, n), low)