import logging

import os
import sys
import time
import psutil
import subprocess

import glob
import shutil
import psutil

from enum import Enum
from enum import IntEnum

import threading
import queue as q
import janus
import itertools
import ctypes

import json
import websockets
from contextlib import suppress

import math
import pyautogui
import cv2

import pyppeteer
from fake_useragent import UserAgent

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

class TaskScheduler:
    def __init__(self):
        self.task_heap = [] #heap
        self.task_map = {}

        self.counter = itertools.count()
        self.REMOVED = "<removed-task>"
        
        self.active_task = None

        self.stop_event = threading.Event()

        self.exception = None

        self.logger = logging.getLogger("ateball.utils.taskscheduler")

    def add_task(self, task, priority):
        'Add a new task or update the priority of an existing task'
        if task in self.task_map:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.task_map[task] = entry

        q.heappush(self.task_heap, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.task_map.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.task_heap:
            priority, count, task = q.heappop(self.task_heap)
            if task is not self.REMOVED:
                del self.task_map[task]
                return task
        raise KeyError('pop from an empty priority queue')

    # def cancel_task(self):
    #     self.logger.debug("cancelling task")
    #     thread_id = self.active_task.ident
    #     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
    #     if res > 1:
    #         ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
    #         self.active_task = None
    #         self.logger.debug("task cancelled")

    def start(self):
        self.logger.info("Waiting to process task")

        while not self.stop_event.is_set():
            try:
                task = self.pop_task()

                self.logger.debug(f"task starting")
                self.active_task = threading.Thread(target=task, daemon=True)
                self.active_task.start()
                self.active_task.join()
                    
                self.logger.debug("task complete")
            except KeyError as e:
                pass
            except Exception as e:
                self.logger.error(f"Error processing task - {type(e).__name__} - {e}")

    def shutdown(self):
        self.stop_event.set()
    
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
            self.logger.info("sending message: {msg}")
            while not self.stop_event.is_set():
                try:
                    msg = self.outgoing.get()
                    sys.stdout.write(msg)
                except q.Empty:
                    pass
        except Exception as e:
            self.start_exception_event.set()
            self.logger.exception(f"error handling ipc messages: {e}")

    def quit(self):
        self.stop_event.set()

# class WebSocketServer:
#     def __init__(self, port):
#         self.server = None
#         self.port = port

#         self.instance_id = None

#         self.outgoing = asyncio.Queue() # outgoing msg from ateball
#         self.inc_processing = asyncio.Queue() # incoming msg
#         self.inc_processed = asyncio.Queue() # incoming response msg
        
#         self.connection = None # connected client
#         self.message_handler = None

#         self.start_event = asyncio.Event()
#         self.start_exception_event = asyncio.Event() 
#         self.start = OrEvent(self.start_event, self.start_exception_event)

#         self.connected_event = asyncio.Event()

#         self.stop_event = asyncio.Event()
#         self.disconnect_event = asyncio.Event()

#         self.quit_event = asyncio.Event()

#         self.exception = None

#         self.logger = logging.getLogger("ateball.utils.wsserver")

#     async def serve(self):
#         try:
#             self.logger.info("Starting websocket server...")
            
#             self.server = await websockets.serve(self.handler, "localhost", self.port, reuse_address=True)
#             self.logger.info("Websocket server listening...")
#             self.start.notify(self.start_event)
#         except Exception as e:
#             # if type(e) != asyncio.CancelledError:
#             self.start.notify(self.start_exception_event)
#             self.logger.error(f"error starting websocket server: {e}")
#             self.exception = WSServerSetupError("error setting up ws server")

#     async def handler(self, websocket):  
#         try:
#             if self.connection and not self.connection.closed:
#                 self.logger.warning("attempted connection - still communicating with existing connection")
#                 await self.connection.close(code=1011, reason="connection denied") 

#             self.logger.debug("CLIENT CONNECTED")
            
#             self.connection = websocket
#             self.connected_event.set()
#             self.disconnect_event.clear()

#             self.message_handler = MessageHandler(self.connection, self.outgoing, self.inc_processing, self.inc_processed, self.disconnect_event)

#             while self.connection.open and (not self.stop_event.is_set() and not self.disconnect_event.is_set()):
#                 # process whichever comes first, incoming msg or outgoing msg
#                 incoming = asyncio.create_task(self.message_handler.receiveIncomingMsg())
#                 outgoing = asyncio.create_task(self.message_handler.receiveOutgoingMsg())
#                 send = asyncio.create_task(self.message_handler.sendOutgoingMsg())
#                 done, pending = await asyncio.wait(
#                     [incoming, outgoing, send],
#                     return_when=asyncio.FIRST_COMPLETED,
#                 )

#                 if incoming in done:
#                     msg = incoming.result()
#                     if msg:
#                         await self.message_handler.processIncoming(msg)
#                 else:
#                     incoming.cancel()
                
#                 if outgoing in done:
#                     msg = outgoing.result()
#                     await self.message_handler.processOutgoing(msg)
#                 else:
#                     outgoing.cancel()

#                 if send in done:
#                     msg = send.result()
#                     await self.message_handler.processOutgoing(msg)
#                 else:
#                     send.cancel()
#         except Exception as e:
#             self.logger.exception(f"error handling messages: {e}")
#         finally:
#             if self.connection and self.connection.closed:
#                 self.connection = None
#             # await self.inc_processing.put({"action" : "cancel"})

#     async def shutdown(self):
#         self.stop_event.set()

#         if self.server is not None:
#             self.server.close()
#             await self.server.wait_closed()

#         self.quit_event.set()
#         self.logger.debug("Websocket server shutdown")

class ResponseType(int, Enum):
    INIT: int = 0
    INFO: int = 1
    PROMPT: int = 2
class ResponseAction(int, Enum):
    STATUS: int = 0
    PLAY: int = 1
    BUSY: int = 2
class ResponseStatus(int, Enum):
    SUCCESS: int = 1
    FAILED: int = 2

# class MessageHandler:
#     def __init__(self, websocket, outgoing, processing_queue, processed_queue, disconnect_event):
#         self.websocket = websocket

#         #outgoing
#         self.send = outgoing

#         #incoming
#         self.incoming = asyncio.Queue() #incoming msgs from browser
#         self.processing = processing_queue #pending
#         self.processed = processed_queue #processed
#         self.outgoing = asyncio.Queue() #outgoing msgs to browser

#         self.disconnect_event = disconnect_event

#         self.logger = logging.getLogger("ateball.utils.wsserver.handler")

#     async def receiveIncomingMsg(self):
#         try:
#             msg = json.loads(await self.websocket.recv())
#         except websockets.exceptions.ConnectionClosedOK as e:
#             self.logger.debug("closing connection - ok")
#             self.disconnect_event.set()
#             self.connected_event.clear()
#         except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as e:
#             self.logger.debug("closing connection - error")
#             self.disconnect_event.set()
#             self.connected_event.clear()
#         else:
#             return msg

#     async def processIncoming(self, msg):
#         try:
#             await self.processing.put(msg)

#             response = await self.processed.get()
#             if (response):
#                 await self.outgoing.put(response)
#         except websockets.exceptions.ConnectionClosedOK as e:
#             self.logger.debug("closing connection - ok")
#             self.disconnect_event.set()
#             self.connected_event.clear()
#         except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as e:
#             self.logger.debug("closing connection - error")
#             self.disconnect_event.set()
#             self.connected_event.clear()

#     async def receiveOutgoingMsg(self):
#         try:
#             msg = await self.outgoing.get()
#         except websockets.exceptions.ConnectionClosedOK as e:
#             self.logger.debug("closing connection - ok")
#             self.disconnect_event.set()
#             self.connected_event.clear()
#         except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as e:
#             self.logger.debug("closing connection - error")
#             self.disconnect_event.set()
#             self.connected_event.clear()
#         else:
#             return msg

#     async def processOutgoing(self, msg):
#         try:
#             await self.websocket.send(json.dumps(msg))
#             self.logger.debug(f"outgoing msg sent - {msg}")
#         except websockets.exceptions.ConnectionClosedOK as e:
#             self.logger.debug("closing connection - ok")
#             self.disconnect_event.set()
#             self.connected_event.clear()
#         except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as e:
#             self.logger.debug("closing connection - error")
#             self.disconnect_event.set()
#             self.connected_event.clear()

#     async def sendOutgoingMsg(self):
#         try:
#             msg = await self.send.get()
#         except websockets.exceptions.ConnectionClosedOK as e:
#             self.logger.debug("closing connection - ok")
#             self.disconnect_event.set()
#             self.connected_event.clear()
#         except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as e:
#             self.logger.debug("closing connection - error")
#             self.disconnect_event.set()
#             self.connected_event.clear()
#         else:
#             return msg
    
class Wall():
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
        return os.path.join("images", filename)

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

class Misc:
    @staticmethod
    def hasMetEndCondition(end_conditions):
        return any([ec() for ec in end_conditions])