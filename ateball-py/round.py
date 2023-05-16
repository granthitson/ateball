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
import path
import utils
from constants import constants

logger = logging.getLogger("ateball.round")

class Round(threading.Thread):

    def __init__(self, ipc, data, *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.ipc = ipc

        self.regions = constants.regions

        # round constants
        self.table = data["table_data"]
        
        self.turn_num = data['turn_num']
        self.round_path = str(Path(data["path"], f"round-{self.turn_num}"))
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

        self.chosen_ball = None

        self.viable_paths = []
        self.unviable_paths = []

        self.clear_table = (0,0)
        self.can_pick_up = False

        self.round_cancel = threading.Event()
        self.round_complete = threading.Event()
        self.round_exception = threading.Event()
        self.round_over_event = utils.OrEvent(self.round_complete, self.round_cancel, self.round_exception)

        self.logger = logging.getLogger("ateball.round")

    def run(self):
        try:
            start_time = time.time()

            os.makedirs(self.round_path, exist_ok=True)

            self.logger.info(f"Turn #{self.turn_num}")
            self.ipc.send_message({
                "type" : "ROUND-START", 
                "data" : {
                    "turn_num" : self.turn_num
                }
            })

            self.images["table"] = self.images["game"].copy()[self.regions.table[1]:self.regions.table[1]+self.regions.table[3], self.regions.table[0]:self.regions.table[0]+self.regions.table[2]]
            self.images["targets_bot"] = self.images["game"].copy()[self.regions.targets_bot[1]:self.regions.targets_bot[1]+self.regions.targets_bot[3], self.regions.targets_bot[0]:self.regions.targets_bot[0]+self.regions.targets_bot[2]]
            self.images["targets_opponent"] = self.images["game"].copy()[self.regions.targets_opponent[1]:self.regions.targets_opponent[1]+self.regions.targets_opponent[3], self.regions.targets_opponent[0]:self.regions.targets_opponent[0]+self.regions.targets_opponent[2]]
            self.images["targets_pocketed"] = self.images["game"].copy()[self.regions.targets_pocketed[1]:self.regions.targets_pocketed[1]+self.regions.targets_pocketed[3], self.regions.targets_pocketed[0]:self.regions.targets_pocketed[0]+self.regions.targets_pocketed[2]]

            cv2.imwrite(str(Path(self.round_path, "round_start.png")), self.images["game"])
            cv2.imwrite(str(Path(self.round_path, "table.png")), self.images["table"])
            cv2.imwrite(str(Path(self.round_path, "targets_pocketed.png")), self.images["targets_pocketed"])
            cv2.imwrite(str(Path(self.round_path, "targets_bot.png")), self.images["targets_bot"])
            cv2.imwrite(str(Path(self.round_path, "targets_opponent.png")), self.images["targets_opponent"])

            self.round_over_event.notify(self.round_complete)
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.round_over_event.notify(self.round_exception)
        finally:
            self.logger.info("ROUND COMPLETE")
            self.ipc.send_message({"type" : "ROUND-COMPLETE"})


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
        os.makedirs(self.round_path + "\paths", exist_ok=True)
        os.makedirs(self.round_path + "\paths\\ball-hole", exist_ok=True)
        os.makedirs(self.round_path + "\paths\\ball-hole\\rotated", exist_ok=True)
        os.makedirs(self.round_path + "\paths\\cue-ball", exist_ok=True)
        os.makedirs(self.round_path + "\paths\\cue-ball\\rotated", exist_ok=True)

        viable, unviable = [], []
        for hole in self.hole_locations:
            for points in hole.points:
                for point in points:
                    p = path.Path(self.cueball, ball, hole, point, self.round_path)

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
            clean = cv2.imread(self.round_path + "pooltable.png", 1)

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

            clean = cv2.imread(self.round_path + "pooltable.png", 1)

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
            cv2.imwrite(f"{self.round_path}\paths\\ball-hole\\rotated\\{path.ball.name}-{path.hole.name}-{targetIndex}_rotated.png", result)

            cv2.imwrite(f"{self.round_path}\paths\\ball-hole\\{path.ball.name}-{path.hole.name}-{targetIndex}_path.png", clean)

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
        cv2.imwrite(self.round_path + "pooltableOutlined.png", self.round_image)

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
        
        pyautogui.screenshot(self.round_path + "confirmation.png",
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